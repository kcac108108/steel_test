"""
규격 텍스트에서 사이즈 추출 서비스

추출 순서:
  1. 정확 매칭 (확정 데이터 사전)
  2. 정규식 기반 추출 (매뉴얼 규칙 적용)
  3. LLM 폴백
"""

import re
import json
import time
from pathlib import Path
from typing import Optional
from openai import OpenAI, RateLimitError
import pandas as pd
from app.core.config import settings


def _build_exact_lookup(confirmed_dir: str = "confirmed") -> dict[str, str]:
    """확정 파일 전체에서 규격→사이즈 정확 매칭 사전 구축"""
    files = sorted(Path(confirmed_dir).glob("**/*.xlsx"))
    if not files:
        return {}

    dfs = []
    for f in files:
        try:
            df = pd.read_excel(f, dtype=str)
            if len(df.columns) < 5:
                continue
            cols = df.columns.tolist()
            # 첫 컬럼이 숫자(번호)면 규격=2, 아니면 규격=1
            sample = str(df.iloc[0, 0]) if len(df) > 0 else ""
            if sample.strip().replace(".", "").isdigit():
                spec_col, size_col = 2, 4
            else:
                spec_col, size_col = 1, 4
            sub = df[[cols[spec_col], cols[size_col]]].copy()
            sub.columns = ["spec", "size"]
            dfs.append(sub)
        except Exception:
            continue

    if not dfs:
        return {}

    all_data = pd.concat(dfs, ignore_index=True)
    has_size = (
        all_data["size"].notna()
        & (all_data["size"].str.strip() != "")
        & (~all_data["size"].str.strip().isin(["0", "0.0"]))
    )
    lookup = all_data[has_size].copy()
    lookup["spec_key"] = lookup["spec"].str.strip().str.upper()
    lookup["size_val"] = lookup["size"].str.strip()
    return lookup.drop_duplicates("spec_key", keep="last").set_index("spec_key")["size_val"].to_dict()


# 모듈 로드 시 사전 구축 (한 번만)
_EXACT_LOOKUP: dict[str, str] = {}


def _get_exact_lookup() -> dict[str, str]:
    global _EXACT_LOOKUP
    if not _EXACT_LOOKUP:
        print("  [사이즈] 확정 데이터 사전 로드 중...")
        _EXACT_LOOKUP = _build_exact_lookup()
        print(f"  [사이즈] 사전 {len(_EXACT_LOOKUP):,}건 준비 완료")
    return _EXACT_LOOKUP


_SIZE_LLM_PROMPT = """규격품명 텍스트에서 사이즈만 추출하세요.

규칙:
1. 숫자 간 연결은 X(대문자)로 연결
2. MM 단위는 숫자만 남김 (12.5MM → 12.5)
3. M/MTR/METER → ×1000 변환 (2M → 2000)
4. CM → ×10 변환
5. INCH("), IN → 숫자 뒤에 IN 붙임 (2" → 2IN)
6. FEET('), FT → 숫자 뒤에 FT 붙임
7. Coil/COIL → C (맨 뒤에 표기)
8. OD 값은 앞에 D 붙임, ID는 무시
9. NPS, SCH, STD, XXS 등 특수단위는 그대로 유지
10. 사이즈가 명확하지 않으면 null 반환
11. 숫자 정렬은 오름차순 (단, SIZE: 키워드 뒤 또는 T/W/L 명시 시 그 순서 유지)

반드시 JSON 형식으로만 응답하세요.
JSON 응답 형식 (사이즈 있음): {"size": "12.5X1524X3000"}
JSON 응답 형식 (판단불가): {"size": null}"""


def _normalize(s: str) -> str:
    s = s.upper().strip()
    # 전각 콜론 → 반각: THICKNESS：11.1MM → THICKNESS:11.1MM
    s = s.replace('：', ':')
    # 언더스코어를 공백으로: OD:1.125IN_L:8IN → OD:1.125IN L:8IN (레이블 구분자)
    s = s.replace('_', ' ')
    # Ø/⌀/Φ (직경 기호) → DIA
    s = s.replace('Ø', 'DIA').replace('⌀', 'DIA').replace('Φ', 'DIA').replace('φ', 'DIA')
    # × (Unicode 곱셈 기호 U+00D7) → X (치수 구분자 통일)
    s = s.replace('×', 'X')
    # ㎜ (Unicode 밀리미터 U+339C) → MM (단위 통일)
    s = s.replace('㎜', 'MM')
    # W/L 레이블 뒤 천단위 쉼표 제거 (European decimal 변환 전에 먼저 처리)
    # W 1,255MM → W 1255MM, L 4,620MM → L 4620MM (폭/길이는 천단위 구분자)
    s = re.sub(r'\b(W|L)\s+(\d{1,4}),(\d{3})(MM|CM|IN|FT)\b', r'\1 \2\3\4', s, flags=re.IGNORECASE)
    # PLATE/SHEET 맥락 1자리+콤마+3자리: 천단위 구분자 (먼저 처리, 유럽식 소수점 규칙 전에)
    # 예: 1,277MM X → 1277MM (판재 폭/길이, 단 비-PLATE 맥락은 아래 유럽식 소수점 규칙으로)
    if re.search(r'\bPLATES?\b', s, re.IGNORECASE):
        s = re.sub(r'\b([1-9]),(\d{3})(MM)\b', r'\1\2\3', s, flags=re.IGNORECASE)
    # 유럽식 소수점 쉼표: 3,100MM X → 3.100MM X (뒤에 X가 있을 때만 변환, 천단위 콤마와 구분)
    # L:4,825MM처럼 단독으로 끝나는 경우는 천단위 콤마로 간주
    # 반드시 천단위 콤마 제거 전에 처리해야 함
    s = re.sub(r'\b(\d{1,2}),(\d{3})(MM)(?=\s*[Xx])', r'\1.\2\3', s)
    # 천단위 콤마 제거: 6,000 → 6000 (알파벳/숫자 직후 콤마는 구분자로 보존)
    s = re.sub(r'(?<![A-Z\d])(\d{1,3}),(\d{3})(?!\d)', r'\1\2', s)
    # X 뒤 천단위 콤마 제거: 120MMX6,700MM → 120MMX6700MM
    s = re.sub(r'(?<=[Xx])(\d{1,3}),(\d{3})(?!\d)', r'\1\2', s)
    # AMS 공차 코드 제거: 0.079T( → '' [AMS5599-0.079T(0.79"X48"X144") 공차 접두코드]
    s = re.sub(r'(?<![A-Z\d])\d+\.\d+T(?=\()', '', s, flags=re.IGNORECASE)
    # 두께/직경 접미사 T/D 제거: 0.4T → 0.4, 48.6D → 48.6, 6T → 6 (정수 포함)
    # 단, 알파벳/하이픈 직후 T는 강종코드이므로 제외 (S45C-T, 3/4H-T 등)
    s = re.sub(r'(\d+\.\d+)[TD](?=[X\s,*/]|$)', r'\1', s)
    s = re.sub(r'(?<![A-Z\d\-])(\d+)T(?=[X\s,*/]|$)', r'\1', s)
    # MML(길이 단위 MM+L) → MM 통일: 4020MML → 4020MM
    s = re.sub(r'MML\b', 'MM', s, flags=re.IGNORECASE)
    # MMM 이상 중복 M → MM 통일: 3.15MMM → 3.15MM
    s = re.sub(r'M{3,}', 'MM', s, flags=re.IGNORECASE)
    # 소수점 주변 공백 제거: 19. 0MM → 19.0MM (스캔/OCR 오류 교정)
    s = re.sub(r'(\d)\. (\d)', r'\1.\2', s)
    # 선행 점 소수 보정: .063 → 0.063 (알파벳/숫자 직후 점은 구분자이므로 제외)
    s = re.sub(r'(?<![A-Z\d.])\.(\d)', r'0.\1', s)
    # X 뒤 선행 점 소수 보정: X.356 → X0.356 (치수 구분자 X 뒤 알파벳 직후 점)
    s = re.sub(r'(?<=[Xx])\.(\d)', r'0.\1', s)
    # 숫자 직후 M 단위가 키워드로 이어지면 공백 삽입: 6.0MGRADE → 6.0M GRADE
    s = re.sub(r'(?<=\d)(M)(?=GRADE|SPEC|SHAPE|MODEL|TEMPER|HEAT|CERT)', r'\1 ', s, flags=re.IGNORECASE)
    # 치수-MT: 분리: 237MT:STAINLESS → 237 MT:STAINLESS (키워드 strip이 정확히 동작하도록)
    s = re.sub(r'(\d)(MT)\s*:', r'\1 \2:', s, flags=re.IGNORECASE)
    # COILat 등 COIL 뒤에 의미없는 알파 분리: COILat → COIL AT
    s = re.sub(r'\b(COIL)([A-BD-Z][A-Z]{1,3})\b', r'\1 \2', s, flags=re.IGNORECASE)
    # PIPE{alpha} 분리: PIPEOUTER → PIPE OUTER, PIPESA335 → PIPE SA335 (단어 경계 없이 붙은 키워드)
    s = re.sub(r'\bPIPE(?=[A-Z])', 'PIPE ', s, flags=re.IGNORECASE)
    # ROPE{강종코드} 분리: ROPESUS304 → ROPE SUS304 (와이어로프 뒤 강종코드가 붙어있는 경우)
    s = re.sub(r'\b(ROPE)([A-Z]{2,5}\d[A-Z0-9]*)', r'\1 \2', s, flags=re.IGNORECASE)
    # AISI{grade}{구성코드} 분리: AISI3167X7 → AISI316 7X7 (강종코드+구성코드 연결)
    s = re.sub(r'\b(AISI\d{3,4})(\d+[Xx]\d+)', r'\1 \2', s, flags=re.IGNORECASE)
    # 숫자 뒤 MM+순수알파 분리: 5.80MMSAMPLE → 5.80MM SAMPLE (단위 뒤 알파 접미사만 분리)
    # IN은 제외 (INCONEL, INNER 등 오분리 방지), 숫자 뒤 MM에만 적용
    s = re.sub(r'(\d)(MM)([A-Z]{3,})(?![A-Z]*\d)', r'\1\2 \3', s, flags=re.IGNORECASE)
    # {n}DMM → OD{n}MM (직경 단위 DMM: 530 DMM → OD530MM, 530DMMSIZE → OD530MM SIZE)
    s = re.sub(r'\b(\d+)\s*DMM(?!\d)', r'OD\1MM ', s, flags=re.IGNORECASE)
    # {n}KG 무게 표기 제거: 15KG, 250KG, X250KG/DRUM, 600KG*2 (KG/M 선밀도는 제외)
    s = re.sub(r'[Xx]?\s*\d+\.?\d*\s*KG\b(?![\s/]*M\b)(?:\s*[*/]\s*\d+)?', '', s, flags=re.IGNORECASE)
    # 치수 표기에서 앞에 0이 붙은 2자리 숫자: 022 → 0.22 (X 앞뒤에 오는 경우)
    # 단, 첫 자리가 0인 경우(001~009)는 lot번호로 간주하여 변환 안 함
    # 단, 하이픈 뒤에 오는 경우(부품번호 일부)는 변환 안 함: -014 → -014 (not -0.14)
    s = re.sub(r'(?<![.\d\-])0([1-9]\d)(?=[Xx\s])', r'0.\1', s)
    # NO.N 표면처리 등급 제거: NO.1/2B FINISH, NO.1 MILL EDGE (FINISH 뒤에 있는 경우)
    # 또는 NO.1/NO.2B 단독 (스테인리스 표면처리 등급만, 한정적)
    s = re.sub(r'\bNO\.?\s*(?:1[A-D]?|2[BD]?)\b(?:\s+(?:FINISH|POLISH|MILL\s*EDGE))?', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\bNO\.?\s*(?:4|6|7|8)\b\s+(?:FINISH|POLISH|MILL\s*EDGE)\b', '', s, flags=re.IGNORECASE)
    # (FINISH N) 괄호 내 표면처리 코드 제거: (FINISH 1), (FINISH 2B) 등
    s = re.sub(r'\(\s*FINISH\s*\d+\w?\s*\)', '', s, flags=re.IGNORECASE)
    # {n}DIA 접미사 제거 (숫자 바로 뒤에 붙은 DIA만): 76.200DIA → 76.200
    s = re.sub(r'([\d])DIA\b(?!\s*[.:=])', r'\1', s, flags=re.IGNORECASE)
    # 선두 리스트 번호 제거: "80)41.20MM..." → "41.20MM..." (인벤토리 목록 순번)
    s = re.sub(r'^\d+\)\s*', '', s)
    # 8자리 이상 카탈로그/부품번호 제거 (선두 위치만): 62590271 0023,OIL PIPE → 0023,OIL PIPE
    s = re.sub(r'^\d{8,}\s*', '', s.strip())
    # 0{n}-0{n} 선두 로트범위 코드 제거: 0202-001696 → '' (선두 0 포함 로트번호 범위)
    s = re.sub(r'^0\d{2,3}-0\d{4,6}\b\s*', '', s.strip())
    # WIRE{n} 시리즈 번호 제거: WIRE1 ER308 → WIRE ER308 (단독 숫자만, 2자리 이상 유지)
    # 단, 소수점 뒤에 오는 경우 제외: WIRE2.0MM (실제 치수) → 그대로 유지
    s = re.sub(r'\bWIRE(\d)\b(?!\.\d)', 'WIRE', s, flags=re.IGNORECASE)
    # 선두 수량 + ER 용접봉 코드: 2 ER308 → ER308 (수량이 치수로 잡히는 방지)
    s = re.sub(r'^\d{1,2}\s+(?=ER\d{3})', '', s.strip())
    # SAE1018 강종코드 + 치수 연결: SAE10189.0MM → SAE1018 9.0MM (숫자 분리)
    s = re.sub(r'\b(SAE\d{4})(\d)', r'\1 \2', s, flags=re.IGNORECASE)
    # 탄소강 강종 범위 제거: C1010~1020 → '' (공백으로 둘러싸인 경우만)
    s = re.sub(r'\bC\d{4}~\d{4}\b', '', s, flags=re.IGNORECASE)
    # 유럽식 소수점 (1~2자리 뒤): 57,15MM → 57.15MM (3자리는 천단위로 처리)
    s = re.sub(r'\b(\d{1,4}),(\d{1,2})(MM|CM|IN|FT)\b', r'\1.\2\3', s, flags=re.IGNORECASE)
    # 단위 직후 알파코드 분리: 300MMHWT020626 → 300MM HWT020626 (그레이드코드 제거 위해)
    s = re.sub(r'(MM|IN|FT|CM)([A-Z]{2,5}\d{3,})', r'\1 \2', s, flags=re.IGNORECASE)
    # 단위 직후 ABOUT 분리: 2.30MMABOUT → 2.30MM ABOUT
    s = re.sub(r'(MM|IN|FT|CM)(ABOUT)\b', r'\1 \2', s, flags=re.IGNORECASE)
    # SEAMLESS{grade} 연결된 스텐인리스 등급 분리: SEAMLESS316 → SEAMLESS (316은 그레이드)
    s = re.sub(r'\bSEAMLESS(\d{3}[A-Z]?\b)', r'SEAMLESS ', s, flags=re.IGNORECASE)
    # S{0.0xx} 두께 공차 접두코드 완전 제거: S0.050, S0.080 → '' (tolerance prefix, 치수 아님)
    s = re.sub(r'(?<!\w)S(0\.\d+)\b', '', s, flags=re.IGNORECASE)
    # {a.bc}/{d.ef} X {w} X {l} 공차범위: 두번째 값이 실 사양 → 0.48/0.052 X 36 X 120 → 0.052 X 36 X 120
    s = re.sub(r'(?<!\w)(\d+\.\d+)/(\d+\.\d+)(\s*[Xx]\s*[\d.]+\s*[Xx]\s*[\d.]+)', r'\2\3', s)
    # IN EXTERNAL DIAMETER 구문 제거: "20 MM in external diameter" → "20 MM" (전치사 IN이 INCH로 오인식 방지)
    s = re.sub(r'\bIN\s+EXTERNAL\s+DIAMETER\b', '', s, flags=re.IGNORECASE)
    # FM{n} 마감코드 제거: FM 3, FM3 → '' (COLD ROLLED 마감, 소수점 앞은 치수이므로 제외)
    s = re.sub(r'\bFM\s*\d{1,2}(?!\.\d)\b', '', s, flags=re.IGNORECASE)
    # SC H\d → SCH\d (타이핑 오류로 공백 삽입된 SCH 복원: SC H40 → SCH40)
    s = re.sub(r'\bSC\s+H(\d)', r'SCH\1', s, flags=re.IGNORECASE)
    # 복합분수 인치 표기: 1 5/8" → 1-5/8IN, 1 1/4" → 1-1/4IN (정수+분수 복합 인치)
    s = re.sub(r'\b(\d)\s+(\d+/\d+)\s*"', r'\1-\2IN', s)
    # 파이프 공칭인치+스케줄: 6"*SCH40, 6" *STD, 12"*SCH20 → 6IN *SCH40 (SCH/STD/XS 앞의 " 보존)
    s = re.sub(r'(\d+(?:\.\d+)?)\s*"\s*(?=\*?\s*(?:SCH|STD|XS|XXS))', r'\1IN ', s, flags=re.IGNORECASE)
    # PIPE/TUBE 맥락 인치 OD: 5"*8T*3000 → 5IN*8*3000 (파이프 인치OD × 두께 × 길이)
    if re.search(r'\b(?:PIPE|TUBE)\b', s, re.IGNORECASE):
        s = re.sub(r'(\d+(?:\.\d+)?)"(?=\s*\*)', r'\1IN', s)
    # 구조재 폭 표기: BOOM 133"*4.5 형태에서 " 뒤에 * 가 오면 치수 구분자 → IN 변환 안함
    s = re.sub(r'(\d+)"\s*(?=\*)', r'\1 ', s)
    # PIPE 맥락 두께+길이 연결 표기: 12.76000 → 12.7 6000 (소수 1자리 + 4자리 길이)
    if re.search(r'\bPIPE\b', s, re.IGNORECASE):
        s = re.sub(r'\b(\d+\.\d)([456789]\d{3})\b', r'\1 \2', s)
    # PIPE/TUBE 맥락 T접두사 두께 제거: DIA38.1 X T3 X 1750 → DIA38.1 X 3 X 1750
    if re.search(r'\b(?:PIPE|TUBE|TUBING)\b', s, re.IGNORECASE):
        s = re.sub(r'\bT(\d+\.?\d*)\b(?=\s*[Xx*]|\s*MM\b)', r'\1', s, flags=re.IGNORECASE)
    # SHEET 맥락 유럽식 점-천단위: X.YYZ MM (마지막 자리 ≠ 0) → XYYY MM
    # 예: 1.219MM → 1219MM, 2.438MM → 2438MM (단, 8.800MM 등 trailing 0은 소수점으로 유지)
    if re.search(r'\bSHEET\b', s, re.IGNORECASE):
        s = re.sub(r'\b([1-9])\.(\d{2}[1-9])\s*(MM)\b', r'\1\2\3', s, flags=re.IGNORECASE)
    # CERT NO:XXXXX 인증번호 제거: CERT NO:N26054A64, CERT NO:JSL-JRD/QA/2025-26 → '' (슬래시 포함)
    s = re.sub(r'\bCERT\.?\s*NO\.?\s*[:#]\s*[\w/.-]+', '', s, flags=re.IGNORECASE)
    # X NNNN(CUT) 절단 길이 제거: X 2950(CUT) → ' ' (파이프 절단 길이는 치수 아님, 공백으로 분리)
    s = re.sub(r'\s*[Xx]\s*\d+\s*\(CUT\)\b', ' ', s, flags=re.IGNORECASE)
    # 중복 코일 스펙 트런케이션: N*NMM*C, N*NMM*C, ... → N*NMM*C (첫번째만 유지)
    # 예: 0.15*255MM*C, 0.15*210MM*C, COLD ROLLED-COIL → 0.15*255MM*C
    s = re.sub(r'([\d.]+\s*[*Xx]\s*[\d.]+\s*MM\s*[*Xx]\s*C)\s*,.*', r'\1', s, flags=re.IGNORECASE)
    # LENGTH/LG 천단위 쉼표 제거 후 L로 정규화: LENGTH4,870MM → L4870MM (_extract_twl 인식용)
    s = re.sub(r'\bLENGTH\s*(\d+),(\d{3})(?=\D|$)', r'L\1\2', s, flags=re.IGNORECASE)
    s = re.sub(r'\bLENGTH\s*:?\s*(?=\d)', 'L', s, flags=re.IGNORECASE)
    # LG 레이블 → L 정규화: LG4000MM → L4000MM (한국/일본 강재 치수 표기, _extract_twl 인식용)
    s = re.sub(r'\bLG\s*:?\s*(?=\d)', 'L', s, flags=re.IGNORECASE)
    # INNER DIAMETER/DIA 제거: 외경+내경 동시 표기 시 내경은 치수 불필요 (외경+두께가 정답)
    s = re.sub(r'\bINNER\s+(?:DIAMETER|DIA)\s*:?\s*[\d.]+\s*MM\b', '', s, flags=re.IGNORECASE)
    # OD 맥락에서 ID/CHD(내경)/R(반경)/H(공차등급) 제거: OD14 ID4.2, CHD:4.20MM, R13.000MM, H8
    # OD13.3처럼 OD 직후 숫자가 붙어도 인식 (word boundary 없이)
    if re.search(r'\bOD(?:\b|(?=[\d.]))', s, re.IGNORECASE):
        s = re.sub(r'\b(?:ID|CHD)\s*:?\s*[\d.]+\s*(?:MM|CM|IN(?:CH)?)?\b', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\bR\s*[\d,]+(?:\.\d+)?\s*(?:MM|CM|IN)?\b', '', s, flags=re.IGNORECASE)  # R13,000/R13.000MM 반경
        s = re.sub(r'\bH\d{1,2}\b', '', s, flags=re.IGNORECASE)                              # H8 공차등급
    # THICKNESS 키워드 → T 정규화: THICKNESS:11.1MM → T11.1MM (_extract_twl 인식용)
    s = re.sub(r'\bTHICKNESS\s*:?\s*(?=\d)', 'T', s, flags=re.IGNORECASE)
    # PIPE 맥락에서 ,L{n}MM,T{n}MM → ,T{n}MM,L{n}MM 순서 스왑 (표준: OD×T×L)
    if re.search(r'\bPIPE\b', s, re.IGNORECASE):
        s = re.sub(r'(,+\s*)(L\s*[\d.]+\s*MM\b)(\s*,+\s*)(T\s*[\d.]+\s*MM\b)', r'\1\4\3\2', s, flags=re.IGNORECASE)
    # DIAMETER 키워드 → OD 정규화: DIAMETER :60MM → OD60MM, BARDIAMETER :60 → BAR OD60
    # 공백 삽입으로 word boundary 확보 (BAROD → BAR OD)
    s = re.sub(r'DIAMETER\s*:?\s*(?=\d)', ' OD', s, flags=re.IGNORECASE)
    # WIDTH 키워드 → W 정규화: WIDTH:38MM → W38MM (_extract_twl 인식용)
    s = re.sub(r'\bWIDTH\s*:?\s*(?=\d)', 'W', s, flags=re.IGNORECASE)
    # F{n}X{n}X{n}-{suffix} 부품형번 제거: F12X125X170-L, F12X125X170-2 → '' (실치수는 PLATE 뒤 별도 표기)
    s = re.sub(r'\bF(\d+(?:[Xx]\d+)+)-(?:[A-Z]\d*|\d+)\b', '', s, flags=re.IGNORECASE)
    # USD 가격 괄호 제거: (USD8,859.20), (FOB CHARGE:USD413.14) → '' (단가/운임 치수 오추출 방지)
    s = re.sub(r'\((?:FOB\s+CHARGE\s*:?\s*)?USD[\d,.]+\)', '', s, flags=re.IGNORECASE)
    # AM{10+자리} 제품코드 제거: AM913202004179020 → '' (코일/강재 제품번호, 치수 오추출 방지)
    s = re.sub(r'\bAM\d{10,}\b', '', s, flags=re.IGNORECASE)
    # T{n}MM X {w}MM X L COIL → W 레이블 추가: _extract_twl이 trailing zero 보존하며 추출
    # (COIL trailing zero strip 코드 통과 전에 TWL 경로로 유도)
    s = re.sub(r'\bT([\d.]+)(MM)\s*[Xx]\s*([\d.]+)(MM)\s*[Xx]\s*L\s+COIL\b',
               r'T\1\2 X W\3\4 X L COIL', s, flags=re.IGNORECASE)
    # T{n}MMXW/XWT/XL 공백 삽입: XW1420→X W1420, XWT2.11→X WT2.11, XL4000→X L4000 (_extract_twl 인식용)
    # (?<![A-Z]): 알파벳 직후 X는 제외 (MAXW 등 단어 내부 X 보호)
    s = re.sub(r'(?<![A-Z])X(WT|[WL])(?=[\d.])', r'X \1', s, flags=re.IGNORECASE)
    # ALUSI 열간성형 코드 제거: HF-950-1300-MNB-S, TM-2014, REV6 (치수 아닌 재료 규격 코드)
    s = re.sub(r'\bHF-\d+-\d+(?:-\w+)*\b', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\bTM-\d{4}\b', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\bREV\s*\d+\b', '', s, flags=re.IGNORECASE)
    # AYHS 강종코드 뒤 숫자 분리: AYHS5 → AYHS (5가 치수로 오인되는 방지)
    s = re.sub(r'\bAYHS(\d+)\b', 'AYHS', s, flags=re.IGNORECASE)
    # KD/SKD 강종코드 제거: KD61, KD11, SKD11, SKD61 등 (금형강/공구강 코드, 치수로 오인 방지)
    s = re.sub(r'\bS?KD\d{1,4}\b', '', s, flags=re.IGNORECASE)
    # C/S NO.: 케이스 번호 제거: C/S NO.:2025001131 → ''
    s = re.sub(r'\bC/S\s+NO\.?\s*:?\s*\S+', '', s, flags=re.IGNORECASE)
    # ORDER NO: 주문번호 이후 제거: ORDER NO: TS25110007 → 제거
    s = re.sub(r'\bORDER\s+NO\s*:\s*\S+', '', s, flags=re.IGNORECASE)
    # BARSIZE: → BAR SIZE: (공백 없이 연결된 SIZE: 키워드 분리)
    s = re.sub(r'\bBARSIZE\s*:', 'BAR SIZE:', s, flags=re.IGNORECASE)
    # PLATE{digit} → PLATE {digit}: PLATE100 → PLATE 100 (공백 없이 연결된 치수 분리)
    s = re.sub(r'\bPLATE(\d)', r'PLATE \1', s, flags=re.IGNORECASE)
    # PLATE 뒤 {6자리}[A-Z]{6자리} 부품번호 제거: PLATE 100112G591561 → PLATE
    s = re.sub(r'(?<=\bPLATE\s)\d{5,}[A-Z]\d{5,}\b', '', s, flags=re.IGNORECASE)
    # N BOX(ES) OF → 수량 제거: (2 BOXES OF 50M) → (50M), (1 BOX OF 50M) → (50M)
    s = re.sub(r'\(\s*\d+\s*BOX(?:ES)?\s+OF\s+', '(', s, flags=re.IGNORECASE)
    # 제품 카탈로그 NO.NNNN-XX-NNNN-NX 형태 제거: TOMBO NO.1600-JZ-3020-1V 등
    s = re.sub(r'\bNO\.?\s*\d{4,}(?:-[A-Z0-9]+){2,}\b', '', s, flags=re.IGNORECASE)
    # TOMBO NO.NNNN-XX 형태 (하이픈 1개인 경우): NO.1600-JZ-4020 등
    s = re.sub(r'\bNO\.?\s*\d{4,}-[A-Z]{1,4}-\d{3,}(?:-[A-Z0-9]+)?\b', '', s, flags=re.IGNORECASE)
    # NO. OF COILS:N 코일 수량 표기 제거: NO. OF COILS:26 → '' (치수 아님)
    s = re.sub(r'\bNO\.\s*OF\s*COILS?\s*:\s*\d+', '', s, flags=re.IGNORECASE)
    # STEEL GRADE 접두사 제거: STEEL GRADE.SKD61 → '' (강종 라벨, 치수 아님)
    s = re.sub(r'\bSTEEL\s+GRADE\s*\.?\s*', '', s, flags=re.IGNORECASE)
    # {n}"/강종코드 형태: 1.5"/434MR → 1.5IN, 1.5"/M300R → 1.5IN (인치+강종코드 복합 표기)
    # 숫자 시작 코드(434MR) 또는 알파벳+숫자 시작 코드(M300R) 모두 처리
    s = re.sub(r'([\d.]+)\s*"\s*/\s*(?:\d{3,4}[A-Z]*|[A-Z]\d{2,4}[A-Z]*)\b', r'\1IN', s, flags=re.IGNORECASE)
    # ASL{n} 계열 와이어 강종 제거: ASL42, ASL52 등 (wire/rod 고합금 코드)
    s = re.sub(r'\bASL\d{2,4}\b', '', s, flags=re.IGNORECASE)
    # ALLOY{grade} 복합 단어에서 강종 코드 분리: ALLOYFN315 → ALLOY (FN315 제거)
    s = re.sub(r'\bALLOY[A-Z]{1,5}\d{3,6}[A-Z]{0,2}\b', 'ALLOY', s, flags=re.IGNORECASE)
    # PHI 직경 중복 제거: PHI 8 PHI 8 → PHI 8 (같은 직경 두 번 표기, 입력 오류)
    s = re.sub(r'\bPHI\s+([\d.]+)\s+PHI\s+\1\b', r'PHI \1', s, flags=re.IGNORECASE)
    # COIL 맥락 W {n}M: W 240M → W 240MM (코일 폭은 미터 아님, 240m는 비현실적)
    # M→mm 변환 전에 먼저 처리해야 240M→240000MM 오변환 방지
    if re.search(r'\bCOILS?\b', s, re.IGNORECASE):
        s = re.sub(r'\b(W\s*)(\d+)\s*M\b(?!M)', r'\1\2MM', s, flags=re.IGNORECASE)
    # {n}({-tol}/+{tol2})M 공차+미터 패턴: 6.00(-0/+30)M → 6000~6030MM
    s = re.sub(
        r'([\d.]+)\s*\(\s*-\s*(\d+)\s*/\s*\+\s*(\d+)\s*\)\s*M\b(?!M)',
        lambda m: f'{round(float(m.group(1))*1000)}~{round(float(m.group(1))*1000)+int(m.group(3))}MM',
        s, flags=re.IGNORECASE
    )
    # {n}(-0/+{tol}MM) 공차 패턴 (MM단위): 6.00(-0/+30MM) → 6000~6030MM
    s = re.sub(
        r'([\d.]+)\s*\(\s*-\s*(\d+)\s*/\s*\+\s*(\d+)\s*MM\s*\)',
        lambda m: f'{round(float(m.group(1))*1000)}~{round(float(m.group(1))*1000)+int(m.group(3))}MM',
        s, flags=re.IGNORECASE
    )
    # N미터(M 단독) → mm 변환: 4.7M→4700MM, X6M→X6000MM
    # (?<![A-WY-Z0-9.-]) : X 이외 알파, 숫자, 소수점, 하이픈 뒤는 미변환
    # (SA-213M, A333M 등 강종코드 내 M이 변환되는 오류 방지)
    s = re.sub(
        r'(?<![A-WY-Z0-9.-])(\d+\.?\d*)\s*M(?!\w)',
        lambda m: str(round(float(m.group(1)) * 1000)) + 'MM',
        s, flags=re.IGNORECASE
    )
    # (P{n}) 스풀크기 코드 제거: (P3.5) → '' (와이어/용접봉 스풀 크기, 치수 아님)
    s = re.sub(r'\(\s*P[\d.]+\s*\)', '', s, flags=re.IGNORECASE)
    # ({n}G) 소포장 중량 괄호 제거: (250G), (10G) → '' (그램 단위 소포장, 치수 아님)
    s = re.sub(r'\(\s*\d+\s*G\s*\)', '', s, flags=re.IGNORECASE)
    # (MATERIAL {grade}) 괄호 내 재질/목적 정보 제거: (MATERIAL 304STAINLESS...) → ''
    s = re.sub(r'\(\s*MATERIAL\s*[^)]+\)', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\(\s*PURPOSE[^)]+\)', '', s, flags=re.IGNORECASE)
    # (±{n}[MM/IN]) 공차 괄호 제거: (±0.009MM), (±0.008) → '' (치수 아닌 허용오차)
    s = re.sub(r'\(\s*[±]\s*[\d.]+\s*(?:MM|CM|IN(?:CH)?)?\s*\)', '', s, flags=re.IGNORECASE)
    # 치수 뒤 순수 숫자 괄호 제거: 20.79(18.83)X → 20.79X (OD 뒤 내경/내치수 괄호 표기)
    s = re.sub(r'(?<=[\d])\s*\(\s*[\d.]+\s*\)(?=\s*(?:[Xx*\s]|$))', '', s)
    # 단독 인치 공차 괄호 제거: (0.004INCH), (0.004") → '' (공차 표기, 치수 아님)
    s = re.sub(r'\(\s*[\d.]+\s*(?:INCH?|")\s*\)', '', s, flags=re.IGNORECASE)
    # {n}/{m}HD/HN/HR 경도 표기 제거: 1/8HD, 1/4HN, 1/2HR → '' (치수가 아닌 경도 코드)
    s = re.sub(r'\b\d+/\d+H[DNR]\b', '', s, flags=re.IGNORECASE)
    # McMaster/Grainger 카탈로그 번호 제거: 1162K39, 50415K25, 89785K845
    # {4-6자리숫자}{알파(X 제외)}{2-4자리숫자} 형태 (X는 치수 구분자이므로 제외)
    s = re.sub(r'\b\d{4,6}[A-WY-Z]\d{2,4}\b', '', s, flags=re.IGNORECASE)
    # {알파}{3-4숫자}{알파(X 제외)}{2-3숫자} 카탈로그 코드 제거: A263P04, A264P03 (D204X50 같은 치수 보호)
    s = re.sub(r'\b[A-WY-Z]\d{3,4}[A-WY-Z]\d{2,3}\b', '', s, flags=re.IGNORECASE)
    # AMS 재료 규격코드 제거: AMS 5643, AMS5670H/5671H → '' (미군 재료규격, 치수 아님)
    s = re.sub(r'\bAMS\s*\d{4,5}[A-Z]?(?:/\d{4,5}[A-Z]?)?\b', '', s, flags=re.IGNORECASE)
    # UNS 합금 번호 제거: UNS N06625, UNS#N07718, UNS-N-07718 → '' (합금 통합번호, 치수 아님)
    s = re.sub(r'\bUNS#?[-\s]*[A-Z]-?\d{5}\b', '', s, flags=re.IGNORECASE)
    # HIGHPERRM {n} 소재명 뒤 숫자 제거: HIGHPERRM 49 → '' (치수 아닌 합금계열 인덱스)
    s = re.sub(r'\bHIGHPERRM\s+\d+\b', '', s, flags=re.IGNORECASE)
    # EN 규격코드 제거: EN 10269, EN10060 → '' (유럽 강재규격, 5자리 숫자)
    s = re.sub(r'\bEN\s*1\d{4}\b', '', s, flags=re.IGNORECASE)
    # DIN 규격코드 제거: DIN 17100, DIN1013, DIN 1.4550 → '' (독일 강재규격, 소수 등급 포함)
    s = re.sub(r'\bDIN\s*\d+(?:\.\d+)?\b', '', s, flags=re.IGNORECASE)
    # 크롬-몰리브덴 내열강 등급코드 제거: CRMOV5-7, CRMOV5-11 → ''
    s = re.sub(r'\bCRMOV\d+-\d+\b', '', s, flags=re.IGNORECASE)
    # PH 석출경화강 강종코드 제거: 17-4PH, 15-5PH, 13-8PH → '' (범위로 오변환 방지 + 제거)
    s = re.sub(r'\b\d+-\d+PH\b', '', s, flags=re.IGNORECASE)
    # EN 재료번호 제거: 1.7709/21, 1.4541 등 (1.XXXX 형태 유럽 재료번호, 슬래시 있을 때만)
    s = re.sub(r'\b1\.\d{4}/\d{1,4}\b', '', s, flags=re.IGNORECASE)
    # KT{n} 제품 코드 제거: KT4, KT12 → '' (치수 아닌 제품 카테고리 코드)
    s = re.sub(r'\bKT\d{1,2}\b', '', s, flags=re.IGNORECASE)
    # 솔더 합금 조성 체인 제거: 99.79SN/0.2CU/0.01PB → '' (합금 성분비, 치수 아님)
    s = re.sub(r'\b\d+(?:\.\d+)?[A-Z]{1,3}(?:/\d+(?:\.\d+)?[A-Z]{1,3}){2,}\b', '', s, flags=re.IGNORECASE)
    # LFC{n} 솔더 플럭스 브랜드코드 제거: LFC2 → ''
    s = re.sub(r'\bLFC\d+\b', '', s, flags=re.IGNORECASE)
    # TS{n}TUBING 카탈로그 코드 제거: TS110TUBING → '' (튜빙 카탈로그 코드)
    s = re.sub(r'\bTS\d+TUBING\b', '', s, flags=re.IGNORECASE)
    # C{n}-{n}PKG SAFE-T-CABLE 카탈로그 코드 제거: C10-218PKG → '' (와이어 로프 클립 제품코드)
    s = re.sub(r'\bC\d+-\d+PKG\b', '', s, flags=re.IGNORECASE)
    # {n} PACKS OF {n} 수량 표기 제거: 3 PACKS OF 50 → '' (치수 아님)
    s = re.sub(r'\b\d+\s+PACKS?\s+OF\s+\d+\b', '', s, flags=re.IGNORECASE)
    # S{n}T/{n}/{n}/ 와이어 시리즈 코드 제거: S59T/030/000/ → '' (SAFE-T-CABLE 형식)
    s = re.sub(r'\bS\d+T(?:/\d+)+/', '', s, flags=re.IGNORECASE)
    # A-{n}TI 티타늄 합금 코드 제거: A-59TI → '' (Active Wire 합금 코드)
    s = re.sub(r'\bA-\d+TI\b', '', s, flags=re.IGNORECASE)
    # VZ{n} 합금 코드 제거: VZ2120 → '' (Vitrobraze 합금 번호)
    s = re.sub(r'\bVZ\d+\b', '', s, flags=re.IGNORECASE)
    # S{n}H{n}H 부품번호 제거: S60023H2010H → '' (Vitrobraze 브레이징 포일 부품코드)
    s = re.sub(r'\bS\d+H\d+H\b', '', s, flags=re.IGNORECASE)
    # {n}(U) 수량단위 괄호 제거: 4(U) → '' (unit 수량 표기)
    s = re.sub(r'\b\d+\(U\)', '', s, flags=re.IGNORECASE)
    # VITROBRAZE {n} 브랜드+모델 제거: VITROBRAZE 2120 → '' (브레이징 합금 브랜드)
    s = re.sub(r'\bVITROBRAZE\s*\d*\b', '', s, flags=re.IGNORECASE)
    # MUSTER: {int} {digit} X{n} → MUSTER:0.{int}{digit}X{n} (OCR 공백 포함 소수점 치수)
    # 예: MUSTER: 0 1 X152 → 0.1X152
    s = re.sub(r'\bMUSTER\s*:?\s*(\d+)\s+(\d)(?=\s*[Xx])', r'\1.\2', s, flags=re.IGNORECASE)
    # 선두 수량 / {치수} 분리: 3 / 0.1 X152 → 0.1 X152 (공백+슬래시인 경우만, 분수 1/16 보호)
    s = re.sub(r'^\s*\d+\s+/\s*(?=[\d.])', '', s.strip())
    # MATERIAL:O{n} 관재 재질코드 제거: MATERIAL:O54, MATERIAL:O61 → MATERIAL: (DIN/EN 관 소재등급)
    s = re.sub(r'(?<=MATERIAL:)O\d{2,3}\b', '', s, flags=re.IGNORECASE)
    # KP{8+자리} 구매처 부품번호 제거: KP989690048936 → '' (Kaman 구매처 추적번호)
    s = re.sub(r'\bKP\d{8,}\b', '', s, flags=re.IGNORECASE)
    # KPW{alphanum} 칼라 카탈로그 코드 제거: KPW040MB026510 → ''
    s = re.sub(r'\bKPW[A-Z0-9]+\b', '', s, flags=re.IGNORECASE)
    # {알파2-5}{숫자2-4}-{숫자2-4} 카탈로그 코드 제거: FPA206-45 (3단계 칼라코드는 아래에서 처리)
    s = re.sub(r'\b[A-Z]{2,5}\d{2,4}-\d{2,4}\b(?!-[\d.])', '', s, flags=re.IGNORECASE)
    # 00XX-XXXXXGL 카탈로그 코드 제거: 0052-10699GL → '' (선행 0 + 하이픈 + 5자리 + 알파)
    s = re.sub(r'\b0+\d{2,4}-\d{4,6}[A-Z]{1,4}\b', '', s, flags=re.IGNORECASE)
    # NI{n}/CR{n}/FE{n}... 연속 합금조성 체인 제거: NI80/CR20, NI74/CR15/FE 7/TI/AL/NB → ''
    s = re.sub(r'\bNI\d{1,3}(?:\s*/\s*[A-Z]{1,3}\d*)+', '', s, flags=re.IGNORECASE)
    # {n}/TI/AL/NB 등 숫자 뒤 원소 체인 잔류 제거: 7/TI/AL/NB → '' (NI... 제거 후 남는 잔류)
    s = re.sub(r'\b\d+(?:/[A-Z]{1,3}){2,}\b', '', s, flags=re.IGNORECASE)
    # ;-{n}WIRE 솔더 와이어 시리즈 코드 제거: ;-3.0WIRE → '' (세미콜론 뒤 시리즈 변형코드)
    s = re.sub(r';-[\d.]+WIRE\b', '', s, flags=re.IGNORECASE)
    # 분수형 화학 성분 표기 제거: 1/2MO, 1/4CR → '' (성분비가 분수 형태, 정수형보다 먼저 처리)
    s = re.sub(r'\b\d+/\d+(?:MO|NI|CR|MN|CU|AL|CO)\b', '', s, flags=re.IGNORECASE)
    # 화학 조성 퍼센트 표기 제거: 1NI, 2CR, 0.5V 등 (합금 성분비, 치수 아님)
    # (?<![/]) lookbehind: 1/2MO 같은 분수 뒤 성분기호는 보호 (1/2MO → 1/ 잔류 방지)
    s = re.sub(r'(?<![/])\b\d+(?:\.\d+)?(?:NI|MO|CR|MN|CU|AL|CO)\b', '', s, flags=re.IGNORECASE)
    # KNCLSS 칼라 코드 치수 변환: KNCLSS8-30-35 → 8MM X 30MM (bore×OD, \b 없음: COLLARSKNCLSS 대응)
    # KNCLSS 칼라 코드: bore-A-B 형식에서 A가 bore에 가까우면(차이<5) OD는 B, 아니면 OD는 A
    # KNCLSS8-30-35 → 30-8=22≥5 → 8X30 / KNCLSS8-10-25 → 10-8=2<5 → 8X25
    s = re.sub(
        r'KNCLSS(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\b',
        lambda m: (
            f'{m.group(1)}MM X {m.group(3)}MM'
            if float(m.group(2)) - float(m.group(1)) < 5
            else f'{m.group(1)}MM X {m.group(2)}MM'
        ),
        s, flags=re.IGNORECASE
    )
    # 칼라/링 모델 코드 치수 변환: AMSC10-14-30 → 14X30, NCLM20-35-50 → 35X50
    # (전체 코드는 삭제하되 뒤 두 숫자를 치수로 변환, NCLM/NCLB/AMSC 계열)
    s = re.sub(
        r'\b(?:AMSC|NCLB|NCLM|FNCLM|FNCLSS)\d+-([\d.]+)-([\d.]+)\b',
        lambda m: f'{m.group(1)}MM X {m.group(2)}MM',
        s, flags=re.IGNORECASE
    )
    # FNCLM-V{bore}-D{OD}-L{len}-... 라벨형 칼라 코드 치수 변환: FNCLM-V12.0-D30-L12-CC2 → 30MM X 12MM
    s = re.sub(
        r'\bFNCLM-V[\d.]+-D(\d+)-L(\d+(?:\.\d+)?)(?:-[A-Z]+\d*)*\b',
        lambda m: f'{m.group(1)}MM X {m.group(2)}MM',
        s, flags=re.IGNORECASE
    )
    # TASC 칼라 코드 치수 변환: TASC3-1-19 → 3MM X 19MM (첫번째×마지막, 중간 무시)
    # \b 없음: COLLARSTASC3-1-19처럼 앞에 단어가 붙어 있어도 처리
    s = re.sub(
        r'TASC(\d+)-(\d+)-(\d+)\b',
        lambda m: f'{m.group(1)}MM X {m.group(3)}MM',
        s, flags=re.IGNORECASE
    )
    # H{n}-{m} 열처리/경도 조건코드 제거: H180-200, H900-1000 (경도/온도 범위 코드)
    # 단, SCH{n} 등 단독 용도 제외 (뒤에 하이픈+숫자 조합만 대상)
    s = re.sub(r'\bH(\d{2,4})-(\d{2,4})\b', '', s, flags=re.IGNORECASE)
    # ALLOY X-{n} 합금 지정코드 제거: X-750, X -750 (공백 포함), X-625 (합금 계열명, 치수 아님)
    if re.search(r'\bALLOY\b', s, re.IGNORECASE):
        s = re.sub(r'\bX\s*-\s*\d{3,4}\b', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\bC\d{2,4}\b', '', s, flags=re.IGNORECASE)  # C22, C276 합금 계열코드
        # ALLOY {n} 뒤 단독 숫자 합금 번호 제거: ALLOY 22 → ALLOY, ALLOY 718 → ALLOY
        s = re.sub(r'(?<=\bALLOY\s)\d{2,4}\b', '', s, flags=re.IGNORECASE)
    # @ {n}°F/°C 열처리/어닐링 온도 제거: Annealed @ 1800°F → '' (치수 아닌 열처리 조건)
    # [^a-zA-Z0-9\s]* 는 °·‽ 같은 특수문자(도 기호, 인코딩 깨진 문자) 허용
    s = re.sub(r'@\s*[^a-zA-Z0-9\s]*\d{3,4}[^a-zA-Z0-9\s]*[FC]\b', '', s, flags=re.IGNORECASE)
    # NI {n} 단독 합금 계열 번호 제거: NI 42, NI 36 → '' (NI42/CR chain 패턴과 구별: 슬래시 없는 경우만)
    s = re.sub(r'\bNI\s+\d{2,3}\b(?!\s*/)', '', s, flags=re.IGNORECASE)
    # {n} IN ({m}MM) → {m}MM 변환: 10 IN (254MM) → 254MM (인치 표기 뒤 괄호 내 MM 값 채택)
    s = re.sub(r'\b\d+(?:\.\d+)?\s*IN\s*\(\s*(\d+(?:\.\d+)?)\s*MM\s*\)', r'\1MM', s, flags=re.IGNORECASE)
    # NIKROTHAL 브랜드 내 N{n} 합금 등급 제거: NIKROTHAL STRIP N80 → NIKROTHAL STRIP (80은 Ni 함량)
    if re.search(r'\bNIKROTHAL\b', s, re.IGNORECASE):
        s = re.sub(r'\bN\d{2,3}\b', '', s, flags=re.IGNORECASE)
    # OD:A*B*C → OD:A*C (외경*내경*길이에서 내경 제거): OD:55*25.5*147 → OD:55*147
    s = re.sub(r'\bOD\s*:?\s*([\d.]+)\s*\*\s*[\d.]+\s*\*\s*([\d.]+)', r'OD:\1*\2', s, flags=re.IGNORECASE)
    # CONARC 용접봉 전용 코드 제거: 511420-2 CONARC 49C 2.5X350XVPMD(RO) → 2.5X350
    if re.search(r'\bCONARC\b', s, re.IGNORECASE):
        s = re.sub(r'^\d{5,7}-\d{1,2}\s+', '', s.strip(), flags=re.IGNORECASE)   # 선두 카탈로그번호
        s = re.sub(r'\bCONARC\s+\d+[A-Z]?\b', '', s, flags=re.IGNORECASE)         # CONARC 49C 브랜드+강종
        s = re.sub(r'[Xx]?VPMD\s*\([A-Z]+\)', '', s, flags=re.IGNORECASE)          # XVPMD(RO) 포장코드
    # H242 RPB{n}N{digits} NICKEL ALLOY STRIP 부품코드 제거: H242 RPB500N06000010.00600" → .00600"
    if re.search(r'NICKEL\s+ALLOY\s+STRIP\b', s, re.IGNORECASE):
        s = re.sub(r'\bH\d+\s+RPB\d+[A-Z]\d+\b', '', s, flags=re.IGNORECASE)
    # SA/SB {grade}-P{class} ASME 파이프 등급 제거: SA335-P22, SA335-P92, SB407-N08810 → ''
    s = re.sub(r'\bS[AB]\s*-?\s*\d{2,3}[-/][A-Z]\d{1,2}\b', '', s, flags=re.IGNORECASE)
    # ASTM 규격코드 제거: ASTM-B-637, ASTM A312 → '' (미국 재료규격)
    s = re.sub(r'\bASTM[-\s]?[A-Z]-?\d+\b', '', s, flags=re.IGNORECASE)
    # TYPE{n} 제품 타입 분류코드 제거: TYPE2, TYPE 1 → ''
    s = re.sub(r'\bTYPE\s*\d+\b', '', s, flags=re.IGNORECASE)
    # OE- 용접재료 브랜드+강종 코드 제거: OE-SD3, OE-S1CrMo91 → ''
    s = re.sub(r'\bOE-\S+', '', s, flags=re.IGNORECASE)
    # 선두 알파숫자 혼합 부품번호 제거: RM41A100290A, W000285394 (선두 {알파1-3}{숫자1-3}{알파}{숫자5+} 형태)
    s = re.sub(r'^[A-Z]{1,3}\d{1,3}[A-Z]\d{5,}[A-Z]?\b\s*', '', s.strip(), flags=re.IGNORECASE)
    # P252420SEAMLESS 같이 \b 없는 경우: 알파/공백/끝 앞 단독 [알파1][숫자6+] 선두코드 제거
    s = re.sub(r'^[A-Z]\d{6,}(?=[A-Z\s]|$)\s*', '', s.strip(), flags=re.IGNORECASE)
    # B{2자리} 배치코드 prefix 제거: B30.15X15.50 → .15X15.50 (제품코드 A{9digit} 제거 후 남는 B{2digit} 접미사)
    # 선두 코드 제거(line 422) 이후에 실행해야 \b가 정상 동작
    s = re.sub(r'\bB\d{2}(?=\.\d)', '', s)
    # ST{n} 독일 구조강 등급코드 제거: ST52, ST35, ST37, ST52CF, ST52E → '' (DIN ST 계열)
    # (?<![A-Za-z]): 앞에 알파가 없을 때만 (8.5ST52CF → ST52CF 제거, STAINLESS는 제외)
    s = re.sub(r'(?<![A-Za-z])ST\d{2,3}(?:CF|E|N|H)?\b', '', s, flags=re.IGNORECASE)
    # SACH-NR.{숫자} 독일 부품번호, /DR{숫자} 도면번호 참조 제거
    s = re.sub(r'\bSACH-NR\.\d+\b', '', s, flags=re.IGNORECASE)
    s = re.sub(r'/DR\d+\b', '', s, flags=re.IGNORECASE)
    # /PC + NO. OF BUNDLES:{n} 연결형 수량 제거: /PCNO. OF BUNDLES:10 → ''
    s = re.sub(r'/\s*PC(?:\s*NO\.\s*OF\s*BUNDLES?\s*:\s*\d+)?', '', s, flags=re.IGNORECASE)
    # 독립 NO. OF BUNDLES:{n} 묶음 수량 제거 (치수 아님)
    s = re.sub(r'\bNO\.\s*OF\s*BUNDLES?\s*:\s*\d+', '', s, flags=re.IGNORECASE)
    # {n}BUNDLES {m}[L] 묶음+총수량 제거: 29BUNDLES 786L, 29BUNDLES786L, 13BUNDLES 500 → ''
    s = re.sub(r'\b\d+\s*BUNDLES?\s*(?:\d+L?)?\b', '', s, flags=re.IGNORECASE)
    # {n}PIECES LENGTH(M):{m} 묶음수+총파이프길이 함께 제거: 131PIECES LENGTH(M):786 → ''
    # 주의: LENGTH(M):{m} 단독(PIECES 없음)은 개별 피스 길이이므로 제거하지 않음
    s = re.sub(r'\b\d+\s*PIECES?\s+LENGTH\s*\(\s*M\s*\)\s*:\s*[\d.]+\b', '', s, flags=re.IGNORECASE)
    # {n}PIECES 독립 수량 제거: 131PIECES (위 패턴으로 처리 안 된 잔류분)
    s = re.sub(r'\b\d+\s*PIECES?\b', '', s, flags=re.IGNORECASE)
    # NOT MOR(E) THAN {n}MM 규격 상한 표기 제거: CROSS-SECTION/NOT MOR THAN 114.3MM → '' (오타 포함)
    s = re.sub(r'\bNOT\s+MOR(?:E)?\s+THAN\b.*', '', s, flags=re.IGNORECASE)
    # 와이어로프 구조코드 제거: +IWRC, IWRC6, 6XFI(25), FI(29), O/O (치수가 아닌 구성코드)
    # IWRC{직경}{단위} → 직경+단위 보존 (IWRC 구성코드 뒤에 직경이 붙은 경우)
    s = re.sub(r'\+?IWRC([\d.]+)(?=\s*(?:MM|CM|IN|FT)\b)', r' \1', s, flags=re.IGNORECASE)
    s = re.sub(r'\+?IWRC\d*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\b\d+[Xx*]FI\(\d+\)\b', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\bFI\(\d+\)\b', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\bO/O\b', '', s, flags=re.IGNORECASE)
    # GALV./WIRE/ROPE/CABLE/UNGALV 맥락 구성코드 제거
    if re.search(r'\b(?:GALV|UNGALV|WIRE|ROPE|CABLE)\b', s, re.IGNORECASE):
        s = re.sub(r'\+\d*(?:FC|PP|NFC)', '', s, flags=re.IGNORECASE)         # +7FC/+7PP/+NFC/+NFC 섬유코어 (숫자 없는 +NFC 포함)
        # WIRE ROPE 업계 표준 스트랜드 구성코드 명시 제거: 1X7, 7X7, 6X29(FI), 8X19 등
        s = re.sub(r'\b(?:1[Xx]7|7[Xx]7|6[Xx]7|6[Xx]12|6[Xx]19|6[Xx]24|6[Xx]25|6[Xx]29|6[Xx]36|7[Xx]19|1[Xx]19|8[Xx]19|8[Xx]7)(?:\([A-Z]+\))?[A-Z]?\b',
                   '', s, flags=re.IGNORECASE)
        s = re.sub(r'(?<!\.)\b\d{1,2}[Xx]\d{1,2}(?=\d)', '', s, flags=re.IGNORECASE)  # 7X73.18 직결 잔류분 (소수점 뒤 보호)
        s = re.sub(r'[Xx*]\s*\d{6,}\s*MM\b', '', s, flags=re.IGNORECASE)    # X1000000MM/*1000000MM 총길이 (100m+ 제거, 45m 피스 보존)
        if re.search(r'\bCABLE\b', s, re.IGNORECASE):
            s = re.sub(r'^\d{5,6}\b\s*', '', s.strip())                    # CABLE 선두 카탈로그번호
    # 파이프/튜브 코일 총길이 제거: 12.7MMX0.89MMX140000MM → 12.7MMX0.89MM (6자리+ = 100m+)
    # 앞에 MM 치수가 있을 때만 적용 (X로 이어지는 마지막 값이 6자리+)
    s = re.sub(r'(?<=MM)[Xx]\d{6,}\s*MM\b', '', s, flags=re.IGNORECASE)
    # DRY 와이어 맥락: {quantity}X{length}M{construction} {diameter}MM DRY 형태에서 직경만 추출
    # 예: 6X1000M6X24+7PP 14MM DRY → 14MM
    if re.search(r'\bDRY\b', s, re.IGNORECASE):
        dry_m = re.search(r'([\d.]+)\s*MM\b(?=\s+DRY\b)', s, re.IGNORECASE)
        if dry_m:
            s = dry_m.group(1) + 'MM'
    # SCS{n} 용접/납땜 와이어 등급코드 제거: SCS7 CORE WIRE → CORE WIRE (강종 아닌 제품 시리즈)
    s = re.sub(r'\bSCS\d{1,2}\b', '', s, flags=re.IGNORECASE)
    # CLF{4+자리} Castolin 브랜드 모델코드 제거: CLF5160 → ''
    s = re.sub(r'\bCLF\d{4,}\b', '', s, flags=re.IGNORECASE)
    # SR-{n}SUPER 솔더링 로드 브랜드 제거: SR-34SUPER → ''
    s = re.sub(r'\bSR-\d+\w*\b', '', s, flags=re.IGNORECASE)
    # LFM-{n} 납땜 합금 모델 제거: LFM-22 → ''
    s = re.sub(r'\bLFM-\d+\b', '', s, flags=re.IGNORECASE)
    # {n}% 플럭스/합금 퍼센트 제거: 3.5% → '' (와이어 구성 비율, 치수 아님)
    # (?<!\.)\b: 소수점 뒤 숫자는 제외 (99.93%에서 93%만 제거되는 오작동 방지)
    s = re.sub(r'(?<!\.)\b\d+(?:\.\d+)?%(?!\w)', '', s)
    # P/O NO 이하 구매주문 참조 제거: P/O NO LTML26-NA0309 → ''
    s = re.sub(r'\bP/O\s*NO\b.*', '', s, flags=re.IGNORECASE)
    # THERMACLAD {n} 브랜드+모델 제거: THERMACLAD 457 → ''
    s = re.sub(r'\bTHERMACLAD\s+\d+\b', '', s, flags=re.IGNORECASE)
    # {n}# 파운드 중량 표기 제거: 75# → '', 500# → '' (납땜 와이어 스풀 중량, # 뒤 word boundary 없음)
    s = re.sub(r'\b\d+#', '', s)
    # POP{n}KG / POP {n}KG / 단독 POP 포장 표기 제거
    s = re.sub(r'\bPOP\s*[\d.]*\s*(?:KG)?\b', '', s, flags=re.IGNORECASE)
    # CP{4+자리} 솔더 제품코드 제거: CP2000, CP2000 → ''
    s = re.sub(r'\bCP\d{4,}\b', '', s, flags=re.IGNORECASE)
    # CONSTRUCTION 키워드 이후 제거: CONSTRUCTION 7 X 19 X 0.25 MM (와이어 구성코드 블록)
    s = re.sub(r'\bCONSTRUCTION\b.*', '', s, flags=re.IGNORECASE)
    # MM2/MM² 전선/강도 단면적 제거: 10 MM2, 1570/1770MM2 → '' (단면적 또는 인장강도, 치수 아님)
    s = re.sub(r'\b\d+(?:\.\d+)?(?:/\d+)?\s*MM2\b', '', s, flags=re.IGNORECASE)
    # (1.XXXX) 유럽 재료번호 괄호 제거: (1.4401), (1.4307) → ''
    s = re.sub(r'\(\s*1\.\d{4}\s*\)', '', s)
    # 유럽 소수점 1자리 + 공백 + 단위: 5,0 MM → 5.0 MM (기존 규칙은 단위 직결만 처리)
    s = re.sub(r'\b(\d{1,4}),(\d)(?=\s+(?:MM|CM|IN|FT)\b)', r'\1.\2', s, flags=re.IGNORECASE)
    # EN 구조강 등급 제거: S355J2H, S235JR, S690Q → '' (강도+충격기호 조합, 치수 아님)
    s = re.sub(r'\bS\d{3}[A-Z]\d?[A-Z]?\b', '', s, flags=re.IGNORECASE)
    # (nEA), (nPCS) 수량 괄호 제거: (2EA), (3PCS) → '' (치수 아님)
    s = re.sub(r'\(\s*\d+\s*(?:EA|PCS?|NOS?)\s*\)', '', s, flags=re.IGNORECASE)
    # AWT/AVGWT {m}-{l}L 벽두께-길이 형식 변환: AWT 101.6-5700L → WT 101.6 X 5700 (범위로 오인식 방지)
    s = re.sub(r'\b(?:AVG|A)?WT\s+([\d.]+)\s*-\s*([\d.]+)L\b', r'WT \1 X \2', s, flags=re.IGNORECASE)
    # SIZE:{n}(A|KG) 단위중량 표기 제거: SIZE:37ALENGTH → SIZE:LENGTH (단위중량은 치수 아님)
    s = re.sub(r'(?<=SIZE:)\d+(?:A|KG)(?=[A-Z])', '', s, flags=re.IGNORECASE)
    # BATCH NO/BATCH: 로트번호 제거: BATCH NO:M1SW231037, BATCH:251000136/5 → ''
    s = re.sub(r'\bBATCH\s*(?:NO\.?\s*)?:?\s*[\w/.-]+', '', s, flags=re.IGNORECASE)
    # NUMBER OF BARS:{n} 수량 제거: NUMBER OF BARS:47 → ''
    s = re.sub(r'\bNUMBER\s+OF\s+BARS?\s*:\s*\d+', '', s, flags=re.IGNORECASE)
    # {n}D 표면처리 코드 제거: 2D VIM → '' (냉연 표면처리 등급, 치수 아님)
    # \s 옵션 제외: 연속공백(7M  PS)에서 오인식 방지
    s = re.sub(r'\b\d[A-Z]\b(?=\s+(?:VIM|ESR|ANNEALED|COLD|ROLLED|PICKLED|DESCALED|$))', '', s, flags=re.IGNORECASE)
    # HEAT NO. 열번호 제거: HEAT NO.338645, CUT HEAT NO.338645 → '' (word boundary 없이 처리)
    s = re.sub(r'HEAT\s+NO\.?\s*[\w/.-]+', '', s, flags=re.IGNORECASE)
    # {fraction}CUT 절단 지정 제거: 1/2CUT, 1/2CUTHEAT → '' (뒤 word boundary 없이 처리)
    s = re.sub(r'\b\d+/\d+\s*CUT', '', s, flags=re.IGNORECASE)
    # CUT:{n}L+BALANCE 절단+잔량 지정 제거: BARCUT:1500L+BALANCE → BAR (1500은 치수 아닌 절단 지시)
    # \b 미사용: BARCUT:처럼 CUT 앞에 word char가 붙어도 처리
    s = re.sub(r'CUT\s*:\s*[\d.]+[LM]\b\s*\+\s*BALANCE\b', '', s, flags=re.IGNORECASE)
    # SP-CUT 특수절단 코드 제거: SP-CUT → ''
    s = re.sub(r'\bSP-CUT\b', '', s, flags=re.IGNORECASE)
    # PMI{n} 양성적 소재 확인 번호 제거: PMI39, PMI40 → '' (치수 아님)
    s = re.sub(r'\bPMI\d+\b', '', s, flags=re.IGNORECASE)
    # -CP{n} 마감/코팅 suffix 제거: 1000L-CP1 → 1000L (KOVAR 등 정밀관 마감코드)
    s = re.sub(r'-CP\d+\b', '', s, flags=re.IGNORECASE)
    # 끝 쉼표 뒤 5-6자리 독립 카탈로그번호 제거: ,100004 → '' (와이어 COIL OXIDE 등)
    s = re.sub(r',\s*\d{5,6}\s*$', '', s.strip())
    # 쉼표 뒤 0-시작 4자리 코드 제거: ,0023 → '' (zero-padded 내부 코드)
    s = re.sub(r',\s*0\d{3}\b', '', s)
    # 선두 0-시작 4자리 코드+쉼표 제거: 0023, → '' (62590271 제거 후 남는 lot 코드)
    s = re.sub(r'^\s*0\d{3,4}\s*,\s*', '', s.strip())
    # 말미 8자리 독립 숫자 제거: 26257483 → '' (인벤토리/SAP 참조번호)
    s = re.sub(r'\s+\d{8}\s*$', '', s.strip())
    # Tinplate 도금코드+공차 제거: ,T3,CA,1.1/1.1 → '' (T[1-9] 뒤에 쉼표+코드 패턴)
    s = re.sub(r',\s*T[1-9]\s*,.*$', '', s, flags=re.IGNORECASE)
    # 쉼표 구분 단일알파+1-2자리숫자 제품 코드 제거: ,F4, → '' (I,F4, 형태)
    s = re.sub(r',\s*[A-Z]\d{1,2}\s*(?=,)', '', s, flags=re.IGNORECASE)
    # 유럽식 소수점 쉼표(2자리) 변환: 3,70 → 3.70 (3자리는 천단위이므로 제외)
    s = re.sub(r'(\d),(\d{2})(?!\d)', r'\1.\2', s)
    # W{숫자} BR 독일어 폭 표기 제거: W1 BR (Breite 1m = 폭 코드) → ''
    s = re.sub(r'\bW\d+\s+BR\b', '', s, flags=re.IGNORECASE)
    # 선두 4자리 강종번호 제거: 1215 COLD DRAWN STEEL ROUND BAR → '' (SAE/AISI 강종 접두)
    s = re.sub(r'^\d{4}\b\s*(?=(?:COLD|HOT|DRAWN|ROUND|BAR|WIRE|PIPE|TUBE|STEEL|ALLOY|FREE|CARBON)\b)',
               '', s.strip(), flags=re.IGNORECASE)
    # 이중 점 오타 수정: 1..96 → 1.96 (입력 오류 교정)
    s = re.sub(r'(\d+)\.\.(\d+)', r'\1.\2', s)
    # NTI-{숫자}{알파} 모델 suffix 제거: NTI-2R → '' (제품 시리즈 코드)
    s = re.sub(r'\bNTI-\d+[A-Z]?\b', '', s, flags=re.IGNORECASE)
    # TUBE OD/ID 이중직경 표기: R25/12,5X90 → 25X90 (R{OD}/{ID}, 유럽식 콤마 소수점 포함)
    s = re.sub(r'\bR(\d+(?:\.\d+)?)/[\d,]+', r'\1', s, flags=re.IGNORECASE)
    # ST 제거 후 남는 R{n}X{n} 앞 R 접두사 제거: R66X8.5 → 66X8.5 (독일어 Rohr=관)
    s = re.sub(r'(?:^|\s)R(?=\d+[Xx][\d.])', r' ', s.strip(), flags=re.IGNORECASE)
    # 중량 표기 제거: 25KG, 226.8KG → '' (코일 단위중량, 치수 아님)
    s = re.sub(r'\b\d+(?:\.\d+)?\s*KG\b(?![\s/]*M\b)', '', s, flags=re.IGNORECASE)
    # 괄호 내 참조번호 제거: [#1125301] → ''
    s = re.sub(r'\[\s*#\d+\s*\]', '', s)
    # 선행 소수점 0 보완 (코드 제거 후 잔류): .00600" → 0.00600"
    s = re.sub(r'(?<![A-Z\d.])\.(\d)', r'0.\1', s)
    return re.sub(r'\s+', ' ', s)


def _convert_unit(num_str: str, unit: str) -> str:
    """단위 변환 → 최종 토큰"""
    unit = (unit or '').strip().upper()

    # 틸다 범위 표기: 0.3590~0.3610 (각 파트 변환 후 재결합)
    if '~' in num_str and '/' not in num_str:
        lo, hi = num_str.split('~', 1)
        return f'{_convert_unit(lo.strip(), unit)}~{_convert_unit(hi.strip(), unit)}'

    # 하이픈 범위 표기: 4.95-5.0 → 4.95~5, 6.000-6.050M → 6000~6050 (단위 적용)
    if '-' in num_str and '/' not in num_str:
        parts = num_str.split('-', 1)
        if len(parts) == 2 and all(re.match(r'^\d+\.?\d*$', p) for p in parts):
            u = unit.upper()
            if u in ('M', 'MTR', 'METER', 'CM'):
                # M/CM 단위: 각 파트에 단위 변환 (6000-6.050M → 6000~6050)
                lo = _convert_unit(parts[0].strip(), unit)
                hi = _convert_unit(parts[1].strip(), unit)
            else:
                # MM 또는 단위 없음: 정수값은 정수화 (4.95-5.0 → 4.95~5)
                def _fmt_num(n: str) -> str:
                    try:
                        v = float(n)
                        return str(int(v)) if v == int(v) else n
                    except ValueError:
                        return n
                lo = _fmt_num(parts[0].strip())
                hi = _fmt_num(parts[1].strip())
            return f'{lo}~{hi}'

    # 분수 형태 (예: 1/2, 1/4) — 인치는 확정 데이터와 동일하게 분수 표기 유지
    if '/' in num_str:
        if unit in ('"', 'IN', 'INCH', 'INCHES'):
            return f"{num_str}IN"
        if unit in ("'", 'FT', 'FEET', 'FOOT'):
            return f"{num_str}FT"
        # M/CM 변환이 필요한 경우 평가
        try:
            parts = num_str.split('/')
            val = float(parts[0]) / float(parts[1])
        except (ValueError, ZeroDivisionError):
            return num_str
        if unit in ('M', 'MTR', 'METER'):
            result = val * 1000
            rounded = round(result)
            return str(rounded) if abs(result - rounded) < 0.01 else f'{result:.1f}'
        if unit == 'CM':
            result = val * 10
            return str(int(result)) if result == int(result) else f'{result:.1f}'
        return num_str

    try:
        val = float(num_str)
    except ValueError:
        return num_str

    if unit in ('M', 'MTR', 'METER'):
        result = val * 1000
        rounded = round(result)
        return str(rounded) if abs(result - rounded) < 0.01 else f'{result:.1f}'
    if unit == 'CM':
        result = val * 10
        return str(int(result)) if result == int(result) else f"{result:.1f}"
    if unit in ('MM', ''):
        return num_str
    if unit in ('"', 'IN', 'INCH', 'INCHES'):
        return f"{num_str}IN"
    if unit in ("'", 'FT', 'FEET', 'FOOT'):
        return f"{num_str}FT"
    if unit == 'A':
        return f"{num_str}A"
    return f"{num_str}{unit}"


# 강종/규격 코드 패턴 — 사이즈 숫자에서 제거 대상
_GRADE_CODE_PAT = re.compile(
    r'\b(?:'
    r'[A-Z]{1,5}\d{3,6}[A-Z]{0,2}'   # SS400, SM490A, A240, G3454, S31803, AMS5659
    r'|S\d{1,2}C'                      # S10C, S20C, S45C (탄소강)
    r'|SK[DHMS]\d{2}'                  # SKD11, SKD61, SKH51 (JIS 공구강)
    r'|304[A-Z]{0,2}|316[A-Z]{0,2}|321[A-Z]{0,2}|347|430|410(?![\d.])|201(?![\d.])|202(?![\d.])'
    r'|4130|4140|4330|4340|8620'
    r'|C\d{4}(?:~\d{4})?(?![\d.])'     # 탄소강: C1010, C1020, C1010~1020 등
    r'|C-\d{2,3}'                       # Hastelloy C-22, C-276 등
    r')\b'
    r'|(?<!\d)1\.\d{4}(?!\d)',         # DIN 스테인리스 강종: 1.4404, 1.4571, 1.4301
    re.IGNORECASE
)


def _strip_grade_codes(text: str) -> str:
    """텍스트에서 강종/규격코드 숫자를 제거"""
    # X{digits} 치수 구분자 임시 보호: X237 → X_237, X12000MM → X_12000MM (강종코드로 오인 방지)
    # 주의: trailing \b 제거 — X12000MM에서 MM이 word char이라 \b가 없어 보호 실패 방지
    text = re.sub(r'\b(X)(\d+)', r'\1_\2', text)
    # SCH{n} 스케줄 코드 임시 보호: SCH160 → SCH__160 (grade code로 오인 삭제 방지)
    text = re.sub(r'\bSCH(\d+)\b', r'SCH__\1', text)
    # OD{n} 외경 접두사 임시 보호: OD406 → OD__406 (강종코드 오인 삭제 방지)
    text = re.sub(r'\bOD(\d)', r'OD__\1', text)
    # DIA{n} 직경 접두사 임시 보호: DIA105 → DIA__105 (강종코드 오인 삭제 방지)
    text = re.sub(r'\bDIA(\d)', r'DIA__\1', text)
    # GR./GR 등급 표시 제거: GR.1, GR.2, GR 1 등
    text = re.sub(r'\bGR\.?\s*\d+\b', '', text, flags=re.IGNORECASE)
    # SK 탄소공구강 제거: SK4, SK40 등 (치수 직전에 오는 경우)
    text = re.sub(r'\bSK\d{1,2}(?=[.\d])', '', text, flags=re.IGNORECASE)
    # STKM 기계구조용 강관 코드 제거: STKM13C, STKM13C-T 등
    text = re.sub(r'\bSTKM\d{1,2}[A-Z]?(?:-[A-Z])?\b', '', text, flags=re.IGNORECASE)
    # HYM/MYM 고강도 선재 강종코드 제거: HYM35, HYM2, MYM2AI 등
    text = re.sub(r'\b[HM]YM\w*\d*\b', '', text, flags=re.IGNORECASE)
    # HSS 고속도강 M-시리즈 강종코드 제거: M2, M42, M35, M2AI, M33 등 (& 또는 , 로 연결된 경우)
    # "AISI M2 & M42 & M35" → M2, M42, M35 등은 단독으로 나타날 때 강종코드
    text = re.sub(r'\b(?:AISI\s+)?M\d{1,2}(?:AI|V)?\b', '', text, flags=re.IGNORECASE)
    # SAE 자동절삭강 완전 제거: SAE12L14, SAE12L15 등 (L이 포함된 4-5자리 복합 코드)
    text = re.sub(r'\bSAE\s*\d+[A-Z]\d+\b', '', text, flags=re.IGNORECASE)
    # SAE 보론강 강종코드 제거: 10B21, 10B22, 10B46 등
    text = re.sub(r'\b\d{2}B\d{2}\b', '', text, flags=re.IGNORECASE)
    # SAE 52100 베어링강 강종코드 제거 (5자리 숫자로 치수로 오인됨)
    text = re.sub(r'\b521\d{2}\b', '', text, flags=re.IGNORECASE)
    # SAE 탄소강 강종코드 괄호 제거: (1006), (1008), (1018), (1020), (1045) 등
    text = re.sub(r'\(\s*10[0-9][0-9]\s*\)', '', text, flags=re.IGNORECASE)
    # NILO 니켈합금 강종코드 제거: NILO 42, NILO 36 등
    text = re.sub(r'\bNILO\s+\d+\b', '', text, flags=re.IGNORECASE)
    # ASTM F{2자리}-{2자리} 규격 코드 제거: F30-96, F15-12 등 (ASTM 맥락)
    if re.search(r'\bASTM\b', text, re.IGNORECASE):
        text = re.sub(r'\bF\d{2,3}-\d{2,3}\b', '', text, flags=re.IGNORECASE)
    # KD FINISH 표면처리 등급코드 제거: KD FINISH 5-G-00E8G → (치수 아님)
    text = re.sub(r'\bKD\s+FINISH\b', '', text, flags=re.IGNORECASE)
    # 영숫자 표면처리 코드 제거: 5-G-00E8G, 5-G-00E12G 등 (숫자-알파-숫자알파 형태)
    text = re.sub(r'\b\d+-[A-Z]-[\dA-Z]+\b', '', text, flags=re.IGNORECASE)
    # INCONEL 니켈합금 코드 제거: INCONEL718, INCONEL625 등
    text = re.sub(r'\bINCONEL\s*\d{3,4}\b', '', text, flags=re.IGNORECASE)
    # ALLOY 합금 번호: ALLOY 625, ALLOY 718, ALLOY 276 등
    text = re.sub(r'\bALLOY\s+\d{3,4}\b', '', text, flags=re.IGNORECASE)
    # 17-4 PH / 17/4 PH 스테인리스 강종 코드 제거
    text = re.sub(r'\b17[-/]4\s+PH\b', '', text, flags=re.IGNORECASE)
    # 합금 비율 코드 (숫자/숫자): 18/8, 70/30, 60/40 등 (부등호 없이 전체 단어)
    text = re.sub(r'\b(?:18/8|70/30|60/40|80/20|90/10|63/37)\b', '', text, flags=re.IGNORECASE)
    # UDDEHOLM 브랜드명 제거: UDDEHOLM (Stavax 등 제품을 포함하는 회사명)
    text = re.sub(r'\bUDDEHOLM\b', '', text, flags=re.IGNORECASE)
    # VANADIS/STAVAX/CORRAX 등 Uddeholm 강종명 제거: VANADIS 23, STAVAX ESR (ESR 같은 공정코드 포함)
    text = re.sub(r'\b(?:VANADIS|STAVAX|CORRAX|ELMAX|ORVAR|CALDIE)(?:\s+(?:ESR|VAR|VIM|CVM|AOD)\b)?\s*\d*', '', text, flags=re.IGNORECASE)
    # 한자리/두자리 숫자 + MN 망간강 코드 제거: 65MN, 55MN → (치수 앞 강종)
    text = re.sub(r'\b\d{2,3}MN\b', '', text, flags=re.IGNORECASE)
    # API 5L 라인파이프 스펙 코드 제거: API5LGR-B, 5LGR-B (5가 치수로 오인됨)
    text = re.sub(r'\bAPI\s*5L(?:GR-[A-Z])?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b5L(?:GR-[A-Z])?\b', '', text, flags=re.IGNORECASE)
    # 합금 비율 코드: NIAL95/5, CU70/30 등 (알파+숫자+/+숫자)
    text = re.sub(r'\b[A-Z]{2,5}\d{1,3}/\d+\b', '', text, flags=re.IGNORECASE)
    # 아연도금 코팅 등급 코드 제거: HIZN-N3, HIZN-N8 등 (HOT DIP GALVANIZING 등급)
    text = re.sub(r'\bHIZN-N\d+\b', '', text, flags=re.IGNORECASE)
    # 용접봉/와이어 모델 코드: ERNI-1, ER70S-6, AK-10 등 (알파+하이픈+숫자)
    text = re.sub(r'\b[A-Z]{2,5}-\d+\b', '', text, flags=re.IGNORECASE)
    # OCR 합금 와이어 강종코드: OCR25AL50, OCR21AL8NB, OCR27AL7MO2 등
    text = re.sub(r'\bOCR\d{2}[A-Z]{2}\d+(?:[A-Z]{2,3}\d*)?\b', '', text, flags=re.IGNORECASE)
    # AMS 항공우주 규격 코드: AMS-6350, AMS5659 (hyphen 포함)
    text = re.sub(r'\bAMS[-\s]?\d{3,5}\b', '', text, flags=re.IGNORECASE)
    # JIS 공구강 접두어 분리: SKD1122 → SKD11 22 (단어 경계 문제 우회)
    text = re.sub(r'\b(SK[DHMS]\d{2})(?=\d)', r'\1 ', text, flags=re.IGNORECASE)
    # 그레이드코드 + 숫자 직접 연결 분리: SWAK2738M230 → SWAK2738M 230
    # X는 치수 구분자이므로 trailing alpha에서 제외
    text = re.sub(r'([A-Z]{1,5}\d{3,6}[A-WY-Z]{1,2})(?=\d)', r'\1 ', text, flags=re.IGNORECASE)
    # 알파-숫자-알파-숫자 복합 코드 제거: PST23F85405 → PST23F85 제거 (끝에 치수 숫자 남김)
    # DIA/OD/ID/NPS 등 차원 키워드는 제외 (DIA80MMX4010에서 DIA80MMX4 제거 방지)
    text = re.sub(r'\b(?!(?:DIA|OD|ID|NPS|DIN|TIG|MIG)\b)[A-Z]{2,5}\d{1,4}[A-Z]{1,4}\d{1,4}(?=\d{3,})', '', text, flags=re.IGNORECASE)
    # 단자리 알파-숫자-알파 코드 제거: LP4M→, AK30MOD→, (치수 직전 코드)
    # DIA/OD/ID/NPS 등 차원 키워드는 제외
    text = re.sub(r'\b(?!(?:DIA|OD|ID|NPS|DIN|TIG|MIG)\b)[A-Z]{2,5}\d{1,2}[A-Z]{1,4}(?=\d)', '', text, flags=re.IGNORECASE)
    result = _GRADE_CODE_PAT.sub(' ', text)
    # 임시 보호 복원
    result = result.replace('X_', 'X')
    result = result.replace('SCH__', 'SCH')
    result = result.replace('OD__', 'OD')
    return result.replace('DIA__', 'DIA')


def _parse_tokens(text: str, preserve_order: bool = False) -> Optional[str]:
    """
    텍스트에서 숫자(+단위) 토큰을 추출해 X로 연결.
    preserve_order=True 이면 정렬 없이 추출 순서 그대로.
    """
    # 코일 표기 통일
    text = re.sub(r'\d*\s*COILS?', 'XC', text, flags=re.IGNORECASE)
    text = re.sub(r'\bCOIL\b', 'C', text, flags=re.IGNORECASE)

    # 숫자(범위포함, 분수포함) + 단위 / 단독 스케줄 토큰 파싱
    combined_pat = re.compile(
        r'(?P<sch>\bSCH\d+\b|\bXXS\b|\bSTD\b|\bNPS\s*\d+\b)'  # 단독 스케줄/NPS 토큰
        r'|(?P<num>[\d]+/[\d]+|[\d]+\.?[\d]*(?:A(?!\w))?(?:[~-][\d]+\.?[\d]*)?)'
        r'\s*(?P<unit>MM(?!\w)|CM(?!\w)|M(?!\w|M)|MTR|METER'
        r'|INCH(?:ES)?|IN(?!\w)|"|\''
        r'|FT|FEET|FOOT'
        r'|SCH\d*|NPS|STD|XXS|XS|BWG|SWG'
        r'|C(?!\w))?',
        re.IGNORECASE
    )

    tokens = []
    for m in combined_pat.finditer(text):
        if m.group('sch'):
            tokens.append(m.group('sch').upper())
            continue
        num = m.group('num') or ''
        unit = (m.group('unit') or '').strip()
        if not num:
            continue
        num = num.rstrip('.')  # 후행 점 제거: 8. → 8, 127. → 127
        if not num:
            continue
        # 00, 000 등 0값 토큰 제거 (부품번호 lot suffix)
        try:
            if '/' not in num and '-' not in num and float(num) == 0.0:
                continue
        except ValueError:
            pass
        tokens.append(_convert_unit(num, unit))

    # C(Coil) 분리
    non_c = [t for t in tokens if t.upper() != 'C']
    has_c = len(non_c) < len(tokens)

    if not non_c:
        return 'C' if has_c else None

    if not preserve_order:
        def sort_key(t):
            m = re.match(r'^([\d.]+)', t)
            return float(m.group(1)) if m else float('inf')
        non_c = sorted(non_c, key=sort_key)

    if has_c:
        non_c.append('C')
    result = 'X'.join(non_c)
    # NPS 토큰과 XXS 사이의 여분 X 제거: NPS3XXXS → NPS3XXS (X separator가 XXS와 겹치는 경우)
    result = re.sub(r'\b(NPS\d+)X(XXS)(?=X|\s|$)', r'\1\2', result, flags=re.IGNORECASE)
    return result


def _extract_twl(text: str) -> Optional[str]:
    """
    T/W/L/OD/THK 명시 패턴: 순서 그대로 추출 (정렬 없음)
    예) T14.5MM X W3191MM X L11675MM  →  14.5X3191X11675
        OD 60.3MM X W 2.77MM X L 6000MM  →  60.3X2.77X6000
        THK: 5.0MM,W 1255MM X L 4620MM  →  5.0X1255X4620
    """
    pat = re.compile(
        r'(?:\([TWLH]\d*\)|\b(?:[TWLH]|OD|THK|WT)(?=[\d\s.:\-]))'  # T/W/L/H/OD/THK/WT 뒤에 숫자(콜론 허용)
        r'\s*:?\s*'
        r'([\d]+\.?[\d]*)'                          # 숫자
        r'\s*'
        r'(MM(?!\w)|CM(?!\w)|(?<!\w)M(?!\w|M)|MTR|METER'
        r'|INCH(?:ES)?|IN(?!\w)|"|\''
        r'|FT|FEET)?',
        re.IGNORECASE
    )
    matches = pat.findall(text)
    if len(matches) < 2:
        return None
    tokens = [_convert_unit(num, unit) for num, unit in matches]
    return 'X'.join(tokens)


def _clean_size_block(block: str) -> str:
    """SIZE 블록에서 수량/무게/부품번호 등 노이즈 제거"""
    # N.N(N.N-N.N) 공차/허용범위 괄호 제거: 3.0(2.85-2.90)MM → 3.0MM
    block = re.sub(r'(\b[\d.]+)\s*\([\d.]+[-~][\d.]+\)', r'\1', block)
    # PREPAINTED 이면 코팅 두께 괄호 제거: 0.50(0.48)X → 0.50X
    block = re.sub(r'(\b[\d.]+)\s*\(\s*[\d.]+\s*\)(\s*[Xx])', r'\1\2', block)
    # (NNNMM) 괄호 단위 치수 → 괄호 제거: (350MM) → 350MM (강종코드 괄호 제거 전에 처리)
    block = re.sub(r'\(\s*(\d+(?:\.\d+)?)\s*(MM|CM|IN|FT)\s*\)', r' \1\2', block, flags=re.IGNORECASE)
    # (RW) 롤폭 표시자 제거: 0.30X901(RW)X710 → 0.30X901X710
    block = re.sub(r'\s*\(RW\)\s*', '', block, flags=re.IGNORECASE)
    # TRIMMED/SLIT EDGE 이후 제거 (SIZE 블록 뒤에 붙는 가공 표기)
    block = re.sub(r',?\s*TRIMMED\b.*$', '', block, flags=re.IGNORECASE)
    # TYPE N.N INSPECTION 인증서 표기 이후 제거: TYPE 3.1 INSPECTION → 제거
    block = re.sub(r',?\s*\bTYPE\s+[\d.]+\s+INSPECTION\b.*$', '', block, flags=re.IGNORECASE)
    # *COIL/* COIL 뒤 불필요한 설명 제거하고 XC로 압축: *COIL MILL EDGE → XC
    block = re.sub(r'\s*[*Xx]\s*COILS?\b.*$', 'XC', block, flags=re.IGNORECASE)
    # Tinplate 품질코드 제거: T3,CA,SF,1.1/2.8 형태 (치수 뒤 표면처리/공차 코드)
    block = re.sub(r'T[1-8]\s*,\s*[A-Z]{1,4}\s*,.*$', '', block, flags=re.IGNORECASE)
    # A261S10/A262S05 등 강선 카탈로그 코드 제거
    block = re.sub(r'\bA\d{3}S\d{2}\b', '', block, flags=re.IGNORECASE)
    # W{6+digits} 카탈로그 번호 제거: W45010007 등
    block = re.sub(r'\bW\d{6,}\b', '', block, flags=re.IGNORECASE)
    # AMC 부품번호 제거: AMC-260327-81A 등
    block = re.sub(r'\bAMC-\d+-\d+[A-Z]?\b', '', block, flags=re.IGNORECASE)
    # 괄호 안 WITH ... 제거: (WITH 100% X-RAY) 등
    block = re.sub(r'\(\s*WITH\b[^)]*\)', '', block, flags=re.IGNORECASE)
    # REV: 버전 표기 제거: REV:1, REV: 2
    block = re.sub(r'\bREV\s*:\s*\d+', '', block, flags=re.IGNORECASE)
    # SCH{n}S → SCH{n} 정규화: SCH10S → SCH10
    block = re.sub(r'\bSCH(\d+)S\b', r'SCH\1', block, flags=re.IGNORECASE)
    # 아연도금 코팅 등급 제거: Z80, Z12, Z8 등 (갈바나이즈드 코팅량 표기)
    block = re.sub(r'\bZ\d{1,3}\b', '', block, flags=re.IGNORECASE)
    # AZ/ZM 도금 코팅 두께 코드 제거: AZ90, AZ40, ZM80 등
    block = re.sub(r'\bAZ\d{2,3}\b|\bZM\d{2,3}\b', '', block, flags=re.IGNORECASE)
    # INVAR 합금 제품 코드 제거: INVARM93/VIR, INVARM93/PLA 등
    block = re.sub(r'\bINVAR\w*/[A-Z]+\b', '', block, flags=re.IGNORECASE)
    # T/S 경도/인장강도 표기 제거: T/S:1/8HARD
    block = re.sub(r'\bT/S\s*:\s*\S+', '', block, flags=re.IGNORECASE)
    # KGF 인장강도 범위 괄호 제거: (85-95KGF/MM2), (70-80KGF/MM2) 등
    block = re.sub(r'\(\s*[\d.]+[-~][\d.]+\s*KGF[^)]*\)', '', block, flags=re.IGNORECASE)
    # P/N: 부품번호 이후 제거: P/N:7203475 등 (코드 뒤 치수 없음)
    block = re.sub(r',?\s*\bP/N\s*:.*$', '', block, flags=re.IGNORECASE)
    # P/O NO. 구매주문번호 이후 제거: P/O NO. TP-20260306-01 등
    block = re.sub(r'\s*,?\s*\bP/O\s+NO\.?\b.*$', '', block, flags=re.IGNORECASE)
    # PO-KONNNNNN-NNN 구매주문번호 제거: PO-KO20260316-014 등
    block = re.sub(r'\bPO-[A-Z]{2,}\d{6,}-\d+\b', '', block, flags=re.IGNORECASE)
    # PART NO./PART NUMBER: 이후 부품번호 제거: PART NUMBER: 89895K604 - → 제거
    block = re.sub(r'\bPART\s+(?:NO\.?|NUMBER)\s*:\s*\S+\s*-?\s*', '', block, flags=re.IGNORECASE)
    # AWG 게이지 표기 제거: 21GA, 23GA (숫자 뒤에 공백+알파 또는 끝인 경우)
    block = re.sub(r'\b(\d{1,2})\s*GA\.?\b(?=\s|$|,)', '', block, flags=re.IGNORECASE)
    # HEAT# 이후 제거: HEAT#: 10975950C → (제거) — 열처리 번호는 치수 아님
    block = re.sub(r'\s*,?\s*\bHEAT\s*#?:?\s*\S+.*$', '', block, flags=re.IGNORECASE)
    # OD(INCH) NPS 크기 제거: OD(INCH)4" X OD(MM)114.3 → OD(MM)114.3
    block = re.sub(r'\bOD\s*\(\s*INCH\s*\)\s*[\d.]+\s*["\'"]?\s*[Xx]?\s*', '', block, flags=re.IGNORECASE)
    # NPS 명목 파이프 크기 제거: 4IN, 114.3X6.02 → 114.3X6.02
    block = re.sub(r'^\d{1,2}IN\s*,\s*(?=\d)', '', block, flags=re.IGNORECASE)
    # *N,TS: 패턴 제거: *27,TS:1400MPA → '' (TS 앞 1-2자리 여분 값은 수량/참조, 4자리 이상은 치수)
    block = re.sub(r'[*Xx]\s*\d{1,2}\b(?=\s*,\s*TS\s*:)', '', block, flags=re.IGNORECASE)
    # LENGTH/COIL WT.: 이후 제거: SIZE:7.000MM, LENGTH/COIL WT.: 400-600 KGS → 7.000MM
    block = re.sub(r',?\s*\bLENGTH\s*/\s*COIL\s+WT\.?\b.*$', '', block, flags=re.IGNORECASE)
    # (LENGTH IN...) 측정 단위 설명 괄호 제거: 118.2"(LENGTH IN INCHES) → 118.2"
    block = re.sub(r'\(\s*LENGTH\b[^)]*\)', '', block, flags=re.IGNORECASE)
    # 1/2H, 1/4H 경도 코드 제거: SIZE:13MM X 4030MM 1/2H → 13MM X 4030MM
    block = re.sub(r'\b1/[24]\s*H\b', '', block, flags=re.IGNORECASE)
    # L {a}/{b}[A-Z]? 진분수 품질/열처리 코드 제거: L 1/2C, L 1/4H (진분수 → 치수 아님)
    block = re.sub(r'\bL\s+(\d+)/(\d+)[A-Z]?\b',
                   lambda m: '' if int(m.group(1)) < int(m.group(2)) else m.group(0),
                   block, flags=re.IGNORECASE)
    # +{n}/-{n} 또는 +/-{n} 공차 표기 제거: 0.83 +0/-0.12 → 0.83, 44.5+/-0.2 → 44.5
    block = re.sub(r'\s*\+[\d.]*/\s*-[\d.]+', '', block)
    # TS 인장강도 이후 제거: X C TS:359-477MPA → X C
    block = re.sub(r'\s*\bTS\s*:.*$', '', block, flags=re.IGNORECASE)
    # USE: 이후 제거
    block = re.sub(r'\s*,?\s*\bUSE\s*:.*$', '', block, flags=re.IGNORECASE)
    # 말미 소수 수량 괄호 제거: 31*31(4) → 31*31, *60(6) → *60 (포장 수량 1-2자리)
    block = re.sub(r'\s*\(\d{1,2}\)\s*$', '', block)
    # 치수 뒤 수량+괄호 제거 (먼저): 3,724(196PCS/BUNDLE) 등
    block = re.sub(r'\s+\d[\d,]*\s*\(\d+\s*(?:PCS?|EA|BUNDLE|SHEET)\w*/?\w*\)', '', block, flags=re.IGNORECASE)
    # 수량 표기 제거: 2PCS, 3EA, 5PC, 10 SHEET, (196PCS/BUNDLE) 등
    block = re.sub(r'[,\s]\s*\d+\s*(?:PCS?|EA|NOS?|PIECES?|SETS?|BUNDLES?|SHEETS?)\s*(?:/\w+)?(?=[,\s]|$)', ' ', block, flags=re.IGNORECASE)
    block = re.sub(r'\s*\(?\d+\s*(?:PCS?|EA|NOS?|PIECES?)\s*(?:/\w+)?\)?', '', block, flags=re.IGNORECASE)
    # 괄호 안 무게 표기 제거: (9162KGS), (25KG COIL) 등
    block = re.sub(r'\s*\(\s*\d+\.?\d*\s*(?:KGS?|TONS?|LBS?|LB)(?:\s+COIL)?\s*\)', '', block, flags=re.IGNORECASE)
    # 철선 커팅 제품 설명 제거: CUT WIRE (치수 추출 방해)
    block = re.sub(r'\bCUT\s+WIRE\b', '', block, flags=re.IGNORECASE)
    # 무게×치수 형식 제거: 9.8KGX550MM → 550MM (무게 값이 치수로 잡히는 방지)
    block = re.sub(r'\b[\d.]+\s*KGS?\s*[Xx]\s*', '', block, flags=re.IGNORECASE)
    # NC/LS 코일/롯 수량 제거: 8C/LS, 10C/LS 등
    block = re.sub(r'\b\d+C/LS\b', '', block, flags=re.IGNORECASE)
    # N COLS 코일 수량 제거: 15COLS, 10COLS 등 (코일 수량 약어)
    block = re.sub(r'\b\d+\s*COLS?\b', '', block, flags=re.IGNORECASE)
    # 무게/수량 표기 제거: 1650KG, 600KG*2(무게×수량), 12.5TON, 5KC, ABOUT 3KG/BOBBIN 등
    block = re.sub(r'\s*(?:ABOUT\s*)?\d+\.?\d*\s*(?:KGS?|TON|LBS?|LB|KC)\b(?:\s*[*]\s*\d+)?(?:/\w+)?', '', block, flags=re.IGNORECASE)
    # BOBBIN 이후 내용 제거
    block = re.sub(r'\s*(?:ABOUT\s.*)?\bBOBBIN\b.*$', '', block, flags=re.IGNORECASE)
    # 표면처리 표기 제거: ACID PICKLED, ACID WASHED 등 (치수 뒤에 나타나는 처리 표기)
    block = re.sub(r'\s+ACID\s+(?:PICKLED?|WASHED?)\b.*$', '', block, flags=re.IGNORECASE)
    # ABOU1" 형태 잡음 제거 (ABOUT 오타/잘림): ABOU1", ABOUT 등
    block = re.sub(r'\bABOU[T]?\w*\s*["\']?', '', block, flags=re.IGNORECASE)
    # 동일 단위 하이픈 범위 결합: 0.3590"-0.3610" → 0.3590~0.3610"
    block = re.sub(r'([\d.]+)(["\'])\s*-\s*([\d.]+)(["\'])', r'\1~\3\4', block)
    # {n}MM~{n}MM → {n}~{n}MM 범위 표기 통일: 5530MM~5596MM → 5530~5596MM
    block = re.sub(r'([\d.]+)\s*MM\s*~\s*([\d.]+)\s*MM\b', r'\1~\2MM', block, flags=re.IGNORECASE)
    # *{n}C 또는 X{n}C 코일 수량 제거: *2C → *C, X4C → XC (수량+코일 표기)
    block = re.sub(r'([*X])\s*\d+\s*(C\b)', r'\1\2', block, flags=re.IGNORECASE)
    # *C 또는 XC 이후 그레이드 표기 제거: *C SGCC → *C
    block = re.sub(r'([*X]\s*C)\s+[A-Z].*$', r'\1', block, flags=re.IGNORECASE)
    # 코일 내경 제거: X C 359X477 → XC (C 뒤의 추가 치수는 코일 내경)
    block = re.sub(r'(\bX\s*C\b)\s+[\d].*$', r'\1', block, flags=re.IGNORECASE)
    # 버전 번호 제거: V8.5, REV.2 등
    block = re.sub(r'\bV[\d.]+\b', '', block, flags=re.IGNORECASE)
    # P-코드 제품번호 제거: P20260405-1, P20260405 (P + 7자리 이상 숫자)
    block = re.sub(r'\bP\d{7,}(?:-\d+)?', '', block, flags=re.IGNORECASE)
    # GB/T 중국 국가표준 코드 제거: GB T 17853, GB/T 106 등
    block = re.sub(r'\bGB\s*[/T]?\s*\d[\d.-]*\b', '', block, flags=re.IGNORECASE)
    # GB 뒤 비숫자(중국어 깨짐 등) 이후 끝까지 제거: GB ???? → 제거
    block = re.sub(r'\s*\bGB\s+(?![/T\d]).*$', '', block, flags=re.IGNORECASE)
    # 파이프 스케줄 표기 정규화: S80S, S40S, S160 → SCH80, SCH40, SCH160 (S45C 등 강종코드는 제외)
    block = re.sub(r'\bS(\d{2,3})S?\b', lambda m: f'SCH{m.group(1)}', block, flags=re.IGNORECASE)
    # XXS, XS 단독 스케줄 표기 보존 (정규화 불필요, tok_pat에서 이미 처리)
    # R/L (랜덤 길이) 표기 제거
    block = re.sub(r'\bR/L\b', '', block, flags=re.IGNORECASE)
    # 괄호 공차 표기 → 범위 변환: 6000(+50MM, -0) → 6000~6050, 4000(+50,-0) → 4000~4050
    block = re.sub(
        r'(\d{3,5})\s*\(\s*\+\s*(\d+)\s*(?:MM)?\s*,\s*-\s*\d+\s*\)',
        lambda m: f'{m.group(1)}~{int(m.group(1)) + int(m.group(2))}',
        block, flags=re.IGNORECASE
    )
    # 공차 +{n} 표기를 범위로 변환: 4000+50 → 4000~4050 (길이+공차)
    # 단, 와이어로프 구성 코드 등 알파 직후는 제외
    block = re.sub(
        r'(?<![A-Z])(\d{3,5})\+(\d{2,3})(?![A-Z0-9])',
        lambda m: f'{m.group(1)}~{int(m.group(1)) + int(m.group(2))}',
        block, flags=re.IGNORECASE
    )
    # 괄호 안 퍼센트 제거: (20.3%), (99.5%) 등
    block = re.sub(r'\(\s*\d+\.?\d*\s*%\s*\)', '', block)
    # 뒤따르는 퍼센트 제거: SIZE: 230X400X1250MM99.93% → SIZE: 230X400X1250MM
    block = re.sub(r'(?<=[A-Z\d])\s*\d{2,3}\.\d+\s*%', '', block)
    # HEAT NO / CAST NO / ITEM NO 이후 전부 제거 (뒤에 이어지는 값까지)
    block = re.sub(r'\b(?:HEAT|CAST|ITEM)\s*NO\b.*$', '', block, flags=re.IGNORECASE)
    # PMI 시험번호 제거: PMI45, PMI 46 등
    block = re.sub(r'\bPMI\s*\d+\b', '', block, flags=re.IGNORECASE)
    # NO.1 FINISH / NO.2B FINISH 등 표면처리 표기 제거: NO.1 FINISH, NO.2 POLISH 등
    block = re.sub(r'\bNO\.?\s*\d+\w*\s+(?:FINISH|POLISH)\b', '', block, flags=re.IGNORECASE)
    # OD/ID 3자리 이상 슬래시 표기: 268/250 → OD(큰 값), 1/4는 분수이므로 건드리지 않음
    block = re.sub(
        r'\b(\d{3,})/(\d{3,})\b',
        lambda m: m.group(1) if int(m.group(1)) >= int(m.group(2)) else m.group(2),
        block
    )
    # XS/EX/WX 카탈로그번호 제거: XS25120040, EX25051SCM415, WX145303 등 (X+5자리이상)
    block = re.sub(r'\bXS\d{4,}\b', '', block, flags=re.IGNORECASE)
    block = re.sub(r'\bEX\d{4,}\w*\b', '', block, flags=re.IGNORECASE)
    block = re.sub(r'\bWX\d{5,}\b', '', block, flags=re.IGNORECASE)
    # WX/EX 카탈로그 코드 제거 후 남는 로트번호 제거: 001, 002 등 (선행 0이 있는 3자리 단독 숫자)
    block = re.sub(r'(?<!\S)0\d{2}(?!\d)', '', block)
    # [PL] 언어 태그 이후 제거: 1/2" [PL] RURA NIERDZEWNA 1/2" → 1/2" (폴란드어 번역 제거)
    block = re.sub(r'\s*\[PL\].*$', '', block, flags=re.IGNORECASE)
    # 대괄호 내 코드/번호 제거: [0204], [A1], [123] 등 (치수 아님)
    block = re.sub(r'\s*\[[\w\s\d]+\]', '', block)
    # 선두 6자리 부품번호 제거: 699397 18/8-321N → (제거) (단위 없이 시작하는 경우만)
    block = re.sub(r'^\d{6}\b(?!\s*(?:MM|CM|M\b|IN|FT))', '', block.strip()).strip()
    # 7자리 이상 연속 숫자 (부품번호/날짜코드/시리얼) 제거 (X 뒤는 치수구분자이므로 허용)
    block = re.sub(r'(?<![A-WY-Z])\d{7,}', '', block)
    # ASTM/ASME 규격 번호 제거: SB 575, SA 312, SE 309 등 (2~4자리 숫자)
    block = re.sub(r'\bS[ABE]\s+\d{2,4}\b', '', block, flags=re.IGNORECASE)
    # AMS 재료규격 코드 제거: AMS 5536R, AMS5536 등 (4~5자리 숫자)
    block = re.sub(r'\bAMS\s*\d{4,5}[A-Z]?\b', '', block, flags=re.IGNORECASE)
    # ASTM-B-{n} 규격 코드 제거: ASTM-B-435 → ''
    block = re.sub(r'\bASTM-[A-Z]-\d{2,4}\b', '', block, flags=re.IGNORECASE)
    # UNS# 합금 번호 제거: UNS#N06002 → ''
    block = re.sub(r'\bUNS#[A-Z]\d{5}\b', '', block, flags=re.IGNORECASE)
    # 하이픈+연도 제거: A333M-2024 → A333M (-2024 = 규격 개정년도)
    block = re.sub(r'-(?:19|20)\d{2}\b', '', block)
    # {n}BUNDLES 수량 제거: 13BUNDLES, 4BUNDLES → '' (수량 표기, 치수 아님)
    block = re.sub(r'\b\d+\s*BUNDLES?\b(?:\s+\d+)?', '', block, flags=re.IGNORECASE)
    # /{grade} 강종 코드 접미사 제거: 1.5IN/434MR → 1.5IN (치수 뒤 슬래시+강종코드)
    block = re.sub(r'(?<=[A-Z\d])/\d{3,4}[A-Z]{0,3}\b', '', block)
    # 끝 치수에 붙은 스테인리스 강종 접미사 제거: X50316L → X50 (316L = SUS강종)
    block = re.sub(r'(X\d+(?:\.\d+)?)(?:316L?|304L?|347H?|321H?)\s*$', r'\1', block, flags=re.IGNORECASE)
    # 괄호 안 강종 코드 제거: (904L), (625L), (316) 등 (숫자+알파 조합) — 공백으로 치환
    # 1~4자리 숫자만 (5자리 이상은 단위 포함 치수일 수 있음: (50000MM))
    block = re.sub(r'\(\s*\d{1,4}[A-Z]{1,3}\s*\)', ' ', block, flags=re.IGNORECASE)
    # 괄호 안 구성 표기 제거: (1.6MM*7), (7*1.6MM) 등 stranded wire 구성
    block = re.sub(r'\([\d.]+\s*(?:MM|CM|M)?\s*[*Xx]\s*\d+\)', '', block, flags=re.IGNORECASE)
    block = re.sub(r'\(\d+\s*[*Xx]\s*[\d.]+\s*(?:MM|CM|M)?\)', '', block, flags=re.IGNORECASE)
    # #번호 게이지 제거: 50#, #50 (WIRE50#2.0 → 2.0)
    block = re.sub(r'\d+\s*#', '', block)
    # CT/숫자 수량 표기 제거: 30CT/10 → 제거
    block = re.sub(r'\d+\s*CT/\d+', '', block, flags=re.IGNORECASE)
    # X N드럼 수량 제거: X 4DRUMS → 제거
    block = re.sub(r'\s*[Xx]\s*\d+\s*DRUMS?\b', '', block, flags=re.IGNORECASE)
    # 인장강도 표기 제거: 1860MPA, 1570MPa
    block = re.sub(r'\b\d+\.?\d*\s*MPA\b', '', block, flags=re.IGNORECASE)
    # CERT 앞 숫자 제거: 3.1 CERT → CERT (인증서 번호가 치수로 잡히는 방지)
    block = re.sub(r'[\d.]+\s+CERT\b', 'CERT', block, flags=re.IGNORECASE)
    # 와이어 로프 구성 코드 제거: 6X19+PP, 7X7+IWRC 등
    block = re.sub(r'\b\d+[Xx]\d+\+[A-Z]+\b', '', block, flags=re.IGNORECASE)
    # (2 REELS) 등 괄호 내 릴 수량 제거 (치수 앞에 나타나는 경우): (2 REELS) 0.045 → 0.045
    block = re.sub(r'\(\s*\d+\s*REELS?\s*\)', '', block, flags=re.IGNORECASE)
    # REEL/QUANTITY 이후 제거 (와이어/코일 수량 noise): 40REELS, 24REELS, REELSXM: 등 포함
    # \d+REEL: 숫자 바로 붙은 경우(40REELS), \bREEL: 공백 구분된 경우
    block = re.sub(r'\s*,?\s*(?:\d+REEL[A-Z]*|\bREEL[A-Z]*)\b.*$', '', block, flags=re.IGNORECASE)
    block = re.sub(r'\s*,?\s*\bQUANTITY\b.*$', '', block, flags=re.IGNORECASE)
    # 수량 COILS 제거 (2자리 이상 숫자 + COILS = 수량, 코일 형태 아님): 81COILS → ''
    block = re.sub(r',?\s*\b\d{2,}\s*COILS?\b', '', block, flags=re.IGNORECASE)
    # L NCOIL 표기 제거: L 2COIL → '' (L + 단소수 + COIL → 길이가 아닌 코일 수량)
    block = re.sub(r'\s*[Xx]\s*L\s+\d{1,2}\s*COILS?\b', '', block, flags=re.IGNORECASE)
    # MATERIAL NO 이후 제거: MATERIAL NO:KMT403B823D00 → 제거 (잔류 숫자 오추출 방지)
    block = re.sub(r',?\s*\bMATERIAL\s+NO\b.*$', '', block, flags=re.IGNORECASE)
    # NETT WEIGHT 이후 제거
    block = re.sub(r'\s*,?\s*\bNETT\s+WEIGHT\b.*$', '', block, flags=re.IGNORECASE)
    # C/SIZE 제거 (Coil 크기 표기, 치수 아님)
    block = re.sub(r'\s*,?\s*\bC/SIZE\b', '', block, flags=re.IGNORECASE)
    # DRUM 이후 제거: 300 kg drum withinner tube → (무게 제거 후) drum... → 제거
    block = re.sub(r'\s*\bDRUM\b.*$', '', block, flags=re.IGNORECASE)
    # AS DRAWING 이후 제거: 1-1/4" AS DRAWING → 1-1/4"
    block = re.sub(r'\s*\bAS\s+DRAWING\b.*$', '', block, flags=re.IGNORECASE)
    # X N(CUT) 절단 길이 제거: X 2950(CUT) → '' (파이프 절단 길이는 치수 아님)
    block = re.sub(r'\s*[Xx]\s*\d+\s*\(CUT\)\b.*$', '', block, flags=re.IGNORECASE)
    # (CUT) 표기 이후 제거: 2950(CUT) → 2950
    block = re.sub(r'\s*\(CUT\)\b.*$', '', block, flags=re.IGNORECASE)
    # 괄호 내 알파+점 코드 제거: (N.C.V), (H.R.) 등 약어 코드
    block = re.sub(r'\(\s*(?:[A-Z]\.)+[A-Z]?\s*\)', '', block, flags=re.IGNORECASE)
    # 끝부분 강종/표면처리 코드 제거: 7000MM-316L/N1/ME → 7000MM
    block = re.sub(r'[-]\s*\d{3,4}[A-Z]{0,2}(?:/[A-Z0-9]{1,4})+\s*$', '', block, flags=re.IGNORECASE)
    # 잔류 하이픈 짧은 숫자 코드 제거: -17-2, -034-17 등 (강종코드 제거 후 남는 lot/작업번호)
    block = re.sub(r'\s+-\d{1,3}-\d{1,3}\b', '', block)
    # 연속 점 뒤 숫자 노이즈 제거: 2000..1 → 2000 (PC수량 제거 후 잔류)
    block = re.sub(r'(\d+)\.{2,}\d*', r'\1', block)
    # PC수량 제거 후 남는 말미 .0. 소수 노이즈 제거: 2000.0. → 2000
    block = re.sub(r'(\d+)\.0+\.\s*$', r'\1', block)
    return block.strip().rstrip('.,;').strip()


def _extract_size_block(text: str) -> Optional[str]:
    """
    SIZE: / SIZE; / SIZE= 키워드 뒤 블록 추출.
    블록이 이미 숫자X숫자 형태면 → 순서 유지 (정렬 안 함).
    """
    # SIZE[;:] {dia}MM ROUND/SQUARE {len_m}[-~{len_m2}] 패턴
    # 원형봉/사각봉 직경 + 미터 길이 (단위 M 암묵적): SIZE; 24.000MM ROUND 6.00 H9 → 24.000X6000
    round_sq_m = re.search(
        r'SIZE\s*[;:=]\s*([\d.]+)\s*MM\s+(?:ROUND|SQUARE)\s+([\d.]+)(?:\s*[-~]\s*([\d.]+))?',
        text, re.IGNORECASE
    )
    if round_sq_m:
        dia, l1, l2 = round_sq_m.group(1), round_sq_m.group(2), round_sq_m.group(3)
        try:
            fl1 = float(l1)
            if 2.0 <= fl1 <= 13.0:  # 미터 단위 범위 (2M~13M)
                l1_mm = str(round(fl1 * 1000))
                if l2:
                    l2_mm = str(round(float(l2) * 1000))
                    return f'{dia}X{l1_mm}~{l2_mm}'
                return f'{dia}X{l1_mm}'
        except ValueError:
            pass

    # SIZE/DIMENSION/LENGH?T: {dia} X {l1} X {l2}MM 전용 패턴
    # 원형봉: 직경 X 최소길이 X 최대길이 → {dia}X{l1}~{l2}
    sdl_m = re.search(
        r'SIZE\s*/\s*DIMENSION(?:\s*/\s*LENGH?T)?\s*[:;=]\s*([\d.]+)\s*[Xx]\s*([\d.]+)\s*[Xx]\s*([\d.]+)\s*MM\b',
        text, re.IGNORECASE
    )
    if sdl_m:
        dia, l1, l2 = sdl_m.group(1), sdl_m.group(2), sdl_m.group(3)
        try:
            fl1, fl2 = float(l1), float(l2)
            # 마지막 두 값이 길이 범위인지 확인: 둘 다 1000 이상, 차이 비율 15% 이내
            if fl1 >= 1000 and fl2 >= 1000 and abs(fl1 - fl2) / max(fl1, fl2) <= 0.15:
                return f'{dia}X{l1}~{l2}'
        except ValueError:
            pass

    m = re.search(
        r'SIZE(?:[/\s]+DIMENSION(?:[/\s]+LENGH?T)?)?\s*\(?(?:MM|CM|IN|FT)?\)?\s*[.:;=]\s*(.+?)(?=\s*(?:PS\s*:|MT\s*:|MODEL\s*:|SHAPE\s*:|GRADE\s*:|TEMPER\s*:|COATING\s*:|FINISH\s*:|HEAT\s*(?:NO?\s*)?:|CAST\s*NO\b|ITEM\s*NO\b|CERT\s*NO?\b|$))',
        text, re.IGNORECASE
    )
    if not m:
        # SIZE {n} X {m} [unit] — 구분자 없는 형태 (예: SUPRA STRIP SIZE 9.00 X 2000 MM)
        size_nodel_m = re.search(
            r'\bSIZE\s+([\d.]+)\s*[Xx]\s*([\d.]+)(?:\s*[Xx]\s*([\d.]+))?\s*(MM|CM|IN|FT)?\b',
            text, re.IGNORECASE
        )
        if size_nodel_m:
            vals = [size_nodel_m.group(1), size_nodel_m.group(2)]
            if size_nodel_m.group(3):
                vals.append(size_nodel_m.group(3))
            try:
                vals = [f'{float(v):g}' for v in vals]
            except ValueError:
                pass
            return 'X'.join(vals)
        return None

    block = _clean_size_block(m.group(1).strip().rstrip('.,;'))

    # D {dia}MM 단독 표기 (직경만): D8.0MM → 8, D25.4MM → 25.4
    d_only_m = re.match(r'^D\s*([\d.]+)\s*MM?\s*$', block.strip(), re.IGNORECASE)
    if d_only_m:
        val = d_only_m.group(1)
        try:
            fval = float(val)
            return str(int(fval)) if fval == int(fval) else f'{fval:g}'
        except ValueError:
            return val

    # D NMM X L NM 형식 (GUIDE SHAFT 등): L의 단위 M을 MM으로 간주
    dl_m = re.match(r'^D\s*([\d.]+)\s*MM?\s*[Xx]\s*L\s*([\d.]+)\s*M\s*$', block.strip(), re.IGNORECASE)
    if dl_m:
        return f'{dl_m.group(1)}X{dl_m.group(2)}'

    # THICKNESS N.NNMM, WIDTH/W NNNN.NNMM 전문 라벨: THICKNESS 9.00MM, WIDTH 2000.00MM → 9X2000
    # _normalize가 THICKNESS→T, WIDTH→W로 변환하므로 T/W 약어도 허용
    thkw_m = re.match(
        r'^(?:THICKNESS\s+|T\s?)([\d.]+)\s*MM,?\s*(?:WIDTH\s+|W\s?)([\d.]+)\s*MM\s*$',
        block.strip(), re.IGNORECASE
    )
    if thkw_m:
        def _fmt_g(v):
            try:
                f = float(v)
                return str(int(f)) if f == int(f) else f'{f:g}'
            except ValueError:
                return v
        return f'{_fmt_g(thkw_m.group(1))}X{_fmt_g(thkw_m.group(2))}'

    # {n}MM TX{n}MM WX{n}MM L 형식 (T/W/L이 숫자 뒤에 붙는 경우): 12MM TX1852MM WX2486MM L → 12X1852X2486
    twl_suffix_m = re.match(
        r'^([\d.]+)\s*MM\s*T[Xx]([\d.]+)\s*MM\s*W[Xx]([\d.]+)\s*MM\s*L\s*$',
        block.strip(), re.IGNORECASE
    )
    if twl_suffix_m:
        return f'{twl_suffix_m.group(1)}X{twl_suffix_m.group(2)}X{twl_suffix_m.group(3)}'

    # T/W/L 명시인 경우
    result = _extract_twl(block)
    if result:
        # T/W 2차원만 추출된 경우 trailing X NMM 길이 보완: T 0.15MM X W 810MM X 3350MM → 0.15X810X3350
        if result.count('X') == 1:
            _twl_label_pat = re.compile(
                r'(?:\([TWLH]\d*\)|\b(?:[TWLH]|OD|THK|WT)(?=[\d\s.:\-]))\s*:?\s*[\d.]+\s*(?:MM(?!\w)|CM(?!\w)|(?<!\w)M(?!\w|M)|IN(?!\w)|FT)?',
                re.IGNORECASE
            )
            _twl_end = 0
            for _m in _twl_label_pat.finditer(block):
                _twl_end = _m.end()
            if _twl_end:
                _leftover = block[_twl_end:].strip()
                _trail_m = re.match(
                    r'[Xx]\s*([\d.]+(?:[~-][\d.]+)?)\s*(MM(?!\w)|CM(?!\w)|IN(?!\w)|FT)?\b',
                    _leftover, re.IGNORECASE
                )
                if _trail_m:
                    _trailing = _convert_unit(_trail_m.group(1), _trail_m.group(2) or 'MM')
                    result = f'{result}X{_trailing}'
        # OD/WT + LENGTH 범위 미추출 보완: SIZE 블록에 LENGTH: 범위가 있으면 추가
        if not re.search(r'~', result) and not re.search(r'XC$', result, re.IGNORECASE):
            len_range_m = re.search(r'\bLENGTH\s*:\s*(\d+)\s*~\s*(\d+)', block, re.IGNORECASE)
            if len_range_m:
                result = f'{result}X{len_range_m.group(1)}~{len_range_m.group(2)}'
        return _append_coil_if_shape(result, text) or result

    # T{t}MM X{w}MM 패턴 (W 접두어 없음): T14.7MM X1217MM → thickness x width
    t_bare_m = re.match(
        r'^T\s*([\d.]+)\s*MM\s*[Xx]\s*([\d.]+)\s*MM(?:\s*,.*)?$',
        block.strip(), re.IGNORECASE
    )
    if t_bare_m:
        t, w = t_bare_m.group(1), t_bare_m.group(2)
        return _append_coil_if_shape(f'{t}X{w}', text) or f'{t}X{w}'

    # 블록 끝 단독 C (코일 표시자) 분리: "X C", "*C", 공백+C
    has_coil = bool(re.search(r'(?:\s*[Xx*]\s*|\s+)C\s*$', block, re.IGNORECASE))
    if has_coil:
        block = re.sub(r'(?:\s*[Xx*]\s*|\s+)C\s*$', '', block, flags=re.IGNORECASE).strip()

    # SHAPE:SHEET 낱장 제품은 SIZE 블록의 XC 무시 (SLIT/CRC 코일납품은 제외)
    def _coil_flag(text_full: str) -> bool:
        if not has_coil:
            return False
        if (re.search(r'SHAPE\s*:\s*SHEET', text_full, re.IGNORECASE)
                and not re.search(r'\b(?:SLIT|CRC)\b', text_full, re.IGNORECASE)):
            return False
        return True

    # 이미 숫자X숫자 형태 → 순서 유지
    if re.match(r'^[\d.~/]+(?:[Xx][\d.~/]+)+[Xx]?[Cc]?$', block.strip()):
        result = block.strip().upper().replace('x', 'X')
        return (result + 'XC') if _coil_flag(text) else result

    # 그 외 → 강종코드 제거 후 토큰 추출 (순서 유지)
    cleaned = _strip_grade_codes(block)
    result = _parse_tokens(cleaned, preserve_order=True)
    if result and _coil_flag(text):
        result = result + 'XC'
    return result


def _is_part_number_spec(text: str) -> bool:
    """부품번호/시리얼 포함 규격 판별 — regex 추출 불가, LLM으로 넘김"""
    # KP로 시작하는 시리얼 번호 (예: KPJ05897026256, KPW010V0000268)
    has_kp_serial = bool(re.search(r'\bKP[A-Z0-9]{8,}', text))
    # 10자리 이상 순수 숫자 시리얼
    has_long_serial = bool(re.search(r'(?<![A-Z\d])\d{10,}(?!\d)', text))
    # 명시적 치수 단위 없음
    has_unit = bool(re.search(r'\b(?:MM|CM|MTR|METER|IN(?:CH)?|FT)\b', text, re.IGNORECASE))
    # X 연결 숫자 없음 (예: 2.0X1219X2438)
    has_x = bool(re.search(r'\d[Xx]\d', text))
    # SIZE: 키워드 없음
    has_size_kw = bool(re.search(r'\bSIZE\s*[:;=]', text, re.IGNORECASE))

    return (has_kp_serial or has_long_serial) and not (has_unit or has_x or has_size_kw)


def _strip_trailing_zeros(size: str) -> str:
    """Strip trailing decimal zeros from dimension numbers.
    '0.200X40XC' -> '0.2X40XC', '20X20X8.000IN' -> '20X20X8IN'
    """
    def _fmt(m):
        try:
            return f'{float(m.group(1)):g}{m.group(2) or ""}'
        except ValueError:
            return m.group(0)
    return re.sub(r'(\d+\.\d+)(IN|FT)?', _fmt, size)


def extract_size_regex(spec_text: str) -> Optional[str]:
    """정규식 기반 사이즈 추출"""
    text = _normalize(spec_text)

    # WAVEGUIDE 길이 추출: GLG-100/P51MJ3WAVEGUIDE ... L= 4200MM → 4200
    if re.search(r'WAVEGUIDE', text, re.IGNORECASE):
        wg_m = re.search(r'\bL\s*=\s*([\d.]+)\s*MM\b', text, re.IGNORECASE)
        if wg_m:
            v = wg_m.group(1)
            try:
                fv = float(v)
                return str(int(fv)) if fv == int(fv) else v
            except ValueError:
                return v

    # T-star 형식 전용 핸들러: {T}T-{W}*{H}[*{L}] — 색상코드/수량/접미사 이후 무시
    # 예: 14T-22*29/241-6(6) → 14X22X29, 19T-32*31*31(4) → 19X32X31X31
    tstar_m = re.match(r'^(\d+)\s*T-\s*(\d+)\s*\*\s*(\d+)(?:\s*\*\s*(\d+))?\b', text.strip(), re.IGNORECASE)
    if tstar_m:
        dims = [g for g in tstar_m.groups() if g]
        return 'X'.join(dims)

    # N.NNNxN.NNNx0.00[NNN]... 코일 스펙: 세 번째 차원이 0에 가까운 경우 → T×W×C
    # normalize로 0.00,304 → 0.00304로 변환되므로 0.00\d* 패턴 사용
    # SIZE: 접두어 유무 모두 처리 (예: SIZE: 1.000x1219.000x0.00,MODEL: 304/2B/TE)
    def _fmt_int_if_whole(v: str) -> str:
        try:
            f = float(v)
            return str(int(f)) if f == int(f) else f'{f:g}'
        except ValueError:
            return v

    coil_twx0_m = re.search(
        r'(?:\bSIZE\s*:\s*)?([\d.]+)[Xx]([\d.]+)[Xx]0\.0{2,}\d*(?=[,/\s]|$)',
        text, re.IGNORECASE
    )
    if coil_twx0_m:
        t_val = _fmt_int_if_whole(coil_twx0_m.group(1))
        w_val = _fmt_int_if_whole(coil_twx0_m.group(2))
        return f'{t_val}X{w_val}XC'

    # ALLOY STRIP COIL 인치 패턴: {t}"X{w}"XCOIL → {t}INX{w}INXC
    # 정수 두께→3dp 유지 (38.00000→38.000), 소수→:g (9.85000→9.85, 0.003→0.003)
    # 첫 숫자: 정수부 최대 4자리 (부품번호 N06000010 오인식 방지)
    inch_coil_m = re.search(
        r'(?<!\d)(\d{1,4}(?:\.\d+)?)\s*"\s*[Xx]\s*([\d.]+)\s*"\s*[Xx]?\s*COIL',
        text, re.IGNORECASE
    )
    if inch_coil_m:
        def _fmt_ic(s):
            g = f'{float(s):g}'
            return g if '.' in g else f'{float(s):.3f}'
        return f'{_fmt_ic(inch_coil_m.group(1))}INX{_fmt_ic(inch_coil_m.group(2))}INXC'

    # ALLOY COIL 공차 범위 패턴: T {a}"/{b}" X {w}" W ... → {b}INX{w}INXC
    # 예: ALLOY X-750 ... T 0.0036"/0.00644" X 12.00" W → 0.00644INX12.00INXC
    if re.search(r'\bALLOY\b.*COIL', text, re.IGNORECASE):
        alloy_coil_m = re.search(
            r'\bT\s+(?:[\d.]+"/)?([\d.]+)"\s*[Xx]\s*([\d.]+)"\s*W',
            text, re.IGNORECASE
        )
        if alloy_coil_m:
            t_ac = alloy_coil_m.group(1)
            w_ac = alloy_coil_m.group(2)
            try:
                t_ac = f'{float(t_ac):g}'
            except ValueError:
                pass
            return f'{t_ac}INX{w_ac}INXC'

    # BA STRIP 전용 핸들러: BA STRIP {t}[MM] [X {w}W] [X C] → {t}X{w}XC 또는 {t}
    # 예: UNS N06625 BA STRIP 0.152 X 480W X C → 0.152X480XC
    # 예: UNS N06625 BA STRIP 0.102 X C → 0.102
    if re.search(r'\bBA\s+STRIP\b', text, re.IGNORECASE):
        ba_t = re.search(r'\bBA\s+STRIP\s+([\d.]+)', text, re.IGNORECASE)
        if ba_t:
            t_ba = ba_t.group(1)
            try:
                t_ba = f'{float(t_ba):g}'
            except ValueError:
                pass
            ba_w = re.search(r'([\d.]+)\s*(?:MM\s*)?W\b', text, re.IGNORECASE)
            if ba_w and ba_w.group(1) != ba_t.group(1):
                w_ba = ba_w.group(1)
                try:
                    w_ba = f'{float(w_ba):g}'
                except ValueError:
                    pass
                return f'{t_ba}X{w_ba}XC'
            return t_ba

    # {code} Plate N x N x Nmm 형식 (유럽식 구매주문서): 1R0053322 Plate 3 x 1500 x3000mm → 3X1500X3000
    plate_xyz_m = re.search(
        r'\bPlate\s+([\d,]+(?:\.[\d]+)?)\s*[xX]\s*([\d,]+)\s*[xX]\s*([\d,]+)\s*mm\b',
        text, re.IGNORECASE
    )
    if plate_xyz_m:
        def _eu_num(v: str) -> str:
            v = v.replace(',', '.')
            try:
                f = float(v)
                return str(int(f)) if f == int(f) else f'{f:g}'
            except ValueError:
                return v
        t_p = _eu_num(plate_xyz_m.group(1))
        w_p = plate_xyz_m.group(2)
        l_p = plate_xyz_m.group(3)
        return f'{t_p}X{w_p}X{l_p}'

    # GRADE:{code} {dim1} X {dim2} [X {dim3}] MM 인라인 치수 패턴 (GRADE 트런케이션 전에 추출)
    # 예: GRADE:Z-M4 77.20 X 377.83 X 1555.75MM → 77.20X377.83X1555.75
    grade_inline_m = re.search(
        r'\bGRADE\s*:\s*\S+\s+([\d.]+)\s+[Xx]\s+([\d.]+)(?:\s+[Xx]\s+([\d.]+))?\s*MM\b',
        text, re.IGNORECASE
    )
    if grade_inline_m:
        d1 = grade_inline_m.group(1)
        d2 = grade_inline_m.group(2)
        d3 = grade_inline_m.group(3)
        return f'{d1}X{d2}X{d3}' if d3 else f'{d1}X{d2}'

    # GRADE:?  {grade_desc} {float}MM [X/*] {float}MM [X/*] C 코일 패턴
    # 예: GRADE:CRC SUS430 2B MILL 1.20MM X 1219.00MM X C → 1.20X1219.00XC
    # 예: GRADE 204R1 2B WITH PAPER INTERLEAVED SLIT EDGE 0.60MM*1100MM*C → 0.60X1100XC
    grade_mill_m = re.search(
        r'\bGRADE\s*:?\s*.*?([\d.]+)\s*MM\s*[Xx*]\s*([\d.]+)\s*MM\s*[Xx*]\s*C\b',
        text, re.IGNORECASE
    )
    if grade_mill_m:
        return f'{grade_mill_m.group(1)}X{grade_mill_m.group(2)}XC'

    # SWCH/보론강 강종코드/직경 패턴: SWCH18A/2.8MMXC → 2.8, SWCH35K 45MM → 45, 10B21/5.5MMXC → 5.5
    # 강종코드(슬래시 또는 공백 뒤) 무시하고 직경만 추출
    swch_m = re.search(r'\bSWCH\w*(?:/|\s+)([\d.]+)\s*MM', text, re.IGNORECASE)
    if swch_m:
        return swch_m.group(1)
    b21_m = re.search(r'\b\d{2}B\d{2}/\s*([\d.]+)\s*MM', text, re.IGNORECASE)
    if b21_m:
        return b21_m.group(1)

    # PATENTED STEEL WIRE 직경 추출: DIA(MM){strength_code}{dia} 형식
    # 예: WIREDIA(MM)82A1.4 → 1.4, WIREDIA(MM)92A2.00 → 2
    # \bWIRE 뒤 DIA가 붙어 word boundary 없으므로 후행 \b 사용 안 함
    if re.search(r'\bPATENTED\s+STEEL\s+WIRE', text, re.IGNORECASE):
        pat_m = re.search(r'(?:\d{2,3}[A-Z]\s*)?([\d.]+)\s*$', text.strip(), re.IGNORECASE)
        if pat_m:
            val = pat_m.group(1)
            try:
                fval = float(val)
                return str(int(fval)) if fval == int(fval) else f'{fval:g}'
            except ValueError:
                return val

    # 제품코드 N-NNN + 단독 치수MM 패턴: 3-347 2.20MM 50KG*10 → 2.2
    # 3-347, 3-348 같은 제품번호(1-3자리-3자리) 뒤에 오는 MM 치수가 실제 사이즈
    prod_code_mm_m = re.search(r'^\d{1,2}-\d{3}\s+([\d.]+)\s*MM\b', text.strip(), re.IGNORECASE)
    if prod_code_mm_m:
        val = prod_code_mm_m.group(1)
        try:
            fval = float(val)
            return str(int(fval)) if fval == int(fval) else f'{fval:g}'
        except ValueError:
            return val

    # H-BEAM/I-BEAM/형강 전용 핸들러: N×N×tw/tf 슬래시 분리자 + 선택적 M 길이
    # 예: H200X200X8/12 → 200X200X8X12,  300X150X10/18.5 10M → 300X150X10X18.5X10000
    # 컨텍스트: H-BEAM, I-BEAM, SHAPE:H SECTION, JIS G3101 등 구조용 형강 표준
    _beam_ctx = bool(
        re.search(r'\b(?:H[-\s]?BEAM|I[-\s]?BEAM|SHAPE\s*:\s*(?:H\s*SECTION|I\s*BEAM))\b', text, re.IGNORECASE)
        or re.search(r'\bJIS\s+G\s*3(?:101|192|194|350|444)\b', text, re.IGNORECASE)
    )
    if _beam_ctx:
        hbeam_slash_m = re.search(
            r'([\d.]+)\s*[Xx]\s*([\d.]+)\s*[Xx]\s*([\d.]+)/([\d.]+)'  # d1×d2×tw/tf
            r'(?:\s*([\d]{4,6})\s*MM\b)?'  # normalize된 M길이 (N M → N000MM)
            r'(?:\s+(\d{1,2})(?:\s|$))?',  # 또는 소정수 M단위 (3~20)
            text, re.IGNORECASE
        )
        if hbeam_slash_m:
            d1, d2, tw, tf = hbeam_slash_m.group(1), hbeam_slash_m.group(2), hbeam_slash_m.group(3), hbeam_slash_m.group(4)
            l_mm = hbeam_slash_m.group(5)
            l_small = hbeam_slash_m.group(6)
            dims = [d1, d2, tw, tf]
            if l_mm:
                dims.append(l_mm)
            elif l_small:
                try:
                    n = int(l_small)
                    if 3 <= n <= 20:
                        dims.append(str(n * 1000))
                except ValueError:
                    pass
            return 'X'.join(dims)

    # SIZE:H{n}MMXL{n}MM 형식 (H=직경, L=길이): H200MMXL10100MM → 200X10100
    # SHAPE:H SECTION 컨텍스트 + H×L 두 차원만
    if re.search(r'SHAPE\s*:\s*H\s*SECTION', text, re.IGNORECASE):
        hml_m = re.search(
            r'\bH\s*([\d.]+)\s*MM\s*[Xx]?\s*L\s*([\d.]+)\s*MM\b',
            text, re.IGNORECASE
        )
        if hml_m:
            return f'{hml_m.group(1)}X{hml_m.group(2)}'

    # JIS G 구조용 형강 표준: NxNxN N (SIZE: 키워드 없이 치수+M단위 길이, 슬래시 없음)
    # JIS G 3101 SS400 300X90X9 10 → 300X90X9X10000
    if re.search(r'\bJIS\s+G\s+3(?:101|192|194|350|444)\b', text, re.IGNORECASE):
        jis_beam_m = re.search(
            r'(\d+)\s*[Xx*]\s*(\d+)\s*[Xx*]\s*(\d+(?:\.\d+)?)\s+(\d{1,2})\s*$',
            text, re.IGNORECASE
        )
        if jis_beam_m:
            d1, d2, d3, l = jis_beam_m.groups()
            try:
                l_mm = int(l) * 1000
                return f'{d1}X{d2}X{d3}X{l_mm}'
            except ValueError:
                pass

    # CCRG 클래드 강판: (T{base}+{clad})X(W{a},{bcd})X(L{e},{fgh}) → 합산두께X폭X길이
    # 예: (T85+3)X(W3,180)X(L5,700) → 88X3180X5700
    clad_m = re.search(
        r'\(T(\d+)\+(\d+)\)\s*[Xx]\s*\(W(\d+),(\d{3})\)\s*[Xx]\s*\(L(\d+),(\d{3})\)',
        text, re.IGNORECASE
    )
    if clad_m:
        t = int(clad_m.group(1)) + int(clad_m.group(2))
        w = clad_m.group(3) + clad_m.group(4)
        l = clad_m.group(5) + clad_m.group(6)
        return f'{t}X{w}X{l}'

    # ST52 BK+S 튜브: O.D{od}xI.D{id}xL{l},{lll} → OD X Length
    # 예: O.D130.0xI.D110.0xL6,200(10PC) → 130.0X6200
    od_id_l_m = re.search(
        r'O\.D\s*([\d.]+)\s*[Xx]\s*I\.D\s*[\d.]+\s*[Xx]\s*L\s*(\d+),(\d{3})',
        text, re.IGNORECASE
    )
    if od_id_l_m:
        od = od_id_l_m.group(1)
        l = od_id_l_m.group(2) + od_id_l_m.group(3)
        return f'{od}X{l}'

    # SIZE:N.NNN*N.NNN*N 미터 단위 대형 플레이트 (SHAPE:PLATE 컨텍스트)
    # 예: SIZE:6.800 * 3.700 * 13 → 6800X3700X13 (앞 두 값은 미터, 마지막은 mm)
    if re.search(r'SHAPE\s*:\s*PLATE', text, re.IGNORECASE):
        plate_m_m = re.search(
            r'SIZE\s*:\s*(\d+\.\d{3})\s*\*\s*(\d+\.\d{3})\s*\*\s*(\d+)',
            text, re.IGNORECASE
        )
        if plate_m_m:
            l = str(round(float(plate_m_m.group(1)) * 1000))
            w = str(round(float(plate_m_m.group(2)) * 1000))
            t = plate_m_m.group(3)
            return f'{l}X{w}X{t}'

    # MAEU 해운 컨테이너 화물 각도형강: MAEU266195296/0012/19322695-012/ANGLE,...;ANGLE-UNE Q 100X50X8T
    if re.search(r'\bMAEU\d{6,}\b', text, re.IGNORECASE) and re.search(r'\bANGLE\b', text, re.IGNORECASE):
        rest = re.sub(r'\bMAEU\d+(?:/[^/\s,;]{1,30}){2,}/\s*', '', text, flags=re.IGNORECASE)
        # H/W/T 레이블 패턴: H100 X W50 X T8MM → 100X50X8
        hwt_m = re.search(r'\bH\s*([\d.]+)\s*X\s*W\s*([\d.]+)\s*X\s*T\s*([\d.]+)', rest, re.IGNORECASE)
        if hwt_m:
            return f'{hwt_m.group(1)}X{hwt_m.group(2)}X{hwt_m.group(3)}'
        rest_clean = re.sub(r'(\d)T\b', r'\1', rest)  # 8T → 8 (두께 suffix 제거)
        angle_m = re.search(r'\b([\d.]+)[Xx]([\d.]+)[Xx]([\d.]+)\b', rest_clean, re.IGNORECASE)
        if angle_m:
            h, b, t = angle_m.group(1), angle_m.group(2), angle_m.group(3)
            if all(float(v) < 1000 for v in [h, b, t]):
                return f'{h}X{b}X{t}'

    # MAEU 해운 화물 직사각/정사각 파이프: THK N MM;H N MM;W N MM 패턴
    # THK1 2.5 MM 처럼 공백으로 분리된 숫자도 허용 (예: THK12.5 → THK1 2.5)
    if re.search(r'\bMAEU\d{6,}\b', text, re.IGNORECASE):
        thk_hw_m = re.search(
            r'\bTHK\s*([\d]+\s*[\d.]*)\s*MM\s*;H\s*([\d.]+)\s*MM\s*;W\s*([\d.]+)\s*MM\b',
            text, re.IGNORECASE
        )
        if thk_hw_m:
            thk = thk_hw_m.group(1).replace(' ', '')
            return f'{thk}X{thk_hw_m.group(2)}X{thk_hw_m.group(3)}'

    # PFC 채널형강: PFC{H}X{B}X{t} + LENGTH: NM → HxBxtxL_mm
    # \b 대신 음수 전방탐색: 3.1PFC100X50X10 에서 1P 사이에 word boundary 없음
    pfc_m = re.search(r'(?<![A-Z])PFC\s*([\d.]+)[Xx]([\d.]+)[Xx]([\d.]+)', text, re.IGNORECASE)
    if pfc_m:
        h, b, t = pfc_m.group(1), pfc_m.group(2), pfc_m.group(3)
        len_m2 = re.search(r'\bLENGTH\s*:\s*([\d.]+)\s*(M\b|MM\b)', text, re.IGNORECASE)
        if len_m2:
            lv, lu = len_m2.group(1), len_m2.group(2).upper()
            l_mm = str(round(float(lv) * 1000)) if lu == 'M' else lv
            return f'{h}X{b}X{t}X{l_mm}'
        return f'{h}X{b}X{t}'

    # (W)N.NM X STEEL(TH)N.NNMM X TOTAL(TH)N.NNMM 폭×두께범위 표기 (SEK02 형식)
    sek_wth_m = re.search(
        r'\(W\)\s*([\d.]+)\s*M\s*[Xx]\s*(?:\w+\s*)?\(TH\)\s*([\d.]+)\s*MM\s*[Xx]\s*(?:\w+\s*)?\(TH\)\s*([\d.]+)\s*MM',
        text, re.IGNORECASE
    )
    if sek_wth_m:
        w, th1, th2 = sek_wth_m.group(1), sek_wth_m.group(2), sek_wth_m.group(3)
        return f'{w}X{th1}~{th2}'

    # CHANNEL 4차원 치수 블록: HxBxtxL (카탈로그번호 혼재 스펙에서 직접 추출)
    # 예: BSYK840 6862UU52L005-01 1-25 150X75X9X6000(BSYK840) → 150X75X9X6000
    if re.search(r'\bCHANNEL\b', text, re.IGNORECASE):
        ch4_m = re.search(r'\b(\d+)[Xx](\d+)[Xx](\d+)[Xx](\d+)\b', text)
        if ch4_m:
            return f'{ch4_m.group(1)}X{ch4_m.group(2)}X{ch4_m.group(3)}X{ch4_m.group(4)}'

    # McMaster-Carr 스퀘어 튜브: SIZE:T N/N" X N-N/N" X N-N/N" L NFT
    # 예: SIZE:T 1/4" X 2" X 2" L 8FT → 1/4INX2INX2INX8FT
    sq_tube_m = re.search(
        r'SIZE\s*:?\s*T\s*:?\s*([\d/]+)"\s*[Xx]\s*([\d/]+(?:-[\d/]+)?)\s*"\s*[Xx]\s*([\d/]+(?:-[\d/]+)?)\s*"\s*(?:[Xx]\s*)?L\s*([\d/]+)\s*FT',
        text, re.IGNORECASE
    )
    if sq_tube_m:
        t, w1, w2, l = sq_tube_m.group(1), sq_tube_m.group(2), sq_tube_m.group(3), sq_tube_m.group(4)
        return f'{t}INX{w1}INX{w2}INX{l}FT'

    # T(min/max") X W N" X L N" 판재 두께범위 표기 (PLATE-BAFFLE RING 등)
    t_range_wl_m = re.search(
        r'\bT\s*\(\s*([\d.]+)\s*/\s*([\d.]+)\s*"\s*\)\s*[Xx]\s*W\s*([\d.]+)\s*"\s*[Xx]\s*L\s*([\d.]+)\s*"',
        text, re.IGNORECASE
    )
    if t_range_wl_m:
        t_max = t_range_wl_m.group(2)  # 최대값
        w, l = t_range_wl_m.group(3), t_range_wl_m.group(4)
        return f'{t_max}INX{w}INX{l}IN'

    # GA 게이지 ASTM SHEET: SIZE:{w}INCH X {l}INCH + THICKNESS:{n}GA → {n}X{w}INX{l}IN
    # (normalize가 THICKNESS:N GA → TN GA로 변환하므로 T패턴도 처리)
    if re.search(r'\bSHEET\b', text, re.IGNORECASE):
        ga_m = re.search(r'(?:THICKNESS\s*:?\s*|(?<!\w)T)(\d+)\s*GA\b', text, re.IGNORECASE)
        size_inch_m = re.search(r'SIZE\s*:\s*([\d.]+)\s*INCH\s*[Xx]\s*([\d.]+)\s*INCH', text, re.IGNORECASE)
        if ga_m and size_inch_m:
            return f'{ga_m.group(1)}X{size_inch_m.group(1)}INX{size_inch_m.group(2)}IN'

    # SNAP FIT SPIGOT: D{od}X{l} → OD x Length (원래 순서 유지)
    if re.search(r'SPIGOT', text, re.IGNORECASE):
        spigot_m = re.search(r'\bD\s*(\d+)\s*[Xx]\s*(\d+)\b', text, re.IGNORECASE)
        if spigot_m:
            return f'{spigot_m.group(1)}X{spigot_m.group(2)}'

    # BIMETALLIC 클래드강: N.N(BACK STEEL M.NMM)*W*L → MXW XL (BACK STEEL 두께가 실 치수)
    back_steel_m = re.search(
        r'\(BACK\s+STEEL\s+([\d.]+)\s*MM\)\s*\*\s*(\d+)\s*\*\s*(\d+)',
        text, re.IGNORECASE
    )
    if back_steel_m:
        t_str, w, l = back_steel_m.groups()
        t_val = float(t_str)
        t_fmt = str(int(t_val)) if t_val == int(t_val) else t_str
        return f'{t_fmt}X{w}X{l}'

    # SFHI 부품번호: SFHI-26-3019CR-1050Y-1300T-...-U{t}MM*{w}MM*C → tXwXC
    if re.search(r'\bSFHI\b', text, re.IGNORECASE):
        sfhi_m = re.search(r'\bU(\d+)MM\s*\*\s*(\d+)MM\s*\*\s*C\b', text, re.IGNORECASE)
        if sfhi_m:
            return f'{sfhi_m.group(1)}X{sfhi_m.group(2)}XC'

    # MSD SPRAYCARD: 'SPRAYCARD NxNxN CDF...' → NxNxN 그대로 (원래 순서 유지)
    if re.search(r'SPRAYCARD', text, re.IGNORECASE):
        sc_m = re.search(r'SPRAYCARD\s*([\d.]+)\s*[Xx]\s*([\d.]+)\s*[Xx]\s*([\d.]+)', text, re.IGNORECASE)
        if sc_m:
            return f'{sc_m.group(1)}X{sc_m.group(2)}X{sc_m.group(3)}'

    # SIZE:NXNxC 코일 — WEIGHT/LENGTH 이후 정보 무시
    size_xc_m = re.search(r'\bSIZE\s*:\s*([\d.]+)\s*[Xx]\s*([\d.]+)\s*[Xx]\s*C\b', text, re.IGNORECASE)
    if size_xc_m:
        return f'{size_xc_m.group(1)}X{size_xc_m.group(2)}XC'

    # LANYARD: N/NN" ROPE DIAMETER, N" LONG → 직경X길이 (인치)
    if re.search(r'LANYARD', text, re.IGNORECASE):
        rope_m = re.search(r'([\d/]+)"\s*ROPE\s*DIAMETER,?\s*([\d/]+)"\s*LONG', text, re.IGNORECASE)
        if rope_m:
            return f'{rope_m.group(1)}INX{rope_m.group(2)}IN'
        # DIA + LG 패턴: 3/64 DIA NYLON COATED 6"LG → 3/64INX6IN
        dia_lg_m = re.search(r'([\d/]+)\s+DIA\b.*?([\d./]+)\s*"?\s*LG\b', text, re.IGNORECASE)
        if dia_lg_m:
            return f'{dia_lg_m.group(1)}INX{dia_lg_m.group(2)}IN'
        # 직경만 MM 단독 표기: LANYARD 10 IN (254MM) → 254
        lan_mm = re.search(r'([\d.]+)\s*MM\b', text, re.IGNORECASE)
        if lan_mm:
            try:
                return f'{float(lan_mm.group(1)):g}'
            except ValueError:
                return lan_mm.group(1)

    # PRODUCT DIMENSIONS {dia} DIA X {len} IN LONG: 잉곳/바 단순 직경×길이
    # 예: PRODUCT DIMENSIONS 3 DIA X 18 IN LONG → 3X18IN
    pd_dim_m = re.search(
        r'PRODUCT\s+DIMENSIONS?\s+([\d./]+)\s+DIA\s+[Xx]\s+([\d./]+)\s+IN\b',
        text, re.IGNORECASE
    )
    if pd_dim_m:
        return f'{pd_dim_m.group(1)}X{pd_dim_m.group(2)}IN'

    # WIRE TURNBUCKLE {frac}*{n}MM*{l}M(M): 인치 분수 OD × 두께 × 길이
    # 예: 3/4*8MM*6M (L) → 3/4INX8X6000
    if re.search(r'\bTURNBUCKLE\b', text, re.IGNORECASE):
        tb_m = re.search(
            r'\b(\d+/\d+)\s*\*\s*([\d.]+)\s*MM\s*\*\s*([\d.]+)\s*(MM\b(?!M)|M\b(?!M))',
            text, re.IGNORECASE
        )
        if tb_m:
            frac, size2, l_raw, l_unit = tb_m.groups()
            l_val = _convert_unit(l_raw, l_unit.upper().replace(' ', ''))
            return f'{frac}INX{size2}X{l_val}'

    # WALL PIPE: {h}X{w}X{wall} WALL X {len}M(ETER) → {max(h,w)}INX{wall}INX{len*1000}
    # 예: 8X4X1/2 WALL X 6METER137316ASTM A500 → 8INX1/2INX6000
    if re.search(r'\bWALL\b', text, re.IGNORECASE) and re.search(r'\bPIPE\b', text, re.IGNORECASE):
        wall_m = re.search(
            r'\b(\d+(?:\.\d+)?)\s*[Xx]\s*(\d+(?:\.\d+)?)\s*[Xx]\s*(\d+/\d+|\d+(?:\.\d+)?)\s+WALL\s+[Xx]\s*([\d.]+)\s*M(?:ETER)?',
            text, re.IGNORECASE
        )
        if wall_m:
            h, w, wall, l_raw = wall_m.groups()
            l_val = _convert_unit(l_raw, 'M')
            try:
                od = h if float(h) >= float(w) else w
            except ValueError:
                od = h
            return f'{od}INX{wall}INX{l_val}'

    # TUBING {frac}" OD with ID present: OD만 추출 (1/16 OD, 0.010" ID → 1/16IN)
    # normalize가 TS110TUBING을 제거하므로 원본 spec_text 사용
    if re.search(r'TUBING\b', spec_text, re.IGNORECASE) and re.search(r'\bOD\b.*\bID\b', text, re.IGNORECASE):
        tubing_od_m = re.search(r'([\d/]+(?:\.\d+)?)\s*"?\s+OD\b', text, re.IGNORECASE)
        if tubing_od_m:
            return f'{tubing_od_m.group(1)}IN'

    # SOLDER-WIRE D{n} 직경 추출: WAVE SOLDER-WIRE,D3.0, → 3 (콤마로 구분된 D+숫자)
    if re.search(r'\bSOLDER\b', text, re.IGNORECASE):
        sol_m = re.search(r',D([\d.]+),', text, re.IGNORECASE)
        if sol_m:
            val = sol_m.group(1)
            try:
                return f'{float(val):g}'
            except ValueError:
                return val

    # SEAMLESS PIPE OD*ID*WT*L: 내경(ID) 제거, OD×WT×L 추출
    # 예: 68.0*48.4*9.8MM 25MM → 68.0X9.8X25 (OD=68.0, ID=48.4 제거, WT=9.8, L=25)
    if re.search(r'\bSEAMLESS\b', text, re.IGNORECASE):
        pipe_oidwt_m = re.search(
            r'([\d.]+)\s*\*\s*([\d.]+)\s*\*\s*([\d.]+)\s*MM\b\s+([\d.]+)\s*MM\b',
            text, re.IGNORECASE
        )
        if pipe_oidwt_m:
            od, mid_val, wt, length = pipe_oidwt_m.groups()
            # 검증: OD > ID > WT (중간값이 ID인지 확인)
            try:
                if float(od) > float(mid_val) > float(wt):
                    return f'{od}X{wt}X{length}'
            except ValueError:
                pass

    # CUT TO {n}IN: 절단 최종 길이 추출 → 39.53IN CUT TO pattern
    # 예: ... CUT TO 39.53IN(+,1250/-.0000IN) → 39.53IN
    cut_to_m = re.search(r'\bCUT\s+TO\s+([\d.]+)\s*IN\b', text, re.IGNORECASE)
    if cut_to_m:
        return f'{cut_to_m.group(1)}IN'

    # SCH 파이프 괄호 실치수: 50AX SCH80X 6000MM (60.50MM*5.5MM) → 60.50X5.5X6000
    # 공칭치수(NPS+SCH) 옆 괄호에 실OD×WT가 있을 때 우선 사용
    sch_actual_m = re.search(
        r'SCH\d+X?\s*([\d.]+)\s*MM\b[^(]*\(\s*([\d.]+)\s*MM\s*[*Xx]\s*([\d.]+)\s*MM\s*\)',
        text, re.IGNORECASE
    )
    if sch_actual_m:
        l, d1, d2 = sch_actual_m.groups()
        return f'{d1}X{d2}X{l}'

    # {frac}" OD + {n}" WALL + {n}FT LENGTH 순서 보존: 1/2" OD, 0.049" WALL, 1FT → 1/2INX0.049INX1FT
    od_wall_ft_m = re.search(
        r'([\d/]+)\s*"\s+OD[,\s]+([\d./]+)\s*"\s+WALL[^,\n]*[,\s]+([\d.]+)\s*(?:FT|FOOT)\s+L(?:ENGTH|ONG)?\b',
        text, re.IGNORECASE
    )
    if od_wall_ft_m:
        od, wt, l = od_wall_ft_m.groups()
        return f'{od}INX{wt}INX{l}FT'

    # SIZE: D {n}CMX L{n}CM: 직경(cm) × 길이(cm) → mm 변환: D 1CMX L63CM → 10X630
    d_cm_m = re.search(r'\bD\s*([\d.]+)\s*CMX?\s*L\s*([\d.]+)\s*CM\b', text, re.IGNORECASE)
    if d_cm_m:
        od_val = round(float(d_cm_m.group(1)) * 10)
        l_val = round(float(d_cm_m.group(2)) * 10)
        return f'{od_val}X{l_val}'

    # {n}MM OD + {n}MM WALL + {n}MM LENGTH 순서 보존: 6MM OD, 0.25MM WALL, 500MM → 6X0.25X500
    od_wall_len_m = re.search(
        r'([\d.]+)\s*MM\s+OD[,\s]+([\d.]+)\s*MM\s+WALL[^,\n]*[,\s]+([\d.]+)\s*MM\s+L(?:ENGTH)?\b',
        text, re.IGNORECASE
    )
    if od_wall_len_m:
        od, wt, l = od_wall_len_m.groups()
        return f'{od}X{wt}X{l}'

    # GEWA ENHANCED-FINNED TUBES: OD × 총길이 추출 (O54 재질코드 제외, LENGTH만 사용)
    # 예: GEWA-PB 3/4" MATERIAL:O54 ... L7400MM → 3/4INX7400
    if re.search(r'\bGEWA\b', text, re.IGNORECASE):
        gewa_od = re.search(r'(\d+/\d+)\s*"', text, re.IGNORECASE)
        gewa_len = re.search(r'(?<!\w)L\s*([\d.]+)\s*MM\b', text, re.IGNORECASE)
        if gewa_od and gewa_len:
            return f'{gewa_od.group(1)}INX{gewa_len.group(1)}'

    # PLATE SIZE 다중 길이 목록 → min~max 범위: SIZE:7200MM,7700MM,...,5200MM → 5200~7700
    if re.search(r'\bPLATE\b', text, re.IGNORECASE):
        multi_len_m = re.search(
            r'SIZE\s*:\s*([\d.]+MM(?:\s*,\s*[\d.]+MM){3,})',
            text, re.IGNORECASE
        )
        if multi_len_m:
            vals = [int(float(v)) for v in re.findall(r'([\d.]+)MM', multi_len_m.group(1))]
            if len(vals) >= 4:
                return f'{min(vals)}~{max(vals)}'

    # CHANNEL 4-dim 치수: CHANNEL,{a}X{b}X{c}X{d}" → actual profile dimensions
    # 예: A36, CHANNEL,3X1.596X.356X240" → 3X1.596X0.356X240IN
    if re.search(r'\bCHANNEL\b', text, re.IGNORECASE):
        ch_m = re.search(
            r'\bCHANNEL\s*,?\s*([\d.]+[Xx][\d.]+[Xx][\d.]+[Xx][\d.]+)"',
            text, re.IGNORECASE
        )
        if ch_m:
            return ch_m.group(1).upper() + 'IN'

    # SLITTING WIDTH: 슬리팅 폭이 실제 구매 치수 (SIZE 외형치수보다 우선)
    # 예: SLITTING WIDTH 170MMSIZE:740X690X770MM → 170
    sw_m = re.search(r'\bSLITTING\s+W(?:IDTH)?\s*([\d.]+)\s*MM\b', text, re.IGNORECASE)
    if sw_m:
        return sw_m.group(1)

    # PRODUCT DIMENSIONS NB+SCH: NB {n} IN, SCH {n}, CUT TO LENGTH {l} IN → {n}INXSCH{n}X{l}IN
    # 예: PRODUCT DIMENSIONS NB 1.5 IN, SCH 40, CUT TO LENGTH 236.23 IN → 1.5INXSCH40X236.23IN
    if re.search(r'PRODUCT\s+DIMENSIONS?\b', text, re.IGNORECASE):
        pd_nb_m = re.search(
            r'PRODUCT\s+DIMENSIONS?\s+NB\s+([\d./]+)\s+IN[,\s]+SCH\s+(\d+)'
            r'(?:[,\s]+CUT\s+TO\s+L(?:ENGTH\s+)?([\d.]+)\s+IN'
            r'|[\s,]*[Xx]\s+([\d.]+)\s+IN\s+CUT\s+TO\s+L(?:ENGTH)?)',
            text, re.IGNORECASE
        )
        if pd_nb_m:
            nb, sch, l1, l2 = pd_nb_m.groups()
            l = l1 or l2
            return f'{nb}INXSCH{sch}X{l}IN'

    # KINDORF CHANNEL {n}GA X {d1} X {d2} X {l}FT: GA 게이지 제외, 프로파일 치수만 추출
    # 예: KINDORF CHANNEL - 12 GA. X 1 5/8" X 1 5/8" X 3 FT → 1-5/8INX1-5/8INX3FT
    if re.search(r'\bKINDORF\b', text, re.IGNORECASE):
        kd_m = re.search(
            r'\d+\s*GA\.?\s*[Xx]\s+([\d]+(?:-\d+/\d+)?IN)\s*[Xx]\s*([\d]+(?:-\d+/\d+)?IN)\s*[Xx]\s*([\d.]+\s*FT)',
            text, re.IGNORECASE
        )
        if kd_m:
            d1, d2, lft = kd_m.groups()
            return f'{d1}X{d2}X{lft.replace(" ", "")}'

    # RECTANGLE {w}X{h}: 직사각형 단면 빔/바 치수 추출 (단위중량 등 뒤 숫자 무시)
    # 예: UB 500 PLUS RECTANGLE 145X75 28.02 → 145X75
    rect_m = re.search(r'\bRECTANGLE\b\s+([\d.]+[Xx][\d.]+)', text, re.IGNORECASE)
    if rect_m:
        return rect_m.group(1).upper()

    # CANNULA BLANK / BIOPSY CANNULA / NEEDLE BLANK: NN GA [TW/RW] X N.NNN["?] → 길이만 추출 (게이지 무시)
    if re.search(r'CANNULA\b|NEEDLE\s+BLANK\b', text, re.IGNORECASE):
        ca_m = re.search(r'\d+\s*GA\.?\s*\w*\s*[Xx]\s*([\d.]+)"?', text, re.IGNORECASE)
        if ca_m:
            return f'{ca_m.group(1)}IN'

    # THERMACLAD / CP{4+} 솔더 와이어: 분수 직경만 추출 → {frac}IN
    # 예: THERMACLAD 457 1/8 75# → 1/8IN,  7/64 CP2000 500# → 7/64IN
    # (normalize가 THERMACLAD를 제거하므로 원본 spec_text 사용)
    if re.search(r'\bTHERMACLAD\b|\bCP\d{4,}\b', spec_text, re.IGNORECASE):
        frac_m = re.search(r'\b(\d+/\d+)\b', spec_text, re.IGNORECASE)
        if frac_m:
            return f'{frac_m.group(1)}IN'

    # BIOPSY STYLET: N.NNNN DIA. X N.NNN" or 4자리코드DIA X N.NNN → 직경X길이 (부품번호 제외)
    # 예: 0395DIA X 7.313 → 0.0395IN × 7.313IN (4자리 정수 = 0.XXXX인치)
    if re.search(r'BIOPSY\s+STYLET\b', text, re.IGNORECASE):
        # 4자리 코드 형식: 0395(DIA) X 7.313 → 0.0395INX7.313IN
        bs_4d = re.search(r'\b(\d{4})\s*(?:DIA\.?)?\s*[Xx]\s*([\d.]+)"?', text, re.IGNORECASE)
        if bs_4d:
            dia_val = f'{int(bs_4d.group(1)) / 10000:g}'
            return f'{dia_val}INX{bs_4d.group(2)}IN'
        bs_m = re.search(r'([\d.]+)\s*DIA\.?\s*[Xx]\s*([\d.]+)"?', text, re.IGNORECASE)
        if bs_m:
            return f'{bs_m.group(1)}INX{bs_m.group(2)}IN'

    # SEAMLESS STEEL TUBES: 도면번호(/) 이후 실치수 추출
    # 예: ...PLAIN ENDS/13138-M0001-01 193.7 X 24.0 X 6150 → 193.7X24.0X6150
    if re.search(r'SEAMLESS\s+STEEL\s+TUBES?\b', text, re.IGNORECASE):
        tube_dim_m = re.search(
            r'/[\w-]+\s+([\d.]+)\s*[Xx]\s*([\d.]+)\s*[Xx]\s*(\d{3,6})(?!\d)',
            text, re.IGNORECASE
        )
        if tube_dim_m:
            d, t, l = tube_dim_m.group(1), tube_dim_m.group(2), tube_dim_m.group(3)
            return f'{d}X{t}X{l}'

    # NANO RIBBON: (N-NUM) UM 두께 마이크로미터 변환 → mm (소수 2자리)
    # 예: NANO RIBBON 49.5MM (16-18UM) → 0.02X49.5
    nano_m = re.search(
        r'NANO\s+RIBBON\b.*?([\d.]+)\s*MM\s*\(\s*[\d.]+\s*[-~]\s*([\d.]+)\s*UM\s*\)',
        text, re.IGNORECASE
    )
    if nano_m:
        width_f = float(nano_m.group(1))
        width = f'{width_f:g}'  # trailing zero 제거: 10.0 → '10'
        um_max = float(nano_m.group(2))
        thickness = f'{um_max / 1000:.2f}'  # μm → mm, 소수 2자리
        return f'{thickness}X{width}'

    # CLEAN SIZE N → 바 직경 추출: 102559 BAR/718 ... CLEAN SIZE 0.5100
    clean_size_m = re.search(r'\bCLEAN\s+SIZE\s+([\d.]+)\b', text, re.IGNORECASE)
    if clean_size_m:
        val = clean_size_m.group(1)
        try:
            return f'{float(val):g}'
        except ValueError:
            return val

    # {val}MM ABOUT → 선두 단독 치수 (BOBBIN/COIL 제품): trailing zero 제거
    bobbin_m = re.match(r'^([\d.]+)\s*MM\s+ABOUT\b', text, re.IGNORECASE)
    if bobbin_m:
        val = bobbin_m.group(1)
        try:
            fval = float(val)
            return str(int(fval)) if fval == int(fval) else f'{fval:g}'
        except ValueError:
            return val

    # P.I.# 와이어로프 구매주문서: P.I.#XXXXXXTA30MM ... → 30 (TA 뒤가 직경)
    pi_m = re.search(r'\bP\.I\.\s*#.*?TA(\d+\.?\d*)\s*MM\b', text, re.IGNORECASE)
    if pi_m:
        val = pi_m.group(1)
        try:
            fval = float(val)
            return str(int(fval)) if fval == int(fval) else val
        except ValueError:
            return val

    # LENGTH:{range}M SIZE:DIA {n}MM 조합 (단조/봉강 규격): 길이범위 + 직경
    # 예: QDH LENGTH:4.39-5.37M SIZE:DIA 70MM → 4390~5370X70
    len_dia_m = re.search(
        r'LENGTH\s*:\s*([\d.]+(?:-[\d.]+)?)\s*(M\b|MM\b).*?SIZE\s*:\s*DIA\s*([\d.]+)\s*(MM\b|CM\b)?',
        text, re.IGNORECASE
    )
    if len_dia_m:
        l_val = _convert_unit(len_dia_m.group(1), len_dia_m.group(2).upper())
        d_val = _convert_unit(len_dia_m.group(3), len_dia_m.group(4) or 'MM')
        return f'{l_val}X{d_val}'

    # DIA.OD * WT * L 형식 용접튜브: DIA.25.4 * 1.1T * 4M → 25.4X1.1X4000
    dia_t_l_m = re.search(
        r'\bDIA\.?\s*([\d.]+)\s*\*\s*([\d.]+)T?\s*\*\s*([\d.]+)\s*(M\b|MM\b)',
        text, re.IGNORECASE
    )
    if dia_t_l_m:
        od, wt, lv, lu = dia_t_l_m.group(1), dia_t_l_m.group(2), dia_t_l_m.group(3), dia_t_l_m.group(4)
        l_converted = _convert_unit(lv, lu.upper())
        return f'{od}X{wt}X{l_converted}'

    # RTJ (Ring Type Joint) 관이음쇠 호칭경: RTJ2", RTJ1 1/4", RTJ2 1/2"
    rtj_m = re.search(r'\bRTJ\s*(\d+)\s*(?:(\d+)/(\d+))?\s*"', text, re.IGNORECASE)
    if rtj_m:
        whole = rtj_m.group(1)
        num, den = rtj_m.group(2), rtj_m.group(3)
        if num and den:
            return f"{whole}-{num}/{den}IN"
        return f"{whole}IN"

    # NCL 제품 코드 치수 패턴 (COLLAR/ADJUST RING): NCLM10-12-12 → 12X12
    ncl_m = re.search(r'\b[A-Z]*NCL[A-Z0-9\-\.]*', text, re.IGNORECASE)
    if ncl_m:
        nums = re.findall(r'[\d.]+', ncl_m.group())
        if len(nums) >= 2:
            return f'{nums[-2]}X{nums[-1]}'

    # FAMSC/FPASC/FASC/AMSC/ACSC 칼라 코드: -D{d}-L{l} 형식 → D×L
    # 예: FAMSC-V30-D40-L19.6 → 40X19.6, FPASC-V40-D46-L22 → 46X22
    famsc_dl_m = re.search(
        r'(?:FAMSC|FPASC|FASC|AMSC|ACSC)\S*-D\s*(\d+(?:\.\d+)?)-L\s*(\d+(?:\.\d+)?)\b',
        text, re.IGNORECASE
    )
    if famsc_dl_m:
        return f'{famsc_dl_m.group(1)}X{famsc_dl_m.group(2)}'

    # COLLAR/ADJUST RING + 하이픈 숫자: 마지막 두 값이 치수 (NCL 이외 코드 포함)
    # COLLAR\w* — COLLARSAMSC, COLLARSACSC 등 복합어도 인식
    if re.search(r'\bCOLLAR\w*|\bADJUST\s*RING\w*\b', text, re.IGNORECASE):
        seqs = re.findall(r'[\d.]+(?:-[\d.]+)+', text)
        for seq in seqs:
            nums = re.findall(r'[\d.]+', seq)
            if len(nums) >= 3:
                return f'{nums[-2]}X{nums[-1]}'

    # SWOVM-VS/VSE, SWOSC-VHVE 스프링 와이어: 제품코드 뒤 치수가 사이즈
    swovm_m = re.search(r'\bSWO(?:VM|SC)-\w+\b', text, re.IGNORECASE)
    if swovm_m:
        rest = text[swovm_m.end():]
        # {num}X{num}MM (직사각형 단면) 또는 {num}MM (원형 단면)
        dim_m = re.search(r'([\d.]+)\s*[Xx]\s*([\d.]+)\s*MM\b|([\d.]+)\s*MM\b', rest, re.IGNORECASE)
        if dim_m:
            if dim_m.group(1):
                a, b = dim_m.group(1).strip(), dim_m.group(2).strip()
                return f'{a}X{b}'
            else:
                val = dim_m.group(3)
                try:
                    fval = float(val)
                    return str(int(fval)) if fval == int(fval) else f'{fval:g}'
                except ValueError:
                    return val

    # CDW HONED/SRB 냉간인발 강관: {OD} {ID} {range} CDW HONED/SRB → OD X range
    cdw_m = re.match(
        r'^(\d+)\s+\d+\s+([\d.]+(?:-[\d.]+)?)\s+CDW\s+(?:HONED|SRB)\b',
        text.strip(), re.IGNORECASE
    )
    if cdw_m:
        od = cdw_m.group(1)
        rng = cdw_m.group(2)
        if '-' in rng:
            lo, hi = rng.split('-', 1)
            rng_str = f'{lo.strip()}~{hi.strip()}'
        else:
            rng_str = rng
        return f'{od}X{rng_str}'

    # FULL HARD STEEL COIL{grade}-{code}{thickness}*{width}*C 형식
    # 예: COILS45C-C12.510*1250.0*C → 2.510X1250.0XC (C1이 코드, 2.510이 두께)
    fhsc_m = re.search(
        r'\bFULL\s+HARD\s+STEEL\s+COIL[\w-]+-[A-Z]+\d([\d.]+)\*([\d.]+)\*C\b',
        text, re.IGNORECASE
    )
    if fhsc_m:
        return f'{fhsc_m.group(1)}X{fhsc_m.group(2)}XC'

    # OUTER*INNER*LENGTH: 외경*내경*길이 → 외경X길이 (내경 무시)
    outer_inner_m = re.search(
        r'\bOUTER\s*\*\s*INNER\s*\*\s*(?:LE?\s*NGTH|LENGTH|L)\b\s*([\d.]+)\s*\*\s*[\d.]+\s*\*\s*([\d.]+)',
        text, re.IGNORECASE
    )
    if outer_inner_m:
        return f'{outer_inner_m.group(1)}X{outer_inner_m.group(2)}'

    # OD×WT×Length 패턴: OD52MM X WT3.15MM X 8500MM → 52X3.15X8500 (word boundary 없이)
    od_wt_m = re.search(
        r'OD\s*([\d.]+)\s*MM+\s*[Xx]\s*WT\s*([\d.]+)\s*MM+\s*[Xx]\s*([\d.]+)\s*MM',
        text, re.IGNORECASE
    )
    if od_wt_m:
        od, wt, length = od_wt_m.group(1), od_wt_m.group(2), od_wt_m.group(3)
        return f'{od}X{wt}X{length}'

    # O.D.{n}XW.T.{n}MM IN LENGTH {n}M[ETERS] 패턴: SMO 254 파이프 등
    odwt_len_m = re.search(
        r'O\.D\.\s*([\d.]+)\s*[Xx]\s*W\.T\.\s*([\d.]+)\s*MM.*?\bIN\s+LENGTH\s+([\d.]+)\s*M',
        text, re.IGNORECASE
    )
    if odwt_len_m:
        od, wt = odwt_len_m.group(1), odwt_len_m.group(2)
        lm = float(odwt_len_m.group(3))
        lmm = str(int(lm * 1000)) if lm * 1000 == int(lm * 1000) else f'{lm * 1000:.1f}'
        return f'{od}X{wt}X{lmm}'

    # HORNED/HONED/COLD DRAWN 파이프: {n1}*{n2} → 큰 값(OD)만 반환
    if re.search(r'\b(?:HORNED|HORNING|HONED|HONING)\b|COLD\s+DRAWN\s+SEAMLESS', text, re.IGNORECASE):
        horned_m = re.match(r'^([\d.]+)\s*\*\s*([\d.]+)\s*(?:MM\b)?', text.strip(), re.IGNORECASE)
        if horned_m:
            n1, n2 = float(horned_m.group(1)), float(horned_m.group(2))
            od = horned_m.group(1) if n1 >= n2 else horned_m.group(2)
            return od

    # HLR 압력파이프: D{dia} 직경만 반환 (압력 단위 등 잡음 제외)
    if re.search(r'\bHLR\d+\b', text, re.IGNORECASE):
        hlr_m = re.search(r'\bD([\d.]+)\s*(?:MM\b|$)', text, re.IGNORECASE)
        if hlr_m:
            val = hlr_m.group(1)
            try:
                fval = float(val)
                return str(int(fval)) if fval == int(fval) else f'{fval:g}'
            except ValueError:
                return val

    # EUREKAMATIC 용접봉: 첫번째 숫자(분수 포함)가 직경
    if re.search(r'\bEUREKAMATIC\b', text, re.IGNORECASE):
        eurekamatic_m = re.search(r'([\d]+/[\d]+|[\d.]+)', text, re.IGNORECASE)
        if eurekamatic_m:
            val = eurekamatic_m.group(1)
            return f'{val}IN' if '/' in val else val

    # TIG 아르곤 용접와이어: (AWS ER...){dia} 패턴에서 직경 추출 (SIZE 블록은 로드 길이)
    if re.search(r'\bTIG\b.*\bWELDING\s+WIRE|\bARGON[-\s]+ARC\s+WELDING\s+WIRE', text, re.IGNORECASE):
        tig_m = re.search(r'\(AWS\s+ER[\w-]+\)\s*([\d.]+)', text, re.IGNORECASE)
        if tig_m:
            val = tig_m.group(1)
            try:
                fval = float(val)
                return str(int(fval)) if fval == int(fval) else f'{fval:g}'
            except ValueError:
                return val

    # STAINLESS STEEL WIRE: 직경만 추출 (grade/catalog/T/S 코드 무시)
    # \b 없이 WIRE 뒤 바로 강종코드가 붙는 경우도 포함: WIRE304L, WIRE251027-01
    if re.search(r'\bSTAINLESS\s+STEEL\s+WIRE', text, re.IGNORECASE):
        # 패턴 1: {dia}MM 마지막 (슬래시/비슬래시 모든 형식 공통)
        ssw_end = re.search(r'([\d.]+)\s*MM\s*$', text.strip(), re.IGNORECASE)
        if ssw_end:
            val = ssw_end.group(1)
            try:
                fval = float(val)
                return str(int(fval)) if fval == int(fval) else f'{fval:g}'
            except ValueError:
                return val
        # 패턴 2: WIRE {dia} A2-{n} (A2-70/A2-50: 단위 없는 경우)
        ssw_a2 = re.search(r'\bWIRE\s+([\d.]+)\s+A2-\d+\b', text, re.IGNORECASE)
        if ssw_a2:
            val = ssw_a2.group(1)
            try:
                fval = float(val)
                return str(int(fval)) if fval == int(fval) else f'{fval:g}'
            except ValueError:
                return val
        # 패턴 BT: WIRE BT-{dia} (카탈로그 코드 접두): BT-2.3 → 2.3
        ssw_bt = re.search(r'\bWIRE\s+BT-\s*([\d.]+)\b', text, re.IGNORECASE)
        if ssw_bt:
            val = ssw_bt.group(1)
            try:
                fval = float(val)
                return str(int(fval)) if fval == int(fval) else f'{fval:g}'
            except ValueError:
                return val
        # 패턴 ORDER: WIRE{grade} {dia}ORDER NO → diameter only (예: WIRE304L 0.80ORDER NO)
        ssw_order = re.search(r'\bWIRE\w*\s+([\d.]+)\s*ORDER\b', text, re.IGNORECASE)
        if ssw_order:
            val = ssw_order.group(1)
            try:
                fval = float(val)
                return str(int(fval)) if fval == int(fval) else f'{fval:g}'
            except ValueError:
                return val
        # 패턴 DIA-suffix: {dia}[DC] MM DIA 형식 (catalog code 접미): WIRE251027-01 9.06DC MM DIA → 9.06
        # (?!\s*[Xx]): DIA X {len}MM 다차원 패턴은 SIZE 블록 핸들러에 위임 (2.30MM DIA X 11.25MM 오인식 방지)
        ssw_dia_suf = re.search(r'([\d.]+)\s*(?:\w+\s+)?MM\s+DIA\b(?!\s*[Xx])', text, re.IGNORECASE)
        if ssw_dia_suf:
            val = ssw_dia_suf.group(1)
            try:
                fval = float(val)
                return str(int(fval)) if fval == int(fval) else f'{fval:g}'
            except ValueError:
                return val
        # 패턴 3: 마지막 단독 숫자 (단위 없는 경우): 2.00, 0.9
        ssw_bare = re.search(r'[,\s]([\d.]+)\s*$', text.strip(), re.IGNORECASE)
        if ssw_bare:
            val = ssw_bare.group(1)
            try:
                fval = float(val)
                return str(int(fval)) if fval == int(fval) else f'{fval:g}'
            except ValueError:
                return val

    # EX-NO. 와이어로프 품번: {dia}MM 직경만 반환 (길이/수량/구성코드 제외)
    if re.search(r'\bEX[-\s.]?NO\.?\b', text, re.IGNORECASE):
        exno_dia = re.search(r'([\d.]+)\s*MM\b', text, re.IGNORECASE)
        if exno_dia:
            val = exno_dia.group(1)
            try:
                fval = float(val)
                return str(int(fval)) if fval == int(fval) else f'{fval:g}'
            except ValueError:
                return val

    # JIS 파이프 공칭mm 표기: {n}AXSCH{n}S?XL{n}MM → {n}AXSCH{n}X{l}
    # 예: 150AXSCH5SXL4000MM → 150AXSCH5X4000, 15AXSCH5SXL4000MM → 15AXSCH5X4000
    jis_pipe_m = re.search(r'\b(\d+)A\s*[Xx]\s*SCH(\d+)S?\s*[Xx]\s*L?\s*(\d+)\s*MM\b', text, re.IGNORECASE)
    if jis_pipe_m:
        return f'{jis_pipe_m.group(1)}AXSCH{jis_pipe_m.group(2)}X{jis_pipe_m.group(3)}'

    # OD {n}" S{n}MM 패턴 (SCH 표기에 MM 붙은 경우): OD 14" S40MM → 14INXSCH40
    od_sch_mm_m = re.search(r'\bOD\s*([\d.]+)\s*"\s*S(\d{2,3})MM\b', text, re.IGNORECASE)
    if od_sch_mm_m:
        return f'{od_sch_mm_m.group(1)}INXSCH{od_sch_mm_m.group(2)}'

    # SCH 파이프 패턴: {dia}"/{dia}*/{dia}IN {S/SCH}{n}S? [WLD/SMLS] PIPE {len}MM
    sch_pipe_m = re.search(
        r'([\d]+(?:-\d+/\d+)?|\d+/\d+|[\d.]+)\s*(?:IN|"|\*)\s*(?:X\s*)?'
        r'(?:SCH|S)(\d+)S?\b\s+(?:(?:WLD|SMLS|SEAMLESS|WELDED)\s+)?PIPE\s+'
        r'([\d.]+)\s*MM\b',
        text, re.IGNORECASE
    )
    if sch_pipe_m:
        dia_raw = sch_pipe_m.group(1)
        sch = sch_pipe_m.group(2)
        length = sch_pipe_m.group(3)
        try:
            lf = float(length)
            length_s = str(int(lf)) if lf == int(lf) else length
        except ValueError:
            length_s = length
        return f'{dia_raw}INXSCH{sch}X{length_s}'

    # {공칭인치}"{OD}MM×{WT}MM×{L}MM: 공칭 인치 크기 무시, 실제 MM 치수만 사용
    # 예: 14"355.6MMx5MMx6000MM → 355.6X5X6000 (ASTM A312 스테인리스 파이프)
    nom_mm_pipe = re.search(
        r'\b\d+"\s*([\d.]+)\s*MM\s*[Xx]\s*([\d.]+)\s*MM\s*[Xx]\s*([\d.]+)\s*MM',
        text, re.IGNORECASE
    )
    if nom_mm_pipe:
        return f'{nom_mm_pipe.group(1)}X{nom_mm_pipe.group(2)}X{nom_mm_pipe.group(3)}'

    # WIRE ROPE / WR / UNGALV 핸들러: 직경(+길이) 추출 (UNGALV=ungalvanized 와이어 로프 포함)
    if re.search(r'\b(?:WIRE\s+ROPE|WR|UNGALV)\b', text, re.IGNORECASE):
        # OD/DIA/PHI 명시 직경 우선
        wr_od_m = re.search(r'(?:OD|DIA|PHI)\s*([\d.]+)', text, re.IGNORECASE)
        if wr_od_m:
            try:
                fv = float(wr_od_m.group(1))
                return str(int(fv)) if fv == int(fv) else f'{fv:g}'
            except ValueError:
                return wr_od_m.group(1)
        # 직경 범위 표기: 1.2-1.5MM → 1.2~1.5
        wr_range = re.search(r'([\d.]+)-([\d.]+)\s*MM\b', text, re.IGNORECASE)
        if wr_range:
            try:
                lo = f'{float(wr_range.group(1)):g}'
                hi = f'{float(wr_range.group(2)):g}'
                return f'{lo}~{hi}'
            except ValueError:
                pass
        # 인치 분수 직경: 1/16*200M, 3/64 DIA 등 (분수 뒤 * 구분자 또는 단독)
        wr_frac = re.search(r'\b(\d+/\d+)\b', text)
        if wr_frac:
            return f'{wr_frac.group(1)}IN'
        # D{n}X{length} 직경+길이 패턴: D11.2X45M → 11.2X45000, D009.0x150000mm → 9
        # 길이 < 100000mm(100m)이면 절단 피스 → 직경+길이 반환, 이상이면 코일 → 직경만
        d_m = re.search(r'\bD0*(\d+\.?\d*)\s*[Xx]\s*([\d.]+)\s*(MM|M\b)?', text, re.IGNORECASE)
        if d_m:
            dia_val = d_m.group(1)
            len_val = d_m.group(2)
            len_unit = d_m.group(3) or ''
            try:
                dia_f = float(dia_val)
                dia_s = str(int(dia_f)) if dia_f == int(dia_f) else f'{dia_f:g}'
                len_mm_s = _convert_unit(len_val, len_unit.upper())
                len_mm_f = float(len_mm_s)
                if len_mm_f < 100000:
                    len_s = str(int(len_mm_f)) if len_mm_f == int(len_mm_f) else len_mm_s
                    return f'{dia_s}X{len_s}'
                return dia_s
            except (ValueError, TypeError):
                return dia_val
        wr_m = re.search(r'([\d.]+)\s*MM\b', text, re.IGNORECASE)
        if wr_m:
            val = wr_m.group(1)
            # 두 번째 MM 값이 있고 < 100m이면 절단 피스 길이로 포함
            remaining = text[wr_m.end():]
            wr_len_m = re.search(r'([\d.]+)\s*MM\b', remaining, re.IGNORECASE)
            try:
                dia_f = float(val)
                dia_s = str(int(dia_f)) if dia_f == int(dia_f) else f'{dia_f:g}'
                if wr_len_m:
                    len_f = float(wr_len_m.group(1))
                    if len_f < 100000:
                        len_s = str(int(len_f)) if len_f == int(len_f) else str(len_f)
                        return f'{dia_s}X{len_s}'
                return dia_s
            except ValueError:
                return val

    # {n}MM DIA X COIL 패턴: 5.50MM DIA X COIL → 5.5 (코일 제품 직경, DIA+COIL 조합만)
    mm_dia_coil_m = re.search(r'([\d.]+)\s*MM\s+DIA\s+X\s+COIL\b', text, re.IGNORECASE)
    if mm_dia_coil_m:
        val = mm_dia_coil_m.group(1)
        try:
            fval = float(val)
            return str(int(fval)) if fval == int(fval) else f'{fval:g}'
        except ValueError:
            return val

    # GUIDE WIRE {n} 또는 {n} GUIDE WIRE: 단독 숫자가 직경 (단위 없는 경우도 허용)
    if re.search(r'\bGUIDE\s+WIRE\b', text, re.IGNORECASE):
        gw_m = re.search(r'\b([\d.]+)\b', text)
        if gw_m:
            try:
                return f'{float(gw_m.group(1)):g}'
            except ValueError:
                pass

    # WIRE DIA {num}MM → 단선 직경 (SWRCH 등 와이어 규격) — 숫자 직후 WIRE도 허용
    # DIA\.? : DIA.2MM 형태에서 점을 소비해 .2→2 오인식 방지
    wire_dia_m = re.search(
        r'(?<![A-Z])WIRE\s+DIA\.?\s*([\d.]+)\s*(MM|CM|(?<!\w)M(?!\w)|IN(?:CH)?|FT|")?',
        text, re.IGNORECASE
    )
    if wire_dia_m:
        val = wire_dia_m.group(1)
        unit = wire_dia_m.group(2) or ''
        result = _convert_unit(val, unit)
        try:
            fval = float(result)
            return str(int(fval)) if fval == int(fval) else f'{fval:g}'
        except ValueError:
            return result

    # SPRING,{dia}MM 패턴: SS SPRING,0.7MM → 0.7
    spring_m = re.search(r'\bSPRING\s*,\s*([\d.]+)\s*(MM|CM|(?<!\w)M(?!\w)|IN(?:CH)?|FT|")?', text, re.IGNORECASE)
    if spring_m:
        return _convert_unit(spring_m.group(1), spring_m.group(2) or '')

    # VEROPOWER 와이어로프 브랜드: 직경 {n}MM 추출
    if re.search(r'\bVEROPOWER\b', text, re.IGNORECASE):
        vero_dia = re.search(r',\s*(\d+\.?\d*)\s*MM\b', text, re.IGNORECASE)
        if vero_dia:
            val = vero_dia.group(1)
            try:
                fval = float(val)
                return str(int(fval)) if fval == int(fval) else val
            except ValueError:
                return val

    # 와이어 로프 구성코드 ({n}X{n}+ 또는 표준 구성코드 1X7/7X7/6X19 등) 있을 때 직경
    # 6X24+7FC-16MM → 16, GAL.06x24+FC 30MM → 30, 1X7 0.7MM → 0.7
    _WR_CONSTRUCTIONS = r'\b(?:1X7|1X19|7X7|7X19|6X7|6X12|6X19|6X24|6X25|6X36|6X37|6X49)\b'
    if re.search(r'\b\d+[Xx]\d+\+', text) or re.search(_WR_CONSTRUCTIONS, text, re.IGNORECASE):
        # 직경 범위 표기 우선: 1.2-1.5MM → 1.2~1.5
        wr_range2 = re.search(r'([\d.]+)-([\d.]+)\s*MM\b', text, re.IGNORECASE)
        if wr_range2:
            try:
                lo = f'{float(wr_range2.group(1)):g}'
                hi = f'{float(wr_range2.group(2)):g}'
                return f'{lo}~{hi}'
            except ValueError:
                pass
        wire_dia = re.search(r'[-,\s](\d+\.?\d*)\s*MM(?!\w)', text, re.IGNORECASE)
        if not wire_dia:
            wire_dia = re.search(r'\bDIA\.\s*:?\s*([\d.]+)', text, re.IGNORECASE)
        if wire_dia:
            val = wire_dia.group(1)
            try:
                fval = float(val)
                return str(int(fval)) if fval == int(fval) else val
            except ValueError:
                return val

    # JIS 용접 와이어 강종코드 + 직경: YGW-12 (YT-12) 1.2MM → 1.2
    # JIS 등급: YGW, YT(JIS), MG(JQ 계열), MF(플럭스 코어드)
    if re.search(r'\b(?:YGW|YT|YF|MG\d|MF[-\s]\d)', text, re.IGNORECASE):
        jis_ww_m = re.search(r'\b(\d+\.?\d*)\s*MM\b', text, re.IGNORECASE)
        if jis_ww_m:
            val = jis_ww_m.group(1)
            try:
                return f'{float(val):g}'
            except ValueError:
                return val

    # AUTROD/BÖHLER 등 용접봉 브랜드 + {n}MM 직경
    if re.search(r'\bAUTROD\b', text, re.IGNORECASE):
        autrod_m = re.search(r'\b(\d+\.?\d*)\s*MM\b', text, re.IGNORECASE)
        if autrod_m:
            val = autrod_m.group(1)
            try:
                return f'{float(val):g}'
            except ValueError:
                return val

    # WELDING WIRE {grade},{dia}mm 패턴: ER308,1.6mm / ER70S-6,1.6mm → 1.6
    # \b 제거: WELDING WIREJQ... 형태도 인식 (WIRE 뒤 단어경계 없을 때)
    if re.search(r'\bWELDING\s*WIRE', text, re.IGNORECASE):
        ww_m = re.search(r'(?:ER|E)[\w-]*\s*,\s*([\d.]+)\s*(MM|CM|(?<!\w)M(?!\w)|IN(?:CH)?|FT|")', text, re.IGNORECASE)
        if ww_m:
            return _convert_unit(ww_m.group(1), ww_m.group(2) or '')
        # ERxx D{dia}MM 형태: ER70S-6D1.2MM → 1.2 (D= diameter prefix, comma 없음)
        ww_m3 = re.search(r'(?:ER|E)[A-Z0-9\-]+D\s*([\d.]+)\s*MM\b', text, re.IGNORECASE)
        if ww_m3:
            try:
                return f'{float(ww_m3.group(1)):g}'
            except ValueError:
                return ww_m3.group(1)
        # (ERxx)N.NMM, (AWS ERxx)N.NMM 형태: (ER70S-6)1.0mm → 1 (괄호 내 ER 분류코드 다음 직경)
        # 단위 없는 경우도 허용: (AWS ER308)0.9(5KG) → 0.9
        ww_m2 = re.search(r'\((?:AWS\s+)?E[A-Z0-9\-]+\)[\s,]*([\d.]+)\s*(MM|CM|(?<!\w)M(?!\w)|IN(?:CH)?|FT|")?', text, re.IGNORECASE)
        if ww_m2 and ww_m2.group(1):
            val = _convert_unit(ww_m2.group(1), ww_m2.group(2) or '')
            try:
                fval = float(val)
                return f'{fval:g}'
            except ValueError:
                return val

    # BARE ROD 분수 직경 + 인치 길이: EUREKA 1/8 X 36"C" BARE ROD → 1/8INX36IN
    if re.search(r'\bBARE\s+ROD\b', text, re.IGNORECASE):
        bare_m = re.search(r'\b(\d+/\d+)\s*[Xx]\s*([\d.]+)\s*"', text, re.IGNORECASE)
        if bare_m:
            return f'{bare_m.group(1)}INX{bare_m.group(2)}IN'

    # 단독 인치 분수 직경: 5/32IN (MM/CM/FT 치수가 없는 경우만, 인치 분수가 유일 치수)
    # WELDING ROD 등 소모재 분야에서 분수 인치가 직경으로 쓰임
    frac_in_only = re.search(r'\b(\d+/\d+)\s*IN\b(?:CH)?\b', text, re.IGNORECASE)
    if frac_in_only and not re.search(r'[\d.]+\s*(?:MM|CM|FT)\b', text, re.IGNORECASE):
        return f'{frac_in_only.group(1)}IN'

    # P.I.# 와이어로프 직경 패턴: P.I.#...{n}MM → n (직경만 추출)
    if re.search(r'\bP\.I\.\s*#', text, re.IGNORECASE):
        pi_dia_m = re.search(r'(\d+\.?\d*)\s*MM\b', text, re.IGNORECASE)
        if pi_dia_m:
            val = pi_dia_m.group(1)
            try:
                fval = float(val)
                return str(int(fval)) if fval == int(fval) else val
            except ValueError:
                return val

    # {n}MM DIA X {l}MM 패턴 (직경이 DIA 앞, 길이가 뒤): 9.9MM DIA X 1752.6MM → 9.9X1752.6
    ndia_x_m = re.search(r'\b([\d.]+)\s*MM\s+DIA\s+[Xx]\s*([\d.]+)\s*MM\b', text, re.IGNORECASE)
    if ndia_x_m:
        d = f'{float(ndia_x_m.group(1)):g}'
        l = f'{float(ndia_x_m.group(2)):g}'
        return f'{d}X{l}'

    # {n} DIA (단독 직경, 단위 없음): 20 DIA IN EXTERNAL DIAMETER → 20
    # (?<![/\d]): 분수 분모 오인식 방지 (3/64 DIA에서 64 대신 3을 잡는 것 방지)
    if re.search(r'\bDIA\b', text, re.IGNORECASE) and not re.search(r'DIA\s*\d', text, re.IGNORECASE):
        n_dia_m = re.search(r'(?<![/\d])(\d+\.?\d*)("?)\s+DIA\b', text, re.IGNORECASE)
        if n_dia_m:
            val, inch_mark = n_dia_m.group(1), n_dia_m.group(2)
            if inch_mark:
                return f'{val}IN'  # 인치 직경: 0.030" DIA → 0.030IN (trailing zero 보존)
            try:
                return f'{float(val):g}'
            except ValueError:
                pass

    # {n}MM DIA (DIA가 값 뒤에 오는 패턴): SUS304 W1 0.300 MM DIA → 0.3, TOKUSEN ... 0.035 MM DIA. → 0.035
    # X 연결 치수가 없는 단독 직경 표기에만 적용 (다차원 치수 혼재 시 제외)
    if not re.search(r'\d[Xx]\d', text):
        mm_dia_suffix_m = re.search(r'([\d.]+)\s*MM\s+DIA\.?\s*$', text, re.IGNORECASE)
        if mm_dia_suffix_m:
            val = mm_dia_suffix_m.group(1)
            try:
                fval = float(val)
                return f'{fval:g}'
            except ValueError:
                return val

    # 201/202 강종 접두어 처리: 20127.6*0.68*4020 → 27.6X0.68X4020, 201162*1.452B → 162X1.45XC
    grade201_m = re.match(r'^(201|202)(\d+\.?\d*(?:[*Xx][\d.]+)+)([BG]?)\s*$', text.strip(), re.IGNORECASE)
    if grade201_m:
        dim_str = grade201_m.group(2)
        suffix = grade201_m.group(3).upper()
        result = _parse_tokens(dim_str.replace('*', 'X'), preserve_order=True)
        if result:
            return (result + 'XC') if suffix == 'B' else result

    # 0-ag. {n}*{n}*{n}[MM] 별표 구분자 3차원 패턴 (GRADE: 키워드가 치수를 덮는 문제 방지)
    # 예: GRADE: SA-213M T11 31.8*2.4*25900MM → 31.8X2.4X25900
    if re.search(r'\d\s*\*\s*\d', text):
        star3_m = re.search(
            r'([\d.]+)\s*\*\s*([\d.]+)\s*\*\s*([\d.]+)\s*(?:MM\b)?(?:\s*\*\s*([\d.]+)\s*(?:MM\b)?)?',
            text, re.IGNORECASE
        )
        if star3_m and not re.search(r'\bSIZE\s*:', text, re.IGNORECASE):
            d1, d2, d3, d4 = star3_m.group(1), star3_m.group(2), star3_m.group(3), star3_m.group(4)
            if d4:
                return f'{d1}X{d2}X{d3}X{d4}'
            return f'{d1}X{d2}X{d3}'

    # 0-ah. {n}*{n} {large_mm}MM 2차원 별표+공백 길이 패턴 (_is_part_number_spec 전에 처리)
    # 예: 273*15.09 6000MM 13BUNDLES → 273X15.09X6000 (파이프 OD*두께, 앞 시리얼번호 무시)
    if re.search(r'\d\s*\*\s*\d', text) and not re.search(r'\bSIZE\s*:', text, re.IGNORECASE):
        star2_l_m = re.search(
            r'([\d.]+)\s*\*\s*([\d.]+)\s+(\d{3,6})\s*MM\b',
            text, re.IGNORECASE
        )
        if star2_l_m:
            return f'{star2_l_m.group(1)}X{star2_l_m.group(2)}X{star2_l_m.group(3)}'

    # 0-ai. WIRE {n}( ... ) 형태: 직경 다음 재질설명 괄호 (치수는 n만)
    # 예: WIRE 0.35( MATERIAL 304 STAINLESS...) → 0.35
    wire_paren_m = re.search(r'\bWIRE\s+([\d.]+)\s*\(', text, re.IGNORECASE)
    if wire_paren_m:
        return wire_paren_m.group(1)

    # 부품번호/시리얼 형식 → regex 불가, LLM으로
    if _is_part_number_spec(text):
        return None

    # SCH 파이프 분수인치 패턴: {frac}" SCH {n}[S] X {l} FT → {frac}INXSCH{n}X{l}FT
    # 복합 분수 인치 전처리 전에 처리: 1-1/2" 그대로 유지
    smls_sch_m = re.search(
        r'((?:\d+-\d+/\d+|\d+/\d+|\d+\.?\d*))\s*"\s*SCH\s*(\d+)S?\s*[Xx]\s*([\d.]+)\s*FT\b',
        text, re.IGNORECASE
    )
    if smls_sch_m:
        dia, sch, l = smls_sch_m.group(1), smls_sch_m.group(2), smls_sch_m.group(3)
        try:
            lf = float(l)
            l_s = str(int(lf)) if lf == int(lf) else l
        except ValueError:
            l_s = l
        return f'{dia}INXSCH{sch}X{l_s}FT'

    # ROUND BAR 분수 인치: 3-1/2" ROUND BAR → 3-1/2IN (소수 변환 없이 표기 유지)
    if re.search(r'\bROUND\s+BAR\b', text, re.IGNORECASE):
        rb_frac_m = re.search(r'\b(\d+-\d+/\d+)\s*"', text, re.IGNORECASE)
        if rb_frac_m:
            return f'{rb_frac_m.group(1)}IN'

    # 복합 분수 인치 전처리: 1-1/2" → 1.5IN (SIZE 블록 처리 전에 적용)
    text = re.sub(
        r'\b(\d+)-(\d+)/(\d+)\s*(?:"|IN(?:CH)?(?:ES)?)',
        lambda m: f'{int(m.group(1)) + int(m.group(2))/int(m.group(3)):.4g}IN',
        text, flags=re.IGNORECASE
    )

    # 1-a-0. SPEC:{강종} {두께}MM X {폭}MM X COIL 핫코일 패턴 (SPHC, SS400 등)
    # SIZE: 키워드 없이 SPEC: 뒤에 강종+치수가 오는 경우: step4에서 SPEC 이후 제거되어 pred=None 방지
    coil_spec_m = re.search(
        r'\bSPEC\s*:\s*\w+\s+([\d.]+)\s*MM\s*[Xx]\s*([\d.]+)\s*MM\s*[Xx]\s*COIL\b',
        text, re.IGNORECASE
    )
    if coil_spec_m:
        t_raw, w_raw = coil_spec_m.group(1), coil_spec_m.group(2)
        try:
            # 3자리 정수: 600→6, 100→1 (소수점 없이 mm×100 인코딩된 두께값)
            if re.match(r'^\d{3}$', t_raw) and float(t_raw) <= 2500:
                t_val = float(t_raw) / 100
                t_str = str(int(t_val)) if t_val == int(t_val) else f'{t_val:g}'
            else:
                t_str = t_raw
            return f'{t_str}X{w_raw}XC'
        except ValueError:
            pass

    # 1-a-1. GRADE:{code} {OD}M{1,2} X {WT}M{1,2} X {L}M (N PCS) 파이프/튜브 패턴
    # 예: GRADE:GR.6 219.1MMX11.5MMX8.70M (43PCS) → 219.1X11.5X8700
    # 첫 두 값은 단위 변환 없이 사용 (219.1M → 219.1mm 오인식 방지), 마지막만 M→×1000
    grade_pipe_m = re.search(
        r'\bGRADE\s*:\s*[\w.]+\s+([\d.]+)\s*M{1,2}\s*[Xx]\s*([\d.]+)\s*M{1,2}\s*[Xx]\s*([\d.]+)\s*(M\b(?!M)|MM\b)',
        text, re.IGNORECASE
    )
    if grade_pipe_m:
        od, wt, l_raw, l_unit = grade_pipe_m.groups()
        try:
            l_val = _convert_unit(l_raw, l_unit.upper().replace(' ', ''))
            return f'{od}X{wt}X{l_val}'
        except (ValueError, AttributeError):
            pass

    # 1-a-2. GRADE:{code} {t}MM X {w}MM X COIL 코일 패턴 (SPEC: 와 동일 구조)
    # 예: GRADE:S30400 1.50MM X 1524MM X COIL → 1.50X1524XC
    coil_grade_m = re.search(
        r'\bGRADE\s*:\s*\S+\s+([\d.]+)\s*MM\s*[Xx]\s*([\d.]+)\s*MM\s*[Xx]\s*COIL\b',
        text, re.IGNORECASE
    )
    if coil_grade_m:
        t_raw, w_raw = coil_grade_m.group(1), coil_grade_m.group(2)
        return f'{t_raw}X{w_raw}XC'

    # 1-a-3. OD(MM){n} WT(MM){n} SIZE(MM){l1}-{l2} 무계목 강관 패턴
    # 예: GRADE 37MN OD(MM)232 WT(MM)5.4 SIZE(MM)8940-8890 → 232X5.4X8940~8890
    odwt_paren_m = re.search(
        r'\bOD\s*\(MM\)\s*([\d.]+)\s*WT\s*\(MM\)\s*([\d.]+)\s*SIZE\s*\(MM\)\s*([\d.]+)\s*[-~]\s*([\d.]+)',
        text, re.IGNORECASE
    )
    if odwt_paren_m:
        od, wt, l1, l2 = odwt_paren_m.groups()
        return f'{od}X{wt}X{l1}~{l2}'

    # 1-a-4. GRADE:{code}SIZE(MM) N X N X N 패턴 (괄호 내 MM 단위 명시)
    # 예: GRADE:SCM420SIZE(MM) 330 X 370 X 2390 → 330X370X2390
    grade_size_paren_m = re.search(
        r'\bGRADE\s*:\s*\S+SIZE\s*\(MM\)\s*([\d.]+)\s*[Xx]\s*([\d.]+)\s*[Xx]\s*([\d.]+)',
        text, re.IGNORECASE
    )
    if grade_size_paren_m:
        d1, d2, d3 = grade_size_paren_m.groups()
        return f'{d1}X{d2}X{d3}'

    # 1-a-4b. GRADE:{code} FINISH:{code} N.NMM*NNNNMM*COIL → NXNNNNxC
    grade_finish_coil_m = re.search(
        r'\bGRADE\s*:\s*\S+\s+FINISH\s*:\s*\S+\s+([\d.]+)\s*MM\s*\*\s*([\d.]+)\s*MM\s*\*\s*COIL\b',
        text, re.IGNORECASE)
    if grade_finish_coil_m:
        t, w = grade_finish_coil_m.groups()
        return f'{t}X{w}XC'

    # 1-a-4c. GRADE:{code} [추가코드] N.N [MM] X NNNN [MM] [UP] X C → NXNNNNXC (코일 치수)
    # 예: GRADE : 190EM 4.7 MM X 1250 MM X C → 4.7X1250XC
    # 예: GRADE: 304 NO.2B 1.2 X 1219 UP X C → 1.2X1219XC
    grade_mm_x_coil_m = re.search(
        r'\bGRADE\s*:\s*\S+(?:\s+[A-Z][\w.]+)?\s+([\d.]+)\s*(?:MM\s*)?[Xx]\s*([\d.]+)\s*(?:MM\s*)?(?:UP\s*)?[Xx]\s*C\b',
        text, re.IGNORECASE)
    if grade_mm_x_coil_m:
        t, w = grade_mm_x_coil_m.groups()
        return f'{t}X{w}XC'

    # 1-a-4d. GRADE:{code} N.NMM X NNNNMM (2차원 치수, 코일 없음)
    grade_mm_2d_m = re.search(
        r'\bGRADE\s*:\s*\S+\s+([\d.]+)\s*MM\s*[Xx]\s*([\d.]+)\s*MM\b',
        text, re.IGNORECASE)
    if grade_mm_2d_m:
        t, w = grade_mm_2d_m.groups()
        return f'{t}X{w}'

    # 1-a-5. GRADE:{code} N*N*N 직사각봉/단조재: GRADE:1.2714 405*605*3510 → 405X605X3510
    grade_nxn_m = re.search(
        r'\bGRADE\s*:\s*[\w.]+\s+([\d.]+)\s*\*\s*([\d.]+)\s*\*\s*([\d.]+)\b',
        text, re.IGNORECASE)
    if grade_nxn_m:
        d1, d2, d3 = grade_nxn_m.groups()
        return f'{d1}X{d2}X{d3}'

    # 1-a. GRADE: {code},{OD}*{ID}*{T}MM, {L} 링/파이프 규격
    grade_ring_m = re.search(
        r'\bGRADE\s*:\s*\w+\s*,\s*([\d.]+)\s*\*\s*[\d.]+\s*\*\s*([\d.]+)\s*MM\s*,\s*([\d.]+(?:-[\d.]+)?)\s*(M(?:\b|TR|ETER)|MM)',
        text, re.IGNORECASE
    )
    if grade_ring_m:
        od, t, l_str, l_unit = grade_ring_m.groups()
        l_val = _convert_unit(l_str, l_unit)
        return f'{od}X{t}X{l_val}'

    # 1-b. DIA.{d} * {t}T * {l}M 파이프 규격: DIA.25.4 * 1.1T * 4M → 25.4X1.1X4000
    dia_t_l_m = re.search(
        r'DIA\.\s*([\d.]+)\s*\*\s*([\d.]+)\s*T\s*\*\s*([\d.]+)\s*(M(?:\b|TR|ETER)|MM|CM)',
        text, re.IGNORECASE
    )
    if dia_t_l_m:
        d, t, l, l_unit = dia_t_l_m.groups()
        l_val = _convert_unit(l, l_unit)
        return f'{d}X{t}X{l_val}'

    # 1-c. SIZE: ID{n}*OD{n}*L{n}MM 형태: SIZE: ID80*OD95*L698MM → 95X698
    size_id_od_m = re.search(
        r'\bSIZE\s*[:;=]\s*ID\s*([\d.]+)\s*\*\s*OD\s*([\d.]+)\s*\*\s*L\s*([\d.]+)',
        text, re.IGNORECASE
    )
    if size_id_od_m:
        od, l = size_id_od_m.group(2), size_id_od_m.group(3)
        return f'{od}X{l}'

    # 0-pre-a. SA179/SA334 형식: OD{n}MM X WT{m}MM[...] X{l}MM (열교환기관 OD/WT/Length)
    # SIZE 블록보다 먼저 체크: OD→WT→길이 3차원 우선
    od_wt_l_pre = re.search(
        r'(?<![A-Z])OD\s*([\d.]+)\s*MM\s*[Xx]\s*WT\s*([\d.]+)\s*MM[^Xx]*[Xx]\s*([\d.]+)\s*MM',
        text, re.IGNORECASE
    )
    if od_wt_l_pre:
        return f'{od_wt_l_pre.group(1)}X{od_wt_l_pre.group(2)}X{od_wt_l_pre.group(3)}'

    # 0-pre-b. 각도형강/채널 {h}X{h}X{t}MM 패턴 (M 단위 길이)
    # 케이스 1: ANGLE {l1.xx}-{l2.xx} (소수점 범위) → nXnXtX{l1*1000}~{l2*1000}
    angle_range_m = re.search(
        r'(\d+)\s*[Xx]\s*(\d+)\s*[Xx]\s*(\d+(?:\.\d+)?)\s*MM\s+ANGLE\s+(\d+\.\d+)\s*-\s*(\d+\.\d+)',
        text, re.IGNORECASE
    )
    if angle_range_m:
        h1, h2, t, l1, l2 = angle_range_m.groups()
        try:
            l1_mm = str(round(float(l1) * 1000))
            l2_mm = str(round(float(l2) * 1000))
            return f'{h1}X{h2}X{t}X{l1_mm}~{l2_mm}'
        except ValueError:
            pass

    # 케이스 2: {h}X{h}X{t}MM ... LENGTH:{l}M (단일 M 길이)
    angle_len_m = re.search(
        r'(\d+(?:\.\d+)?)\s*[Xx]\s*(\d+(?:\.\d+)?)\s*[Xx]\s*(\d+(?:\.\d+)?)\s*MM'
        r'.*?LENGTH\s*[:=]?\s*(\d+(?:\.\d+)?)\s*M\b(?!M)',
        text, re.IGNORECASE
    )
    if angle_len_m:
        h1, h2, t, l = angle_len_m.groups()
        try:
            l_mm = str(round(float(l) * 1000))
            return f'{h1}X{h2}X{t}X{l_mm}'
        except ValueError:
            pass

    # 0-x. SIZE: D NMM X T NMM [X L NM] 파이프 표기 (직경 x 두께 x 길이)
    # _extract_size_block의 _extract_twl이 D(diameter) 키워드를 인식하지 못해 누락되는 문제 보완
    dtp_m = re.search(
        r'\bSIZE\s*[;:=]\s*D\s*([\d.]+)\s*MM\s*[Xx]\s*T\s*([\d.]+)\s*MM'
        r'(?:\s*,?\s*[Xx]\s*L\s*([\d.]+)\s*(M\b|MM\b))?',
        text, re.IGNORECASE
    )
    if dtp_m:
        d, t, lv, lu = dtp_m.group(1), dtp_m.group(2), dtp_m.group(3), dtp_m.group(4) or ''
        if lv:
            l_mm = str(round(float(lv) * 1000)) if lu.upper() == 'M' else lv
            return f'{d}X{t}X{l_mm}'
        return f'{d}X{t}'

    # 0-y. GRAIN ORIENTED SILICON STEEL: B{n}R{t} 강종코드 제거 후 두께×폭 추출
    # 예: SIZE: B20R0.70 0.20MMX1050MM-1140MMXC → 0.20X1050~1140XC
    # 예: SIZE: 20-80 0.20MMX1080MM-1140MMXC → 0.20X1080~1140XC
    if re.search(r'GRAIN\s+ORIENT', text, re.IGNORECASE):
        grain_m = re.search(
            r'SIZE\s*[.:(MM)]*\s*(?:[A-Z]\d+[A-Z][\d.]+|[\d]+-[\d]+)?\s*([\d.]+)\s*MM[Xx*]([\d.]+)\s*MM(?:[-~]([\d.]+)\s*MM)?',
            text, re.IGNORECASE
        )
        if grain_m:
            t_val, w1, w2 = grain_m.group(1), grain_m.group(2), grain_m.group(3)
            w_str = f'{w1}~{w2}' if w2 else w1
            result = f'{t_val}X{w_str}'
            return _append_coil_if_shape(result, text) or result

    # 0-z. CIRCULAR ROD IN COILS: SIZE(MM): D NMM → OD만 추출 (외경)
    # 예: SHAPE:CURCULAR ROD IN COILS SIZE(MM): D 20MM NO. OF COILS:26 → 20
    if re.search(r'SHAPE\s*:\s*(?:CURCULAR|CIRCULAR)\s+ROD', text, re.IGNORECASE):
        circ_rod_m = re.search(
            r'SIZE\s*(?:\([^)]*\))?\s*[;:=]\s*(?:D\s*)?([\d.]+)\s*(?:MM)?',
            text, re.IGNORECASE
        )
        if circ_rod_m:
            val = circ_rod_m.group(1)
            try:
                fval = float(val)
                return str(int(fval)) if fval == int(fval) else f'{fval:g}'
            except ValueError:
                return val

    # 0-z0. ASTM A269 Seamless Tube: O.D. {od} WallT. {wt} [Condition {cond} Length(m) {l}]
    # 두 가지 length 형식 처리:
    #   1) Length(m) 6 Kg/m ...  → 6m = 6000mm
    #   2) normalize 후 L{n}M 형식
    # 예: O.D. 25 WallT. 3 Condition AP Length(m) 6 → 25X3X6000
    #     O.D. 9.53 WallT. 0.89 Condition BA → 9.53X0.89
    a269_m = re.search(
        r'\bO\.D\.\s*([\d.]+)\s+WALLT\.\s*([\d.]+)'
        r'(?:\s+CONDITION\s+\w+\s+(?:LENGTH\s*\(\s*[Mm]\s*\)\s*([\d.]+)|L([\d.]+)\s*M\b(?!M)))?',
        text, re.IGNORECASE
    )
    if a269_m:
        od = a269_m.group(1)
        wt = a269_m.group(2)
        length = a269_m.group(3) or a269_m.group(4)
        if length and float(length) <= 15:  # 15m 이하만 piece length로 간주 (초과는 공급 총길이)
            l_val = round(float(length) * 1000)
            return f'{od}X{wt}X{l_val}'
        return f'{od}X{wt}'

    # 0-z1. OD(MM):{od} WT(MM):{wt} LG(MM):{l} (TP316 seamless tube catalog format)
    # 예: TP316/316LOD(MM):8.00 WT(MM):1.00 LG(MM):6100 → 8X1X6100
    odwtlg_m = re.search(
        r'OD\s*\(\s*MM\s*\)\s*:\s*([\d.]+)\s+WT\s*\(\s*MM\s*\)\s*:\s*([\d.]+)\s+LG\s*\(\s*MM\s*\)\s*:\s*([\d.]+)',
        text, re.IGNORECASE
    )
    if odwtlg_m:
        od = f'{float(odwtlg_m.group(1)):g}'
        wt = f'{float(odwtlg_m.group(2)):g}'
        lg = f'{float(odwtlg_m.group(3)):g}'
        return f'{od}X{wt}X{lg}'

    # 0-z2. SNAP FIT SPIGOT D{OD}X{L}{grade}: 직경×길이 추출 (316L 등 강종 suffix 무시)
    # 예: SNAP FIT SPIGOTSF D154X50316L → 154X50, D219X50316L → 219X50
    if re.search(r'\bSNAP\s+FIT\b', text, re.IGNORECASE):
        sf_m = re.search(
            r'\bD\s*(\d+(?:\.\d+)?)\s*[Xx]\s*(\d+(?:\.\d+)?)(?:316L?|304L?|347H?|321H?|310S?)?\b',
            text, re.IGNORECASE
        )
        if sf_m:
            return f'{sf_m.group(1)}X{sf_m.group(2)}'

    # 0-aa. TETHER CABLE L={n}: L=150, L=200 형식 길이 추출
    # 예: TETHER CABLE FORM:B WITH CRIMP TERMINAL L=150, STAINLESS STEEL, 1.4301 → 150
    if re.search(r'\bTETHER\s+CABLE\b', text, re.IGNORECASE):
        tether_m = re.search(r'\bL\s*=\s*(\d+)', text, re.IGNORECASE)
        if tether_m:
            return tether_m.group(1)

    # 0-ab. STRANDED WIRE({n}MM): 단면 직경 추출
    # 예: STRANDED WIRE(12.7MM)-1860mpa-U → 12.7
    stranded_m = re.search(r'\bSTRANDED\s+WIRE\s*\(\s*([\d.]+)\s*MM\s*\)', text, re.IGNORECASE)
    if stranded_m:
        return stranded_m.group(1)

    # 0-ac. WELDED PIPE PIPE-{n}T: 두께(T) 추출
    # 예: WELDED PIPE PIPE-8T → 8, WELDED PIPE PIPE-10T → 10
    pipe_t_m = re.search(r'\bPIPE-(\d+)T\b', text, re.IGNORECASE)
    if pipe_t_m:
        return pipe_t_m.group(1)

    # 0-ad. W/R (Wire Rope) + DIA {n}MM: 직경만 추출 (35X7 구성 코드 제외)
    # 예: W/R GALV 35X7 C, DIA 14.0MM, LENGTH 1500M/REEL → 14
    if re.search(r'\bW/R\b', text, re.IGNORECASE):
        wr_dia_m = re.search(r'\bDIA\s+([\d.]+)\s*MM\b', text, re.IGNORECASE)
        if wr_dia_m:
            val = wr_dia_m.group(1)
            try:
                fval = float(val)
                return str(int(fval)) if fval == int(fval) else f'{fval:g}'
            except ValueError:
                return val

    # 0-ae0. OD:{od}MM, [ID:{id}MM,] WT:{wt}MM (DIN EN 10305 tube catalog): OD만 추출 (ID 무시)
    # 예: OD:25.000MM, ID:10.000MM, WT:7.500MM → 25X7.5 (ID가 _normalize에 의해 제거돼도 동작)
    din_od_wt = re.search(
        r'\bOD\s*:\s*([\d.]+)\s*MM[,\s]+(?:ID\s*:\s*[\d.]+\s*MM[,\s]+)?WT\s*:\s*([\d.]+)\s*MM\b',
        text, re.IGNORECASE
    )
    if din_od_wt:
        od = f'{float(din_od_wt.group(1)):g}'
        wt = f'{float(din_od_wt.group(2)):g}'
        return f'{od}X{wt}'

    # 0-ae. DIA(MM):{n}MM: 직경 추출
    # 예: DIA(MM):45MM → 45, DIA(MM):100MM → 100
    dia_mm_kw = re.search(r'\bDIA\s*\(\s*MM\s*\)\s*:\s*([\d.]+)\s*MM\b', text, re.IGNORECASE)
    if dia_mm_kw:
        val = dia_mm_kw.group(1)
        try:
            fval = float(val)
            return str(int(fval)) if fval == int(fval) else f'{fval:g}'
        except ValueError:
            return val

    # 0-af. PELLET DIA × TH: 펠릿 직경×두께
    # 예: NI/CR(80/20WT%) PELLET 3MMDIAX 3MMTH → 3X3
    pellet_m = re.search(r'([\d.]+)\s*MM\s*DIA\s*[Xx]?\s*([\d.]+)\s*MM\s*TH\b', text, re.IGNORECASE)
    if pellet_m:
        return f'{pellet_m.group(1)}X{pellet_m.group(2)}'

    # 1-pre-0a. D{n} X {l}MM H6 CIRCULAR BAR: 직경×길이 추출 (공차등급 H6 포함)
    # 예: D4 X 70MM H6 MT:ALLOY STEEL SHAPE:CIRCULAR BAR SIZE:70MM → 4X70
    d_h6_bar = re.search(r'(?:^|\s)D\s*(\d+(?:\.\d+)?)\s+X\s+([\d.]+)\s*MM\s+H\d\b', text, re.IGNORECASE)
    if d_h6_bar and re.search(r'\bCIRCULAR\s+BAR\b', text, re.IGNORECASE):
        return f'{d_h6_bar.group(1)}X{d_h6_bar.group(2)}'

    # 1-pre-0b. BOHLER/FLAT RANDOM LENGTH: SIZE:ROUND/FLAT {d}[X{w}] X {lo}-{hi}MM RANDOM LENGTH
    # 예: SIZE:ROUND 71 X 3.000-4.000MM RANDOM LENGTH → 71X3000~4000
    #     SIZE:FLAT 1.260 X 305 X 2.000-5.000MM RANDOM LENGTH → 1.260X305X2000~5000
    random_len_m = re.search(
        r'\bSIZE\s*:\s*(?:ROUND|FLAT)\s+([\d.,]+)(?:\s*X\s*([\d.]+))?\s*X\s*([\d.]+)\s*-\s*([\d.]+)\s*MM\s+RANDOM',
        text, re.IGNORECASE
    )
    if random_len_m:
        d1 = random_len_m.group(1).replace(',', '.')
        d2 = random_len_m.group(2)
        lo = float(random_len_m.group(3))
        hi = float(random_len_m.group(4))
        if lo < 50 and hi < 50:  # 미터 단위
            lo_mm = round(lo * 1000)
            hi_mm = round(hi * 1000)
            if d2:
                return f'{d1}X{d2}X{lo_mm}~{hi_mm}'
            return f'{d1}X{lo_mm}~{hi_mm}'

    # 1-pre-0c. {od}MM * {wt}MM * {l}MM pipe (마지막 값이 소수 → 미터 단위)
    # 예: 355.60MM * 8.00MM * 5.95MM → 355.60X8.00X5950
    pipe_mm_star = re.search(
        r'([\d.]+)\s*MM\s*\*\s*([\d.]+)\s*MM\s*\*\s*([\d.]+)\s*MM\b',
        text, re.IGNORECASE
    )
    if pipe_mm_star:
        l_val = float(pipe_mm_star.group(3))
        if l_val < 20:  # 미터 단위로 간주
            return f'{pipe_mm_star.group(1)}X{pipe_mm_star.group(2)}X{round(l_val * 1000)}'

    # 1-pre. SIZE:H{h}MMXW{w}MMXT{t}MMXL{l}MM 4차원 핸들러 (SIZE 블록 추출 전 우선 처리)
    # _extract_size_block의 _strip_grade_codes가 XL12000MM 등을 강종코드로 오인 제거하는 문제 방지
    hwt_l_pre = re.search(
        r'\bSIZE\s*:\s*H\s*([\d.]+)\s*MM\s*[Xx]\s*W\s*([\d.]+)\s*MM\s*[Xx]\s*T\s*([\d.]+)\s*MM\s*[Xx]?\s*L\s*([\d.]+)\s*MM\b',
        text, re.IGNORECASE
    )
    if hwt_l_pre:
        h, w, t, l = hwt_l_pre.group(1), hwt_l_pre.group(2), hwt_l_pre.group(3), hwt_l_pre.group(4)
        return f'{h}X{w}X{t}X{_convert_unit(l, "MM")}'

    # 1-pre-2b. SIZE: D {d}MM * L {lo}-{hi} 미터 범위 (대시 형식)
    # 예: D 120 MM * L 5.50-6.00 → 120X5500~6000
    d_l_dash_m = re.search(
        r'\bSIZE\s*:\s*D\s*([\d.]+)\s*MM\s*[Xx*]\s*L\s*([\d.]+)\s*-\s*([\d.]+)\b',
        text, re.IGNORECASE
    )
    if d_l_dash_m:
        lo = float(d_l_dash_m.group(2))
        hi = float(d_l_dash_m.group(3))
        if lo < 50 and hi < 50:  # 미터 단위 (소값 → mm 변환)
            return f'{d_l_dash_m.group(1)}X{round(lo*1000)}~{round(hi*1000)}'

    # 1-pre-2a. SIZE:D{d}MM * L{m}(-{t1}/+{t2})M 공차+미터 길이 패턴
    # 예: SIZE: D 10.00MM * L 6.00(-0/+30)M → 10.00X6000~6030
    d_l_tol_m = re.search(
        r'\bSIZE\s*:\s*D\s*([\d.]+)\s*MM\s*[Xx*]\s*L\s*([\d.]+)\s*\(\s*-\s*(\d+)\s*/\s*\+\s*(\d+)\s*\)\s*M\b(?!M)',
        text, re.IGNORECASE
    )
    if d_l_tol_m:
        d = d_l_tol_m.group(1)
        l_m = float(d_l_tol_m.group(2))
        tol_plus = int(d_l_tol_m.group(4))
        l_mm = round(l_m * 1000)
        return f'{d}X{l_mm}~{l_mm + tol_plus}'

    # 1-pre-2. SIZE:D{d}MM X L{l}[MM] 4차원 핸들러 (SIZE 블록 추출 전 우선 처리)
    # _strip_grade_codes가 L12000MM를 제거하는 문제 방지 (XL12000MM이 grade code 패턴에 매칭됨)
    # 범위 값 지원: L 6000~6030MM → 6000~6030 (normalize에서 공차 변환 후 처리)
    d_l_pre = re.search(
        r'\bSIZE\s*:\s*D\s*([\d.]+)\s*MM\s*[Xx*]\s*L\s*([\d.]+(?:~[\d.]+)?)\s*(?:MM\b)?',
        text, re.IGNORECASE
    )
    if d_l_pre:
        return f'{_convert_unit(d_l_pre.group(1), "MM")}X{_convert_unit(d_l_pre.group(2), "MM")}'

    # 1-pre-3-0. SIZE:DIA{d}MM X {l}MM L 1/2CUT 패턴 (SUS630 반절단): SIZE:DIA13MM X 4,030MM L 1/2CUT → 13X4030
    dia_lcut_m = re.search(
        r'\bSIZE\s*:\s*DIA\s*([\d.]+)\s*MM\s*[Xx]\s*([\d,]+)\s*MM(?:\s+L)?\s+\d+/\d+CUT\b',
        text, re.IGNORECASE
    )
    if dia_lcut_m:
        dia = dia_lcut_m.group(1)
        length = dia_lcut_m.group(2).replace(',', '')
        return f'{dia}X{length}'

    # 1-pre-3a. SIZE:{d}MMXL{l}MM 패턴 (L=Length, D prefix 없는 형태): SIZE:280MMXL6000MM → 280X6000
    size_mml_m = re.search(
        r'\bSIZE\s*:\s*([\d.]+)\s*MM\s*X\s*L\s*([\d.]+)\s*MM\b',
        text, re.IGNORECASE
    )
    if size_mml_m:
        d = f'{float(size_mml_m.group(1)):g}'
        l = f'{float(size_mml_m.group(2)):g}'
        return f'{d}X{l}'

    # 1-pre-3b. DIA.{n} SHAPE: ROUND BAR, SIZE: {n}MM 패턴 (DIA가 SIZE 앞에 옴)
    # 예: DIA.42 SHAPE: ROUND BAR, SIZE: 5,600MM 170PCS → 42X5600
    dia_rb_m = re.search(
        r'\bDIA\s*\.\s*([\d.]+)\s+SHAPE\s*:\s*ROUND\s+BAR.*?SIZE\s*:\s*([\d,]+)\s*MM',
        text, re.IGNORECASE
    )
    if dia_rb_m:
        length = dia_rb_m.group(2).replace(',', '')
        return f'{dia_rb_m.group(1)}X{length}'

    # 1-pre-4a. {N} FEET LENGTH 패턴: 2 FEET LENGTH → 2FT
    feet_len_m = re.search(r'\b(\d+(?:\.\d+)?)\s*FEET?\s+(?:IN\s+)?LENGTH\b', text, re.IGNORECASE)
    if feet_len_m:
        return f'{feet_len_m.group(1)}FT'

    # 1-pre-4b. MODEL NO:,강종,NMM 패턴: MODEL NO.SBI15, S55C, 4030MM → 4030
    model_grade_dim_m = re.search(r'\bMODEL\b[^,]*,\s*\w+,\s*(\d{3,6})\s*MM\b', text, re.IGNORECASE)
    if model_grade_dim_m:
        return model_grade_dim_m.group(1)

    # 1-pre-4c. <NMM 최대폭 표기: FLAT ROLLED <600MM WIDE → 600
    lt_mm_m = re.search(r'<\s*(\d+(?:\.\d+)?)\s*MM\b', text, re.IGNORECASE)
    if lt_mm_m:
        return lt_mm_m.group(1)

    # 1-pre-4d. BAR/ROUND BAR + (MATERIAL ...) + {n} X {n} MM 패턴
    # 예: SPRING STEEL ROUND BAR FOR STABILIZER (MATERIAL GRADE :STEEL SAE9254) 20.10 X 3250 MM → 20.10X3250
    if re.search(r'\b(?:ROUND\s+)?BAR\b', text, re.IGNORECASE):
        bar_mat_m = re.search(
            r'\bBAR\b.*?\([^)]*(?:MATERIAL|GRADE)[^)]*\)\s*([\d.]+)\s*[Xx]\s*([\d.]+)\s*MM\b',
            text, re.IGNORECASE
        )
        if bar_mat_m:
            d = f'{float(bar_mat_m.group(1)):g}'
            l = bar_mat_m.group(2)
            return f'{d}X{l}'
        # Case 2: ROUND/PEELED BAR ... n X n MM (괄호 제거 후, FOR {application} 포함)
        bar_for_m = re.search(
            r'\b(?:ROUND|PEELED)\s+BAR\b.*?\b([\d.]+)\s+[Xx]\s+([\d.]+)\s*MM\b',
            text, re.IGNORECASE
        )
        if bar_for_m:
            d = bar_for_m.group(1)  # 원본 정밀도 유지: 20.10 → '20.10'
            l = bar_for_m.group(2)
            return f'{d}X{l}'

    # 1. SIZE: 키워드 블록
    result = _extract_size_block(text)
    if result:
        # SHAPE:ROUND/CIRCULAR BAR + SIZE:{dia}X{n}MM 에서 n이 2~12이면 미터 단위로 간주: 6→6000
        # (예: SIZE:140*6MM → ROUNDBAR이므로 6M = 6000mm)
        if re.search(r'SHAPE\s*:\s*(?:ROUND|CIRCULAR)\s+BAR', text, re.IGNORECASE):
            rb_m = re.match(r'^([\d.]+)X(\d{1,2})$', result)
            if rb_m:
                n = int(rb_m.group(2))
                if 2 <= n <= 12:
                    result = f'{rb_m.group(1)}X{n * 1000}'
        # H-BEAM/I-BEAM/CHANNEL 컨텍스트: 마지막 차원이 소정수(3~20)이면 미터 단위로 변환
        # 예: SIZE: H125X125X6.5/9 10 → 125X125X6.5X9X10000, C300X90X9 10 → 300X90X9X10000
        _beam_trailing_ctx = bool(
            re.search(r'\b(?:H[-\s]?BEAM|I[-\s]?BEAM|SHAPE\s*:\s*(?:H\s*SECTION|I\s*BEAM|L\s*SECTION))\b', text, re.IGNORECASE)
            or re.search(r'\bCHANNEL\s+SIZE\s*:', text, re.IGNORECASE)
        )
        if _beam_trailing_ctx:
            hb_m = re.match(r'^((?:[\d.]+X)+)(\d{1,2})$', result)
            if hb_m:
                last_n = int(hb_m.group(2))
                if 3 <= last_n <= 20:
                    result = f'{hb_m.group(1)}{last_n * 1000}'
        # WIRE trailing zero 제거: 7.000→7, 0.90X350→0.9X350, 2.30X11.25→2.3X11.25
        # WIRE ROD 또는 WIRE+SIZE 맥락에서 적용 (단일/다차원 모두)
        _wire_ctx = (
            re.search(r'SHAPE\s*:\s*WIRE\s*ROD', text, re.IGNORECASE)
            or re.search(r'\bWIRE\b[^X]*SIZE\s*:', text, re.IGNORECASE)
        )
        if _wire_ctx and '.' in result:
            if 'X' not in result:
                try:
                    result = f'{float(result):g}'
                except ValueError:
                    pass
            else:
                parts = result.split('X')
                result = 'X'.join(
                    f'{float(p):g}' if re.match(r'^[\d.]+$', p) else p
                    for p in parts
                )
        # PIPE/TUBE SHAPE인데 XC 포함이면 제거 (SEAMLESS COIL TUBE에서 *C는 코일 표시 아님)
        if (result.upper().endswith('XC')
                and re.search(r'SHAPE\s*:\s*\w*\s*(?:PIPE|TUBE)', text, re.IGNORECASE)
                and not re.search(r'SHAPE\s*:\s*(?:IN\s+)?COIL\b', text, re.IGNORECASE)):
            result = result[:-2]
        # 단일 숫자 결과: trailing zero 제거 (0.50→0.5)
        if re.match(r'^[\d.]+(?:IN|FT)?$', result) and '.' in result:
            try:
                result = f'{float(result):g}'
            except ValueError:
                pass
        return _append_coil_if_shape(result, text) or result

    # 2-0a. SA179/SA334 형식: OD{n}MM X WT{m}MM[...] X{l}MM (열교환기관 OD/WT/Length)
    # _extract_twl보다 먼저 체크: OD/WT로 시작하면 길이까지 포함한 3차원 매칭 우선
    od_wt_l_m = re.search(
        r'(?<![A-Z])OD\s*([\d.]+)\s*MM\s*[Xx]\s*WT\s*([\d.]+)\s*MM[^Xx]*[Xx]\s*([\d.]+)\s*MM',
        text, re.IGNORECASE
    )
    if od_wt_l_m:
        return f'{od_wt_l_m.group(1)}X{od_wt_l_m.group(2)}X{od_wt_l_m.group(3)}'

    # 2-0a-2. SIZE:D{d}MM X L{l}[MM] 패턴: BAR/ROD 직경×길이 (* 구분자 허용, 후행MM 선택)
    # 예: SIZE:D20MM X L3000MM → 20X3000, SIZE:D36MM X L6000 → 36X6000
    d_l_m = re.search(
        r'\bSIZE\s*:\s*D\s*([\d.]+)\s*MM\s*[Xx*]\s*L\s*([\d.]+)\s*(?:MM\b)?',
        text, re.IGNORECASE
    )
    if d_l_m:
        return f'{_convert_unit(d_l_m.group(1), "MM")}X{_convert_unit(d_l_m.group(2), "MM")}'

    # 2-0a-3. SIZE:H{h}MMXW{w}MMXT{t}MMXL{l}MM 각형 파이프/빔 4차원
    # 예: SIZE:H150MMXW100MMXT6.3MMXL12000MM → 150X100X6.3X12000
    hwt_l_m = re.search(
        r'\bSIZE\s*:\s*H\s*([\d.]+)\s*MM\s*[Xx]\s*W\s*([\d.]+)\s*MM\s*[Xx]\s*T\s*([\d.]+)\s*MM\s*[Xx]?\s*L\s*([\d.]+)\s*MM\b',
        text, re.IGNORECASE
    )
    if hwt_l_m:
        h, w, t, l = hwt_l_m.group(1), hwt_l_m.group(2), hwt_l_m.group(3), hwt_l_m.group(4)
        l_val = _convert_unit(l, 'MM')
        return f'{h}X{w}X{t}X{l_val}'

    # 2-0a-3b. ROUND BAR {dim} OD X {dim} 패턴: A-286 5853 CR - ROUND BAR 0.76OD X 50 → 0.76X50
    round_bar_od_m = re.search(
        r'ROUND\s+BAR\b.*?([\d.]+)\s*OD\s*[Xx]\s*([\d.]+)',
        text, re.IGNORECASE
    )
    if round_bar_od_m:
        return f'{round_bar_od_m.group(1)}X{round_bar_od_m.group(2)}'

    # 2-0a-4. STAINLESS STEEL STRIP IN COIL ... NMMX NMMXCOIL 패턴
    # 파트번호 사이에 끼인 실제 치수만 추출: 5B7KG03 ... 0.5MMX28.7MMXCOIL → 0.5X28.7XC
    strip_coil_m = re.search(
        r'\bSTRIP\s+IN\s+COIL\b.*?([\d.]+)\s*MM\s*[Xx*]\s*([\d.]+)\s*MM\s*[Xx*]\s*COIL\b',
        text, re.IGNORECASE
    )
    if strip_coil_m:
        return f'{strip_coil_m.group(1)}X{strip_coil_m.group(2)}XC'

    # 2. T/W/L 명시 패턴
    result = _extract_twl(text)
    if result:
        return _append_coil_if_shape(result, text) or result

    # 2-0b-pre. PLATE {d1}X{d2} {t} 패턴: X-연결 2차원 뒤에 공백+단독 두께값
    # 예: STEEL PLATE 1219X2438 6T SUS → 1219X2438X6 (normalize가 6T→6 처리 후)
    if re.search(r'\bPLATE\b', text, re.IGNORECASE):
        plate_t_m = re.search(
            r'\b(\d{3,}(?:\.\d+)?)\s*[Xx]\s*(\d{3,}(?:\.\d+)?)\s+(\d{1,3}(?:\.\d+)?)\b',
            text, re.IGNORECASE
        )
        if plate_t_m:
            d1, d2, t = plate_t_m.group(1), plate_t_m.group(2), plate_t_m.group(3)
            try:
                f1, f2, ft = float(d1), float(d2), float(t)
                if ft < f1 and ft < f2:  # t가 가장 작은 값 (두께)
                    return f'{d1}X{d2}X{t}'
            except ValueError:
                pass

    # 2-0b. 무계목 강관: {OD}MM X {WT}MM X {L}M 패턴 (PIPE 키워드 + 마지막 단위 M)
    if re.search(r'\bPIPE', text, re.IGNORECASE):
        # 정사각형/직사각형 강관: PIPE{h}X{w}X{t}MMX{l}M → {h}X{t}X{l*1000} (정사각 h==w)
        rect_pipe_m = re.search(
            r'\bPIPE\s*([\d.]+)\s*[Xx]\s*([\d.]+)\s*[Xx]\s*([\d.]+)\s*MM\s*(?:[Xx]\s*([\d.]+)\s*(M\b(?!M)|MM\b))?',
            text, re.IGNORECASE
        )
        if rect_pipe_m:
            h, w, t = rect_pipe_m.group(1), rect_pipe_m.group(2), rect_pipe_m.group(3)
            l_raw, l_unit = rect_pipe_m.group(4), rect_pipe_m.group(5) or ''
            try:
                if l_raw:
                    lval = _convert_unit(l_raw, l_unit.upper().replace(' ', ''))
                    return (f'{h}X{t}X{lval}' if h == w else f'{h}X{w}X{t}X{lval}')
                else:
                    return (f'{h}X{t}' if h == w else f'{h}X{w}X{t}')
            except (ValueError, AttributeError):
                pass
        pipe_odwt_m = re.search(
            r'(?<!\d)([\d.]+)\s*MM\s*[Xx]\s*([\d.]+)\s*MM\s*[Xx]\s*([\d.]+(?:[./-][\d.]+)?)\s*M\b(?!M)',
            text, re.IGNORECASE
        )
        if pipe_odwt_m:
            od, wt, l = pipe_odwt_m.group(1), pipe_odwt_m.group(2), pipe_odwt_m.group(3)
            try:
                if float(od) > float(wt):
                    l_parts = re.split(r'[-/]', l, maxsplit=1)
                    if len(l_parts) == 2:
                        l1 = str(round(float(l_parts[0]) * 1000))
                        l2 = str(round(float(l_parts[1]) * 1000))
                        return f'{od}X{wt}X{l1}~{l2}'
                    return f'{od}X{wt}X{round(float(l) * 1000)}'
            except ValueError:
                pass
        # 단독 인치 파이프: SUS304 PIPE 18" → 18IN, PIPE 1/2" → 1/2IN
        pipe_inch_m = re.search(
            r'\bPIPE\s+((?:\d+-\d+/\d+|\d+/\d+|\d+\.?\d*))\s*"',
            text, re.IGNORECASE
        )
        if pipe_inch_m:
            return f'{pipe_inch_m.group(1)}IN'

    # 2-0. NMM X NMM X NMM [X NMM] 형식 3~4차원 치수 (순서 보존): 51MM X15MM X 6000MM → 51X15X6000
    pipe3d_m = re.search(
        r'(?<!\d)([\d.]+(?:[~-][\d.]+)?)\s*MM\s*[Xx]?\s*([\d.]+(?:[~-][\d.]+)?)\s*MM\s*[Xx]?\s*([\d.]+(?:[~-][\d.]+)?)\s*MM(?:\s*[Xx]\s*([\d.]+(?:[~-][\d.]+)?)\s*MM)?(?![0-9])',
        text, re.IGNORECASE
    )
    if pipe3d_m:
        a = _convert_unit(pipe3d_m.group(1), 'MM')
        b = _convert_unit(pipe3d_m.group(2), 'MM')
        c = _convert_unit(pipe3d_m.group(3), 'MM')
        d = pipe3d_m.group(4)
        if d:
            return f'{a}X{b}X{c}X{_convert_unit(d, "MM")}'
        return f'{a}X{b}X{c}'

    # 2-1. LENGTH:/DIAMETER: 키워드 패턴 (예: LENGTH:628MM, DIAMETER:3.0MM)
    len_m = re.search(r'LENGTH\s*:\s*([\d.]+)\s*(MM|CM|(?<!\w)M(?!\w)|MTR)?', text, re.IGNORECASE)
    dia_m = re.search(r'DIAMETER\s*:\s*([\d.]+)\s*(MM|CM|(?<!\w)M(?!\w)|MTR)?', text, re.IGNORECASE)
    if len_m and dia_m:
        l_val = _convert_unit(len_m.group(1), len_m.group(2) or '')
        d_val = _convert_unit(dia_m.group(1), dia_m.group(2) or '')
        return f'{l_val}X{d_val}'

    # 2-2. DIN 강종코드 + 치수: "DIN 1.4404 15X3000" → 15X3000
    din_m = re.search(r'\bDIN\s+1\.\d{4}\s+([\d]+(?:\s*[Xx]\s*[\d]+)+)', text, re.IGNORECASE)
    if din_m:
        result = _parse_tokens(din_m.group(1), preserve_order=True)
        if result:
            return _append_coil_if_shape(result, text) or result

    # 2-3. 실린더/튜브 D_inner X (D_)outer X L 패턴: D90XD102XL578 → 102X578
    #      두 직경 중 큰 값(OD)과 L만 유지
    dxdxl_m = re.search(
        r'\bD([\d.]+)[Xx]D?([\d.]+)[Xx]L([\d.]+)\b',
        text, re.IGNORECASE
    )
    if dxdxl_m:
        d1, d2, l = dxdxl_m.group(1), dxdxl_m.group(2), dxdxl_m.group(3)
        od = d2 if float(d2) > float(d1) else d1
        return f'{od}X{l}'

    # 2-4. OD*ID*L 또는 ID*OD*L 형태: OD304*ID215*L2535 → OD X Length (ID 무시)
    od_id_l = re.search(
        r'(?<![A-Z])OD\s*(\d+(?:\.\d+)?)\s*\*\s*ID\s*(\d+(?:\.\d+)?)\s*\*\s*[LH]\s*(\d+(?:\.\d+)?)',
        text, re.IGNORECASE
    )
    if od_id_l:
        od, l = od_id_l.group(1), od_id_l.group(3)
        return _append_coil_if_shape(f'{od}X{l}', text) or f'{od}X{l}'

    id_od_l = re.search(
        r'\bID\s*(\d+(?:\.\d+)?)\s*\*\s*OD\s*(\d+(?:\.\d+)?)\s*\*\s*[LH]\s*(\d+(?:\.\d+)?)',
        text, re.IGNORECASE
    )
    if id_od_l:
        od, l = id_od_l.group(2), id_od_l.group(3)
        return _append_coil_if_shape(f'{od}X{l}', text) or f'{od}X{l}'

    # 3. 규격 자체가 사이즈인 경우 (숫자X숫자 형태)
    #    텍스트 전체가 거의 숫자+X로만 구성된 경우 (IN/FT 단위 포함)
    bare = text.strip()
    if re.match(r'^[\d.~/]+(?:IN|FT|MM)?(?:[Xx][\d.~/]+(?:IN|FT|MM)?)*[Xx]?[Cc]?$', bare, re.IGNORECASE):
        result_bare = bare.upper().replace('x', 'X')
        result_bare = re.sub(r'(\d)MM\b', r'\1', result_bare)
        return result_bare

    # 3-1. 끝부분 X-연결 치수 블록: "모델코드 2.15X1.70X240" → 2.15X1.70X240
    #      trailing 단위 문자(MM, CM 등) 허용
    trail_m = re.search(r'(?<!\d)([\d.]+[Xx][\d.]+(?:[Xx][\d.]+)*)(?:[A-Z]*)\s*$', text)
    if trail_m:
        pos = trail_m.start(1)
        if pos > 0 and text[pos - 1] == ' ':
            cleaned = _strip_grade_codes(trail_m.group(1))
            result = _parse_tokens(cleaned, preserve_order=True)
            if result:
                return _append_coil_if_shape(result, text) or result

    # 3-2. DIA.(단위): 형태 — wire rope 등: DIA.(MM):0.8 → 0.8
    dia_unit_m = re.search(r'DIA\.\s*\(\s*(?:MM|IN|CM|FT)\s*\)\s*[:=]\s*([\d.]+)', text, re.IGNORECASE)
    if dia_unit_m:
        return dia_unit_m.group(1)

    # 4. 앞부분에 강종코드/설명 있고 뒤에 사이즈가 오는 패턴
    #    설명 키워드 이후 텍스트 제거, 강종코드/부품번호/버전 제거 후 추출
    # * 구분자 있으면 순서 보존 (PIPE*22.3*0.4 형태)
    has_star_sep = bool(re.search(r'\d\s*\*\s*\d', text))
    has_cert = bool(re.search(r'\bCERT\b', text, re.IGNORECASE))
    # 공백-X 구분자(50 X 0.90MM 형식)는 원래 순서가 의미 있음
    # 단위(MM/CM/IN/FT) 뒤 공백+X 패턴도 포함: 20MM X 19.5MM (칼라 치수 변환 결과)
    has_space_x = bool(re.search(r'\d(?:MM|CM|IN|FT|M\b)?\s+[Xx]\s+\d', text, re.IGNORECASE))
    # 직결 X 구분자(267.40X12.70 형식)도 원래 순서 유지 (PIPE OD×WT 명시 순서)
    has_direct_x = bool(re.search(r'\d[Xx]\d', text))
    # SEAMLESS/PIPE 맥락: 공백 구분 치수도 표기 순서 유지 (관례상 OD를 먼저 기재)
    has_pipe_seq = bool(re.search(r'\b(?:SEAMLESS|PIPE)\b', text, re.IGNORECASE))
    clean = re.sub(
        r'\b(?:MODEL|MT|PS|SHAPE|GRADE|FOR|ACC|TO|SPEC|DLY\s*CODE)\b.*',
        '', text, flags=re.IGNORECASE
    )
    # 앞부분 부품번호-강종코드 패턴 제거: 79073-SKD1122 → SKD1122
    clean = re.sub(r'^\d{4,8}-(?=[A-Z])', '', clean.strip())
    clean = _clean_size_block(clean)
    clean = _strip_grade_codes(clean)
    # 강종코드 제거 후 남는 잔류 하이픈 숫자 코드 제거: J034 제거 후 -17-2 잔류 등
    clean = re.sub(r'\s+-\d{1,3}-\d{1,3}\b', '', clean)
    # 그레이드 코드 제거 후 선행 점 재보정: OCR25AL50.29MM → .29MM → 0.29MM
    clean = re.sub(r'(?<![A-Z\d])\.(\d)', r'0.\1', clean)
    # clean이 숫자로 시작하면 원래 순서 유지 (예: 139.8MMX15.90MMX6.0M)
    # CERT 포함 시 치수 순서 유지 (254 X 27 MM → 254X27)
    preserve_start = bool(re.match(r'^\s*[\d.]', clean)) or has_cert
    result = _parse_tokens(clean, preserve_order=has_star_sep or preserve_start or has_space_x or has_direct_x or has_pipe_seq)
    if result and ('X' in result or '~' in result):
        # 정수 인치/피트 trailing zero 제거: 8.000IN → 8IN
        result = re.sub(r'(\d+)\.0+(IN|FT)\b', r'\1\2', result)
        # ROUND BAR 공차 분수 제거: 13X4030X1/2 → 13X4030 (SUS630 H900 등)
        if re.search(r'SHAPE\s*:\s*ROUND\s*BAR', text, re.IGNORECASE):
            result = re.sub(r'X\d+/\d+$', '', result)
        # COIL 컨텍스트 trailing zero 제거: 0.200X40XC → 0.2X40XC (COIL/PIPE 혼합 제외)
        if (re.search(r'\bCOIL\b', text, re.IGNORECASE)
                and not re.search(r'\b(?:PIPE|TUBE)\b', text, re.IGNORECASE)):
            parts = result.split('X')
            result = 'X'.join(
                f'{float(p):g}' if re.match(r'^[\d.]+$', p) else p
                for p in parts
            )
        return _append_coil_if_shape(result, text) or result
    # 단일 치수: clean 텍스트가 거의 치수 정보만 남은 경우 반환
    if result and re.match(r'^[\d.]+(?:IN|FT)?$', result):
        clean_alpha = re.sub(r'\d|[.\s/\-]', '', clean)
        ends_with_num = bool(re.search(r'[\d.]+\s*(?:MM|IN|FT|CM|MTR|METERS?|")?$', clean.strip(), re.IGNORECASE))
        # 텍스트가 숫자+단위로 시작하면 (S235JR 2MM STEEL SHEET... 형태) 해당 치수가 명확
        starts_with_dim = bool(re.match(r'^[\d.]+\s*(?:MM|CM|IN|FT|M\b)', clean.strip(), re.IGNORECASE))
        is_wire_coil = bool(re.search(r'\bWIRE\b', text, re.IGNORECASE) and re.search(r'\bCOIL\b', text, re.IGNORECASE))
        if len(clean_alpha) <= 4 or ends_with_num or starts_with_dim or is_wire_coil:  # 단위 글자만 남거나 끝에 치수값이 있는 경우
            try:
                fval = float(result)
                return f'{fval:g}' if '.' in result else result
            except ValueError:
                return result

    # 4-x. diameter N.NMM (키워드 그대로 표기): NCHW-1, diameter 0.5mm, 20KG/D250 → 0.5
    diameter_kw_m = re.search(r'\bdiameter\s+([\d.]+)\s*(MM|CM|M\b|IN(?:CH)?|FT|")', text, re.IGNORECASE)
    if diameter_kw_m:
        val = _convert_unit(diameter_kw_m.group(1), diameter_kw_m.group(2) or '')
        try:
            return f'{float(val):g}'
        except ValueError:
            return val

    # 4-y. WIDTH N MM 단독 폭 표기: AMORPHOUS ALLOY RIBBON,WIDTH 120MM → 120
    width_kw_m = re.search(r'\bWIDTH\s+([\d.]+)\s*(MM|CM|IN|FT|M\b)', text, re.IGNORECASE)
    if width_kw_m:
        val = _convert_unit(width_kw_m.group(1), width_kw_m.group(2))
        try:
            return f'{float(val):g}'
        except ValueError:
            return val

    # 5. DIA / D= 단일 또는 X-연결 치수 (DIA.25X7000MM, DIA.: 3.0, DIA.(MM):0.8 포함)
    # 5-a. DIA.(단위): 형태: ROPE7X7DIA.(MM):0.8 → 0.8 (step 3-2에서도 처리, 중복 방어)
    dia_unit_m = re.search(r'DIA\.\s*\(\s*(?:MM|IN|CM|FT)\s*\)\s*[:=]\s*([\d.]+)', text, re.IGNORECASE)
    if dia_unit_m:
        return dia_unit_m.group(1)
    # 5-b. 일반 DIA/D=/OD 패턴 (OD: OD530MM → 530)
    m = re.search(r'(?:DIA|D\s*=|OD)[\s.:-]*\.?\s*([\d.]+(?:(?:MM|CM|M\b)?\s*[Xx]\s*[\d.]+(?:MM|CM|M)?)*)', text, re.IGNORECASE)
    if m and re.search(r'\d', m.group(1)):  # 실제 숫자가 있어야 함 (단순 '.'만 캡처 방지)
        result = _parse_tokens(m.group(1), preserve_order=True) or m.group(1).replace('X', 'X').strip()
        if result:
            # DIA {n.0} → n (trailing zero 제거)
            try:
                if re.match(r'^[\d.]+$', result):
                    result = f'{float(result):g}'
            except ValueError:
                pass
        if result and re.search(r'SHAPE\s*:\s*(?:IN\s+)?COIL', text, re.IGNORECASE):
            result = result + 'XC'
        return result

    return None


def _append_coil_if_shape(result: Optional[str], text: str) -> Optional[str]:
    """SHAPE:COIL, IN COILS, *C 등 코일 표시 시 XC 미포함이면 추가"""
    if result and 'X' in result and 'XC' not in result.upper():
        # COIL TUBE/SPRING/PIPE/WIRE 등 복합 제품명은 제외
        if re.search(
            r'SHAPE\s*:\s*(?:IN\s+)?COIL'
            r'|SHAPE\s*:\s*XC'
            r'|IN\s*COILS?\d*(?:\s|,|/|$)'      # INCOILS, IN COILS, IN COILS30400/
            r'|IN\s+COLS\b'                       # 오타: IN COLS
            r'|(?<![A-WY-Z])\d*COILS?\b(?!\s+(?:TUBE|SPRING|PIPE|WIRE|BAR|ROD))'  # XCOIL 허용
            r'|(?<![A-Z])COIL(?=\d)',             # COIL0.35 등 숫자 직접 연결
            text, re.IGNORECASE
        ):
            # SHEET IN COILS (3차원 치수): 코일에서 자른 낱장 시트 → XC 불필요
            # (PLATE IN COILS는 코일 형태 제품이므로 XC 유지)
            if re.search(r'\bSHEET\s+IN\s+COILS?\b', text, re.IGNORECASE) and result.count('X') >= 2:
                return result
            return result + 'XC'
        # CIRCULAR BAR + 인장강도(TS:) → 와이어로드 코일 형태
        if (re.search(r'SHAPE\s*:\s*CIRCULAR\s+BAR', text, re.IGNORECASE)
                and re.search(r'\bTS\s*:', text, re.IGNORECASE)):
            return result + 'XC'
        # SLIT + SHAPE:SHEET → 슬리팅 코일 (CRC 냉연코일 납품 형태)
        if re.search(r'\bSLIT\b', text, re.IGNORECASE) and re.search(r'SHAPE\s*:\s*SHEET', text, re.IGNORECASE):
            return result + 'XC'
        # CRC(냉연코일) + SHAPE:SHEET → 코일 형태 납품
        if re.search(r'\bCRC\b', text, re.IGNORECASE) and re.search(r'SHAPE\s*:\s*SHEET', text, re.IGNORECASE):
            return result + 'XC'
        # SIZE 블록 내 *C 또는 XC 접미사 표기 (예: 2.5*158*C, 1.4*553*C)
        if re.search(r'\*\s*C\s*(?:[(\s,]|$)', text, re.IGNORECASE):
            # SHAPE: PIPE/TUBE → *C는 코일 표시 아님 (예: SEAMLESS COIL TUBE의 SHAPE:SEAMLESS PIPE)
            if re.search(r'SHAPE\s*:\s*(?:\w+\s+)*(?:PIPE|TUBE)\b', text, re.IGNORECASE):
                return result
            return result + 'XC'
        # 말미 " X C" 코일 표시 (STRIP 컨텍스트 또는 MMW 직결): {n}MMW X C → XC 추가
        # 예: 0.178MM X 389MMW X C → 0.178X389XC
        if re.search(r'(?:MMW|W)\s+X\s+C\s*$', text, re.IGNORECASE):
            return result + 'XC'
    return result


class SizeExtractor:
    def __init__(self):
        self._client = OpenAI(api_key=settings.openai_api_key)

    def extract_batch(self, spec_texts: list[str]) -> list[tuple[Optional[str], str]]:
        """배치 추출 — 정확매칭 → 정규식 → LLM"""
        lookup = _get_exact_lookup()
        results: list[tuple[Optional[str], str]] = []
        llm_needed: list[int] = []

        exact_cnt = 0
        regex_cnt = 0

        for i, spec in enumerate(spec_texts):
            # 1. 정확 매칭
            key = spec.strip().upper()
            if key in lookup:
                results.append((lookup[key], 'exact'))
                exact_cnt += 1
                continue

            # 2. 정규식
            size = extract_size_regex(spec)
            if size:
                results.append((size, 'regex'))
                regex_cnt += 1
            else:
                results.append((None, 'pending'))
                llm_needed.append(i)

        print(f"  [사이즈] 정확매칭: {exact_cnt}건, 정규식: {regex_cnt}건, LLM 폴백: {len(llm_needed)}건")

        if llm_needed:
            print(f"  [사이즈] 정규식: {len(spec_texts)-len(llm_needed)}건, LLM 폴백: {len(llm_needed)}건")
            for i in llm_needed:
                spec = spec_texts[i]
                for attempt in range(3):
                    try:
                        resp = self._client.chat.completions.create(
                            model=settings.llm_model,
                            messages=[
                                {"role": "system", "content": _SIZE_LLM_PROMPT},
                                {"role": "user", "content": f"규격품명: {spec}"},
                            ],
                            response_format={"type": "json_object"},
                        )
                        data = json.loads(resp.choices[0].message.content)
                        size = data.get("size")
                        if size and str(size).strip() not in ("", "null", "None"):
                            results[i] = (str(size).strip(), 'llm')
                        else:
                            results[i] = (None, 'failed')
                        break
                    except RateLimitError:
                        time.sleep(60 * (attempt + 1))
                    except Exception:
                        results[i] = (None, 'failed')
                        break

        return results
