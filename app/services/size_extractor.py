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
    # 언더스코어를 공백으로: OD:1.125IN_L:8IN → OD:1.125IN L:8IN (레이블 구분자)
    s = s.replace('_', ' ')
    # Ø (직경 기호 U+00D8/U+2300) → DIA: Ø14.1 → DIA14.1
    s = s.replace('Ø', 'DIA').replace('⌀', 'DIA')
    # W/L 레이블 뒤 천단위 쉼표 제거 (European decimal 변환 전에 먼저 처리)
    # W 1,255MM → W 1255MM, L 4,620MM → L 4620MM (폭/길이는 천단위 구분자)
    s = re.sub(r'\b(W|L)\s+(\d{1,4}),(\d{3})(MM|CM|IN|FT)\b', r'\1 \2\3\4', s, flags=re.IGNORECASE)
    # PLATE/SHEET 맥락 1자리+콤마+3자리: 천단위 구분자 (먼저 처리, 유럽식 소수점 규칙 전에)
    # 예: 1,277MM X → 1277MM (판재 폭/길이, 단 비-PLATE 맥락은 아래 유럽식 소수점 규칙으로)
    if re.search(r'\bPLATE\b', s, re.IGNORECASE):
        s = re.sub(r'\b([1-9]),(\d{3})(MM)\b', r'\1\2\3', s, flags=re.IGNORECASE)
    # 유럽식 소수점 쉼표: 3,100MM X → 3.100MM X (뒤에 X가 있을 때만 변환, 천단위 콤마와 구분)
    # L:4,825MM처럼 단독으로 끝나는 경우는 천단위 콤마로 간주
    # 반드시 천단위 콤마 제거 전에 처리해야 함
    s = re.sub(r'\b(\d{1,2}),(\d{3})(MM)(?=\s*[Xx])', r'\1.\2\3', s)
    # 천단위 콤마 제거: 6,000 → 6000 (알파벳/숫자 직후 콤마는 구분자로 보존)
    s = re.sub(r'(?<![A-Z\d])(\d{1,3}),(\d{3})(?!\d)', r'\1\2', s)
    # X 뒤 천단위 콤마 제거: 120MMX6,700MM → 120MMX6700MM
    s = re.sub(r'(?<=[Xx])(\d{1,3}),(\d{3})(?!\d)', r'\1\2', s)
    # 두께/직경 접미사 T/D 제거: 0.4T → 0.4, 48.6D → 48.6, 6T → 6 (정수 포함)
    # 단, 알파벳/하이픈 직후 T는 강종코드이므로 제외 (S45C-T, 3/4H-T 등)
    s = re.sub(r'(\d+\.\d+)[TD](?=[X\s,*/]|$)', r'\1', s)
    s = re.sub(r'(?<![A-Z\-])(\d+)T(?=[X\s,*/]|$)', r'\1', s)
    # MML(길이 단위 MM+L) → MM 통일: 4020MML → 4020MM
    s = re.sub(r'MML\b', 'MM', s, flags=re.IGNORECASE)
    # MMM 이상 중복 M → MM 통일: 3.15MMM → 3.15MM
    s = re.sub(r'M{3,}', 'MM', s, flags=re.IGNORECASE)
    # 선행 점 소수 보정: .063 → 0.063 (알파벳/숫자 직후 점은 구분자이므로 제외)
    s = re.sub(r'(?<![A-Z\d])\.(\d)', r'0.\1', s)
    # 숫자 직후 M 단위가 키워드로 이어지면 공백 삽입: 6.0MGRADE → 6.0M GRADE
    s = re.sub(r'(?<=\d)(M)(?=GRADE|SPEC|SHAPE|MODEL|TEMPER|HEAT|CERT)', r'\1 ', s, flags=re.IGNORECASE)
    # 치수-MT: 분리: 237MT:STAINLESS → 237 MT:STAINLESS (키워드 strip이 정확히 동작하도록)
    s = re.sub(r'(\d)(MT)\s*:', r'\1 \2:', s, flags=re.IGNORECASE)
    # COILat 등 COIL 뒤에 의미없는 알파 분리: COILat → COIL AT
    s = re.sub(r'\b(COIL)([A-BD-Z][A-Z]{1,3})\b', r'\1 \2', s, flags=re.IGNORECASE)
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
    s = re.sub(r'(MM|IN|FT|CM)([A-Z]{2,5}\d{5,})', r'\1 \2', s, flags=re.IGNORECASE)
    # 단위 직후 ABOUT 분리: 2.30MMABOUT → 2.30MM ABOUT
    s = re.sub(r'(MM|IN|FT|CM)(ABOUT)\b', r'\1 \2', s, flags=re.IGNORECASE)
    # 구조재 폭 표기: BOOM 133"*4.5 형태에서 " 뒤에 * 가 오면 치수 구분자 → IN 변환 안함
    s = re.sub(r'(\d+)"\s*(?=\*)', r'\1 ', s)
    # PIPE 맥락 두께+길이 연결 표기: 12.76000 → 12.7 6000 (소수 1자리 + 4자리 길이)
    if re.search(r'\bPIPE\b', s, re.IGNORECASE):
        s = re.sub(r'\b(\d+\.\d)([456789]\d{3})\b', r'\1 \2', s)
    # SHEET 맥락 유럽식 점-천단위: X.YYZ MM (마지막 자리 ≠ 0) → XYYY MM
    # 예: 1.219MM → 1219MM, 2.438MM → 2438MM (단, 8.800MM 등 trailing 0은 소수점으로 유지)
    if re.search(r'\bSHEET\b', s, re.IGNORECASE):
        s = re.sub(r'\b([1-9])\.(\d{2}[1-9])\s*(MM)\b', r'\1\2\3', s, flags=re.IGNORECASE)
    # X NNNN(CUT) 절단 길이 제거: X 2950(CUT) → ' ' (파이프 절단 길이는 치수 아님, 공백으로 분리)
    s = re.sub(r'\s*[Xx]\s*\d+\s*\(CUT\)\b', ' ', s, flags=re.IGNORECASE)
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
        # N.000+ (소수 3자리 이상 모두 0) → 정수화: 350.000→350, 60.000→60 (10 이상만 적용)
        # 3.000 같이 소수점 정밀도가 의미있는 값은 유지
        if '.' in num_str and '/' not in num_str:
            dec_part = num_str.split('.', 1)[1]
            if len(dec_part) >= 3 and all(c == '0' for c in dec_part):
                try:
                    fval = float(num_str)
                    if fval >= 10:
                        return str(int(fval))
                except ValueError:
                    pass
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
    # X{digits} 치수 구분자 임시 보호: X237 → X_237 (강종코드로 오인 방지)
    text = re.sub(r'\b(X)(\d+)\b', r'\1_\2', text)
    # SCH{n} 스케줄 코드 임시 보호: SCH160 → SCH__160 (grade code로 오인 삭제 방지)
    text = re.sub(r'\bSCH(\d+)\b', r'SCH__\1', text)
    # OD{n} 외경 접두사 임시 보호: OD406 → OD__406 (강종코드 오인 삭제 방지)
    text = re.sub(r'\bOD(\d)', r'OD__\1', text)
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
    return result.replace('OD__', 'OD')


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
        r'(?:\([TWL]\d*\)|\b(?:[TWL]|OD|THK|WT)(?=[\d\s.:\-]))'  # T/W/L/OD/THK/WT 뒤에 숫자(콜론 허용)
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
    # TS 인장강도 이후 제거: X C TS:359-477MPA → X C
    block = re.sub(r'\s*\bTS\s*:.*$', '', block, flags=re.IGNORECASE)
    # USE: 이후 제거
    block = re.sub(r'\s*,?\s*\bUSE\s*:.*$', '', block, flags=re.IGNORECASE)
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
    # 무게/수량 표기 제거: 1650KG, 12.5TON, 5KC, ABOUT 3KG/BOBBIN 등
    block = re.sub(r'\s*(?:ABOUT\s*)?\d+\.?\d*\s*(?:KGS?|TON|LBS?|LB|KC)\b(?:/\w+)?', '', block, flags=re.IGNORECASE)
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
    # 선두 6자리 부품번호 제거: 699397 18/8-321N → (제거) (단위 없이 시작하는 경우만)
    block = re.sub(r'^\d{6}\b(?!\s*(?:MM|CM|M\b|IN|FT))', '', block.strip()).strip()
    # 7자리 이상 연속 숫자 (부품번호/날짜코드/시리얼) 제거 (X 뒤는 치수구분자이므로 허용)
    block = re.sub(r'(?<![A-WY-Z])\d{7,}', '', block)
    # ASTM/ASME 규격 번호 제거: SB 575, SA 312, SE 309 등 (2~4자리 숫자)
    block = re.sub(r'\bS[ABE]\s+\d{2,4}\b', '', block, flags=re.IGNORECASE)
    # 괄호 안 강종 코드 제거: (904L), (625L), (316) 등 (숫자+알파 조합) — 공백으로 치환
    block = re.sub(r'\(\s*\d+[A-Z]{1,3}\s*\)', ' ', block, flags=re.IGNORECASE)
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
    return block.strip()


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
        r'SIZE(?:[/\s]+DIMENSION(?:[/\s]+LENGH?T)?)?\s*\(?(?:MM|CM|IN|FT)?\)?\s*[.:;=]\s*(.+?)(?=\s*(?:PS\s*:|MT\s*:|SHAPE\s*:|GRADE\s*:|TEMPER\s*:|COATING\s*:|FINISH\s*:|HEAT\s*(?:NO?\s*)?:|CAST\s*NO\b|ITEM\s*NO\b|CERT\s*NO?\b|$))',
        text, re.IGNORECASE
    )
    if not m:
        return None

    block = _clean_size_block(m.group(1).strip().rstrip('.,;'))

    # D NMM X L NM 형식 (GUIDE SHAFT 등): L의 단위 M을 MM으로 간주
    dl_m = re.match(r'^D\s*([\d.]+)\s*MM?\s*[Xx]\s*L\s*([\d.]+)\s*M\s*$', block.strip(), re.IGNORECASE)
    if dl_m:
        return f'{dl_m.group(1)}X{dl_m.group(2)}'

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
        # OD/WT + LENGTH 범위 미추출 보완: SIZE 블록에 LENGTH: 범위가 있으면 추가
        if not re.search(r'~', result) and not re.search(r'XC$', result, re.IGNORECASE):
            len_range_m = re.search(r'\bLENGTH\s*:\s*(\d+)\s*~\s*(\d+)', block, re.IGNORECASE)
            if len_range_m:
                result = f'{result}X{len_range_m.group(1)}~{len_range_m.group(2)}'
        return result

    # 블록 끝 단독 C (코일 표시자) 분리: "X C", "*C", 공백+C
    has_coil = bool(re.search(r'(?:\s*[Xx*]\s*|\s+)C\s*$', block, re.IGNORECASE))
    if has_coil:
        block = re.sub(r'(?:\s*[Xx*]\s*|\s+)C\s*$', '', block, flags=re.IGNORECASE).strip()

    # 이미 숫자X숫자 형태 → 순서 유지
    if re.match(r'^[\d.~/]+(?:[Xx][\d.~/]+)+[Xx]?[Cc]?$', block.strip()):
        result = block.strip().upper().replace('x', 'X')
        return (result + 'XC') if has_coil else result

    # 그 외 → 강종코드 제거 후 토큰 추출 (순서 유지)
    cleaned = _strip_grade_codes(block)
    result = _parse_tokens(cleaned, preserve_order=True)
    if result and has_coil:
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


def extract_size_regex(spec_text: str) -> Optional[str]:
    """정규식 기반 사이즈 추출"""
    text = _normalize(spec_text)

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

    # T(min/max") X W N" X L N" 판재 두께범위 표기 (PLATE-BAFFLE RING 등)
    t_range_wl_m = re.search(
        r'\bT\s*\(\s*([\d.]+)\s*/\s*([\d.]+)\s*"\s*\)\s*[Xx]\s*W\s*([\d.]+)\s*"\s*[Xx]\s*L\s*([\d.]+)\s*"',
        text, re.IGNORECASE
    )
    if t_range_wl_m:
        t_max = t_range_wl_m.group(2)  # 최대값
        w, l = t_range_wl_m.group(3), t_range_wl_m.group(4)
        return f'{t_max}INX{w}INX{l}IN'

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

    # COLLAR/ADJUST RING + 하이픈 숫자: 마지막 두 값이 치수 (NCL 이외 코드 포함)
    if re.search(r'\b(?:COLLAR|ADJUST\s*RING\w*)\b', text, re.IGNORECASE):
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
    if re.search(r'\bSTAINLESS\s+STEEL\s+WIRE\b', text, re.IGNORECASE):
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

    # SCH 파이프 패턴: {dia}" {S/SCH}{n}S? [WLD/SMLS] PIPE {len}MM
    sch_pipe_m = re.search(
        r'([\d]+(?:-\d+/\d+)?|\d+/\d+|[\d.]+)\s*(?:"|IN(?!CH))\s*(?:X\s*)?'
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

    # WIRE ROPE {dia}MM → 직경만 반환 (길이/수량 제외)
    if re.search(r'\bWIRE\s+ROPE\b', text, re.IGNORECASE):
        # D{n}X 직경 패턴 우선: Wire rope D009.0x150000mm → 9
        d_m = re.search(r'\bD0*(\d+\.?\d*)\s*[Xx]', text)
        if d_m:
            val = d_m.group(1)
            try:
                fval = float(val)
                return str(int(fval)) if fval == int(fval) else val
            except ValueError:
                return val
        wr_m = re.search(r'([\d.]+)\s*MM\b', text, re.IGNORECASE)
        if wr_m:
            val = wr_m.group(1)
            try:
                fval = float(val)
                return str(int(fval)) if fval == int(fval) else f'{fval:g}'
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

    # WIRE DIA {num}MM → 단선 직경 (SWRCH 등 와이어 규격) — 숫자 직후 WIRE도 허용
    wire_dia_m = re.search(
        r'(?<![A-Z])WIRE\s+DIA\s*([\d.]+)\s*(MM|CM|(?<!\w)M(?!\w)|IN(?:CH)?|FT|")?',
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

    # 와이어 로프 구성코드 ({n}X{n}+) 있을 때 직경: 6X24+7FC-16MM → 16, GAL.06x24+FC 30MM → 30
    # 또는 6X19+PP, DIA.: 3.0 형식 (MM 없이 DIA. 표기)
    if re.search(r'\b\d+[Xx]\d+\+', text):
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

    # WELDING WIRE {grade},{dia}mm 패턴: ER308,1.6mm / ER70S-6,1.6mm → 1.6
    if re.search(r'\bWELDING\s*WIRE\b', text, re.IGNORECASE):
        ww_m = re.search(r'(?:ER|E)[\w-]*\s*,\s*([\d.]+)\s*(MM|CM|(?<!\w)M(?!\w)|IN(?:CH)?|FT|")', text, re.IGNORECASE)
        if ww_m:
            return _convert_unit(ww_m.group(1), ww_m.group(2) or '')
        # (ERxx)N.NMM 또는 (ERxx),N.NMM 형태: (ER70S-6)1.0mm → 1 (괄호 내 ER 분류코드 다음 직경)
        ww_m2 = re.search(r'\(E[A-Z0-9\-]+\)[\s,]*([\d.]+)\s*(MM|CM|(?<!\w)M(?!\w)|IN(?:CH)?|FT|")', text, re.IGNORECASE)
        if ww_m2:
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

    # 1. SIZE: 키워드 블록
    result = _extract_size_block(text)
    if result:
        # SHAPE:ROUNDBAR + SIZE:{dia}X{n}MM 에서 n이 2~12이면 미터 단위로 간주: 6→6000
        # (예: SIZE:140*6MM → ROUNDBAR이므로 6M = 6000mm)
        if re.search(r'SHAPE\s*:\s*ROUND\s*BAR', text, re.IGNORECASE):
            rb_m = re.match(r'^([\d.]+)X(\d{1,2})$', result)
            if rb_m:
                n = int(rb_m.group(2))
                if 2 <= n <= 12:
                    result = f'{rb_m.group(1)}X{n * 1000}'
        # PIPE/TUBE SHAPE인데 XC 포함이면 제거 (SEAMLESS COIL TUBE에서 *C는 코일 표시 아님)
        if (result.upper().endswith('XC')
                and re.search(r'SHAPE\s*:\s*\w*\s*(?:PIPE|TUBE)', text, re.IGNORECASE)
                and not re.search(r'SHAPE\s*:\s*(?:IN\s+)?COIL\b', text, re.IGNORECASE)):
            result = result[:-2]
        return _append_coil_if_shape(result, text) or result

    # 2-0a. SA179/SA334 형식: OD{n}MM X WT{m}MM[...] X{l}MM (열교환기관 OD/WT/Length)
    # _extract_twl보다 먼저 체크: OD/WT로 시작하면 길이까지 포함한 3차원 매칭 우선
    od_wt_l_m = re.search(
        r'(?<![A-Z])OD\s*([\d.]+)\s*MM\s*[Xx]\s*WT\s*([\d.]+)\s*MM[^Xx]*[Xx]\s*([\d.]+)\s*MM',
        text, re.IGNORECASE
    )
    if od_wt_l_m:
        return f'{od_wt_l_m.group(1)}X{od_wt_l_m.group(2)}X{od_wt_l_m.group(3)}'

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

    # 2-0. NMM X NMM X NMM 형식 3차원 치수 (순서 보존): 51MM X15MM X 6000MM → 51X15X6000
    pipe3d_m = re.search(
        r'(?<!\d)([\d.]+(?:[~-][\d.]+)?)\s*MM\s*[Xx]?\s*([\d.]+(?:[~-][\d.]+)?)\s*MM\s*[Xx]?\s*([\d.]+(?:[~-][\d.]+)?)\s*MM(?![0-9])',
        text, re.IGNORECASE
    )
    if pipe3d_m:
        a = _convert_unit(pipe3d_m.group(1), 'MM')
        b = _convert_unit(pipe3d_m.group(2), 'MM')
        c = _convert_unit(pipe3d_m.group(3), 'MM')
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
        return bare.upper().replace('x', 'X')

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
    clean = re.sub(
        r'\b(?:MODEL|MT|PS|SHAPE|GRADE|FOR|ACC|TO|SPEC|DLY\s*CODE)\b.*',
        '', text, flags=re.IGNORECASE
    )
    # 앞부분 부품번호-강종코드 패턴 제거: 79073-SKD1122 → SKD1122
    clean = re.sub(r'^\d{4,8}-(?=[A-Z])', '', clean.strip())
    clean = _clean_size_block(clean)
    clean = _strip_grade_codes(clean)
    # 그레이드 코드 제거 후 선행 점 재보정: OCR25AL50.29MM → .29MM → 0.29MM
    clean = re.sub(r'(?<![A-Z\d])\.(\d)', r'0.\1', clean)
    # clean이 숫자로 시작하면 원래 순서 유지 (예: 139.8MMX15.90MMX6.0M)
    # CERT 포함 시 치수 순서 유지 (254 X 27 MM → 254X27)
    preserve_start = bool(re.match(r'^\s*[\d.]', clean)) or has_cert
    result = _parse_tokens(clean, preserve_order=has_star_sep or preserve_start)
    if result and ('X' in result or '~' in result):
        return _append_coil_if_shape(result, text) or result
    # 단일 치수: clean 텍스트가 거의 치수 정보만 남은 경우 반환
    if result and re.match(r'^[\d.]+(?:IN|FT)?$', result):
        clean_alpha = re.sub(r'\d|[.\s/\-]', '', clean)
        ends_with_num = bool(re.search(r'[\d.]+\s*(?:MM|IN|FT|CM|MTR|METERS?|")?$', clean.strip(), re.IGNORECASE))
        if len(clean_alpha) <= 4 or ends_with_num:  # 단위 글자만 남거나 끝에 치수값이 있는 경우
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
    # 5-b. 일반 DIA/D= 패턴
    m = re.search(r'(?:DIA|D\s*=)[\s.:-]*\.?\s*([\d.]+(?:(?:MM|CM|M\b)?\s*[Xx]\s*[\d.]+(?:MM|CM|M)?)*)', text, re.IGNORECASE)
    if m and re.search(r'\d', m.group(1)):  # 실제 숫자가 있어야 함 (단순 '.'만 캡처 방지)
        result = _parse_tokens(m.group(1), preserve_order=True) or m.group(1).replace('X', 'X').strip()
        if result and re.search(r'SHAPE\s*:\s*(?:IN\s+)?COIL', text, re.IGNORECASE):
            result = result + 'XC'
        return result

    return None


def _append_coil_if_shape(result: Optional[str], text: str) -> Optional[str]:
    """SHAPE:COIL, IN COILS, *C 등 코일 표시 시 XC 미포함이면 추가"""
    if result and 'X' in result and 'XC' not in result.upper():
        # COIL TUBE/SPRING/PIPE/WIRE 등 복합 제품명은 제외
        if re.search(r'SHAPE\s*:\s*(?:IN\s+)?COIL|SHAPE\s*:\s*XC|IN\s+COILS?\d*(?:\s|,|$)|(?<![A-Z])\d*COILS?\b(?!\s+(?:TUBE|SPRING|PIPE|WIRE|BAR|ROD))', text, re.IGNORECASE):
            # SHEET IN COILS (3차원 치수): 코일에서 자른 낱장 시트 → XC 불필요
            # (PLATE IN COILS는 코일 형태 제품이므로 XC 유지)
            if re.search(r'\bSHEET\s+IN\s+COILS?\b', text, re.IGNORECASE) and result.count('X') >= 2:
                return result
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
