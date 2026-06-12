"""
LLM 판단 서비스

룰베이스·RAG 모두 실패한 경우에만 GPT에게 강종/사이즈 분류를 요청합니다.
LLM이 "판단 불가" 응답을 반환하면 미분류로 처리합니다.
"""

import json
import os
import pickle
import time
from pathlib import Path
from typing import Optional
from openai import OpenAI, RateLimitError
from app.core.config import settings
from app.models.schemas import ClassifyResult, ClassifyMethod


_GRADE_LIST_CACHE = "grade_list.pkl"


def _load_grade_list(top_n: int = 300) -> list[str]:
    """확정 데이터에서 빈도 상위 강종 목록 로드 (pickle 캐시 사용)"""
    if os.path.exists(_GRADE_LIST_CACHE):
        with open(_GRADE_LIST_CACHE, "rb") as f:
            return pickle.load(f)

    import pandas as pd
    from collections import Counter
    counter: Counter = Counter()
    confirmed_dir = Path("confirmed")
    if confirmed_dir.exists():
        for f in sorted(confirmed_dir.glob("**/*.xlsx")):
            try:
                df = pd.read_excel(f, dtype=str)
                if "강종" in df.columns:
                    valid = df["강종"].dropna().str.strip()
                    valid = valid[(valid != "") & (valid != "0")]
                    counter.update(valid.tolist())
            except Exception:
                pass

    # 빈도 상위 top_n개만 추출
    grade_list = sorted(g for g, _ in counter.most_common(top_n))
    if grade_list:
        with open(_GRADE_LIST_CACHE, "wb") as f:
            pickle.dump(grade_list, f)
        print(f"  [LLM] 강종 목록 구축 완료: {len(grade_list):,}개")
    return grade_list


def _build_system_prompt(grade_list: list[str]) -> str:
    base = """당신은 철강 수입신고 전문가입니다.
입력된 규격품명(영문 자유형식 텍스트)에서 강종(steel grade)과 사이즈(size)를 추출해야 합니다.

규칙:
1. 강종과 사이즈를 명확히 식별할 수 있을 때만 응답하세요.
2. 불확실하거나 철강제품이 아닌 경우 판단 불가를 반환하세요.
3. 응답은 반드시 JSON 형식으로만 하세요.

[규격코드 우선 원칙 - 매우 중요]
텍스트에 아래 규격코드가 명시된 경우, 재질명으로 변환하지 말고 규격코드를 그대로 강종으로 반환하세요.

- AMS 코드 명시 → AMS 코드를 강종으로 반환 (공백 없이, 뒤의 알파벳 접미사 M/D/L 등 제거)
  예) "AMS5659" 또는 "AMS5659T" 명시 → steel_grade: "AMS5659"
  예) "AMS5630M" 명시 → steel_grade: "AMS5630"
  예) "AMS5754D" 명시 → steel_grade: "AMS5754"
  예) "AMS6415" 명시 → steel_grade: "AMS6415" (ALLOY 4340으로 바꾸지 말 것)
  예) "AMS5599" 명시 → steel_grade: "AMS5599" (INCONEL 625로 바꾸지 말 것)

- ASTM 코드 명시 → "ASTM 코드 강종명" 형식으로 반환
  예) "ASTM A240, 304/304L" 명시 → steel_grade: "ASTM A240 304/304L"
  예) "ASTM A513" 명시 → steel_grade: "ASTM A513"
  주의) ASTM 코드 뒤에 "CARBON STEEL", "STAINLESS STEEL", "ALLOY STEEL" 등 재질 설명어는 붙이지 마세요.
  예) "ASTM A1011 CARBON STEEL SHEET..." → steel_grade: "ASTM A1011" (재질 설명 제거)

- ASME 코드 명시 → "ASME 코드" 형식으로 반환
  예) "SA789 UNS S31803" 명시 → steel_grade: "ASME SA789 UNS S31803"

- EN 코드 명시 → EN 코드를 강종으로 반환
  예) "EN10270-1" 명시 → steel_grade: "EN 10270-1"

- JIS 강종 → "JIS 규격번호 강종명" 형식으로 반환
  예) SGCC → steel_grade: "JIS G3302 SGCC"
  예) SKD11 → steel_grade: "JIS G4404 SKD11"
  예) SS400 → steel_grade: "JIS G3101 SS400"
  예) STK490 → steel_grade: "JIS G3444 STK 490"
  예) SPCC → steel_grade: "JIS G3141 SPCC"
  예) SGCH → steel_grade: "JIS G3302 SGCH"
  예) SWRCH10A → steel_grade: "JIS G3507 SWRCH10A"

- UNS 번호 명시 → UNS 번호를 강종으로 반환
  예) "UNS S31803" 명시 → steel_grade: "UNS S31803"

- AISI와 ASTM이 동시에 명시된 경우 → AISI를 우선 선택
  예) "AISI 1045, ASTM A108" 명시 → steel_grade: "AISI 1045"

- 트레이드명(상품명)이 명시된 경우 → UNS/ASTM 번호로 변환하지 말고 트레이드명 그대로 반환
  예) "AL6XN" 명시 → steel_grade: "AL6XN" (UNS N08367로 변환 금지)
  예) "INCONEL 625" 명시 → steel_grade: "INCONEL 625" (UNS N06625로 변환 금지)

- STS/KS 강종에 접미사(-T, -T2 등)가 명시된 경우 → 접미사 포함하여 반환
  예) "STS416-T" 명시 → steel_grade: "STS416-T" (STS416으로 축약 금지)

[선급강 구분 기준 - 매우 중요]
아래 선급강은 텍스트만으로 구분이 어려우므로 반드시 다음 기준을 따르세요:
- 텍스트에 "LR A", "LLOYD", "LR GRADE A" 명시 → LR A
- 텍스트에 "AH32", "GRADE AH32" 명시 → AH32
- 텍스트에 "AH36", "GRADE AH36" 명시 → AH36
- 텍스트에 "EH36", "GRADE EH36" 명시 → EH36
- 텍스트에 "DH32", "GRADE DH32" 명시 → DH32
- 위 키워드가 없고 선급강임이 의심되면 → 판단 불가 반환 (추측 금지)"""

    if grade_list:
        grade_str = ", ".join(grade_list)
        base += f"""

[유효 강종 목록 - 형식 준수 필수]
강종을 반환할 때는 반드시 아래 목록의 강종명과 동일한 형식(대소문자·공백·구분자 포함)을 사용하세요.
목록에 없는 강종이라도 규격코드 우선 원칙에 따라 정확히 식별된 경우에는 반환 가능합니다.
단, 확신이 없으면 null을 반환하세요.

{grade_str}"""

    base += """

응답 형식 (분류 가능할 때):
{"steel_grade": "AMS5659", "size": "0.3125\\" DIA X 144\\"", "confidence": 0.9}

응답 형식 (판단 불가):
{"steel_grade": null, "size": null, "confidence": 0.0}

강종 예시: JIS G3101 SS400, JIS G3302 SGCC, JIS G4404 SKD11, AMS5659, AMS6415, ASTM A240 304/304L, ASME SA789 UNS S31803, SUS304, SCM440, LR A, AH32, EH36 등
사이즈 예시: "2.0T X 1524W X C", "19.05 OD X 1.65T", "100 X 100 X 6.0" 등"""

    return base


class LLMService:
    def __init__(self):
        self._client = OpenAI(api_key=settings.openai_api_key)
        grade_list = _load_grade_list()
        self._system_prompt = _build_system_prompt(grade_list)

    def classify(self, spec_text: str, rag_hints: list[dict] | None = None) -> Optional[ClassifyResult]:
        """
        GPT로 강종/사이즈 분류 시도.
        rag_hints: [{"grade": ..., "similarity": ...}, ...] — RAG 유사 후보 힌트
        신뢰도가 llm_confidence_threshold 미만이면 None 반환 (미분류).
        """
        hint_text = ""
        if rag_hints:
            hint_text = "\n\n[RAG 유사 사례 힌트 - 참고용]\n"
            for h in rag_hints[:3]:
                hint_text += f"  유사도 {h['similarity']:.0%}: {h['grade']}\n"
            hint_text += "* 원문 규격 기준으로 최종 판단하세요. 힌트와 다를 수 있습니다."

        user_content = f"규격품명: {spec_text}{hint_text}"

        for attempt in range(3):
            try:
                response = self._client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    response_format={"type": "json_object"},
                )

                content = response.choices[0].message.content
                data = json.loads(content)

                confidence = float(data.get("confidence", 0.0))
                steel_grade = data.get("steel_grade")
                size = data.get("size")

                if (
                    not steel_grade
                    or confidence < settings.llm_confidence_threshold
                ):
                    return None

                return ClassifyResult(
                    spec_text=spec_text,
                    steel_grade=steel_grade,
                    size=size,
                    method=ClassifyMethod.LLM,
                    confidence=confidence,
                )

            except RateLimitError as e:
                wait = 60 * (attempt + 1)
                print(f"\n  [LLM] Rate limit 초과 - {wait}초 대기 후 재시도 ({attempt + 1}/3)...")
                time.sleep(wait)
            except (json.JSONDecodeError, KeyError, ValueError):
                return None

        return None
