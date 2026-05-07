"""
LLM 판단 서비스

룰베이스·RAG 모두 실패한 경우에만 GPT에게 강종/사이즈 분류를 요청합니다.
LLM이 "판단 불가" 응답을 반환하면 미분류로 처리합니다.
"""

import json
import time
from typing import Optional
from openai import OpenAI, RateLimitError
from app.core.config import settings
from app.models.schemas import ClassifyResult, ClassifyMethod


SYSTEM_PROMPT = """당신은 철강 수입신고 전문가입니다.
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

- UNS 번호 명시 → UNS 번호를 강종으로 반환
  예) "UNS S31803" 명시 → steel_grade: "UNS S31803"

[선급강 구분 기준 - 매우 중요]
아래 선급강은 텍스트만으로 구분이 어려우므로 반드시 다음 기준을 따르세요:
- 텍스트에 "LR A", "LLOYD", "LR GRADE A" 명시 → LR A
- 텍스트에 "AH32", "GRADE AH32" 명시 → AH32
- 텍스트에 "AH36", "GRADE AH36" 명시 → AH36
- 텍스트에 "EH36", "GRADE EH36" 명시 → EH36
- 텍스트에 "DH32", "GRADE DH32" 명시 → DH32
- 위 키워드가 없고 선급강임이 의심되면 → 판단 불가 반환 (추측 금지)

응답 형식 (분류 가능할 때):
{"steel_grade": "AMS5659", "size": "0.3125\" DIA X 144\"", "confidence": 0.9}

응답 형식 (판단 불가):
{"steel_grade": null, "size": null, "confidence": 0.0}

강종 예시: JIS G3101 SS400, JIS G3302 SGCC, JIS G4404 SKD11, AMS5659, AMS6415, ASTM A240 304/304L, ASME SA789 UNS S31803, SUS304, SCM440, LR A, AH32, EH36 등
사이즈 예시: "2.0T X 1524W X C", "19.05 OD X 1.65T", "100 X 100 X 6.0" 등"""


class LLMService:
    def __init__(self):
        self._client = OpenAI(api_key=settings.openai_api_key)

    def classify(self, spec_text: str) -> Optional[ClassifyResult]:
        """
        GPT로 강종/사이즈 분류 시도.
        신뢰도가 llm_confidence_threshold 미만이면 None 반환 (미분류).
        """
        for attempt in range(3):
            try:
                response = self._client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"규격품명: {spec_text}"},
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
                    or not size
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
