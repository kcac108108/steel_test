"""
MODEL 필드 추출기

MODEL:XXX, PS:..., MT:..., SHAPE:..., SIZE:YYY 형식의 규격에서
강종과 사이즈를 직접 추출합니다.
"""

import re
from typing import Optional
from app.models.schemas import ClassifyResult, ClassifyMethod


# JIS 단독 코드 → JIS 규격번호 매핑
_JIS_MAP = {
    "SKD11": "JIS G4404 SKD11",
    "SKD61": "JIS G4404 SKD61",
    "SKH51": "JIS G4403 SKH51",
    "SCM440": "JIS G4105 SCM440",
    "SCM415": "JIS G4105 SCM415",
    "SCM420": "JIS G4105 SCM420",
    "SNCM439": "JIS G4103 SNCM439",
    "SS400": "JIS G3101 SS400",
    "SS490": "JIS G3101 SS490",
    "SGCC": "JIS G3302 SGCC",
    "SGCH": "JIS G3302 SGCH",
    "SPCC": "JIS G3141 SPCC",
    "SPCD": "JIS G3141 SPCD",
    "STK490": "JIS G3444 STK 490",
    "STK400": "JIS G3444 STK 400",
    "STS480": "JIS G3455 STS480",
    "STS370": "JIS G3455 STS370",
}

# MODEL 값 뒤에 오는 메타 키워드 (이 앞까지만 MODEL 값으로 사용)
_META_KEYWORDS = re.compile(
    r'\s+(?:MT|PS|SHAPE|SIZE|CERT|HEAT|PACKING)\s*:', re.IGNORECASE
)


def _clean_model_value(raw: str) -> str:
    """MODEL 값에서 뒤쪽 메타 정보 제거"""
    m = _META_KEYWORDS.search(raw)
    if m:
        raw = raw[:m.start()]
    return raw.strip().rstrip('-,/')


def _extract_size(spec_text: str) -> str:
    """SIZE: 필드 추출"""
    m = re.search(r'SIZE\s*:\s*([^,\n]+)', spec_text, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _parse_grade(model_val: str) -> Optional[str]:
    """
    MODEL 값을 강종으로 변환. 변환 불가시 None 반환.
    """
    v = model_val.strip().upper()

    # 1. SA/A240-{UNS} or SA240-{UNS} → ASTM A240 UNS {UNS}
    m = re.match(r'SA/?A240M?\s*[-/]\s*([A-Z]\d{4,6})', v)
    if m:
        return f"ASTM A240 UNS {m.group(1)}"

    # 2. SA{num}-{grade} → ASME SA{num} {grade}  예) SA302-B → ASME SA302 B
    m = re.match(r'(SA\d+[A-Z]?)\s*[-]\s*([A-Z0-9]+)', v)
    if m:
        return f"ASME {m.group(1)} {m.group(2)}"

    # 3. JIS G{num} {grade} (이미 완전한 형식)
    m = re.match(r'(JIS\s+G\d+\s+\S+)', v)
    if m:
        return m.group(1).strip()

    # 4. 단독 JIS 코드 (SKD11, SCM440 등)
    if v in _JIS_MAP:
        return _JIS_MAP[v]

    # 5. ASTM/ASME로 시작하는 명확한 강종
    m = re.match(r'((?:ASTM|ASME)\s+\S+(?:\s+\S+)?)', v)
    if m:
        return m.group(1).strip()

    # 6. 알파벳+숫자 단순 코드 (P355NL2, DX53D 등) — 너무 짧거나 숫자만이면 제외
    if re.match(r'^[A-Z]{1,4}\d{2,}[A-Z0-9]*$', v) and len(v) >= 4:
        return model_val.strip().upper()

    return None


def extract(spec_text: str) -> Optional[ClassifyResult]:
    """
    MODEL: 필드에서 강종/사이즈 추출. 추출 불가시 None 반환.
    """
    m = re.search(r'MODEL\s*:\s*(.+?)(?:,|$)', spec_text, re.IGNORECASE)
    if not m:
        return None

    raw_model = m.group(1)
    model_val = _clean_model_value(raw_model)

    if not model_val or len(model_val) < 2:
        return None

    grade = _parse_grade(model_val)
    if not grade:
        return None

    size = _extract_size(spec_text)

    return ClassifyResult(
        spec_text=spec_text,
        steel_grade=grade,
        size=size,
        method=ClassifyMethod.RULE,
        confidence=1.0,
    )
