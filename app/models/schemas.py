from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ClassifyMethod(str, Enum):
    RULE = "rule"       # 룰베이스 매칭
    RAG = "rag"         # RAG 유사도
    LLM = "llm"         # LLM 판단
    UNCLASSIFIED = "unclassified"  # 미분류


@dataclass
class ClassifyResult:
    """분류 결과"""
    spec_text: str              # 원본 규격품명
    steel_grade: Optional[str] = None   # 강종
    size: Optional[str] = None          # 사이즈
    method: ClassifyMethod = ClassifyMethod.UNCLASSIFIED
    confidence: float = 0.0
    matched_rule_id: Optional[str] = None  # 매칭된 룰 ID (룰베이스일 때)
    similar_spec: Optional[str] = None     # 유사 규격 (RAG일 때)


@dataclass
class RuleRecord:
    """룰베이스 DB 레코드"""
    rule_id: str
    pattern: str
    steel_grade: str
    size: str


@dataclass
class HistoryRecord:
    """분류 이력 레코드"""
    spec_text: str
    steel_grade: str
    size: str
    method: str = ""
