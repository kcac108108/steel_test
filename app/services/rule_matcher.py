"""
룰베이스 매칭 서비스

Oracle DB의 rule_base 테이블에서 1만여개 규칙을 로드하여
입력 규격품명과 정확/패턴 매칭을 수행합니다.

테이블 구조 (예시):
  CREATE TABLE rule_base (
    rule_id   VARCHAR2(20) PRIMARY KEY,
    pattern   VARCHAR2(500),   -- 매칭 패턴 (정확매칭 or LIKE 패턴)
    steel_grade VARCHAR2(100),
    size      VARCHAR2(200)
  );
"""

import re
from typing import Optional
from app.core.database import db_cursor
from app.models.schemas import ClassifyResult, ClassifyMethod, RuleRecord


class RuleMatcher:
    def __init__(self):
        self._rules: list[RuleRecord] = []
        self._loaded = False

    def load_rules(self) -> None:
        """Oracle DB에서 룰 전체 로드 (메모리 캐싱)"""
        with db_cursor() as cursor:
            cursor.execute(
                "SELECT rule_id, pattern, steel_grade, size_val FROM rule_base ORDER BY rule_id"
            )
            rows = cursor.fetchall()

        self._rules = [
            RuleRecord(
                rule_id=str(row[0]),
                pattern=str(row[1]),
                steel_grade=str(row[2]),
                size=str(row[3]) if row[3] else "",
            )
            for row in rows
        ]
        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load_rules()

    def match(self, spec_text: str) -> Optional[ClassifyResult]:
        """
        규격품명을 룰베이스와 매칭.

        매칭 우선순위:
        1. 정확 매칭 (대소문자 무시)
        2. 패턴 앞부분 포함 매칭 (spec_text.upper() LIKE pattern%)
        """
        self._ensure_loaded()
        text_upper = spec_text.strip().upper()

        # 1단계: 정확 매칭
        for rule in self._rules:
            if rule.pattern.upper() == text_upper:
                return ClassifyResult(
                    spec_text=spec_text,
                    steel_grade=rule.steel_grade,
                    size=None,
                    method=ClassifyMethod.RULE,
                    confidence=1.0,
                    matched_rule_id=rule.rule_id,
                )

        # 2단계: spec_text가 패턴으로 시작하는지 (패턴 prefix 매칭)
        best: Optional[tuple[int, RuleRecord]] = None  # (pattern_len, rule)
        for rule in self._rules:
            pat = rule.pattern.strip().upper()
            if not pat:
                continue
            if text_upper.startswith(pat):
                length = len(pat)
                if best is None or length > best[0]:
                    best = (length, rule)

        if best:
            _, rule = best
            return ClassifyResult(
                spec_text=spec_text,
                steel_grade=rule.steel_grade,
                size=None,
                method=ClassifyMethod.RULE,
                confidence=0.95,
                matched_rule_id=rule.rule_id,
            )

        return None
