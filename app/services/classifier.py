"""
메인 분류기

분류 파이프라인:
  1. Rule 매칭 (Oracle rule_base 모델코드 룩업)
  2. RAG 유사도 검색
  3. LLM 판단
  4. 미분류
"""

import pickle
import os
from app.models.schemas import ClassifyResult, ClassifyMethod
from app.services.rule_matcher import RuleMatcher
from app.services.rag_service import RAGService
from app.services.llm_service import LLMService


class SteelClassifier:
    def __init__(self, use_rule: bool = True, use_rag: bool = True, use_llm: bool = True):
        self._rule = RuleMatcher() if use_rule else None
        self._rag = RAGService() if use_rag else None
        self._llm = LLMService() if use_llm else None

    def classify(self, spec_text: str) -> ClassifyResult:
        """단건 분류"""
        text = spec_text.strip()
        if not text:
            return ClassifyResult(spec_text=spec_text, method=ClassifyMethod.UNCLASSIFIED)

        # 1단계: Rule
        if self._rule:
            result = self._rule.match(text)
            if result:
                return result

        # 2단계: RAG
        if self._rag:
            result = self._rag.search(text)
            if result:
                return result

        # 3단계: LLM
        if self._llm:
            result = self._llm.classify(text)
            if result:
                return result

        # 4단계: 미분류
        return ClassifyResult(spec_text=spec_text, method=ClassifyMethod.UNCLASSIFIED)

    def classify_batch(self, spec_texts: list[str], checkpoint_path: str = "") -> list[ClassifyResult]:
        """배치 분류 - Rule 먼저, RAG는 500건씩, LLM은 미분류 건만 처리"""
        total = len(spec_texts)
        results: list[ClassifyResult] = [
            ClassifyResult(spec_text=t, method=ClassifyMethod.UNCLASSIFIED)
            for t in spec_texts
        ]

        # 1단계: Rule 매칭
        if self._rule:
            print(f"  [RULE] {total:,}건 룰 매칭 시작...")
            for i, text in enumerate(spec_texts):
                if text.strip() and results[i].method == ClassifyMethod.UNCLASSIFIED:
                    rule_result = self._rule.match(text)
                    if rule_result:
                        results[i] = rule_result
            rule_classified = sum(1 for r in results if r.method == ClassifyMethod.RULE)
            print(f"  [RULE] {rule_classified:,}건 분류 완료")

        # 2단계: RAG 배치 검색 - Rule 미분류 건만
        hints_map: dict[int, list[dict]] = {}
        if self._rag:
            rag_indices = [
                i for i, r in enumerate(results)
                if r.method == ClassifyMethod.UNCLASSIFIED and spec_texts[i].strip()
            ]
            rag_texts = [spec_texts[i] for i in rag_indices]
            print(f"  [RAG] {len(rag_texts):,}건 배치 검색 시작...")
            rag_with_hints = self._rag.search_batch_with_hints(rag_texts)
            for idx, (rag_result, hints) in zip(rag_indices, rag_with_hints):
                if rag_result:
                    results[idx] = rag_result
                else:
                    hints_map[idx] = hints
            rag_classified = sum(1 for r in results if r.method == ClassifyMethod.RAG)
            print(f"  [RAG] {rag_classified:,}건 분류 완료")

        # 3단계: LLM - RAG 미분류 건만 병렬 처리
        if self._llm:
            unclassified_indices = [
                i for i, r in enumerate(results)
                if r.method == ClassifyMethod.UNCLASSIFIED and spec_texts[i].strip()
            ]

            # 체크포인트 복원
            checkpoint_saved: dict = {}
            if checkpoint_path and os.path.exists(checkpoint_path):
                with open(checkpoint_path, "rb") as f:
                    checkpoint_saved = pickle.load(f)
                for idx, result in checkpoint_saved.items():
                    results[idx] = result
                done_indices = set(checkpoint_saved.keys())
                unclassified_indices = [i for i in unclassified_indices if i not in done_indices]
                print(f"  [LLM] 체크포인트 복원: {len(checkpoint_saved):,}건 이어서 시작...")

            llm_total = len(checkpoint_saved) + len(unclassified_indices)
            print(f"  [LLM] 미분류 {len(unclassified_indices):,}건 처리 시작...")

            llm_done: dict = dict(checkpoint_saved)

            for count, i in enumerate(unclassified_indices, 1):
                llm_result = self._llm.classify(spec_texts[i], rag_hints=hints_map.get(i))
                if llm_result:
                    results[i] = llm_result
                llm_done[i] = results[i]

                if count % 100 == 0:
                    done_so_far = len(checkpoint_saved) + count
                    print(f"  [LLM] [{done_so_far:,}/{llm_total:,}] 처리 중...")
                    if checkpoint_path:
                        tmp_path = checkpoint_path + ".tmp"
                        with open(tmp_path, "wb") as f:
                            pickle.dump(llm_done, f)
                        os.replace(tmp_path, checkpoint_path)

            llm_classified = sum(1 for r in results if r.method == ClassifyMethod.LLM)
            print(f"  [LLM] {llm_classified:,}건 분류 완료")

        return results

    def save_results(self, results: list[ClassifyResult]) -> None:
        """분류 결과를 Oracle DB classify_history 테이블에 저장"""
        classified = [r for r in results if r.method != ClassifyMethod.UNCLASSIFIED]
        if not classified:
            return

        with db_cursor() as cursor:
            cursor.executemany(
                """INSERT INTO classify_history
                   (spec_text, steel_grade, size, method, confidence, created_at)
                   VALUES (:1, :2, :3, :4, :5, SYSDATE)""",
                [
                    (r.spec_text, r.steel_grade, r.size, r.method.value, r.confidence)
                    for r in classified
                ],
            )
        print(f"[DB] {len(classified)}건 이력 저장 완료")
