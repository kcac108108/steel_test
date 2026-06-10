"""
RAG 유사도 검색 서비스

10년치 분류 이력 데이터를 ChromaDB에 벡터 인덱싱하고,
입력 규격품명과 코사인 유사도 검색을 수행합니다.

ChromaDB 컬렉션 스키마:
  document  : 규격품명 텍스트
  metadata  : {"steel_grade": ..., "size": ..., "method": ...}
  embedding : text-embedding-3-small 벡터
"""

import hashlib
import re
from difflib import SequenceMatcher
from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI
from app.core.config import settings
from app.models.schemas import ClassifyResult, ClassifyMethod, HistoryRecord


class RAGService:
    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=settings.chroma_db_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._openai = OpenAI(api_key=settings.openai_api_key)

    @staticmethod
    def _has_matching_codes(steel_grade: str, spec_text: str) -> bool:
        """강종의 숫자 코드(UNS/AMS 등)가 원문 규격에 모두 존재하는지 확인"""
        code_pattern = re.compile(r'[A-Z]{1,3}\d{4,6}')
        codes_in_grade = code_pattern.findall(steel_grade.upper())
        if not codes_in_grade:
            return False  # 숫자 코드 없으면 판단 불가 → top-1 사용
        spec_upper = spec_text.upper()
        return all(code in spec_upper for code in codes_in_grade)

    @staticmethod
    def _correct_grade_codes(steel_grade: str, spec_text: str) -> str:
        """
        RAG 분류 강종의 숫자 코드가 원문과 다를 때 원문 기준으로 교정.
        예) ASME SA240 UNS S32760 + 원문 S32750 → ASME SA240 UNS S32750
        대상 패턴: UNS 번호(S32750), AMS 번호(AMS5659) 등 알파벳+숫자 조합 코드
        """
        # 알파벳 1~3자 + 숫자 4~6자리 패턴 (S32750, AMS5659, N06625 등)
        code_pattern = re.compile(r'[A-Z]{1,3}\d{4,6}')

        codes_in_grade = code_pattern.findall(steel_grade.upper())
        if not codes_in_grade:
            return steel_grade

        codes_in_spec = code_pattern.findall(spec_text.upper())
        if not codes_in_spec:
            return steel_grade

        corrected = steel_grade.upper()
        for grade_code in codes_in_grade:
            if grade_code in codes_in_spec:
                continue  # 정확 일치 → 교정 불필요

            # spec에서 첫 알파벳이 같고 유사도 높은 코드 탐색
            best_match, best_ratio = None, 0.0
            for spec_code in codes_in_spec:
                if grade_code[0] != spec_code[0]:
                    continue
                ratio = SequenceMatcher(None, grade_code, spec_code).ratio()
                if ratio > best_ratio:
                    best_ratio, best_match = ratio, spec_code

            # 80% 이상 유사 (6자리 기준 1자 차이)하면 원문 코드로 교체
            if best_match and best_ratio >= 0.8:
                corrected = corrected.replace(grade_code, best_match)

        # 원래 대소문자 보존 (steel_grade 원본 기준)
        return corrected if corrected != steel_grade.upper() else steel_grade

    def _embed(self, text: str) -> list[float]:
        resp = self._openai.embeddings.create(
            model=settings.embedding_model,
            input=text,
        )
        return resp.data[0].embedding

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """여러 텍스트를 한 번의 API 호출로 임베딩"""
        resp = self._openai.embeddings.create(
            model=settings.embedding_model,
            input=texts,
        )
        return [item.embedding for item in resp.data]

    def index_history(self, records: list[HistoryRecord], insert_only: bool = False, batch_size: int = 500) -> None:
        """분류 이력 데이터를 ChromaDB에 인덱싱

        insert_only=True: 이미 존재하는 규격은 덮어쓰지 않음 (신규만 추가)
        insert_only=False: 기존 항목도 덮어씀 (초기 구축 시 사용)
        """
        total = len(records)
        skipped = 0
        for i in range(0, total, batch_size):
            batch = records[i : i + batch_size]

            # 배치 내 중복 ID 제거 (같은 spec_text가 여러 번 나올 경우 마지막 것 사용)
            seen: dict[str, int] = {}
            for j, r in enumerate(batch):
                seen[hashlib.md5(r.spec_text.encode()).hexdigest()] = j
            deduped = [batch[j] for j in seen.values()]

            texts = [r.spec_text for r in deduped]
            ids = list(seen.keys())

            # insert_only: 이미 존재하는 ID 제외
            if insert_only:
                existing = set(self._collection.get(ids=ids)["ids"])
                new_indices = [k for k, id_ in enumerate(ids) if id_ not in existing]
                if not new_indices:
                    skipped += len(deduped)
                    print(f"  [{i + len(batch):,}/{total:,}] 진행 중... (신규 0건, 전체 스킵)")
                    continue
                texts = [texts[k] for k in new_indices]
                ids = [ids[k] for k in new_indices]
                deduped = [deduped[k] for k in new_indices]
                skipped += len(seen) - len(new_indices)

            embeddings = self._embed_batch(texts)
            metadatas = [
                {"steel_grade": r.steel_grade, "size": r.size, "method": r.method}
                for r in deduped
            ]
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            print(f"  [{i + len(batch):,}/{total:,}] 진행 중..." + (f" (기존 {skipped}건 스킵)" if insert_only and skipped else ""))
        print(f"[RAG] {total:,}건 인덱싱 완료" + (f" (기존 존재로 스킵: {skipped}건)" if insert_only else ""))

    def search(self, spec_text: str) -> Optional[ClassifyResult]:
        """단건 유사도 검색"""
        results = self.search_batch([spec_text])
        return results[0]

    def search_batch_with_hints(
        self,
        spec_texts: list[str],
        n_hints: int = 3,
        min_hint_sim: float = 0.70,
        batch_size: int = 500,
    ) -> list[tuple[Optional["ClassifyResult"], list[dict]]]:
        """배치 검색 + LLM 힌트 동시 반환 (임베딩 1회 호출로 처리)
        Returns: list of (ClassifyResult | None, [{grade, similarity}, ...])
        """
        if self._collection.count() == 0:
            return [(None, [])] * len(spec_texts)

        all_results: list[tuple[Optional[ClassifyResult], list[dict]]] = []

        for i in range(0, len(spec_texts), batch_size):
            batch_texts = spec_texts[i : i + batch_size]

            valid_indices = [j for j, t in enumerate(batch_texts) if t.strip()]
            valid_texts = [batch_texts[j] for j in valid_indices]

            if not valid_texts:
                all_results.extend([(None, [])] * len(batch_texts))
                continue

            embeddings = self._embed_batch(valid_texts)

            query_results = self._collection.query(
                query_embeddings=embeddings,
                n_results=max(5, n_hints),
                include=["documents", "metadatas", "distances"],
            )

            batch_results: list[tuple[Optional[ClassifyResult], list[dict]]] = [(None, [])] * len(batch_texts)
            for k, j in enumerate(valid_indices):
                spec_text = batch_texts[j]
                distances = query_results["distances"][k]
                metadatas = query_results["metadatas"][k]
                documents = query_results["documents"][k]

                if not distances:
                    batch_results[j] = (None, [])
                    continue

                # 힌트 수집 (임계값 무관, min_hint_sim 이상)
                hints: list[dict] = []
                seen_grades: set[str] = set()
                for dist, meta in zip(distances, metadatas):
                    sim = 1.0 - dist
                    if sim < min_hint_sim:
                        break
                    grade = meta.get("steel_grade", "")
                    if grade and str(grade).strip() not in ("", "0", "0.0", "nan"):
                        if grade not in seen_grades:
                            seen_grades.add(grade)
                            hints.append({"grade": grade, "similarity": round(sim, 3)})
                    if len(hints) >= n_hints:
                        break

                # 분류 결과 (기존 search_batch 로직과 동일)
                classify_result: Optional[ClassifyResult] = None
                if 1.0 - distances[0] >= settings.similarity_threshold:
                    selected_meta, selected_doc, selected_sim = None, None, None
                    for dist, meta, doc in zip(distances, metadatas, documents):
                        sim = 1.0 - dist
                        if sim < settings.similarity_threshold:
                            break
                        grade = meta.get("steel_grade", "")
                        if not grade or str(grade).strip() in ("", "0", "0.0", "nan"):
                            continue
                        if selected_meta is None:
                            selected_meta, selected_doc, selected_sim = meta, doc, sim
                        if self._has_matching_codes(grade, spec_text):
                            selected_meta, selected_doc, selected_sim = meta, doc, sim
                            break

                    if selected_meta is not None:
                        steel_grade = selected_meta.get("steel_grade", "")
                        steel_grade = self._correct_grade_codes(steel_grade, spec_text)
                        classify_result = ClassifyResult(
                            spec_text=spec_text,
                            steel_grade=steel_grade,
                            size=selected_meta["size"],
                            method=ClassifyMethod.RAG,
                            confidence=round(selected_sim, 4),
                            similar_spec=selected_doc,
                        )

                batch_results[j] = (classify_result, hints)

            all_results.extend(batch_results)

        return all_results

    def search_batch(self, spec_texts: list[str], batch_size: int = 500) -> list[Optional[ClassifyResult]]:
        """
        배치 유사도 검색. 500건씩 묶어서 API 호출.
        ChromaDB cosine distance → similarity = 1 - distance
        """
        if self._collection.count() == 0:
            return [None] * len(spec_texts)

        all_results: list[Optional[ClassifyResult]] = []

        for i in range(0, len(spec_texts), batch_size):
            batch_texts = spec_texts[i : i + batch_size]

            # 빈 문자열 제외하고 인덱스 추적
            valid_indices = [j for j, t in enumerate(batch_texts) if t.strip()]
            valid_texts = [batch_texts[j] for j in valid_indices]

            # 빈 텍스트만 있는 배치는 스킵
            if not valid_texts:
                all_results.extend([None] * len(batch_texts))
                continue

            embeddings = self._embed_batch(valid_texts)

            query_results = self._collection.query(
                query_embeddings=embeddings,
                n_results=5,
                include=["documents", "metadatas", "distances"],
            )

            # 결과를 원래 인덱스에 맞게 배치
            batch_results: list[Optional[ClassifyResult]] = [None] * len(batch_texts)
            for k, j in enumerate(valid_indices):
                spec_text = batch_texts[j]
                distances = query_results["distances"][k]
                if not distances:
                    continue

                # top-1이 임계값 미달이면 스킵
                if 1.0 - distances[0] < settings.similarity_threshold:
                    continue

                # top-5 중 임계값 통과하는 후보에서 숫자/문자 코드 매칭 선택
                selected_meta, selected_doc, selected_sim = None, None, None
                for dist, meta, doc in zip(
                    distances,
                    query_results["metadatas"][k],
                    query_results["documents"][k],
                ):
                    sim = 1.0 - dist
                    if sim < settings.similarity_threshold:
                        break  # 유사도 순 정렬이므로 이후는 모두 미달

                    grade = meta.get("steel_grade", "")
                    if not grade or str(grade).strip() in ("", "0", "0.0", "nan"):
                        continue

                    if selected_meta is None:
                        selected_meta, selected_doc, selected_sim = meta, doc, sim

                    # 원문에 강종 코드가 있으면 우선 선택
                    if self._has_matching_codes(grade, spec_text):
                        selected_meta, selected_doc, selected_sim = meta, doc, sim
                        break

                if selected_meta is None:
                    continue

                steel_grade = selected_meta.get("steel_grade", "")
                steel_grade = self._correct_grade_codes(steel_grade, spec_text)

                batch_results[j] = ClassifyResult(
                    spec_text=spec_text,
                    steel_grade=steel_grade,
                    size=selected_meta["size"],
                    method=ClassifyMethod.RAG,
                    confidence=round(selected_sim, 4),
                    similar_spec=selected_doc,
                )

            all_results.extend(batch_results)

        return all_results
