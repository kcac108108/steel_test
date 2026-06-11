"""
강종 추출기

Rule/RAG/LLM이 분류한 강종을 규격 텍스트와 비교하여 검증·보정합니다.
규격 텍스트에 정답 강종이 명시된 경우 이를 우선 사용합니다.

동작 방식:
  1. 확정 데이터 전체에서 강종 어휘 사전 구축 (긴 것 우선)
  2. 규격 텍스트에서 알려진 강종 패턴 탐색
  3. 발견된 강종이 시스템 분류 결과와 다르면 교체
"""

import re
import pickle
import os
from pathlib import Path

import pandas as pd


_GRADE_VOCAB: list[str] | None = None
_GRADE_VOCAB_UPPER: list[tuple[str, str]] | None = None  # (upper, original) pairs
_VOCAB_CACHE_PATH = "grade_vocab.pkl"


def _build_grade_vocab(confirmed_dir: str = "confirmed") -> list[str]:
    """확정 데이터 전체에서 강종 어휘 목록 구축 (긴 것 우선)"""
    files = sorted(Path(confirmed_dir).glob("**/*.xlsx"))
    if not files:
        return []

    grades: set[str] = set()
    for f in files:
        try:
            df = pd.read_excel(f, dtype=str)
            if "강종" in df.columns:
                col = df["강종"]
            elif df.shape[1] >= 4:
                col = df.iloc[:, 3]
            else:
                continue
            for v in col.dropna():
                g = str(v).strip()
                if g and g not in ("nan", "0", "0.0"):
                    grades.add(g)
        except Exception:
            pass

    # 긴 것 우선 정렬 (더 구체적인 강종이 먼저 매칭되도록)
    return sorted(grades, key=len, reverse=True)


def _get_grade_vocab() -> list[tuple[str, str]]:
    """(uppercase, original) 튜플 리스트 반환 (긴 것 우선)"""
    global _GRADE_VOCAB, _GRADE_VOCAB_UPPER
    if _GRADE_VOCAB_UPPER is not None:
        return _GRADE_VOCAB_UPPER

    if os.path.exists(_VOCAB_CACHE_PATH):
        with open(_VOCAB_CACHE_PATH, "rb") as f:
            _GRADE_VOCAB = pickle.load(f)
        print(f"  [강종추출기] 어휘 사전 로드: {len(_GRADE_VOCAB):,}개")
    else:
        print("  [강종추출기] 어휘 사전 구축 중...")
        _GRADE_VOCAB = _build_grade_vocab()
        with open(_VOCAB_CACHE_PATH, "wb") as f:
            pickle.dump(_GRADE_VOCAB, f)
        print(f"  [강종추출기] 어휘 사전 구축 완료: {len(_GRADE_VOCAB):,}개")

    # uppercase 미리 계산
    _GRADE_VOCAB_UPPER = [(g.upper(), g) for g in _GRADE_VOCAB if g and len(g) >= 2]
    return _GRADE_VOCAB_UPPER


def extract_grade_from_spec(spec_text: str) -> str | None:
    """
    규격 텍스트에서 알려진 강종을 탐색.
    발견된 강종 중 가장 긴 것(가장 구체적인 것) 반환.
    """
    vocab = _get_grade_vocab()  # (upper, original) 튜플 리스트
    spec_upper = spec_text.upper()

    for grade_upper, grade_orig in vocab:  # 긴 것부터 탐색
        # 빠른 사전 필터: 포함 안 되면 regex 건너뜀
        if grade_upper not in spec_upper:
            continue
        # 단어 경계 매칭 (숫자/문자 조합 고려)
        pattern = r'(?<![A-Z0-9])' + re.escape(grade_upper) + r'(?![A-Z0-9])'
        if re.search(pattern, spec_upper):
            return grade_orig

    return None


def validate_and_correct(spec_text: str, system_grade: str) -> tuple[str, bool]:
    """
    시스템 분류 강종을 규격 텍스트로 검증·보정.

    Returns:
        (최종강종, 보정여부)
    """
    if not spec_text or not system_grade:
        return system_grade, False

    _get_grade_vocab()  # 캐시 워밍
    spec_upper = spec_text.upper()
    system_upper = system_grade.upper()

    # 1. 시스템 강종이 규격에 있으면 확정 (보정 불필요)
    if system_upper in spec_upper:
        pattern = r'(?<![A-Z0-9])' + re.escape(system_upper) + r'(?![A-Z0-9])'
        if re.search(pattern, spec_upper):
            return system_grade, False

    # 2. 규격에서 알려진 강종 탐색
    found_grade = extract_grade_from_spec(spec_text)
    if found_grade and found_grade.upper() != system_upper:
        return found_grade, True

    return system_grade, False


def rebuild_vocab_cache(confirmed_dir: str = "confirmed") -> None:
    """어휘 사전 강제 재구축"""
    global _GRADE_VOCAB
    print("[강종추출기] 어휘 사전 재구축 중...")
    _GRADE_VOCAB = _build_grade_vocab(confirmed_dir)
    with open(_VOCAB_CACHE_PATH, "wb") as f:
        pickle.dump(_GRADE_VOCAB, f)
    print(f"[강종추출기] 완료: {len(_GRADE_VOCAB):,}개")
