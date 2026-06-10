"""
철강 수입신고 자동 분류 스크립트

사용법:
  python scripts/classify.py --input 전일자.xlsx --output 결과.xlsx
  python scripts/classify.py --input 전일자.xlsx  1# 결과는 자동으로 output/ 저장

입력 엑셀 컬럼:
  - 번호, 수입신고번호, 거래품명, 규격, 규격번호, 강종, 사이즈

출력 엑셀 컬럼:
  - 기존 컬럼 전체 + 강종_RAG, 사이즈_RAG, 분류방법
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import logging
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from app.services.classifier import SteelClassifier
from app.services.size_extractor import SizeExtractor


def run_classify(input_path: str, output_path: str, use_llm: bool, use_rule: bool, use_size: bool) -> None:
    print(f"[분류 시작] 입력: {input_path}")

    df = pd.read_excel(input_path, dtype={"번호": str, "수입신고번호": str, "규격번호": str})

    if "규격" not in df.columns:
        print("[오류] 엑셀에 '규격' 컬럼이 없습니다.")
        sys.exit(1)

    spec_texts = df["규격"].fillna("").astype(str).tolist()
    print(f"  총 {len(spec_texts):,}건 처리 예정")

    def clean(value: str | None) -> str:
        if not value or value.strip() in ("", "0", "0.0"):
            return ""
        return value.strip()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.replace(".xlsx", "_checkpoint.pkl")

    # 강종 분류 (Rule → RAG → LLM)
    classifier = SteelClassifier(use_rule=use_rule, use_rag=True, use_llm=use_llm)
    results = classifier.classify_batch(spec_texts, checkpoint_path=checkpoint_path)

    df["강종_RAG"] = [clean(r.steel_grade) for r in results]
    df["분류방법"] = [r.method.value for r in results]

    # 사이즈 추출 (정확매칭 → 정규식 → LLM)
    if use_size:
        print("\n[사이즈 추출 시작]")
        size_extractor = SizeExtractor()
        size_results = size_extractor.extract_batch(spec_texts)
        df["사이즈_RAG"] = [clean(size) for size, _ in size_results]
        df["사이즈_방법"] = [method for _, method in size_results]
    else:
        print("\n[사이즈 추출 스킵]")
        size_results = [("", "skipped")] * len(spec_texts)

    df.to_excel(output_path, index=False)
    print(f"\n[완료] 결과 저장: {output_path}")

    import os
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    total = len(results)
    by_method: dict[str, int] = {}
    for r in results:
        by_method[r.method.value] = by_method.get(r.method.value, 0) + 1

    print("\n--- 강종 분류 통계 ---")
    for method, count in sorted(by_method.items()):
        pct = count / total * 100
        print(f"  {method:15s}: {count:5d}건 ({pct:.1f}%)")
    print(f"  {'합계':15s}: {total:5d}건")

    size_by_method: dict[str, int] = {}
    for _, method in size_results:
        size_by_method[method] = size_by_method.get(method, 0) + 1

    print("\n--- 사이즈 추출 통계 ---")
    for method, count in sorted(size_by_method.items()):
        pct = count / total * 100
        print(f"  {method:15s}: {count:5d}건 ({pct:.1f}%)")
    print(f"  {'합계':15s}: {total:5d}건")


def main():
    parser = argparse.ArgumentParser(description="철강 수입신고 자동 분류")
    parser.add_argument("--input", required=True, help="입력 엑셀 파일 경로")
    parser.add_argument("--output", help="출력 엑셀 파일 경로 (미지정 시 자동 생성)")
    parser.add_argument("--no-llm", action="store_true", help="LLM 판단 비활성화")
    parser.add_argument("--no-rule", action="store_true", help="Rule 매칭 비활성화 (RAG/LLM만 사용)")
    parser.add_argument("--no-size", action="store_true", help="사이즈 추출 비활성화")

    args = parser.parse_args()

    if not args.output:
        input_p = Path(args.input)
        args.output = str(
            Path("output") / f"{input_p.stem}_분류결과{input_p.suffix}"
        )

    run_classify(
        input_path=args.input,
        output_path=args.output,
        use_llm=not args.no_llm,
        use_rule=not args.no_rule,
        use_size=not args.no_size,
    )


if __name__ == "__main__":
    main()
