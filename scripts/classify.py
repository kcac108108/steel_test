"""
철강 수입신고 자동 분류 스크립트

사용법:
  python scripts/classify.py --input 전일자.xlsx --output 결과.xlsx
  python scripts/classify.py --input 전일자.xlsx  # 결과는 자동으로 output/ 저장

입력 엑셀 컬럼 (기본):
  - 번호, 거래품명, 규격, 강종, 사이즈

입력 엑셀 컬럼 (신고번호 앵커 모드):
  - 번호, 수입신고번호, 거래품명, 규격, 규격번호, 강종, 사이즈

출력 엑셀 컬럼:
  - 기존 컬럼 전체 + 강종_RAG, 사이즈_RAG, 분류방법[, 강종_그룹]
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


def apply_anchor_logic(df: pd.DataFrame) -> pd.Series:
    """
    수입신고번호 그룹 앵커 로직.
    같은 수입신고번호 내에서 강종_RAG 값이 규격 텍스트에 명시된 행을 앵커로 설정하고,
    앵커 ~ 다음 앵커 사이 행들에 앵커 강종을 상속.
    """
    result = df["강종_RAG"].copy()

    for _, group in df.groupby("수입신고번호", sort=False):
        current_anchor = None
        for idx in group.index:
            grade = str(df.at[idx, "강종_RAG"]).strip()
            spec = str(df.at[idx, "규격"]).strip().upper()

            if grade and grade not in ("", "nan") and len(grade) >= 3:
                is_anchor = grade.upper() in spec
            else:
                is_anchor = False

            if is_anchor:
                current_anchor = grade
            elif current_anchor:
                result.at[idx] = current_anchor

    return result


def run_classify(input_path: str, output_path: str, use_llm: bool) -> None:
    print(f"[분류 시작] 입력: {input_path}")

    df = pd.read_excel(input_path, dtype={"번호": str})

    if "규격" not in df.columns:
        print("[오류] 엑셀에 '규격' 컬럼이 없습니다.")
        sys.exit(1)

    # 규격번호 숫자 변환 후 수입신고번호 + 규격번호 기준 정렬
    df["규격번호"] = pd.to_numeric(df["규격번호"], errors="coerce").astype("Int64")
    df = df.sort_values(["수입신고번호", "규격번호"], kind="stable").reset_index(drop=True)
    print(f"  수입신고번호 + 규격번호 기준 정렬 완료")

    spec_texts = df["규격"].fillna("").astype(str).tolist()
    print(f"  총 {len(spec_texts):,}건 처리 예정")

    def clean(value: str | None) -> str:
        if not value or value.strip() in ("", "0", "0.0"):
            return ""
        return value.strip()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.replace(".xlsx", "_checkpoint.pkl")

    classifier = SteelClassifier(use_rag=True, use_llm=use_llm)
    results = classifier.classify_batch(spec_texts, checkpoint_path=checkpoint_path)

    df["강종_RAG"] = [clean(r.steel_grade) for r in results]
    df["사이즈_RAG"] = [clean(r.size) for r in results]
    df["분류방법"] = [r.method.value for r in results]

    # 수입신고번호 앵커 로직 적용
    df["강종_그룹"] = apply_anchor_logic(df)
    anchor_filled = (df["강종_그룹"] != df["강종_RAG"]).sum()
    print(f"  [앵커 로직] 상속 적용: {anchor_filled:,}건")

    df.to_excel(output_path, index=False)
    print(f"[완료] 결과 저장: {output_path}")

    # 체크포인트 삭제
    import os
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # 통계 출력
    total = len(results)
    by_method: dict[str, int] = {}
    for r in results:
        by_method[r.method.value] = by_method.get(r.method.value, 0) + 1

    print("\n--- 분류 결과 통계 ---")
    for method, count in sorted(by_method.items()):
        pct = count / total * 100
        print(f"  {method:15s}: {count:5d}건 ({pct:.1f}%)")
    print(f"  {'합계':15s}: {total:5d}건")


def main():
    parser = argparse.ArgumentParser(description="철강 수입신고 자동 분류")
    parser.add_argument("--input", required=True, help="입력 엑셀 파일 경로")
    parser.add_argument("--output", help="출력 엑셀 파일 경로 (미지정 시 자동 생성)")
    parser.add_argument("--no-llm", action="store_true", help="LLM 판단 비활성화")

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
    )


if __name__ == "__main__":
    main()
                                                                    