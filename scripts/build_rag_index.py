"""
RAG 인덱스 구축 스크립트

steel_sample 폴더의 전체 확정치 엑셀을 읽어서
ChromaDB에 인덱싱합니다.

사용법:
  python scripts/build_rag_index.py --data-dir steel_sample
  python scripts/build_rag_index.py --data-dir steel_sample --dry-run  # 실제 인덱싱 없이 통계만 확인
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import logging
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from app.services.rag_service import RAGService
from app.models.schemas import HistoryRecord


def load_all_files(data_dir: str) -> pd.DataFrame:
    """폴더 내 전체 xlsx 파일 읽어서 합치기"""
    files = sorted(Path(data_dir).glob("**/*.xlsx"))
    if not files:
        print(f"[오류] xlsx 파일을 찾을 수 없습니다: {data_dir}")
        sys.exit(1)

    print(f"[파일 로드] 총 {len(files)}개 파일 발견")

    dfs = []
    for f in files:
        try:
            df = pd.read_excel(f, header=0)
            # 컬럼명 통일 (위치 기준)
            if df.shape[1] >= 5:
                df = df.iloc[:, :5]
                df.columns = ["거래품명", "규격", "강종추정", "강종", "사이즈"]
                dfs.append(df)
                print(f"  OK {f.relative_to(data_dir)} ({len(df):,}건)")
            else:
                print(f"  SKIP {f.relative_to(data_dir)} - 컬럼 수 부족")
        except Exception as e:
            print(f"  FAIL {f.relative_to(data_dir)} - 읽기 실패: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n[합계] 총 {len(combined):,}건 로드 완료")
    return combined


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """전처리: 필터링 → 정제 → 중복 제거"""

    # 1. 강종(D열) 없는 행 제거
    before = len(df)
    df = df.dropna(subset=["강종"])
    df = df[df["강종"].astype(str).str.strip() != ""]
    print(f"[전처리] 강종 없는 행 제거: {before:,} → {len(df):,}건")

    # 2. 규격(B열) 없는 행 제거
    before = len(df)
    df = df.dropna(subset=["규격"])
    df = df[df["규격"].astype(str).str.strip() != ""]
    print(f"[전처리] 규격 없는 행 제거: {before:,} → {len(df):,}건")

    # 3. 텍스트 정제 (공백 정리)
    df["규격"] = df["규격"].astype(str).str.strip()
    df["강종"] = df["강종"].astype(str).str.strip()
    df["사이즈"] = df["사이즈"].fillna("").astype(str).str.strip()

    # 4. 중복 제거 (규격 + 강종 기준)
    before = len(df)
    df = df.drop_duplicates(subset=["규격", "강종"])
    print(f"[전처리] 중복 제거: {before:,} → {len(df):,}건")

    return df


def build_index(data_dir: str, dry_run: bool = False) -> None:
    # 1. 전체 파일 로드
    df = load_all_files(data_dir)

    # 2. 전처리
    print()
    df = preprocess(df)

    # 3. 통계 출력
    print(f"\n[인덱싱 대상] 최종 {len(df):,}건")
    print(f"  강종 고유값: {df['강종'].nunique():,}개")
    print(f"  규격 고유값: {df['규격'].nunique():,}개")

    if dry_run:
        print("\n[dry-run] 실제 인덱싱은 수행하지 않습니다.")
        print("샘플 데이터 (상위 5건):")
        print(df[["규격", "강종", "사이즈"]].head())
        return

    # 4. HistoryRecord 변환
    records = [
        HistoryRecord(
            spec_text=row["규격"],
            steel_grade=row["강종"],
            size=row["사이즈"],
        )
        for _, row in df.iterrows()
    ]

    # 5. ChromaDB 인덱싱
    print("\n[RAG 인덱싱 시작]")
    rag = RAGService()
    rag.index_history(records)
    print("[완료] RAG 인덱싱 완료")


def main():
    parser = argparse.ArgumentParser(description="RAG 인덱스 구축")
    parser.add_argument("--data-dir", default="steel_sample", help="확정치 엑셀 폴더 경로")
    parser.add_argument("--dry-run", action="store_true", help="통계만 확인, 실제 인덱싱 안 함")
    args = parser.parse_args()

    build_index(data_dir=args.data_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
