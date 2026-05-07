"""
RAG 인덱스 갱신 스크립트

confirmed/ 폴더의 확정 엑셀 파일을 읽어서
ChromaDB 인덱스를 갱신합니다.

확정 파일 컬럼: 번호, 거래품명, 규격, 강종, 사이즈
  - 강종이 있는 행만 처리
  - 동일 규격이 이미 인덱스에 있으면 업데이트, 없으면 신규 추가

사용법:
  python scripts/update_rag.py
  python scripts/update_rag.py --confirmed-dir confirmed
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


def load_confirmed_files(confirmed_dir: str) -> pd.DataFrame:
    files = sorted(Path(confirmed_dir).glob("**/*.xlsx"))
    if not files:
        print(f"[오류] 확정 파일이 없습니다: {confirmed_dir}")
        sys.exit(1)

    print(f"[파일 로드] {len(files)}개 파일 발견")
    dfs = []
    for f in files:
        try:
            df = pd.read_excel(f, dtype={"번호": str})
            dfs.append(df)
            print(f"  OK {f.name} ({len(df):,}건)")
        except Exception as e:
            print(f"  FAIL {f.name} - {e}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"[합계] {len(combined):,}건 로드")
    return combined


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["규격", "강종"]:
        if col not in df.columns:
            print(f"[오류] '{col}' 컬럼이 없습니다.")
            sys.exit(1)

    before = len(df)

    df = df.dropna(subset=["강종"])
    df = df[df["강종"].astype(str).str.strip() != ""]
    df = df.dropna(subset=["규격"])
    df = df[df["규격"].astype(str).str.strip() != ""]

    df["규격"] = df["규격"].astype(str).str.strip()
    df["강종"] = df["강종"].astype(str).str.strip()
    if "사이즈" in df.columns:
        df["사이즈"] = df["사이즈"].fillna("").astype(str).str.strip()
    else:
        df["사이즈"] = ""

    df = df.drop_duplicates(subset=["규격"], keep="last")
    print(f"[전처리] {before:,} → {len(df):,}건 (강종 없음/중복 제거)")
    return df


def main():
    parser = argparse.ArgumentParser(description="RAG 인덱스 갱신")
    parser.add_argument("--confirmed-dir", default="confirmed", help="확정 파일 폴더 경로")
    args = parser.parse_args()

    print(f"[RAG 갱신 시작] 폴더: {args.confirmed_dir}")
    df = load_confirmed_files(args.confirmed_dir)
    df = preprocess(df)
    print(f"\n[갱신 대상] {len(df):,}건")

    records = [
        HistoryRecord(
            spec_text=row["규격"],
            steel_grade=row["강종"],
            size=row["사이즈"],
        )
        for _, row in df.iterrows()
    ]

    rag = RAGService()
    rag.index_history(records)
    print("[완료] RAG 갱신 완료")


if __name__ == "__main__":
    main()
