"""
사이즈 사전 pkl 증분 업데이트

새 확정 데이터 파일만 읽어서 기존 pkl에 추가합니다.
pkl이 없으면 confirmed/ 전체로 최초 구축합니다.

사용법:
  python scripts/update_size_cache.py confirmed/2026_05.xlsx   # 특정 파일 추가
  python scripts/update_size_cache.py --rebuild                # 전체 재구축
"""

import argparse
import pickle
import os
import sys
sys.path.insert(0, '.')

import pandas as pd
from pathlib import Path

CACHE_PATH = "size_lookup.pkl"


def read_size_file(path: Path) -> dict[str, str]:
    """xlsx 파일에서 규격→사이즈 dict 추출"""
    df = pd.read_excel(path, dtype=str)
    if len(df.columns) < 5:
        return {}

    cols = df.columns.tolist()
    sample = str(df.iloc[0, 0]) if len(df) > 0 else ""
    if sample.strip().replace(".", "").isdigit():
        spec_col, size_col = 2, 4
    else:
        spec_col, size_col = 1, 4

    sub = df[[cols[spec_col], cols[size_col]]].copy()
    sub.columns = ["spec", "size"]

    has_size = (
        sub["size"].notna()
        & (sub["size"].str.strip() != "")
        & (~sub["size"].str.strip().isin(["0", "0.0"]))
    )
    sub = sub[has_size].copy()
    sub["spec_key"] = sub["spec"].str.strip().str.upper()
    sub["size_val"] = sub["size"].str.strip()
    return sub.drop_duplicates("spec_key", keep="last").set_index("spec_key")["size_val"].to_dict()


def load_cache() -> dict[str, str]:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(lookup: dict[str, str]):
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(lookup, f)


def main():
    parser = argparse.ArgumentParser(description="사이즈 사전 pkl 증분 업데이트")
    parser.add_argument("file", nargs="?", help="추가할 확정 데이터 xlsx 파일 경로")
    parser.add_argument("--rebuild", action="store_true", help="confirmed/ 전체로 재구축")
    args = parser.parse_args()

    if args.rebuild:
        print("[전체 재구축] confirmed/ 폴더 전체 읽는 중...")
        from app.services.size_extractor import _build_exact_lookup
        lookup = _build_exact_lookup()
        save_cache(lookup)
        print(f"[완료] {len(lookup):,}건 저장: {CACHE_PATH}")
        return

    if not args.file:
        parser.print_help()
        return

    target = Path(args.file)
    if not target.exists():
        print(f"[오류] 파일 없음: {target}")
        return

    print(f"[로드] 기존 pkl 로드 중...")
    lookup = load_cache()
    before = len(lookup)
    print(f"  기존: {before:,}건")

    print(f"[추가] {target.name} 읽는 중...")
    new_data = read_size_file(target)
    lookup.update(new_data)
    after = len(lookup)

    save_cache(lookup)
    print(f"[완료] {after:,}건 저장 (신규 추가: {after - before:,}건)")


if __name__ == "__main__":
    main()
