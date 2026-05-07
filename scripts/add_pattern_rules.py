"""
패턴 분석 결과를 Oracle rule_base 테이블에 추가하는 스크립트

사용법:
  python scripts/add_pattern_rules.py --dry-run   # 실제 추가 없이 확인만
  python scripts/add_pattern_rules.py             # 실제 추가
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import oracledb
from app.core.config import settings


def load_existing_patterns(cursor) -> set:
    """기존 rule_base에 있는 패턴 목록"""
    cursor.execute("SELECT UPPER(pattern) FROM rule_base")
    return {row[0] for row in cursor.fetchall()}


def add_pattern_rules(csv_path: str, dry_run: bool = False, min_consistency: float = 95.0) -> None:
    print(f"[패턴 로드] {csv_path}")
    df = pd.read_csv(csv_path)

    # 일관성 필터
    df = df[df["consistency"] >= min_consistency]
    print(f"  일관성 {min_consistency}% 이상: {len(df):,}개 패턴")

    # 빈 패턴 제거
    df = df[df["pattern"].str.strip() != ""]
    df = df[df["best_grade"].str.strip() != ""]

    conn = oracledb.connect(
        user=settings.oracle_user,
        password=settings.oracle_password,
        dsn=settings.oracle_dsn,
    )
    cursor = conn.cursor()

    # 기존 패턴 로드
    existing = load_existing_patterns(cursor)
    print(f"  기존 rule_base 패턴: {len(existing):,}개")

    # 신규 패턴만 필터
    new_rows = df[~df["pattern"].str.upper().isin(existing)]
    print(f"  신규 추가 대상: {len(new_rows):,}개")

    if dry_run:
        print("\n[dry-run] 실제 추가는 하지 않습니다.")
        print("샘플 (상위 10개):")
        for _, row in new_rows.head(10).iterrows():
            print(f"  {row['pattern']} → {row['best_grade']} (일관성: {row['consistency']:.1f}%, {row['best_count']}건)")
        cursor.close()
        conn.close()
        return

    # rule_id 최대 숫자값 조회 (MC000001 형식 가정)
    cursor.execute("SELECT MAX(rule_id) FROM rule_base")
    max_rule_id = cursor.fetchone()[0] or "MC000000"
    # 숫자 부분 추출 후 증가
    prefix = ''.join(c for c in max_rule_id if c.isalpha()) or "MC"
    max_num = int(''.join(c for c in max_rule_id if c.isdigit()) or "0")

    # INSERT
    insert_data = []
    for i, (_, row) in enumerate(new_rows.iterrows()):
        rule_id = f"{prefix}{max_num + i + 1:06d}"
        insert_data.append((rule_id, row["pattern"], row["best_grade"], None))

    cursor.executemany(
        "INSERT INTO rule_base (rule_id, pattern, steel_grade, size_val) VALUES (:1, :2, :3, :4)",
        insert_data,
    )
    conn.commit()

    print(f"\n[완료] {len(insert_data):,}개 패턴 rule_base에 추가 완료!")
    print(f"  rule_base 총 건수: {len(existing) + len(insert_data):,}개")

    cursor.close()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="패턴 분석 결과를 rule_base에 추가")
    parser.add_argument("--csv", default="pattern_analysis_result.csv", help="패턴 CSV 파일 경로")
    parser.add_argument("--dry-run", action="store_true", help="실제 추가 없이 확인만")
    parser.add_argument("--min-consistency", type=float, default=95.0, help="최소 일관성 % (기본: 95)")
    args = parser.parse_args()

    add_pattern_rules(
        csv_path=args.csv,
        dry_run=args.dry_run,
        min_consistency=args.min_consistency,
    )


if __name__ == "__main__":
    main()
