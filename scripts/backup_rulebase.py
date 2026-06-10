"""
Oracle 룰베이스 백업/복원 스크립트

사용법:
  python scripts/backup_rulebase.py                      # 백업
  python scripts/backup_rulebase.py --label 수정전       # 레이블 지정 백업
  python scripts/backup_rulebase.py --list               # 백업 목록 확인
  python scripts/backup_rulebase.py --restore rulebase_backup/20260609_1200_수정전.xlsx  # 복원
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '.')

import pandas as pd
import oracledb
from app.core.config import settings

BACKUP_DIR = Path("rulebase_backup")


def get_connection():
    return oracledb.connect(
        user=settings.oracle_user,
        password=settings.oracle_password,
        dsn=settings.oracle_dsn
    )


def backup(label: str = ""):
    BACKUP_DIR.mkdir(exist_ok=True)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT rule_id, pattern, steel_grade, size_val FROM rule_base ORDER BY rule_id")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    df = pd.DataFrame(rows, columns=["rule_id", "pattern", "steel_grade", "size_val"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{timestamp}_{label}.xlsx" if label else f"{timestamp}.xlsx"
    path = BACKUP_DIR / filename

    df.to_excel(path, index=False)
    print(f"[백업 완료] {path} ({len(df):,}건)")
    return path


def list_backups():
    if not BACKUP_DIR.exists():
        print("[백업 없음] rulebase_backup/ 폴더가 없습니다.")
        return
    backups = sorted(BACKUP_DIR.glob("*.xlsx"))
    if not backups:
        print("[백업 없음]")
        return
    print(f"[백업 목록] 총 {len(backups)}개:")
    for b in backups:
        df = pd.read_excel(b)
        print(f"  {b.name} ({len(df):,}건)")


def restore(path: str):
    target = Path(path)
    if not target.exists():
        print(f"[오류] 파일 없음: {path}")
        return

    df = pd.read_excel(target, dtype=str)
    print(f"[복원] {target.name} ({len(df):,}건) → Oracle RULE_BASE")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("TRUNCATE TABLE rule_base")
    print(f"  기존 데이터 삭제 완료")

    rows = [
        (
            str(row["rule_id"]).strip(),
            str(row["pattern"]).strip(),
            str(row["steel_grade"]).strip(),
            None if pd.isna(row["size_val"]) else str(row["size_val"]).strip()
        )
        for _, row in df.iterrows()
    ]
    cursor.executemany(
        "INSERT INTO rule_base (rule_id, pattern, steel_grade, size_val) VALUES (:1, :2, :3, :4)",
        rows
    )
    conn.commit()
    cursor.close()
    conn.close()
    print(f"[완료] {len(rows):,}건 복원 완료")


def main():
    parser = argparse.ArgumentParser(description="Oracle 룰베이스 백업/복원")
    parser.add_argument("--label", default="", help="백업 파일 레이블")
    parser.add_argument("--list", action="store_true", help="백업 목록 확인")
    parser.add_argument("--restore", default="", help="복원할 백업 파일 경로")
    args = parser.parse_args()

    if args.list:
        list_backups()
    elif args.restore:
        restore(args.restore)
    else:
        backup(args.label)


if __name__ == "__main__":
    main()
