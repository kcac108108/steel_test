"""
ChromaDB 백업 스크립트

chroma_db/ 폴더를 날짜별로 백업합니다.

사용법:
  python scripts/backup_rag.py
  python scripts/backup_rag.py --label 3월완료
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="ChromaDB 백업")
    parser.add_argument("--label", default="", help="백업 이름 (기본: 날짜만)")
    parser.add_argument("--chroma-dir", default="chroma_db", help="ChromaDB 폴더 경로")
    parser.add_argument("--backup-dir", default="chroma_db_backup", help="백업 저장 폴더")
    args = parser.parse_args()

    src = Path(args.chroma_dir)
    if not src.exists():
        print(f"[오류] ChromaDB 폴더가 없습니다: {src}")
        return

    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    folder_name = f"{date_str}_{args.label}" if args.label else date_str
    dst = Path(args.backup_dir) / folder_name

    print(f"[백업 시작] {src} → {dst}")
    shutil.copytree(src, dst)
    print(f"[완료] 백업 완료: {dst}")

    # 백업 목록 출력
    backups = sorted(Path(args.backup_dir).iterdir())
    print(f"\n[백업 목록] 총 {len(backups)}개:")
    for b in backups:
        size_mb = sum(f.stat().st_size for f in b.rglob("*") if f.is_file()) / 1024 / 1024
        print(f"  {b.name} ({size_mb:.0f}MB)")


if __name__ == "__main__":
    main()
