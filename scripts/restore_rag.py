"""
ChromaDB 복원 스크립트

백업된 chroma_db를 복원합니다.

사용법:
  python scripts/restore_rag.py               # 백업 목록 확인
  python scripts/restore_rag.py --list        # 백업 목록 확인
  python scripts/restore_rag.py --name 20260604_1430_3월완료
"""

import argparse
import shutil
from pathlib import Path


def list_backups(backup_dir: Path):
    if not backup_dir.exists():
        print("[백업 없음] 백업 폴더가 없습니다.")
        return []

    backups = sorted(backup_dir.iterdir())
    if not backups:
        print("[백업 없음] 백업이 없습니다.")
        return []

    print(f"[백업 목록] 총 {len(backups)}개:")
    for i, b in enumerate(backups):
        size_mb = sum(f.stat().st_size for f in b.rglob("*") if f.is_file()) / 1024 / 1024
        print(f"  [{i+1}] {b.name} ({size_mb:.0f}MB)")
    return backups


def main():
    parser = argparse.ArgumentParser(description="ChromaDB 복원")
    parser.add_argument("--name", default="", help="복원할 백업 이름")
    parser.add_argument("--list", action="store_true", help="백업 목록만 확인")
    parser.add_argument("--chroma-dir", default="chroma_db", help="ChromaDB 폴더 경로")
    parser.add_argument("--backup-dir", default="chroma_db_backup", help="백업 저장 폴더")
    args = parser.parse_args()

    backup_dir = Path(args.backup_dir)
    backups = list_backups(backup_dir)

    if args.list or not args.name:
        return

    # 복원 대상 찾기
    target = backup_dir / args.name
    if not target.exists():
        print(f"[오류] 백업을 찾을 수 없습니다: {args.name}")
        print("위 목록에서 정확한 이름을 확인해 주세요.")
        return

    chroma_dir = Path(args.chroma_dir)

    # 기존 ChromaDB 삭제 후 복원
    if chroma_dir.exists():
        print(f"[삭제] 기존 ChromaDB 삭제 중...")
        shutil.rmtree(chroma_dir)

    print(f"[복원 시작] {target} → {chroma_dir}")
    shutil.copytree(target, chroma_dir)
    print(f"[완료] 복원 완료!")


if __name__ == "__main__":
    main()
