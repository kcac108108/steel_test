"""
룰베이스 오류 수정 스크립트

1. 잘못된 Rule 확인 및 삭제
2. 신규 Rule 추가 (SN2658/SN2659 → AH32)

사용법:
  python scripts/fix_rulebase.py --dry-run   # 확인만 (실제 변경 없음)
  python scripts/fix_rulebase.py             # 실제 적용
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, '.')

import oracledb
from app.core.config import settings


def get_connection():
    return oracledb.connect(
        user=settings.oracle_user,
        password=settings.oracle_password,
        dsn=settings.oracle_dsn
    )


def find_bad_rules(cursor):
    """잘못 분류되고 있는 Rule 조회"""
    bad_rules = []

    # 1. 광범위한 단일 문자 prefix Rule (MODEL:L, MODEL:A, MODEL:UNS 등)
    cursor.execute("""
        SELECT rule_id, pattern, steel_grade
        FROM rule_base
        WHERE UPPER(pattern) IN (
            'MODEL: L', 'MODEL:L',
            'MODEL: A', 'MODEL:A',
            'MODEL: UNS', 'MODEL:UNS'
        )
        ORDER BY pattern
    """)
    rows = cursor.fetchall()
    for r in rows:
        bad_rules.append({"rule_id": r[0], "pattern": r[1], "steel_grade": r[2], "reason": f"광범위 prefix '{r[1]}' → {r[2]} 오분류"})

    # 2. ASTM A387 GR91로 잘못 분류하는 Rule (304/316 계열)
    cursor.execute("""
        SELECT rule_id, pattern, steel_grade
        FROM rule_base
        WHERE steel_grade = 'ASTM A387 GR91'
        AND (
            UPPER(pattern) LIKE '%304%'
            OR UPPER(pattern) LIKE '%316%'
            OR UPPER(pattern) LIKE '%303%'
        )
        ORDER BY pattern
    """)
    rows = cursor.fetchall()
    for r in rows:
        bad_rules.append({"rule_id": r[0], "pattern": r[1], "steel_grade": r[2], "reason": "스테인리스 → ASTM A387 GR91 오분류"})

    # 3. 303F로 잘못 분류하는 Rule (303 계열)
    cursor.execute("""
        SELECT rule_id, pattern, steel_grade
        FROM rule_base
        WHERE steel_grade = '303F'
        AND UPPER(pattern) NOT LIKE '%303F%'
        ORDER BY pattern
    """)
    rows = cursor.fetchall()
    for r in rows:
        bad_rules.append({"rule_id": r[0], "pattern": r[1], "steel_grade": r[2], "reason": "303 → 303F 오분류"})

    return bad_rules


def find_sn_patterns(cursor):
    """SN2658/SN2659 패턴 확인 (AH32 추가 대상) - MODEL: prefix 포함 여부 체크"""
    cursor.execute("""
        SELECT COUNT(*) FROM rule_base
        WHERE UPPER(pattern) LIKE '%SN2658A111A%'
           OR UPPER(pattern) LIKE '%SN2659A111A%'
    """)
    existing = cursor.fetchone()[0]
    return existing


def delete_wrong_sn_patterns(cursor):
    """잘못 추가된 SN 패턴 삭제 (MODEL: prefix 없는 것)"""
    cursor.execute("""
        DELETE FROM rule_base
        WHERE (UPPER(pattern) LIKE 'SN2658%' OR UPPER(pattern) LIKE 'SN2659%')
        AND UPPER(pattern) NOT LIKE 'MODEL:%'
    """)
    return cursor.rowcount


def main():
    parser = argparse.ArgumentParser(description="룰베이스 오류 수정")
    parser.add_argument("--dry-run", action="store_true", help="확인만 (실제 변경 없음)")
    args = parser.parse_args()

    conn = get_connection()
    cursor = conn.cursor()

    # ── 잘못된 Rule 확인 ─────────────────────────────────────
    print("=" * 60)
    print("[ 삭제 대상 Rule 목록 ]")
    print("=" * 60)
    bad_rules = find_bad_rules(cursor)

    if bad_rules:
        for r in bad_rules:
            print(f"  [{r['reason']}]")
            print(f"    rule_id: {r['rule_id']}")
            print(f"    pattern: {r['pattern']}")
            print(f"    grade  : {r['steel_grade']}")
            print()
        print(f"  총 {len(bad_rules)}건 삭제 예정\n")
    else:
        print("  삭제 대상 없음\n")

    # ── SN2658/SN2659 Rule 추가 확인 ─────────────────────────
    print("=" * 60)
    print("[ 추가 대상 Rule - AH32 ]")
    print("=" * 60)
    existing_sn = find_sn_patterns(cursor)
    if existing_sn > 0:
        print(f"  SN2658/SN2659 패턴이 이미 {existing_sn}건 존재합니다.")
    else:
        print("  SN2658A111A → AH32 (Rule 추가 예정)")
        print("  SN2659A111A → AH32 (Rule 추가 예정)")
    print()

    if args.dry_run:
        print("[dry-run] 실제 변경 없음. --dry-run 제거 후 재실행하면 적용됩니다.")
        cursor.close()
        conn.close()
        return

    # ── 실제 수정 ────────────────────────────────────────────
    # 1. 잘못된 Rule 삭제
    if bad_rules:
        rule_ids = [r["rule_id"] for r in bad_rules]
        cursor.executemany(
            "DELETE FROM rule_base WHERE rule_id = :1",
            [(rid,) for rid in rule_ids]
        )
        print(f"[삭제] {len(rule_ids)}건 삭제 완료")

    # 1-b. 잘못된 SN 패턴(MODEL: prefix 없는 것) 삭제
    deleted_sn = delete_wrong_sn_patterns(cursor)
    if deleted_sn > 0:
        print(f"[삭제] 잘못된 SN 패턴 {deleted_sn}건 삭제")

    # 2. SN2658/SN2659 → AH32 추가 (MODEL: prefix 포함 버전)
    existing_sn = find_sn_patterns(cursor)
    if existing_sn == 0:
        cursor.execute("SELECT MAX(rule_id) FROM rule_base")
        max_id = cursor.fetchone()[0] or "MC000000"
        prefix = ''.join(c for c in max_id if c.isalpha()) or "MC"
        max_num = int(''.join(c for c in max_id if c.isdigit()) or "0")

        new_rules = [
            (f"{prefix}{max_num+1:06d}", "MODEL: SN2658A111A", "AH32", None),
            (f"{prefix}{max_num+2:06d}", "MODEL: SN2659A111A", "AH32", None),
            (f"{prefix}{max_num+3:06d}", "MODEL:SN2658A111A", "AH32", None),
            (f"{prefix}{max_num+4:06d}", "MODEL:SN2659A111A", "AH32", None),
        ]
        cursor.executemany(
            "INSERT INTO rule_base (rule_id, pattern, steel_grade, size_val) VALUES (:1, :2, :3, :4)",
            new_rules
        )
        print(f"[추가] SN2658A111A, SN2659A111A → AH32 추가 완료")

    conn.commit()
    cursor.close()
    conn.close()
    print("\n[완료] 룰베이스 수정 완료. classify.py 재실행으로 정확도 확인하세요.")


if __name__ == "__main__":
    main()
