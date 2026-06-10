"""
룰베이스 갱신 스크립트

confirmed/ 폴더의 확정 엑셀 파일에서 MODEL: 패턴을 추출하여
Oracle rule_base 테이블을 갱신합니다.

동작 방식:
  - 규격 텍스트에서 'MODEL: XXXXX' 패턴 추출
  - 동일 패턴의 강종 일관성 분석 (기본 95% 이상)
  - 신규 패턴 → INSERT
  - 기존 패턴 + 강종 불일치 → 충돌 리포트 출력 (자동 수정 안 함, 담당자 검토)

사용법:
  python scripts/update_rulebase.py
  python scripts/update_rulebase.py --dry-run
  python scripts/update_rulebase.py --confirmed-dir confirmed
"""

import sys
import re
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import oracledb
from app.core.config import settings


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


def extract_model_pattern(spec_text: str) -> str | None:
    """규격 텍스트에서 'MODEL: XXXXX' 패턴 추출 (끝의 쉼표/세미콜론 제거)"""
    m = re.search(r'(MODEL:\s*\S+)', str(spec_text), re.IGNORECASE)
    if m:
        pattern = m.group(1).strip().upper().rstrip(',;')
        return pattern if pattern else None
    return None


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["규격", "강종"]:
        if col not in df.columns:
            print(f"[오류] '{col}' 컬럼이 없습니다.")
            sys.exit(1)

    df = df.dropna(subset=["강종"])
    df = df[df["강종"].astype(str).str.strip() != ""]
    df = df.dropna(subset=["규격"])
    df = df[df["규격"].astype(str).str.strip() != ""]

    df["규격"] = df["규격"].astype(str).str.strip()
    df["강종"] = df["강종"].astype(str).str.strip()

    # MODEL: 패턴 추출
    df["model_pattern"] = df["규격"].apply(extract_model_pattern)
    before = len(df)
    df = df[df["model_pattern"].notna()]
    print(f"[전처리] {before:,}건 중 MODEL: 패턴 있는 행: {len(df):,}건")
    return df


def analyze_consistency(df: pd.DataFrame, min_consistency: float) -> pd.DataFrame:
    """패턴별 강종 일관성 분석"""
    rows = []
    for pattern, group in df.groupby("model_pattern"):
        grades = group["강종"]
        total = len(grades)
        best_grade = grades.mode().iloc[0]
        best_count = (grades == best_grade).sum()
        consistency = best_count / total * 100
        rows.append({
            "pattern": pattern,
            "best_grade": best_grade,
            "best_count": int(best_count),
            "total": total,
            "consistency": round(consistency, 1),
        })

    result = pd.DataFrame(rows)
    print(f"[일관성 분석] 고유 MODEL 패턴: {len(result):,}개")

    filtered = result[result["consistency"] >= min_consistency].copy()
    print(f"[필터링] 일관성 {min_consistency}% 이상: {len(filtered):,}개")
    return filtered


def get_existing_patterns(cursor) -> dict:
    """기존 rule_base 패턴과 강종을 함께 반환 {pattern_upper: steel_grade}"""
    cursor.execute("SELECT UPPER(pattern), steel_grade FROM rule_base")
    return {row[0]: row[1] for row in cursor.fetchall()}


def print_conflict_report(consistent_df: pd.DataFrame, existing: dict) -> None:
    """기존 패턴과 강종이 다른 경우 충돌 리포트 출력"""
    conflicts = []
    for _, r in consistent_df.iterrows():
        pattern_upper = r["pattern"].upper()
        if pattern_upper in existing:
            existing_grade = existing[pattern_upper]
            if existing_grade != r["best_grade"]:
                conflicts.append({
                    "pattern": r["pattern"],
                    "existing_grade": existing_grade,
                    "new_grade": r["best_grade"],
                    "consistency": r["consistency"],
                    "total": r["total"],
                })

    if not conflicts:
        print("\n[충돌 없음] 기존 패턴과 강종이 모두 일치합니다.")
        return

    print(f"\n{'=' * 60}")
    print(f"[ 충돌 감지 리포트 ] — {len(conflicts)}건 (담당자 검토 필요)")
    print(f"{'=' * 60}")
    for c in conflicts:
        print(f"  패턴    : {c['pattern']}")
        print(f"  기존    : {c['existing_grade']}")
        print(f"  신규    : {c['new_grade']} (일관성 {c['consistency']}%, {c['total']}건)")
        print()


def get_next_rule_id(cursor) -> tuple[str, int]:
    cursor.execute("SELECT MAX(rule_id) FROM rule_base")
    max_id = cursor.fetchone()[0] or "MC000000"
    prefix = ''.join(c for c in max_id if c.isalpha()) or "MC"
    max_num = int(''.join(c for c in max_id if c.isdigit()) or "0")
    return prefix, max_num


def main():
    parser = argparse.ArgumentParser(description="룰베이스 갱신 (MODEL: 패턴 기반)")
    parser.add_argument("--confirmed-dir", default="confirmed", help="확정 파일 폴더 경로")
    parser.add_argument("--min-consistency", type=float, default=95.0, help="최소 일관성 %% (기본: 95)")
    parser.add_argument("--dry-run", action="store_true", help="실제 변경 없이 확인만")
    args = parser.parse_args()

    print(f"[룰베이스 갱신 시작] 폴더: {args.confirmed_dir}, 최소 일관성: {args.min_consistency}%")

    df = load_confirmed_files(args.confirmed_dir)
    df = preprocess(df)

    if df.empty:
        print("[완료] MODEL: 패턴이 있는 데이터가 없습니다.")
        return

    consistent_df = analyze_consistency(df, args.min_consistency)
    if consistent_df.empty:
        print("[완료] 일관성 기준을 충족하는 패턴이 없습니다.")
        return

    conn = oracledb.connect(
        user=settings.oracle_user,
        password=settings.oracle_password,
        dsn=settings.oracle_dsn,
    )
    cursor = conn.cursor()

    existing = get_existing_patterns(cursor)
    print(f"[기존 rule_base] {len(existing):,}개 패턴")

    # 충돌 감지 리포트 (기존 패턴 + 강종 불일치)
    print_conflict_report(consistent_df, existing)

    to_insert = consistent_df[~consistent_df["pattern"].str.upper().isin(existing)]

    # 최소 패턴 길이 필터 (짧은 패턴은 너무 광범위하게 매칭되어 오분류 유발)
    MIN_PATTERN_LEN = 15
    too_short = to_insert[to_insert["pattern"].str.len() < MIN_PATTERN_LEN]
    if not too_short.empty:
        print(f"\n[경고] 패턴 길이 {MIN_PATTERN_LEN}자 미만 {len(too_short)}건 제외 (광범위 매칭 위험):")
        for _, r in too_short.iterrows():
            print(f"  '{r['pattern']}' → {r['best_grade']}")
    to_insert = to_insert[to_insert["pattern"].str.len() >= MIN_PATTERN_LEN]

    print(f"\n[기존 패턴 제외] {len(consistent_df) - len(to_insert):,}개 / [신규 INSERT 대상] {len(to_insert):,}개")

    if args.dry_run:
        print("\n[dry-run] 실제 변경은 하지 않습니다.")
        if not to_insert.empty:
            print("INSERT 샘플 (상위 10개):")
            for _, r in to_insert.head(10).iterrows():
                print(f"  {r['pattern']} → {r['best_grade']} (일관성: {r['consistency']}%)")
        else:
            print("신규 추가할 패턴이 없습니다.")
        cursor.close()
        conn.close()
        return

    if to_insert.empty:
        print("[완료] 신규 추가할 패턴이 없습니다.")
        cursor.close()
        conn.close()
        return

    prefix, max_num = get_next_rule_id(cursor)
    insert_data = []
    for i, (_, r) in enumerate(to_insert.iterrows()):
        rule_id = f"{prefix}{max_num + i + 1:06d}"
        insert_data.append((rule_id, r["pattern"], r["best_grade"], None))

    cursor.executemany(
        "INSERT INTO rule_base (rule_id, pattern, steel_grade, size_val) VALUES (:1, :2, :3, :4)",
        insert_data,
    )
    conn.commit()
    print(f"\n[완료] {len(insert_data):,}개 신규 패턴 추가!")
    print(f"  rule_base 총 건수: {len(existing) + len(insert_data):,}개")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()
