import pandas as pd
import glob
import re
from collections import defaultdict

# 모든 xlsx 파일 수집
files = glob.glob('C:/workspace/steelproject/steel_test/steel_sample/**/*.xlsx', recursive=True)
print(f"총 파일 수: {len(files)}")

# 데이터 로드
all_dfs = []
for f in sorted(files):
    try:
        df = pd.read_excel(f, header=0, usecols=[0,1,2,3,4],
                           names=['거래품명','규격','강종추정','강종','사이즈'])
        df['source'] = f
        all_dfs.append(df)
    except Exception as e:
        print(f"  오류 {f}: {e}")

df_all = pd.concat(all_dfs, ignore_index=True)
print(f"전체 행 수: {len(df_all)}")

# 강종이 있는 행만
df_valid = df_all[df_all['강종'].notna() & (df_all['강종'].astype(str).str.strip() != '')].copy()
df_valid = df_valid[df_valid['규격'].notna() & (df_valid['규격'].astype(str).str.strip() != '')].copy()
print(f"강종 있는 행 수: {len(df_valid)}")

def extract_pattern(spec_str):
    """
    규격 문자열에서 선행 패턴 추출
    예시:
      FNCLB-V42-D60-L14  -> FNCLB-V42
      NCLSS8-15-25       -> NCLSS8
      AMSC30-xxx         -> AMSC30
      NCLSS8             -> NCLSS8
    규칙:
      1. 문자+하이픈+문자숫자 형태: 두 번째 하이픈 전까지 (FNCLB-V42)
      2. 문자+숫자 형태 (하이픈 없음): 첫 번째 하이픈 또는 공백 전까지
    """
    s = str(spec_str).strip()

    # 패턴 1: 문자열-문자숫자 형태 (예: FNCLB-V42, NCLSS8-15)
    # 첫 세그먼트가 알파벳만, 두 번째 세그먼트가 알파벳+숫자 혼합
    m = re.match(r'^([A-Za-z]+-[A-Za-z][A-Za-z0-9]*)(?=[-\s]|$)', s)
    if m:
        return m.group(1).upper()

    # 패턴 2: 문자로 시작하고 숫자 포함, 하이픈 앞까지 (예: NCLSS8, AMSC30)
    m = re.match(r'^([A-Za-z]+[0-9]+[A-Za-z0-9]*)(?=[-\s]|$)', s)
    if m:
        return m.group(1).upper()

    # 패턴 3: 순수 알파벳만 (예: SUS, SKD)
    m = re.match(r'^([A-Za-z]{2,})(?=[-\s\d]|$)', s)
    if m:
        candidate = m.group(1).upper()
        # 너무 짧거나 너무 일반적인 것 제외
        if len(candidate) >= 3:
            return candidate

    return None

# 패턴 추출
df_valid['pattern'] = df_valid['규격'].apply(extract_pattern)
df_with_pattern = df_valid[df_valid['pattern'].notna()].copy()
print(f"패턴 추출된 행 수: {len(df_with_pattern)}")
print(f"고유 패턴 수: {df_with_pattern['pattern'].nunique()}")

# 강종 정규화
df_with_pattern['강종_clean'] = df_with_pattern['강종'].astype(str).str.strip().str.upper()

# (패턴, 강종) 그룹별 카운트
group = df_with_pattern.groupby(['pattern', '강종_clean']).size().reset_index(name='count')

# 패턴별 총 카운트
pattern_total = group.groupby('pattern')['count'].sum().reset_index(name='total')

# 패턴별 최빈 강종 및 일관성 계산
def get_consistency(sub):
    total = sub['count'].sum()
    max_count = sub['count'].max()
    best_grade = sub.loc[sub['count'].idxmax(), '강종_clean']
    return pd.Series({
        'best_grade': best_grade,
        'best_count': max_count,
        'total': total,
        'consistency': max_count / total * 100
    })

pattern_stats = group.groupby('pattern').apply(get_consistency).reset_index()
pattern_stats = pattern_stats.merge(pattern_total, on='pattern')

# 95% 이상 일관성 필터
consistent = pattern_stats[pattern_stats['consistency'] >= 95].copy()
consistent = consistent.sort_values('total', ascending=False)

print(f"\n=== 분석 결과 ===")
print(f"총 고유 패턴 수: {pattern_stats['pattern'].nunique()}")
print(f"일관성 95% 이상 패턴 수: {len(consistent)}")
print(f"룰베이스 추가 가능 패턴 수: {len(consistent)}")

print(f"\n=== TOP 30 패턴 (빈도순) ===")
print(f"{'패턴':<20} {'강종':<20} {'건수':>8} {'일관성':>8}")
print("-" * 60)
for _, row in consistent.head(30).iterrows():
    print(f"{row['pattern']:<20} {row['best_grade']:<20} {int(row['total']):>8,} {row['consistency']:>7.1f}%")

# 상세: 일관성 100% vs 95-99%
perfect = consistent[consistent['consistency'] == 100]
near = consistent[(consistent['consistency'] >= 95) & (consistent['consistency'] < 100)]
print(f"\n=== 일관성 요약 ===")
print(f"일관성 100%: {len(perfect)}개")
print(f"일관성 95~99%: {len(near)}개")

# 패턴 유형별 샘플 출력
print(f"\n=== 패턴 유형 샘플 ===")
print("하이픈 포함 패턴 (예: FNCLB-V42):")
hyphen_patterns = consistent[consistent['pattern'].str.contains('-')].head(10)
for _, row in hyphen_patterns.iterrows():
    print(f"  {row['pattern']} -> {row['best_grade']} ({int(row['total'])}건)")

print("\n숫자 포함 패턴 (예: NCLSS8, AMSC30):")
num_patterns = consistent[~consistent['pattern'].str.contains('-') &
                           consistent['pattern'].str.contains(r'\d')].head(10)
for _, row in num_patterns.iterrows():
    print(f"  {row['pattern']} -> {row['best_grade']} ({int(row['total'])}건)")

# 실제 규격 샘플 확인
print(f"\n=== 상위 10개 패턴의 실제 규격 샘플 ===")
for _, row in consistent.head(10).iterrows():
    pat = row['pattern']
    samples = df_with_pattern[df_with_pattern['pattern'] == pat]['규격'].unique()[:3]
    print(f"  {pat} ({row['best_grade']}): {list(samples)}")

# 결과 저장
output_path = 'C:/workspace/steelproject/steel_test/pattern_analysis_result.csv'
consistent.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n결과 저장: {output_path}")
