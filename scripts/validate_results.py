"""
시스템 분류 결과 vs 업무담당자 분류 결과 정확도 비교
"""
import sys
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='output/0501~6_분류결과.xlsx')
args, _ = parser.parse_known_args()

OUTPUT_FILE = args.input

df = pd.read_excel(OUTPUT_FILE, dtype=str)

def clean(val):
    if pd.isna(val):
        return ''
    return str(val).strip()

df['_강종_ans'] = df['강종'].apply(clean)
df['_강종_sys'] = df['강종_RAG'].apply(clean)
df['_사이즈_ans'] = df['사이즈'].apply(clean) if '사이즈' in df.columns else pd.Series([''] * len(df))
df['_사이즈_sys'] = df['사이즈_RAG'].apply(clean) if '사이즈_RAG' in df.columns else pd.Series([''] * len(df))
has_size = '사이즈_RAG' in df.columns

total = len(df)
print(f"전체 건수: {total:,}건\n")

# ── 강종 비교 ──────────────────────────────────────────
# 담당자가 값을 입력한 것만 비교 (빈값은 제외)
grade_has_ans = df['_강종_ans'] != ''
grade_df = df[grade_has_ans].copy()
grade_total = len(grade_df)

grade_match = (grade_df['_강종_ans'] == grade_df['_강종_sys']).sum()
grade_acc = grade_match / grade_total * 100 if grade_total else 0

print("=" * 50)
print("[ 강종 정확도 ]")
print("=" * 50)
print(f"  담당자 입력 건수:  {grade_total:,}건")
print(f"  일치:              {grade_match:,}건")
print(f"  불일치:            {grade_total - grade_match:,}건")
print(f"  정확도:            {grade_acc:.1f}%")
print()

# 분류방법별 강종 정확도
print("  [분류방법별]")
for method in ['rule', 'rag', 'llm', 'unclassified']:
    sub = grade_df[grade_df['분류방법'] == method]
    if len(sub) == 0:
        continue
    m = (sub['_강종_ans'] == sub['_강종_sys']).sum()
    print(f"    {method:15s}: {len(sub):5,}건 중 {m:5,}건 일치 ({m/len(sub)*100:.1f}%)")
print()

# ── 사이즈 비교 ──────────────────────────────────────────
if not has_size:
    print("[ 사이즈 추출 스킵 (--no-size 모드) ]\n")
    sys.exit(0)

size_has_ans = df['_사이즈_ans'] != ''
size_df = df[size_has_ans].copy()
size_total = len(size_df)

size_match = (size_df['_사이즈_ans'] == size_df['_사이즈_sys']).sum()
size_acc = size_match / size_total * 100 if size_total else 0

print("=" * 50)
print("[ 사이즈 정확도 ]")
print("=" * 50)
print(f"  담당자 입력 건수:  {size_total:,}건")
print(f"  일치:              {size_match:,}건")
print(f"  불일치:            {size_total - size_match:,}건")
print(f"  정확도:            {size_acc:.1f}%")
print()

# 사이즈_방법별 정확도
print("  [추출방법별]")
for method in ['exact', 'regex', 'llm', 'failed']:
    sub = size_df[size_df['사이즈_방법'] == method]
    if len(sub) == 0:
        continue
    m = (sub['_사이즈_ans'] == sub['_사이즈_sys']).sum()
    print(f"    {method:15s}: {len(sub):5,}건 중 {m:5,}건 일치 ({m/len(sub)*100:.1f}%)")
print()

# ── 불일치 샘플 출력 ──────────────────────────────────────────
print("=" * 50)
print("[ 사이즈 불일치 샘플 (상위 30건) ]")
print("=" * 50)
size_wrong = size_df[size_df['_사이즈_ans'] != size_df['_사이즈_sys']].copy()
for i, row in size_wrong.head(30).iterrows():
    method = row.get('사이즈_방법', '')
    print(f"  [{method}] ans={row['_사이즈_ans']!r}  sys={row['_사이즈_sys']!r}")
    print(f"    규격: {str(row.get('규격',''))[:80]}")
