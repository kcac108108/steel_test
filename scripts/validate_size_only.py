"""
사이즈 추출만 빠르게 검증 (classify.py 실행 없이)
입력 파일에서 직접 size_extractor 돌려서 담당자 답과 비교
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from app.services.size_extractor import extract_size_regex, _get_exact_lookup

INPUT_FILE = 'input/0501~6.xlsx'

df = pd.read_excel(INPUT_FILE, dtype=str)

def clean(v):
    return str(v).strip() if pd.notna(v) else ''

specs   = df['규격'].apply(clean).tolist()
answers = df['사이즈'].apply(clean).tolist()

lookup = _get_exact_lookup()

total = match = 0
method_stats = {'exact': [0,0], 'regex': [0,0], 'failed': [0,0]}
wrong_samples = []

for spec, ans in zip(specs, answers):
    if not ans or ans == '0':
        continue
    total += 1

    spec_upper = spec.strip().upper()
    if spec_upper in lookup:
        pred = lookup[spec_upper]
        method = 'exact'
    else:
        pred = extract_size_regex(spec) or ''
        method = 'regex' if pred else 'failed'

    ok = (pred == ans)
    method_stats[method][0] += 1
    if ok:
        match += 1
        method_stats[method][1] += 1
    elif len(wrong_samples) < 30:
        wrong_samples.append((method, ans, pred, spec))

acc = match / total * 100 if total else 0
print(f"전체 건수: {total:,}건")
print(f"일치:      {match:,}건")
print(f"불일치:    {total - match:,}건")
print(f"정확도:    {acc:.1f}%")
print()
print("[추출방법별]")
for m, (cnt, hit) in method_stats.items():
    if cnt:
        print(f"  {m:8s}: {cnt:5,}건 중 {hit:5,}건 일치 ({hit/cnt*100:.1f}%)")
print()
print("[ 불일치 샘플 (상위 30건) ]")
for method, ans, pred, spec in wrong_samples:
    print(f"  [{method}] ans={ans!r}  sys={pred!r}")
    print(f"    {spec[:80]!r}")
