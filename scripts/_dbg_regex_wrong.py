"""regex 오답 케이스 전체 목록"""
import os; os.environ['ANONYMIZED_TELEMETRY']='False'
import sys; sys.path.insert(0,'.')
import pandas as pd
from app.services.size_extractor import extract_size_regex, _get_exact_lookup

INPUT_PATH = "input/0413~14(작업원본).xlsx"
df = pd.read_excel(INPUT_PATH, dtype=str)
has_ans = df['사이즈'].notna() & (df['사이즈'].str.strip() != '') & (df['사이즈'].str.strip() != '0')
df_test = df[has_ans].copy()
specs = df_test['규격'].fillna('').astype(str).tolist()
answers = df_test['사이즈'].str.strip().tolist()

print("  사전 로드 중...")
lookup = _get_exact_lookup()
print(f"  사전 {len(lookup):,}건")

wrong = []
none_list = []
for spec, ans in zip(specs, answers):
    key = spec.strip().upper()
    if key not in lookup:
        pred = extract_size_regex(spec) or ''
        if pred == ans:
            pass
        elif not pred:
            none_list.append((spec[:90], ans))
        else:
            wrong.append((spec[:90], pred, ans))

print(f"\nregex 오답: {len(wrong)}건")
print(f"regex None: {len(none_list)}건")
print()
print(f"{'규격':<90} {'예측':<25} {'정답':<25}")
print('-' * 145)
for spec, pred, ans in wrong:
    print(f"{spec:<90} {pred:<25} {ans:<25}")

print(f"\n\n--- None 목록 ---")
for spec, ans in none_list:
    print(f"  {spec[:80]:<80} ans={ans!r}")
