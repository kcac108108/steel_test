"""
April 데이터 대상 사이즈 추출 정확도 측정 (LLM 없이)
"""
import sys, os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import logging
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

sys.path.insert(0, '.')
import pandas as pd
from app.services.size_extractor import extract_size_regex, _get_exact_lookup

INPUT_PATH = "input/0413~14(작업원본).xlsx"
CONFIRMED_COL = "사이즈"    # 담당자 확정 사이즈
SPEC_COL = "규격"

df = pd.read_excel(INPUT_PATH, dtype=str)
print(f"  전체 로드: {len(df):,}건")

# 확정 사이즈 있는 행만
has_ans = df[CONFIRMED_COL].notna() & (df[CONFIRMED_COL].str.strip() != "") & (df[CONFIRMED_COL].str.strip() != "0")
df_test = df[has_ans].copy()
print(f"  정답 있는 행: {len(df_test):,}건")

spec_texts = df_test[SPEC_COL].fillna("").astype(str).tolist()
answers = df_test[CONFIRMED_COL].str.strip().tolist()

# 정확 매칭 사전 로드
print("  사전 로드 중...")
lookup = _get_exact_lookup()
print(f"  사전 {len(lookup):,}건")

exact_cnt = regex_cnt = failed_cnt = 0
exact_ok = regex_ok = 0

errors_regex = []  # regex가 틀린 케이스

for spec, ans in zip(spec_texts, answers):
    key = spec.strip().upper()
    if key in lookup:
        pred = lookup[key]
        exact_cnt += 1
        if pred.strip() == ans:
            exact_ok += 1
    else:
        pred = extract_size_regex(spec) or ""
        if pred:
            regex_cnt += 1
            if pred.strip() == ans:
                regex_ok += 1
            else:
                errors_regex.append((spec[:60], pred, ans))
        else:
            failed_cnt += 1

total = len(spec_texts)
correct = exact_ok + regex_ok
covered = exact_cnt + regex_cnt

print(f"\n--- 사이즈 추출 정확도 ---")
print(f"  정확매칭: {exact_cnt}건 (OK: {exact_ok}건, {exact_ok/exact_cnt*100:.1f}%)" if exact_cnt else "  정확매칭: 0건")
print(f"  정규식:   {regex_cnt}건 (OK: {regex_ok}건, {regex_ok/max(regex_cnt,1)*100:.1f}%)")
print(f"  LLM 필요: {failed_cnt}건")
print(f"  커버리지: {covered}/{total}건 ({covered/total*100:.1f}%)")
print(f"  전체정확도: {correct}/{total}건 ({correct/total*100:.1f}%)")
print(f"  정규식정확도(커버 내): {regex_ok}/{max(regex_cnt,1)}건 ({regex_ok/max(regex_cnt,1)*100:.1f}%)")

print(f"\n--- 정규식 오답 샘플 (상위 30건) ---")
print(f"{'규격':<62} {'예측':<25} {'정답':<25}")
print('-' * 115)
for spec, pred, ans in errors_regex[:30]:
    print(f"{spec:<62} {pred:<25} {ans:<25}")
