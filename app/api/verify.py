"""
검증·통계 API

POST /api/verify/analyze - 파일 업로드 후 강종·사이즈 정확도/커버리지 분석
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile, os
from pathlib import Path

router = APIRouter()


@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        import pandas as pd

        content = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            df = pd.read_excel(tmp_path, dtype=str)
        finally:
            os.unlink(tmp_path)

        required = {"규격", "시스템_강종"}
        missing = required - set(df.columns)
        if missing:
            raise HTTPException(400, f"필수 컬럼 없음: {', '.join(missing)}")

        has_size = "시스템_사이즈" in df.columns
        has_hum_grade = "강종" in df.columns
        has_hum_size = "사이즈" in df.columns
        has_method = "강종_분류방법" in df.columns

        total = len(df)

        def clean(s):
            s = str(s).strip() if s and str(s) not in ("nan", "None", "") else ""
            return s

        df["_sys_grade"]  = df["시스템_강종"].apply(clean)
        df["_hum_grade"]  = df["강종"].apply(clean) if has_hum_grade else ""
        if has_method:
            df["_method"] = df["강종_분류방법"].apply(lambda s: clean(s).lower())

        # 강종 커버리지: 시스템이 분류한 건수
        grade_covered = (df["_sys_grade"] != "").sum()
        grade_coverage = round(grade_covered / total * 100, 1) if total else 0

        # 강종 정확도: 담당자도 값이 있고, 시스템도 값이 있는 건 중 일치율
        grade_base = df[(df["_sys_grade"] != "") & (df["_hum_grade"] != "")]
        grade_match = (grade_base["_sys_grade"] == grade_base["_hum_grade"]).sum()
        grade_accuracy = round(grade_match / len(grade_base) * 100, 1) if len(grade_base) else 0

        # 방법별 정확도
        method_stats = {}
        if has_method:
            for method in ["rule", "rag", "llm"]:
                m_df = df[(df["_method"].str.lower() == method) & (df["_hum_grade"] != "")]
                if len(m_df):
                    match = (m_df["_sys_grade"] == m_df["_hum_grade"]).sum()
                    method_stats[method] = {
                        "count": int(len(m_df)),
                        "match": int(match),
                        "accuracy": round(match / len(m_df) * 100, 1)
                    }

        # 사이즈
        size_stats = None
        if has_size:
            df["_sys_size"] = df["시스템_사이즈"].apply(clean)
            df["_hum_size"] = df["사이즈"].apply(clean) if has_hum_size else ""
            size_covered = (df["_sys_size"] != "").sum()
            size_coverage = round(size_covered / total * 100, 1) if total else 0
            size_base = df[(df["_sys_size"] != "") & (df["_hum_size"] != "")]
            size_match = (size_base["_sys_size"] == size_base["_hum_size"]).sum()
            size_accuracy = round(size_match / len(size_base) * 100, 1) if len(size_base) else 0
            size_stats = {
                "covered": int(size_covered),
                "coverage": size_coverage,
                "base": int(len(size_base)),
                "match": int(size_match),
                "accuracy": size_accuracy,
            }

        # 불일치 목록 전체
        all_mismatch_df = grade_base[grade_base["_sys_grade"] != grade_base["_hum_grade"]]

        # LLM 불일치 패턴 분석: 담당자 정답이 규격 텍스트에 명시되어 있는지 확인
        llm_analysis = None
        if has_method:
            llm_mismatch_df = all_mismatch_df[all_mismatch_df["_method"].str.lower() == "llm"]
            if len(llm_mismatch_df):
                def grade_in_spec(row):
                    spec = str(row["규격"]).upper()
                    # 담당자 강종에서 핵심 키워드 추출 (예: "JIS G3101 SS400" → ["SS400", "G3101"])
                    human = str(row["_hum_grade"]).upper()
                    tokens = [t for t in human.replace("-", " ").split() if len(t) >= 3]
                    return any(t in spec for t in tokens)

                llm_mismatch_df = llm_mismatch_df.copy()
                llm_mismatch_df["_in_spec"] = llm_mismatch_df.apply(grade_in_spec, axis=1)
                in_spec_count = int(llm_mismatch_df["_in_spec"].sum())
                llm_analysis = {
                    "total_mismatch": int(len(llm_mismatch_df)),
                    "answer_in_spec": in_spec_count,
                    "answer_in_spec_pct": round(in_spec_count / len(llm_mismatch_df) * 100, 1),
                    "examples": [
                        {
                            "spec":    row["규격"],
                            "system":  row["_sys_grade"],
                            "human":   row["_hum_grade"],
                            "in_spec": bool(row["_in_spec"]),
                        }
                        for _, row in llm_mismatch_df[llm_mismatch_df["_in_spec"]].head(20).iterrows()
                    ]
                }

        mismatches = [
            {
                "spec":   row["규격"],
                "system": row["_sys_grade"],
                "human":  row["_hum_grade"],
                "method": clean(row.get("강종_분류방법", "")),
            }
            for _, row in all_mismatch_df.head(100).iterrows()
        ]

        return JSONResponse({
            "total": total,
            "grade": {
                "covered":  int(grade_covered),
                "coverage": grade_coverage,
                "base":     int(len(grade_base)),
                "match":    int(grade_match),
                "accuracy": grade_accuracy,
            },
            "size": size_stats,
            "methods": method_stats,
            "mismatches": mismatches,
            "llm_analysis": llm_analysis,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
