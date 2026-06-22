"""
데이터 갱신 API

POST /api/update/start              - 검토 완료 파일 업로드 후 갱신 시작
GET  /api/update/stream/{job_id}    - SSE 진행 상황 스트림
GET  /api/update/conflicts/{job_id} - 룰베이스 충돌 목록 조회
POST /api/update/conflicts/{job_id}/{idx}/apply - 충돌 패턴 신규 강종으로 적용
POST /api/update/conflicts/{job_id}/{idx}/skip  - 충돌 패턴 무시
"""

import asyncio
import json
import queue
import re
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

router = APIRouter()

_jobs: dict = {}
_queues: dict = {}

MIN_PATTERN_LEN = 15
MIN_CONSISTENCY = 95.0


def _emit(job_id: str, event: dict):
    if job_id in _queues:
        _queues[job_id].put(event)


def _extract_model_pattern(spec_text: str):
    m = re.search(r"(MODEL:\s*\S+)", str(spec_text), re.IGNORECASE)
    if m:
        p = m.group(1).strip().upper().rstrip(",;")
        return p if p else None
    return None


@router.post("/start")
async def start_update(
    file: UploadFile = File(...),
    do_rag: str = Form("true"),
    do_rulebase: str = Form("true"),
    do_size: str = Form("true"),
):
    job_id = uuid.uuid4().hex[:8]
    _queues[job_id] = queue.Queue()
    _jobs[job_id] = {"status": "running"}

    # 확정 데이터는 confirmed/ 폴더에 영구 저장 (동일 파일명 덮어쓰기)
    confirmed_dir = Path("confirmed")
    confirmed_dir.mkdir(exist_ok=True)
    save_path = str(confirmed_dir / file.filename)
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    flag_rag = do_rag.lower() not in ("false", "0")
    flag_rulebase = do_rulebase.lower() not in ("false", "0")
    flag_size = do_size.lower() not in ("false", "0")

    def run():
        try:
            import os as _os
            _os.environ["ANONYMIZED_TELEMETRY"] = "False"
            import logging
            logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

            import pandas as pd
            import oracledb
            from app.core.config import settings
            from app.services.rag_service import RAGService
            from app.models.schemas import HistoryRecord

            df = pd.read_excel(save_path, dtype=str)

            # 컬럼명 정규화
            if "규격" not in df.columns or "강종" not in df.columns:
                _emit(job_id, {"type": "error", "message": "'규격' 또는 '강종' 컬럼이 없습니다."})
                return

            df["규격"] = df["규격"].fillna("").astype(str).str.strip()
            df["강종"] = df["강종"].fillna("").astype(str).str.strip()
            df = df[(df["규격"] != "") & (df["강종"] != "") & (df["강종"] != "0")]
            df = df.drop_duplicates(subset=["규격"], keep="last")

            if "사이즈" in df.columns:
                df["사이즈"] = df["사이즈"].fillna("").astype(str).str.strip()
            else:
                df["사이즈"] = ""

            total = len(df)
            _emit(job_id, {"type": "start", "total": total, "filename": file.filename})

            summary = {}

            # ① RAG 갱신
            if flag_rag:
                _emit(job_id, {"type": "rag_start", "total": total})
                records = [
                    HistoryRecord(spec_text=row["규격"], steel_grade=row["강종"], size=row["사이즈"])
                    for _, row in df.iterrows()
                ]

                def rag_progress(e):
                    _emit(job_id, e)

                rag = RAGService()
                stats = rag.index_history(records, insert_only=True, progress_cb=rag_progress)
                summary["rag"] = stats
                _emit(job_id, {"type": "rag_done", "added": stats["added"], "skipped": stats["skipped"], "total": stats["total"]})

            # ② 룰베이스 갱신
            if flag_rulebase:
                _emit(job_id, {"type": "rulebase_start"})
                df_rule = df.copy()
                df_rule["model_pattern"] = df_rule["규격"].apply(_extract_model_pattern)
                df_rule = df_rule[df_rule["model_pattern"].notna()]

                inserted = 0

                if not df_rule.empty:
                    # 일관성 분석
                    rows = []
                    for pattern, group in df_rule.groupby("model_pattern"):
                        grades = group["강종"]
                        total_p = len(grades)
                        best_grade = grades.mode().iloc[0]
                        best_count = int((grades == best_grade).sum())
                        consistency = round(best_count / total_p * 100, 1)
                        rows.append({"pattern": pattern, "best_grade": best_grade,
                                     "best_count": best_count, "total": total_p, "consistency": consistency})
                    import pandas as pd2
                    consistent_df = pd2.DataFrame(rows)
                    consistent_df = consistent_df[consistent_df["consistency"] >= MIN_CONSISTENCY]

                    if not consistent_df.empty:
                        conn = oracledb.connect(
                            user=settings.oracle_user,
                            password=settings.oracle_password,
                            dsn=settings.oracle_dsn,
                        )
                        cursor = conn.cursor()

                        # 기존 패턴 조회 (신규만 INSERT, 기존은 스킵)
                        cursor.execute("SELECT UPPER(pattern) FROM rule_base")
                        existing = {row[0] for row in cursor.fetchall()}

                        # 신규 패턴만 INSERT
                        to_insert = consistent_df[
                            (~consistent_df["pattern"].str.upper().isin(existing)) &
                            (consistent_df["pattern"].str.len() >= MIN_PATTERN_LEN)
                        ]

                        if not to_insert.empty:
                            cursor.execute(
                                "SELECT NVL(MAX(TO_NUMBER(REGEXP_SUBSTR(rule_id, '[0-9]+'))), 0) "
                                "FROM rule_base WHERE REGEXP_LIKE(rule_id, '^[A-Z]+[0-9]+$')"
                            )
                            max_num = int(cursor.fetchone()[0] or 0)
                            prefix = "MC"

                            insert_data = [
                                (f"{prefix}{max_num + i + 1:06d}", r["pattern"], r["best_grade"], None)
                                for i, (_, r) in enumerate(to_insert.iterrows())
                            ]
                            cursor.executemany(
                                "INSERT INTO rule_base (rule_id, pattern, steel_grade, size_val) VALUES (:1, :2, :3, :4)",
                                insert_data,
                            )
                            conn.commit()
                            inserted = len(insert_data)

                        cursor.close()
                        conn.close()

                summary["rulebase"] = {"inserted": inserted}
                _emit(job_id, {"type": "rulebase_done", "inserted": inserted})

            # ③ 사이즈 사전 갱신
            if flag_size:
                _emit(job_id, {"type": "size_start"})
                import pickle
                CACHE_PATH = "size_lookup.pkl"

                cache: dict = {}
                if Path(CACHE_PATH).exists():
                    with open(CACHE_PATH, "rb") as f:
                        cache = pickle.load(f)

                size_added = 0
                for _, row in df.iterrows():
                    spec_key = str(row["규격"]).strip().upper()
                    size_val = str(row["사이즈"]).strip()
                    if spec_key and size_val and size_val not in ("", "0", "0.0"):
                        if spec_key not in cache:
                            size_added += 1
                        cache[spec_key] = size_val

                with open(CACHE_PATH, "wb") as f:
                    pickle.dump(cache, f)

                summary["size"] = {"added": size_added, "total": len(cache)}
                _emit(job_id, {"type": "size_done", "added": size_added, "total": len(cache)})

            # 갱신 이력 저장 (대시보드용)
            try:
                from datetime import datetime
                history_file = Path("data") / "update_history.json"
                history_file.parent.mkdir(exist_ok=True)
                history = []
                if history_file.exists():
                    with open(history_file, "r", encoding="utf-8") as _hf:
                        history = json.load(_hf)
                history.append({
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "filename": file.filename,
                    "rag_added":  summary.get("rag", {}).get("added", 0),
                    "rule_added": summary.get("rulebase", {}).get("inserted", 0),
                    "size_added": summary.get("size", {}).get("added", 0),
                })
                with open(history_file, "w", encoding="utf-8") as _hf:
                    json.dump(history, _hf, ensure_ascii=False, indent=2)
            except Exception:
                pass

            # 통계 캐시 저장 (대시보드 빠른 로딩용)
            try:
                cache_file = Path("data") / "stats_cache.json"
                existing_cache = {}
                if cache_file.exists():
                    with open(cache_file, "r", encoding="utf-8") as _cf:
                        existing_cache = json.load(_cf)
                new_cache = dict(existing_cache)

                if flag_rulebase:
                    conn2 = oracledb.connect(
                        user=settings.oracle_user,
                        password=settings.oracle_password,
                        dsn=settings.oracle_dsn,
                    )
                    cur2 = conn2.cursor()
                    cur2.execute("SELECT COUNT(*) FROM rule_base")
                    new_cache["rulebase_count"] = cur2.fetchone()[0]
                    cur2.close()
                    conn2.close()

                if flag_rag and "rag" in dir():
                    new_cache["rag_count"] = rag._collection.count()

                if flag_size:
                    new_cache["size_dict_count"] = summary.get("size", {}).get("total", existing_cache.get("size_dict_count", 0))

                with open(cache_file, "w", encoding="utf-8") as _cf:
                    json.dump(new_cache, _cf, ensure_ascii=False, indent=2)
            except Exception:
                pass

            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["summary"] = summary
            _emit(job_id, {"type": "done", "summary": summary})

        except Exception as e:
            _jobs[job_id]["status"] = "error"
            _emit(job_id, {"type": "error", "message": str(e)})

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id}


@router.get("/stream/{job_id}")
async def stream_progress(job_id: str):
    if job_id not in _queues:
        raise HTTPException(404, "작업을 찾을 수 없습니다.")

    q = _queues[job_id]

    async def generate():
        loop = asyncio.get_event_loop()

        async def next_event():
            try:
                return await loop.run_in_executor(None, lambda: q.get(timeout=1.0))
            except queue.Empty:
                return None

        while True:
            event = await next_event()
            if event is None:
                yield "data: {\"type\":\"ping\"}\n\n"
                continue
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event.get("type") in ("done", "error"):
                break

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


