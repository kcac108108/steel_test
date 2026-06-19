"""
분류 실행 API

POST /api/classify/start   - 파일 업로드 후 분류 작업 시작
GET  /api/classify/stream/{job_id} - SSE 진행 상황 스트림
GET  /api/classify/download/{job_id} - 결과 파일 다운로드
"""

import asyncio
import json
import queue
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask

router = APIRouter()

_jobs: dict = {}
_queues: dict = {}


def _emit(job_id: str, event: dict):
    if job_id in _queues:
        _queues[job_id].put(event)


@router.post("/start")
async def start_classify(
    file: UploadFile = File(...),
    use_rule: str = Form("true"),
    use_rag:  str = Form("true"),
    use_llm:  str = Form("true"),
    use_size: str = Form("true"),
):
    job_id = uuid.uuid4().hex[:8]
    _queues[job_id] = queue.Queue()
    _jobs[job_id] = {"status": "running", "result_path": None}

    upload_dir = Path("input")
    upload_dir.mkdir(parents=True, exist_ok=True)
    input_path = str(upload_dir / f"{job_id}_{file.filename}")
    content = await file.read()
    with open(input_path, "wb") as f:
        f.write(content)

    output_path = str(Path("output") / f"{job_id}_분류결과.xlsx")
    orig_name = Path(file.filename).stem

    flag_rule = use_rule.lower() not in ("false", "0")
    flag_rag  = use_rag.lower()  not in ("false", "0")
    flag_llm  = use_llm.lower()  not in ("false", "0")
    flag_size = use_size.lower() not in ("false", "0")

    def run():
        try:
            import os as _os
            _os.environ["ANONYMIZED_TELEMETRY"] = "False"
            import logging
            logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

            import pandas as pd
            from app.services.classifier import SteelClassifier
            from app.services.size_extractor import SizeExtractor

            df = pd.read_excel(input_path, dtype={"번호": str, "수입신고번호": str})

            if "규격" not in df.columns:
                _emit(job_id, {"type": "error", "message": "엑셀에 '규격' 컬럼이 없습니다."})
                return

            spec_texts = df["규격"].fillna("").astype(str).tolist()
            total = len(spec_texts)
            _emit(job_id, {"type": "start", "total": total})

            classifier = SteelClassifier(use_rule=flag_rule, use_rag=flag_rag, use_llm=flag_llm)
            results = classifier.classify_batch(spec_texts, progress_cb=lambda e: _emit(job_id, e))

            def clean(v):
                s = str(v).strip() if v else ""
                return "" if s in ("", "0", "0.0", "None", "nan") else s

            df["시스템_강종"] = [clean(r.steel_grade) for r in results]
            df["강종_분류방법"] = [r.method.value for r in results]

            if flag_size:
                _emit(job_id, {"type": "step", "step": "size_start", "done": 0, "total": total})
                size_extractor = SizeExtractor()
                size_results = size_extractor.extract_batch(spec_texts)
                df["시스템_사이즈"] = [clean(size) for size, _ in size_results]
                df["사이즈_분류방법"] = [method for _, method in size_results]
            else:
                size_results = [("", "skipped")] * total

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_excel(output_path, index=False)
            _jobs[job_id]["result_path"] = output_path
            _jobs[job_id]["status"] = "done"

            Path(input_path).unlink(missing_ok=True)

            from collections import Counter
            method_counts = dict(Counter(r.method.value for r in results))
            size_counts = dict(Counter(m for _, m in size_results))

            # 마지막 분류 통계 저장 (대시보드용)
            try:
                data_dir = Path("data")
                data_dir.mkdir(exist_ok=True)
                with open(data_dir / "last_classify_stats.json", "w", encoding="utf-8") as _f:
                    json.dump({"total": total, "method_counts": method_counts, "size_counts": size_counts}, _f, ensure_ascii=False)
            except Exception:
                pass

            _emit(job_id, {
                "type": "done",
                "total": total,
                "method_counts": method_counts,
                "size_counts": size_counts,
                "download_name": f"{orig_name}_분류결과.xlsx",
            })
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


@router.get("/download/{job_id}")
async def download_result(job_id: str):
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "결과 파일이 없습니다.")
    result_path = job.get("result_path")
    if not result_path or not Path(result_path).exists():
        raise HTTPException(404, "결과 파일이 없습니다.")

    download_name = Path(result_path).name
    _jobs.pop(job_id, None)
    _queues.pop(job_id, None)

    def cleanup():
        Path(result_path).unlink(missing_ok=True)

    return FileResponse(
        result_path,
        filename=download_name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        background=BackgroundTask(cleanup),
    )
