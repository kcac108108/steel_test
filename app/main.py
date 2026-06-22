import os
import shutil
import threading
from pathlib import Path
from contextlib import asynccontextmanager
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import logging
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api import router

for _dir in ("input", "output"):
    _p = Path(_dir)
    if _p.exists():
        shutil.rmtree(_p)
    _p.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 캐시 없으면 백그라운드에서 미리 빌드 (첫 대시보드 로딩 빠르게)
    cache_file = Path("data") / "stats_cache.json"
    if not cache_file.exists():
        def _build():
            from app.api.dashboard import _build_cache
            _build_cache({})
        threading.Thread(target=_build, daemon=True).start()
    yield


app = FastAPI(title="철강 강종·사이즈 자동분류 시스템", lifespan=lifespan)
app.include_router(router, prefix="/api")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def index():
    return FileResponse("app/static/index.html")
