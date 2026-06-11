import os
import shutil
from pathlib import Path
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

app = FastAPI(title="철강 강종·사이즈 자동분류 시스템")
app.include_router(router, prefix="/api")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def index():
    return FileResponse("app/static/index.html")
