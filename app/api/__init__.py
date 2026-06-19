from fastapi import APIRouter
from app.api import classify, update, verify, dashboard

router = APIRouter()
router.include_router(dashboard.router, prefix="/dashboard", tags=["대시보드"])
router.include_router(classify.router, prefix="/classify", tags=["분류"])
router.include_router(update.router, prefix="/update", tags=["갱신"])
router.include_router(verify.router, prefix="/verify", tags=["검증"])
