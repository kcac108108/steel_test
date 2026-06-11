from fastapi import APIRouter
from app.api import classify, update, verify

router = APIRouter()
router.include_router(classify.router, prefix="/classify", tags=["분류"])
router.include_router(update.router, prefix="/update", tags=["갱신"])
router.include_router(verify.router, prefix="/verify", tags=["검증"])
