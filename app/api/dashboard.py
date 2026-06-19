import json
import pickle
from pathlib import Path

from fastapi import APIRouter

router = APIRouter()

DATA_DIR = Path("data")
HISTORY_FILE = DATA_DIR / "update_history.json"
LAST_CLASSIFY_FILE = DATA_DIR / "last_classify_stats.json"


@router.get("/stats")
async def get_stats():
    import os
    os.environ["ANONYMIZED_TELEMETRY"] = "False"

    stats = {
        "rulebase":  {"count": 0, "status": "ok"},
        "rag":       {"count": 0, "status": "ok"},
        "size_dict": {"count": 0, "status": "ok"},
        "pipeline": {"oracle": "ok", "chromadb": "ok", "llm": "ok", "size_dict": "ok"},
        "update_history": [],
        "last_classify": None,
    }

    # Oracle 룰베이스 건수
    try:
        import oracledb
        from app.core.config import settings
        conn = oracledb.connect(
            user=settings.oracle_user,
            password=settings.oracle_password,
            dsn=settings.oracle_dsn,
        )
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM rule_base")
        stats["rulebase"]["count"] = cursor.fetchone()[0]
        cursor.close()
        conn.close()
    except Exception:
        stats["rulebase"]["status"] = "error"
        stats["pipeline"]["oracle"] = "error"

    # ChromaDB 건수
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        from app.core.config import settings
        client = chromadb.PersistentClient(
            path=settings.chroma_db_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        col = client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        stats["rag"]["count"] = col.count()
    except Exception:
        stats["rag"]["status"] = "error"
        stats["pipeline"]["chromadb"] = "error"

    # size_lookup.pkl 건수
    try:
        pkl_path = Path("size_lookup.pkl")
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                cache = pickle.load(f)
            stats["size_dict"]["count"] = len(cache)
        else:
            stats["size_dict"]["status"] = "empty"
            stats["pipeline"]["size_dict"] = "empty"
    except Exception:
        stats["size_dict"]["status"] = "error"
        stats["pipeline"]["size_dict"] = "error"

    # 갱신 이력
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                stats["update_history"] = json.load(f)
    except Exception:
        pass

    # 마지막 분류 통계
    try:
        if LAST_CLASSIFY_FILE.exists():
            with open(LAST_CLASSIFY_FILE, "r", encoding="utf-8") as f:
                stats["last_classify"] = json.load(f)
    except Exception:
        pass

    return stats
