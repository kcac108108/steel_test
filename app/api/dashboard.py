import json
import pickle
from pathlib import Path

from fastapi import APIRouter

router = APIRouter()

DATA_DIR = Path("data")
STATS_CACHE_FILE = DATA_DIR / "stats_cache.json"
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

    # ① 건수 — 캐시 파일에서 빠르게 읽기
    if STATS_CACHE_FILE.exists():
        try:
            with open(STATS_CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            stats["rulebase"]["count"]  = cache.get("rulebase_count",  0)
            stats["rag"]["count"]       = cache.get("rag_count",       0)
            stats["size_dict"]["count"] = cache.get("size_dict_count", 0)
        except Exception:
            pass
    else:
        # 캐시 없으면 최초 1회 직접 조회 후 저장
        _build_cache(stats)

    # ② 파이프라인 상태 — 파일/디렉터리 존재 여부로 빠르게 판단
    # Oracle: 캐시에 데이터가 있으면 정상으로 간주
    if stats["rulebase"]["count"] == 0 and not STATS_CACHE_FILE.exists():
        stats["pipeline"]["oracle"] = "error"
        stats["rulebase"]["status"] = "error"

    # ChromaDB: 디렉터리 존재 여부만 확인
    try:
        from app.core.config import settings
        chroma_path = Path(settings.chroma_db_path)
        if not chroma_path.exists() or not any(chroma_path.iterdir()):
            stats["pipeline"]["chromadb"] = "error"
            stats["rag"]["status"] = "error"
    except Exception:
        stats["pipeline"]["chromadb"] = "error"
        stats["rag"]["status"] = "error"

    # size_lookup.pkl: 파일 존재 여부만 확인
    if not Path("size_lookup.pkl").exists():
        stats["pipeline"]["size_dict"] = "empty"
        stats["size_dict"]["status"] = "empty"

    # ③ 갱신 이력
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                stats["update_history"] = json.load(f)
    except Exception:
        pass

    # ④ 마지막 분류 통계
    try:
        if LAST_CLASSIFY_FILE.exists():
            with open(LAST_CLASSIFY_FILE, "r", encoding="utf-8") as f:
                stats["last_classify"] = json.load(f)
    except Exception:
        pass

    return stats


def _build_cache(stats: dict):
    """캐시 파일이 없을 때 최초 1회 직접 조회 후 저장"""
    import os
    os.environ["ANONYMIZED_TELEMETRY"] = "False"

    cache = {}

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
        cache["rulebase_count"] = cursor.fetchone()[0]
        stats["rulebase"]["count"] = cache["rulebase_count"]
        cursor.close()
        conn.close()
    except Exception:
        stats["pipeline"]["oracle"] = "error"
        stats["rulebase"]["status"] = "error"

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
        cache["rag_count"] = col.count()
        stats["rag"]["count"] = cache["rag_count"]
    except Exception:
        stats["pipeline"]["chromadb"] = "error"
        stats["rag"]["status"] = "error"

    try:
        pkl_path = Path("size_lookup.pkl")
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                lookup = pickle.load(f)
            cache["size_dict_count"] = len(lookup)
            stats["size_dict"]["count"] = cache["size_dict_count"]
    except Exception:
        pass

    try:
        DATA_DIR.mkdir(exist_ok=True)
        with open(STATS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
