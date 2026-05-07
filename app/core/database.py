import oracledb
from contextlib import contextmanager
from app.core.config import settings


def get_connection():
    """Oracle DB 연결 반환"""
    conn = oracledb.connect(
        user=settings.oracle_user,
        password=settings.oracle_password,
        dsn=settings.oracle_dsn,
    )
    return conn


@contextmanager
def db_cursor():
    """커서 컨텍스트 매니저"""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()
