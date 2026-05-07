from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = ""

    # Oracle DB
    oracle_user: str = "RAGUSER"
    oracle_password: str = "Rag1234"
    oracle_dsn: str = "127.0.0.1:1521/xepdb1"

    # ChromaDB
    chroma_db_path: str = "./chroma_db"
    chroma_collection_name: str = "steel_history"

    # 분류 임계값
    similarity_threshold: float = 0.90   # RAG 유사도 임계값
    llm_confidence_threshold: float = 0.6  # LLM 신뢰도 임계값

    # 임베딩 모델
    embedding_model: str = "text-embedding-3-large"
    llm_model: str = "gpt-4o-mini"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
