from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    vector_size: int = 384

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "info"

    class Config:
        env_file = ".env"


settings = Settings()
