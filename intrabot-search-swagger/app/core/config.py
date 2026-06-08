from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Cohere — used by CohereLLMProvider
    cohere_api_key: str

    # intrabot-ingestion service — used by IngestionServiceEmbeddingAdapter
    ingestion_service_url: str = "http://localhost:8000"

    # ChromaDB — must point to the same instance as intrabot-ingestion
    chroma_host: str = "localhost"
    chroma_port: int = 8003

    # Service bind
    app_host: str = "0.0.0.0"
    app_port: int = 8002
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
