"""Configuration du service search chargée depuis `.env`."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Paramètres d'exécution du service de recherche RAG.

    `CHROMA_PATH` doit pointer vers le même dossier que `CHROMA_PATH`
    côté intrabot-ingestion (chemin absolu recommandé en local).
    """

    cohere_api_key: str
    ingestion_service_url: str = "http://localhost:8001"
    chroma_path: str = "./data/chroma"
    chroma_collection_name: str = "intrabot"
    app_host: str = "0.0.0.0"
    app_port: int = 8002
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
