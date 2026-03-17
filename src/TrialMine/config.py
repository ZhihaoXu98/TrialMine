"""Application configuration loaded from environment variables and YAML."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central config. Values come from .env, then environment, then defaults."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # External services
    anthropic_api_key: str = Field(..., description="Anthropic API key for LLM agents")
    umls_api_key: str = Field("", description="UMLS API key for concept normalization")
    elasticsearch_url: str = Field("http://localhost:9200")

    # Paths
    db_path: str = Field("data/processed/trialmine.db")
    faiss_index_path: str = Field("data/processed/faiss.index")

    # TODO: add fields for model paths, retrieval top-k values, MLflow URI, etc.
    # Load these from configs/{env}.yaml rather than duplicating here.


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    # TODO: implement lru_cache or use FastAPI Depends
    return Settings()  # type: ignore[call-arg]
