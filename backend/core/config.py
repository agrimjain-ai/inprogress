from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # PostgreSQL
    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str = "postgres"
    postgres_port: int = 5432

    # Neo4j
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    # Qdrant
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333

    # LLM
    groq_api_key: str
    gemini_api_key: str

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()