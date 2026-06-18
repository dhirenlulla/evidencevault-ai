from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for the EvidenceVault backend.

    Values are read from environment variables or the backend .env file.
    Keeping configuration in one place prevents database URLs, API keys,
    and service addresses from being hardcoded throughout the codebase.
    """

    app_name: str = "EvidenceVault AI API"
    app_version: str = "0.2.0"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"

    database_url: str = Field(
        default=(
            "postgresql+psycopg://"
            "evidencevault_user:"
            "evidencevault_local_password"
            "@localhost:5432/"
            "evidencevault_db"
        )
    )

    sql_echo: bool = False

    qdrant_url: str = "http://localhost:6333"
    qdrant_timeout_seconds: float = 5.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached Settings object.

    The first call reads and validates the configuration. Later calls reuse
    the same object instead of repeatedly loading the .env file.
    """
    return Settings()