from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# This file is located at:
# backend/app/core/config.py
#
# parents[0] -> core
# parents[1] -> app
# parents[2] -> backend
BACKEND_DIRECTORY = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    """
    Central configuration for the EvidenceVault backend.

    Values are read from environment variables or the backend .env file.
    """

    app_name: str = "EvidenceVault AI API"
    app_version: str = "0.3.0"
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
    
    qdrant_collection_name : str = (
        "evidencevault_chunks"
    )
    
    qdrant_vector_distance: str = "Cosine"
    
    embedding_model_name: str = (
        "BAAI/bge-small-en-v1.5"
    )

    embedding_dimension: int = Field(
        default=384,
        ge=1,
    )

    embedding_batch_size: int = Field(
        default=16,
        ge=1,
        le=256,
    )

    embedding_device: str = "cpu"

    embedding_normalize: bool = True

    embedding_query_instruction: str = (
        "Represent this sentence for searching "
        "relevant passages: "
    )
    
    upload_directory: str = "uploads"
    max_upload_size_mb: int = 20
    upload_chunk_size_bytes: int = 1_048_576
    
    
    llm_provider: str = "groq"

    groq_api_key: str = ""

    groq_model_name: str = (
        "llama-3.3-70b-versatile"
    )

    llm_temperature: float = 0.0

    llm_max_tokens: int = 1024

    retrieval_top_k: int = 5
    
    rrf_k: int = Field(
        default=60,
        ge=1
    )
    
    llm_timeout_seconds: float = 60.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    
    @property
    def upload_path(self) -> Path:
        """ 
        Return the absolute upload-directory path.
        
        A relative value such as "uploads" is resolved from the
        backend directory rather than the terminal's current location.
        """
        
        configured_path = Path(self.upload_directory)
        
        if configured_path.is_absolute():
            return configured_path.resolve()
        
        return(
            BACKEND_DIRECTORY / configured_path
        ).resolve()
        
        
    @property
    def max_upload_size_bytes(self) -> int:
        """
        Convert the configured size from megabytes to bytes.
        """
            
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """
    Load and validate configuration once and reuse it.
    """
    return Settings()