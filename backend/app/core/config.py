from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Central application configuration.
    
    Values can come from:
    1. Defaults defined below
    2. Environment variables
    3. A local .env file
    
    """
    
    app_name: str = "EvidenceVault AI API"
    app_version: str = "0.1.0"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
@lru_cache
def get_settings() -> Settings:
    
    """
    Return one cached Settings instance.
    
    Caching prevents the application from reading and validating
    environment configuration repeatedly.
    """
    
    return Settings()



# For understanding --->

# What this file does
# This becomes the central place for configuration.

# Later we will add:
# PostgreSQL URL
# Qdrant URL
# LLM API key
# S3 bucket
# Embedding model name
# File-size limit
# Retrieval limits

# We will not hardcode those values throughout the application.