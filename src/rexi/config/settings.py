"""
Configuration settings for REXI system.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    app_name: str = "REXI"
    app_version: str = "0.1.0"
    debug: bool = False
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    
    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "rexi"
    postgres_user: str = "rexi"
    postgres_password: str = "rexi_password"
    
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j_password"
    
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "knowledge_embeddings"
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # AI Models
    openai_api_key: Optional[str] = None
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gpt-3.5-turbo"
    
    # File Processing
    upload_dir: str = "data/uploads"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: list[str] = [
        ".pdf", ".txt", ".md", ".json", ".csv", 
        ".docx", ".doc", ".rtf", ".epub", ".mobi"
    ]
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    access_token_expire_minutes: int = 30
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/rexi.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
