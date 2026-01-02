# =============================================================================
# Configuration Settings for RAG System
# =============================================================================
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    debug: bool = False

    # LLM Settings
    llm_provider: str = "anthropic"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_model: str = "claude-3-5-sonnet-20241022"
    llm_base_url: str = ""

    # Embedding Settings
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Qdrant Settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str = ""

    # PostgreSQL Settings
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "ragdb"
    postgres_user: str = "raguser"
    postgres_password: str = ""

    # Redis Settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""

    # MinIO Settings
    minio_host: str = "localhost"
    minio_port: int = 9000
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "documents"

    # Indexing Settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_tree_depth: int = 5

    # Retrieval Settings
    hybrid_alpha: float = 0.7
    top_k_documents: int = 5
    top_k_sections: int = 10
    top_k_chunks: int = 20

    # Cache Settings
    cache_embedding_ttl: int = 3600
    cache_retrieval_ttl: int = 1800
    cache_semantic_ttl: int = 3600
    semantic_cache_threshold: float = 0.85

    # Verification Settings
    groundedness_threshold: float = 0.85
    enable_tier2_verification: bool = True

    @property
    def postgres_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def redis_url(self) -> str:
        """Get Redis connection URL."""
        return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/0"

    @property
    def qdrant_url(self) -> str:
        """Get Qdrant connection URL."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
