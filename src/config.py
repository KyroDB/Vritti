"""
Configuration management for Episodic Memory service.

Uses Pydantic Settings for type-safe configuration with multiple sources:
- Environment variables (highest priority)
- .env file
- Defaults (lowest priority)
"""

from typing import Literal, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class KyroDBConnection(BaseSettings):
    """KyroDB instance connection configuration."""

    host: str = Field(default="localhost", description="KyroDB host address")
    port: int = Field(default=50051, ge=1, le=65535, description="KyroDB gRPC port")

    @property
    def address(self) -> str:
        """Full connection address for gRPC channel."""
        return f"{self.host}:{self.port}"


class KyroDBConfig(BaseSettings):
    """KyroDB dual-instance configuration for multi-modal embeddings."""

    model_config = SettingsConfigDict(env_prefix="KYRODB_")

    text_host: str = Field(default="localhost")
    text_port: int = Field(default=50051, ge=1, le=65535)
    image_host: str = Field(default="localhost")
    image_port: int = Field(default=50052, ge=1, le=65535)

    # Connection pooling
    max_workers: int = Field(default=10, ge=1, le=100)
    connection_timeout_seconds: int = Field(default=30, ge=1)
    request_timeout_seconds: int = Field(default=60, ge=1)

    @property
    def text_address(self) -> str:
        return f"{self.text_host}:{self.text_port}"

    @property
    def image_address(self) -> str:
        return f"{self.image_host}:{self.image_port}"


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    # Text/code embeddings (sentence-transformers)
    text_model_name: str = Field(default="all-MiniLM-L6-v2")
    text_dimension: int = Field(default=384, ge=1)
    text_batch_size: int = Field(default=32, ge=1)

    # Image embeddings (CLIP)
    image_model_name: str = Field(default="openai/clip-vit-base-patch32")
    image_dimension: int = Field(default=512, ge=1)
    image_batch_size: int = Field(default=16, ge=1)

    # Device allocation
    device: Literal["cpu", "cuda", "mps"] = Field(default="cpu")
    enable_gpu: bool = Field(default=False)

    @field_validator("text_dimension", "image_dimension")
    @classmethod
    def validate_dimension(cls, v: int) -> int:
        if v <= 0 or v > 10000:
            raise ValueError(f"Embedding dimension must be 1-10000, got {v}")
        return v


class LLMConfig(BaseSettings):
    """LLM configuration for reflection generation."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    api_key: str = Field(default="", description="OpenAI API key")
    model_name: str = Field(default="gpt-4-turbo-preview")
    max_tokens: int = Field(default=2000, ge=100, le=8000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout_seconds: int = Field(default=30, ge=1)

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v or v == "your-openai-api-key-here":
            # Don't fail, just warn - reflection will be disabled
            return ""
        return v


class HygieneConfig(BaseSettings):
    """Hygiene policy configuration for decay and promotion."""

    model_config = SettingsConfigDict(env_prefix="HYGIENE_")

    # Decay policies
    decay_check_interval_hours: int = Field(default=168, ge=1)  # Weekly
    archive_age_days: int = Field(default=180, ge=1)  # 6 months
    delete_unused_age_days: int = Field(default=90, ge=1)  # 3 months
    min_retrieval_count_to_keep: int = Field(default=1, ge=0)

    # Promotion (DBSCAN clustering)
    enable_promotion: bool = Field(default=True)
    promotion_interval_hours: int = Field(default=720, ge=1)  # Monthly
    cluster_min_samples: int = Field(default=3, ge=2)
    cluster_eps: float = Field(default=0.3, ge=0.0, le=1.0)


class SearchConfig(BaseSettings):
    """Search and retrieval configuration."""

    model_config = SettingsConfigDict(env_prefix="SEARCH_")

    default_k: int = Field(default=20, ge=1, le=1000)
    max_k: int = Field(default=100, ge=1, le=1000)
    precondition_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    similarity_threshold: float = Field(default=0.6, ge=-1.0, le=1.0)

    # Client-side filtering limits (fetch more, filter down)
    fetch_multiplier: int = Field(default=5, ge=1, le=10)

    @field_validator("max_k")
    @classmethod
    def validate_max_k(cls, v: int, info) -> int:
        default_k = info.data.get("default_k", 20)
        if v < default_k:
            raise ValueError(f"max_k ({v}) must be >= default_k ({default_k})")
        return v


class ServiceConfig(BaseSettings):
    """FastAPI service configuration."""

    model_config = SettingsConfigDict(env_prefix="SERVICE_")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    reload: bool = Field(default=False)
    workers: int = Field(default=1, ge=1)

    # Storage paths
    screenshot_storage_path: str = Field(default="./data/screenshots")
    archive_storage_path: str = Field(default="./data/archive")


class Settings(BaseSettings):
    """Root configuration for Episodic Memory service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    kyrodb: KyroDBConfig = Field(default_factory=KyroDBConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    hygiene: HygieneConfig = Field(default_factory=HygieneConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)

    def validate_configuration(self) -> None:
        """
        Validate cross-field constraints and log warnings.
        Called at application startup.
        """
        # Check LLM API key
        if not self.llm.api_key:
            import logging

            logging.warning(
                "LLM API key not configured - reflection generation will be disabled"
            )

        # Validate embedding dimensions match KyroDB instances
        if self.embedding.text_dimension not in {128, 256, 384, 512, 768, 1536}:
            logging.warning(
                f"Non-standard text embedding dimension: {self.embedding.text_dimension}"
            )

        if self.embedding.image_dimension not in {512, 768, 1024}:
            logging.warning(
                f"Non-standard image embedding dimension: {self.embedding.image_dimension}"
            )


# Global settings instance (lazy-loaded)
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get global settings instance (singleton pattern).

    Returns:
        Settings: Application configuration
    """
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.validate_configuration()
    return _settings
