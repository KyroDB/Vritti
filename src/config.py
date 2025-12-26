"""
Configuration management for Episodic Memory service.

Uses Pydantic Settings for type-safe configuration with multiple sources:
- Environment variables (highest priority)
- .env file
- Defaults (lowest priority)
"""

import logging
from typing import Literal

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

    model_config = SettingsConfigDict(
        env_prefix="KYRODB_",
        env_file=".env",
        env_file_encoding="utf-8"
    )

    text_host: str = Field(default="localhost")
    text_port: int = Field(default=50051, ge=1, le=65535)
    image_host: str = Field(default="localhost")
    image_port: int = Field(default=50052, ge=1, le=65535)

    # Connection pooling
    max_workers: int = Field(default=10, ge=1, le=100)
    connection_timeout_seconds: int = Field(default=30, ge=1)
    request_timeout_seconds: int = Field(default=60, ge=1)

    # TLS/SSL configuration
    enable_tls: bool = Field(
        default=False, description="Enable TLS for KyroDB connections (required for production)"
    )
    tls_ca_cert_path: str | None = Field(
        default=None,
        description="Path to CA certificate for server verification (None = system CA bundle)",
    )
    tls_client_cert_path: str | None = Field(
        default=None, description="Path to client certificate for mutual TLS (optional)"
    )
    tls_client_key_path: str | None = Field(
        default=None, description="Path to client private key for mutual TLS (optional)"
    )
    tls_verify_server: bool = Field(
        default=True,
        description="Verify server certificate (disable only for self-signed certs in dev)",
    )

    @property
    def text_address(self) -> str:
        return f"{self.text_host}:{self.text_port}"

    @property
    def image_address(self) -> str:
        return f"{self.image_host}:{self.image_port}"

    @field_validator("tls_client_cert_path")
    @classmethod
    def validate_tls_cert_path(cls, v: str | None, info) -> str | None:
        """Validate that if client cert is provided, client key must also be provided."""
        if v is not None:
            values = info.data
            if "tls_client_key_path" in values and values["tls_client_key_path"] is None:
                raise ValueError(
                    "tls_client_key_path must be provided when tls_client_cert_path is set (mutual TLS)"
                )
        return v


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=".env",
        env_file_encoding="utf-8"
    )

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
    """
    LLM configuration for multi-perspective reflection generation.

    Uses OpenRouter as unified API gateway for all LLM providers.
    Supports 2-model consensus + 1 cheap tier model.

    Security: API keys are never logged or exposed in errors.
    """

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8"
    )

    # OpenRouter configuration (primary)
    openrouter_api_key: str = Field(
        default="",
        description="OpenRouter API key for unified LLM access"
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL"
    )

    # Consensus models (2 models for multi-perspective reflection)
    consensus_model_1: str = Field(
        default="kwaipilot/kat-coder-pro:free",
        description="First consensus model (structured extraction specialist)"
    )
    consensus_model_2: str = Field(
        default="tngtech/deepseek-r1t2-chimera:free",
        description="Second consensus model (deep reasoning)"
    )

    # Cheap tier model
    cheap_model: str = Field(
        default="x-ai/grok-4.1-fast:free",
        description="Cheap tier model for fast, cost-effective single reflections"
    )

   

    # Shared parameters
    max_tokens: int = Field(
        default=2000,
        ge=100,
        le=8000,
        description="Maximum tokens for reflection generation"
    )

    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperature for reflection (lower = more consistent)"
    )

    timeout_seconds: int = Field(
        default=60,
        ge=1,
        le=180,
        description="Timeout for LLM API calls"
    )

    # Cost limits (security: prevent abuse)
    max_cost_per_reflection_usd: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Maximum allowed cost per reflection (0.0 for free tier)"
    )

    # Retry configuration
    max_retries: int = Field(
        default=3,
        ge=0,
        le=5,
        description="Maximum retries for transient failures"
    )

    retry_backoff_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Initial backoff for exponential retry"
    )

    # Backward compatibility (legacy single API key)
    api_key: str = Field(
        default="",
        description="Legacy OpenAI API key (deprecated, use openrouter_api_key)"
    )


    @field_validator("openrouter_api_key", "api_key")
    @classmethod
    def validate_api_key_security(cls, v: str, info) -> str:
        """
        Security: Validate API key format and prevent common mistakes.

        Never expose API keys in logs or errors.
        """
        if not v:
            return ""

        placeholder_patterns = [
            "your-api-key-here",
            "example",
            "dummy"
        ]

        v_lower = v.lower()
        if any(pattern in v_lower for pattern in placeholder_patterns):
            field_name = info.field_name
            logging.warning(
                f"{field_name} appears to be a placeholder - reflection may be disabled"
            )
            return ""

        if len(v) < 20:
            field_name = info.field_name
            logging.warning(
                f"{field_name} seems too short to be valid - reflection may fail"
            )

        return v

    @property
    def has_any_api_key(self) -> bool:
        """Check if at least one LLM provider is configured."""
        return bool(
            self.openrouter_api_key or
            self.api_key  # Legacy fallback
        )

    @property
    def use_openrouter(self) -> bool:
        """Check if OpenRouter should be used (preferred over direct providers)."""
        return bool(self.openrouter_api_key)

    @property
    def enabled_providers(self) -> list[str]:
        """Get list of enabled LLM providers."""
        providers = []
        if self.openrouter_api_key:
            providers.append("openrouter")
        return providers


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

    # LLM semantic validation
    enable_llm_validation: bool = Field(
        default=True,
        description="Enable LLM semantic validation for preconditions to prevent false blocks in context-sensitive scenarios"
    )
    llm_similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Similarity threshold to trigger LLM validation (only validate high-similarity matches)"
    )
    llm_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum LLM confidence score to accept a match"
    )
    llm_timeout_seconds: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Timeout for LLM validation calls in seconds"
    )
    llm_cache_ttl_seconds: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Time-to-live for LLM validation cache entries (5 minutes default)"
    )
    max_llm_cost_per_day_usd: float = Field(
        default=10.0,
        ge=0.1,
        le=1000.0,
        description="Maximum daily cost for LLM validation (circuit breaker)"
    )

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

    # Request size limits (DoS protection)
    max_request_body_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        ge=1024,
        description="Maximum request body size in bytes (default: 10MB)",
    )
    max_file_upload_size: int = Field(
        default=5 * 1024 * 1024,  # 5MB
        ge=1024,
        description="Maximum file upload size in bytes (default: 5MB)",
    )
    request_timeout_seconds: int = Field(
        default=30, ge=1, le=300, description="Global request timeout in seconds"
    )

    # Storage paths
    screenshot_storage_path: str = Field(default="./data/screenshots")
    archive_storage_path: str = Field(default="./data/archive")
    dead_letter_queue_path: str = Field(
        default="./data/failed_reflections.log",
        description="Path to dead letter queue file for failed reflections"
    )
    
    # Dead letter queue file rotation settings
    dead_letter_queue_max_size_mb: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum size of dead letter queue file in MB before rotation"
    )


class CORSConfig(BaseSettings):
    """CORS configuration for API security."""

    model_config = SettingsConfigDict(env_prefix="CORS_")

    # Allowed origins (comma-separated for multiple origins)
    allowed_origins: str = Field(
        default="*", description="Comma-separated list of allowed origins (* for all, ONLY for dev)"
    )
    allow_credentials: bool = Field(
        default=True, description="Allow credentials (cookies, auth headers)"
    )
    allowed_methods: str = Field(
        default="GET,POST,PUT,DELETE,PATCH,OPTIONS",
        description="Comma-separated list of allowed HTTP methods",
    )
    allowed_headers: str = Field(
        default="*", description="Comma-separated list of allowed headers (* for all)"
    )
    max_age: int = Field(default=600, ge=0, description="Preflight cache duration in seconds")

    @property
    def origins_list(self) -> list[str]:
        """Parse comma-separated origins into list."""
        if self.allowed_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    @property
    def methods_list(self) -> list[str]:
        """Parse comma-separated methods into list."""
        return [method.strip() for method in self.allowed_methods.split(",") if method.strip()]

    @property
    def headers_list(self) -> list[str]:
        """Parse comma-separated headers into list."""
        if self.allowed_headers == "*":
            return ["*"]
        return [header.strip() for header in self.allowed_headers.split(",") if header.strip()]


class LoggingConfig(BaseSettings):
    """Structured logging configuration."""

    model_config = SettingsConfigDict(env_prefix="LOGGING_")

    # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Minimum log level"
    )

    # Output format
    json_output: bool = Field(
        default=True, description="Use JSON output (True for production, False for development)"
    )

    # Console colorization (only for non-JSON output)
    colorized: bool = Field(
        default=False, description="Colorize console output (only for development)"
    )

    # Slow request logging thresholds
    slow_request_warning_ms: float = Field(
        default=100.0, ge=0.0, description="Log warning if request exceeds this latency (ms)"
    )
    slow_request_error_ms: float = Field(
        default=500.0, ge=0.0, description="Log error if request exceeds this latency (ms)"
    )

    # Service metadata (injected into all logs)
    service_name: str = Field(
        default="episodic-memory", description="Service name for log aggregation"
    )
    service_version: str = Field(default="0.1.0", description="Service version")
    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Deployment environment"
    )


class HealthCheckConfig(BaseSettings):
    """Health check configuration with degraded state thresholds."""

    model_config = SettingsConfigDict(env_prefix="HEALTH_")

    # KyroDB latency thresholds (for degraded state detection)
    kyrodb_latency_warning_ms: float = Field(
        default=100.0,
        ge=0.0,
        description="KyroDB latency threshold for warning (degraded state)"
    )
    kyrodb_latency_error_ms: float = Field(
        default=500.0,
        ge=0.0,
        description="KyroDB latency threshold for error (unhealthy state)"
    )

    # Embedding service latency thresholds
    embedding_latency_warning_ms: float = Field(
        default=50.0,
        ge=0.0,
        description="Embedding latency threshold for warning"
    )
    embedding_latency_error_ms: float = Field(
        default=200.0,
        ge=0.0,
        description="Embedding latency threshold for error"
    )

    @field_validator("kyrodb_latency_error_ms")
    @classmethod
    def validate_kyrodb_thresholds(cls, v: float, info) -> float:
        """Ensure error threshold is greater than warning threshold."""
        warning = info.data.get("kyrodb_latency_warning_ms")
        if warning is not None and v <= warning:
            raise ValueError(
                f"kyrodb_latency_error_ms ({v}) must be > kyrodb_latency_warning_ms ({warning})"
            )
        return v

    @field_validator("embedding_latency_error_ms")
    @classmethod
    def validate_embedding_thresholds(cls, v: float, info) -> float:
        """Ensure error threshold is greater than warning threshold."""
        warning = info.data.get("embedding_latency_warning_ms")
        if warning is not None and v <= warning:
            raise ValueError(
                f"embedding_latency_error_ms ({v}) must be > embedding_latency_warning_ms ({warning})"
            )
        return v

    # Cache settings
    cache_ttl_seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=60.0,
        description="Health check result cache TTL in seconds"
    )

    # Probe timeout settings
    liveness_timeout_ms: float = Field(
        default=5000.0,
        ge=100.0,
        description="Maximum time for liveness probe"
    )
    readiness_timeout_ms: float = Field(
        default=10000.0,
        ge=100.0,
        description="Maximum time for readiness probe"
    )

    # Circuit breaker settings
    consecutive_failures_threshold: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Consecutive health check failures before marking unhealthy"
    )
    recovery_threshold: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Consecutive successes needed to recover from unhealthy"
    )




class ReflectionConfig(BaseSettings):
    """Reflection generation configuration for cost optimization."""

    model_config = SettingsConfigDict(env_prefix="REFLECTION_")

    # Tier defaults
    default_tier: Literal["auto", "cheap", "premium"] = Field(
        default="auto",
        description="Default tier: auto (intelligent selection), cheap (OpenRouter free tier), premium (multi-perspective)"
    )

    # Premium triggers (error classes that force premium tier)
    premium_error_classes: list[str] = Field(
        default_factory=lambda: ["data_loss", "security_breach", "production_outage", "corruption"],
        description="Error classes that force premium tier regardless of auto-selection"
    )

    # Quality gates
    min_cheap_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for cheap tier (fallback to premium if lower)"
    )

    min_preconditions: int = Field(
        default=1,
        ge=0,
        le=10,
        description="Minimum number of preconditions required for cheap tier quality gate"
    )

    # Cost controls (circuit breakers)
    max_cost_per_day_usd: float = Field(
        default=50.0,
        ge=1.0,
        le=1000.0,
        description="Maximum daily reflection cost in USD (circuit breaker - alerts when exceeded)"
    )

    max_premium_percentage: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Maximum percentage of reflections using premium tier (alerts if exceeded)"
    )

    # Feature flags
    enable_quality_fallback: bool = Field(
        default=True,
        description="Enable automatic fallback from cheap to premium if quality gates fail"
    )

    enable_cost_tracking: bool = Field(
        default=True,
        description="Enable detailed cost tracking per tier"
    )

    @field_validator("default_tier")
    @classmethod
    def validate_default_tier(cls, v: str) -> str:
        """Validate default tier is a valid option."""
        valid_tiers = {"auto", "cheap", "premium"}
        if v not in valid_tiers:
            raise ValueError(f"default_tier must be one of: {', '.join(valid_tiers)}")
        return v


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
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)
    hygiene: HygieneConfig = Field(default_factory=HygieneConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    cors: CORSConfig = Field(default_factory=CORSConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    health: HealthCheckConfig = Field(default_factory=HealthCheckConfig)
    
    # Admin authentication (optional but recommended for production)
    # Set ADMIN_API_KEY in environment to protect admin endpoints
    admin_api_key: str | None = Field(
        default=None,
        description="API key for admin endpoints (required for admin access)"
    )

    @field_validator("admin_api_key")
    @classmethod
    def validate_admin_api_key_security(cls, v: str | None) -> str | None:
        """
        Security: Validate admin API key format and prevent common mistakes.

        Never expose API keys in logs or errors.
        """
        if not v:
            return None

        placeholder_patterns = [
            "your-api-key-here",
            "admin",
            "example",
            "dummy",
            "test",
        ]

        v_lower = v.lower()
        if any(pattern in v_lower for pattern in placeholder_patterns):
            logging.warning(
                "admin_api_key appears to be a placeholder - admin endpoints will be BLOCKED"
            )
            return None

        if len(v) < 32:
            logging.warning(
                "admin_api_key seems too short to be secure - use at least 32 characters"
            )

        return v

    def validate_configuration(self) -> None:
        """
        Validate cross-field constraints and log warnings.
        Called at application startup.
        """
        # Check LLM API key
        if not self.llm.has_any_api_key:
            logging.warning("LLM API key not configured - reflection generation will be disabled")

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
_settings: Settings | None = None


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
