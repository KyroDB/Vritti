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

    # TLS/SSL configuration (Phase 1 Week 4)
    enable_tls: bool = Field(
        default=False, description="Enable TLS for KyroDB connections (required for production)"
    )
    tls_ca_cert_path: Optional[str] = Field(
        default=None,
        description="Path to CA certificate for server verification (None = system CA bundle)",
    )
    tls_client_cert_path: Optional[str] = Field(
        default=None, description="Path to client certificate for mutual TLS (optional)"
    )
    tls_client_key_path: Optional[str] = Field(
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
    def validate_tls_cert_path(cls, v: Optional[str], info) -> Optional[str]:
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
    """
    LLM configuration for multi-perspective reflection generation.

    Supports three providers for consensus:
    - OpenAI (GPT-4 Turbo)
    - Anthropic (Claude 3.5 Sonnet)
    - Google (Gemini 1.5 Pro)

    Security: API keys are never logged or exposed in errors.
    """

    model_config = SettingsConfigDict(env_prefix="LLM_")

    # OpenAI configuration
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key for GPT-4"
    )
    openai_model_name: str = Field(
        default="gpt-4-turbo-preview",
        description="OpenAI model identifier"
    )

    # Anthropic configuration
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key for Claude"
    )
    anthropic_model_name: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Anthropic model identifier"
    )

    # Google configuration
    google_api_key: str = Field(
        default="",
        description="Google API key for Gemini"
    )
    google_model_name: str = Field(
        default="gemini-1.5-pro",
        description="Google model identifier"
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
        default=30,
        ge=1,
        le=120,
        description="Timeout for LLM API calls"
    )

    # Cost limits (security: prevent abuse)
    max_cost_per_reflection_usd: float = Field(
        default=1.0,
        ge=0.01,
        le=10.0,
        description="Maximum allowed cost per reflection (abort if exceeded)"
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
        description="Legacy OpenAI API key (deprecated, use openai_api_key)"
    )

    @field_validator("openai_api_key", "anthropic_api_key", "google_api_key", "api_key")
    @classmethod
    def validate_api_key_security(cls, v: str, info) -> str:
        """
        Security: Validate API key format and prevent common mistakes.

        Never expose API keys in logs or errors.
        """
        if not v:
            # Empty is OK - reflection will be disabled for that provider
            return ""

        # Check for placeholder values
        placeholder_patterns = [
            "your-api-key-here",
            "example",
            "dummy"
        ]

        v_lower = v.lower()
        if any(pattern in v_lower for pattern in placeholder_patterns):
            import logging
            field_name = info.field_name
            logging.warning(
                f"{field_name} appears to be a placeholder - reflection may be disabled"
            )
            return ""

        # Basic length validation (real API keys are usually 40+ chars)
        if len(v) < 20:
            import logging
            field_name = info.field_name
            logging.warning(
                f"{field_name} seems too short to be valid - reflection may fail"
            )

        return v

    @property
    def has_any_api_key(self) -> bool:
        """Check if at least one LLM provider is configured."""
        return bool(
            self.openai_api_key or
            self.anthropic_api_key or
            self.google_api_key or
            self.api_key  # Legacy fallback
        )

    @property
    def enabled_providers(self) -> list[str]:
        """Get list of enabled LLM providers."""
        providers = []
        if self.openai_api_key or self.api_key:
            providers.append("openai")
        if self.anthropic_api_key:
            providers.append("anthropic")
        if self.google_api_key:
            providers.append("google")
        return providers

    def model_post_init(self, __context):
        """Post-initialization: handle legacy api_key."""
        # If legacy api_key is set but openai_api_key is not, migrate
        if self.api_key and not self.openai_api_key:
            self.openai_api_key = self.api_key
            import logging
            logging.info("Migrated legacy api_key to openai_api_key")


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
        default=False,
        description="Enable LLM semantic validation for preconditions (disabled by default for safety)"
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


class CORSConfig(BaseSettings):
    """CORS configuration for API security (Phase 1 Week 4)."""

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
    """Structured logging configuration (Phase 2 Week 6)."""

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




class ReflectionConfig(BaseSettings):
    """Reflection generation configuration (Phase 5 - Cost Optimization)."""

    model_config = SettingsConfigDict(env_prefix="REFLECTION_")

    # Tier defaults
    default_tier: Literal["auto", "cheap", "premium"] = Field(
        default="auto",
        description="Default tier: auto (intelligent selection), cheap (force Gemini Flash), premium (force multi-perspective)"
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


class StripeConfig(BaseSettings):
    """Stripe billing integration configuration."""

    model_config = SettingsConfigDict(env_prefix="STRIPE_")

    api_key: Optional[str] = Field(
        default=None, description="Stripe secret API key (sk_live_... or sk_test_...)"
    )
    webhook_secret: Optional[str] = Field(
        default=None, description="Stripe webhook signing secret (whsec_...)"
    )
    publishable_key: Optional[str] = Field(
        default=None, description="Stripe publishable key (pk_live_... or pk_test_...)"
    )

    # Subscription tier price IDs (created in Stripe dashboard)
    price_id_starter: Optional[str] = Field(
        default=None, description="Stripe price ID for Starter tier"
    )
    price_id_pro: Optional[str] = Field(default=None, description="Stripe price ID for Pro tier")
    price_id_enterprise: Optional[str] = Field(
        default=None, description="Stripe price ID for Enterprise tier"
    )

    # Billing behavior
    trial_period_days: int = Field(default=14, ge=0, le=90, description="Free trial period in days")
    payment_grace_period_days: int = Field(
        default=3, ge=0, le=30, description="Grace period after payment failure before suspension"
    )

    # Usage-based billing
    enable_metered_billing: bool = Field(
        default=False, description="Enable usage-based metered billing (credits)"
    )
    metered_price_id: Optional[str] = Field(
        default=None, description="Stripe price ID for metered usage (credits)"
    )

    @property
    def is_configured(self) -> bool:
        """Check if Stripe is properly configured."""
        return self.api_key is not None and self.webhook_secret is not None


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
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)  # NEW: Phase 5
    hygiene: HygieneConfig = Field(default_factory=HygieneConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    cors: CORSConfig = Field(default_factory=CORSConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    stripe: StripeConfig = Field(default_factory=StripeConfig)

    def validate_configuration(self) -> None:
        """
        Validate cross-field constraints and log warnings.
        Called at application startup.
        """
        # Check LLM API key
        if not self.llm.api_key:
            import logging

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

        # Check Stripe configuration
        if not self.stripe.is_configured:
            logging.warning(
                "Stripe is not configured - billing features will be disabled. "
                "Set STRIPE_API_KEY and STRIPE_WEBHOOK_SECRET to enable."
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
