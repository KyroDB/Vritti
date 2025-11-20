"""
FastAPI application for Episodic Memory service.

Provides REST API for:
- Episode ingestion (failures, successes, skills)
- Semantic search with precondition matching
- Health monitoring and statistics

Designed for <50ms P99 latency.
"""

import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.auth import (
    get_authenticated_customer,
    get_customer_id_from_request,
)
from src.config import get_settings
from src.ingestion.capture import IngestionPipeline
from src.ingestion.embedding import EmbeddingService
from src.ingestion.tiered_reflection import TieredReflectionService, get_tiered_reflection_service
from src.gating.service import GatingService
from src.kyrodb.client import KyroDBError
from src.kyrodb.router import KyroDBRouter
from src.models.customer import Customer
from src.models.episode import EpisodeCreate
from src.models.gating import ReflectRequest, ReflectResponse
from src.models.search import SearchRequest, SearchResponse
from src.observability.health import (
    HealthCheckResponse,
    LivenessResponse,
    ReadinessResponse,
    get_health_checker,
)
from src.observability.logging import configure_logging, get_logger
from src.observability.logging_middleware import (
    SlowRequestLogger,
    StructuredLoggingMiddleware,
)
from src.observability.metrics import (
    generate_metrics,
    set_kyrodb_health,
    track_ingestion_credits,
    track_search_credits,
    update_customer_quota_usage,
)
from src.observability.middleware import ErrorTrackingMiddleware, PrometheusMiddleware
from src.observability.request_limits import RequestSizeLimitMiddleware
from src.retrieval.search import SearchPipeline
from src.routers import customers_router
from src.storage.database import CustomerDatabase, get_customer_db
from typing import Union, Optional

# Initialize structured logging (Phase 2 Week 6)
settings = get_settings()
configure_logging(
    log_level=settings.logging.level,
    json_output=settings.logging.json_output,
    colorized=settings.logging.colorized,
)
logger = get_logger(__name__)


# Rate limiter configuration
# Per-customer rate limiting based on subscription tier
def get_customer_id_for_rate_limit(request: Request) -> str:
    """
    Extract customer_id for rate limiting.

    Falls back to IP address if not authenticated.
    This allows public endpoints to be rate-limited by IP,
    while authenticated endpoints are rate-limited per customer.
    """
    if hasattr(request.state, "customer"):
        # Authenticated request - rate limit by customer_id
        return request.state.customer.customer_id
    else:
        # Unauthenticated request - rate limit by IP
        return get_remote_address(request)


limiter = Limiter(key_func=get_customer_id_for_rate_limit)


# Global service instances (initialized in lifespan)
kyrodb_router: Optional[KyroDBRouter] = None
embedding_service: Optional[EmbeddingService] = None
reflection_service: Optional[TieredReflectionService] = None
ingestion_pipeline: Optional[IngestionPipeline] = None
search_pipeline: Optional[SearchPipeline] = None
gating_service: Optional[GatingService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown:
    - Connect to KyroDB instances
    - Initialize embedding models
    - Initialize LLM reflection service
    - Set up ingestion and search pipelines
    """
    global kyrodb_router, embedding_service, reflection_service
    global ingestion_pipeline, search_pipeline, gating_service

    settings = get_settings()

    logger.info("=== Episodic Memory Service Starting ===")

    try:
        # Initialize KyroDB router
        logger.info("Initializing KyroDB router...")
        kyrodb_router = KyroDBRouter(config=settings.kyrodb)
        await kyrodb_router.connect()
        logger.info("✓ KyroDB router connected")

        # Initialize embedding service
        logger.info("Initializing embedding service...")
        embedding_service = EmbeddingService(config=settings.embedding)
        logger.info("✓ Embedding service initialized")

        # Warm up embedding models (prevents cold start on first request)
        logger.info("Warming up embedding models...")
        embedding_service.warmup()
        logger.info("✓ Embedding models warmed up")

        # Initialize tiered reflection service (Phase 5)
        if settings.llm.has_any_api_key:
            logger.info("Initializing tiered LLM reflection service...")
            reflection_service = get_tiered_reflection_service(config=settings.llm)
            logger.info(
                f"✓ Tiered reflection service initialized "
                f"(providers: {settings.llm.enabled_providers})"
            )
        else:
            logger.warning(
                "No LLM API keys configured - reflection generation disabled\n"
                "  Set LLM_OPENAI_API_KEY, LLM_ANTHROPIC_API_KEY, or LLM_GOOGLE_API_KEY"
            )
            reflection_service = None

        # Initialize ingestion pipeline
        logger.info("Initializing ingestion pipeline...")
        ingestion_pipeline = IngestionPipeline(
            kyrodb_router=kyrodb_router,
            embedding_service=embedding_service,
            reflection_service=reflection_service,
        )
        logger.info("✓ Ingestion pipeline ready")

        # Initialize search pipeline
        logger.info("Initializing search pipeline...")
        search_pipeline = SearchPipeline(
            kyrodb_router=kyrodb_router,
            embedding_service=embedding_service,
        )
        logger.info("✓ Search pipeline ready")

        # Initialize gating service (Phase 3)
        logger.info("Initializing gating service...")
        gating_service = GatingService(
            search_pipeline=search_pipeline,
            kyrodb_router=kyrodb_router,
        )
        logger.info("✓ Gating service ready")

        logger.info("=== Service Ready ===")

        yield  # Application runs here

    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise

    finally:
        # Shutdown
        logger.info("=== Shutting down ===")

        if kyrodb_router:
            await kyrodb_router.close()
            logger.info("✓ KyroDB connections closed")

        logger.info("=== Shutdown complete ===")


# Create FastAPI app
app = FastAPI(
    title="Episodic Memory API",
    description="Multi-modal episodic memory for AI agents with semantic search",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Rate limiter state
app.state.limiter = limiter

# Rate limit exceeded handler
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware (configured via environment variables)
# Production: Set CORS_ALLOWED_ORIGINS="https://app.example.com,https://api.example.com"
# Development: Defaults to "*" (all origins)
settings = get_settings()
cors_origins = settings.cors.origins_list

# Security warning if wildcard is used
if "*" in cors_origins:
    logger.warning(
        "⚠️  CORS allows ALL origins (*) - configure CORS_ALLOWED_ORIGINS for production!"
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=settings.cors.allow_credentials,
    allow_methods=settings.cors.methods_list,
    allow_headers=settings.cors.headers_list,
    max_age=settings.cors.max_age,
)

logger.info(
    "CORS configured",
    origins=cors_origins,
    credentials=settings.cors.allow_credentials,
)

# Observability and security middleware (Phase 2 Week 5-6, Phase 5)
# Order matters (processed in reverse order of registration):
# 1. RequestSizeLimitMiddleware (innermost) - Rejects oversized requests first
# 2. ErrorTrackingMiddleware - Classifies errors
# 3. PrometheusMiddleware - Tracks metrics
# 4. SlowRequestLogger - Logs slow requests
# 5. StructuredLoggingMiddleware (outermost) - Sets request context
app.add_middleware(StructuredLoggingMiddleware)
app.add_middleware(
    SlowRequestLogger,
    warning_threshold_ms=settings.logging.slow_request_warning_ms,
    error_threshold_ms=settings.logging.slow_request_error_ms,
)
app.add_middleware(PrometheusMiddleware)
app.add_middleware(ErrorTrackingMiddleware)
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_body_size=settings.service.max_request_body_size,
)
logger.info(
    "Observability and security middleware registered",
    middlewares=[
        "RequestSizeLimitMiddleware",
        "ErrorTrackingMiddleware",
        "PrometheusMiddleware",
        "SlowRequestLogger",
        "StructuredLoggingMiddleware",
    ],
)

# Include routers
app.include_router(customers_router)


# Exception handler for KyroDB errors
@app.exception_handler(KyroDBError)
async def kyrodb_error_handler(request: Request, exc: KyroDBError):
    """Handle KyroDB connection/operation errors."""
    logger.error(
        "KyroDB error occurred",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Vector database temporarily unavailable", "error": str(exc)},
    )


# Exception handler for validation errors
@app.exception_handler(ValueError)
async def validation_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    logger.warning(f"Validation error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation failed", "error": str(exc)},
    )


# Health check endpoints (Phase 2 Week 7)
# Kubernetes-ready liveness, readiness, and comprehensive health checks


@app.get(
    "/health/liveness",
    response_model=LivenessResponse,
    tags=["Health"],
    summary="Liveness probe for Kubernetes",
    description="Minimal health check to verify service is alive. Always returns 200 unless service is dead.",
)
async def liveness_probe():
    """
    Liveness probe for Kubernetes.

    This endpoint should ONLY fail if the service is completely dead.
    It performs no I/O operations and returns almost instantly.

    Use for:
    - Kubernetes liveness probes
    - Load balancer health checks

    Performance: <5ms

    Returns:
        LivenessResponse: Liveness status (always alive)
    """
    health_checker = get_health_checker()
    return await health_checker.check_liveness()


@app.get(
    "/health/readiness",
    response_model=ReadinessResponse,
    tags=["Health"],
    summary="Readiness probe for Kubernetes",
    description="Check if service is ready to accept traffic. Verifies critical dependencies.",
    status_code=200,
    responses={
        200: {"description": "Service is ready"},
        503: {"description": "Service is not ready"},
    },
)
async def readiness_probe(response: Response):
    """
    Readiness probe for Kubernetes.

    Checks critical dependencies:
    - KyroDB connections (text and image instances)
    - Customer database connectivity
    - Embedding service availability

    Use for:
    - Kubernetes readiness probes
    - Traffic routing decisions

    Performance: <100ms (includes dependency checks)

    Returns:
        ReadinessResponse: Readiness status with component details
        HTTP 200: Service is ready (healthy or degraded)
        HTTP 503: Service is not ready (unhealthy)
    """
    health_checker = get_health_checker()

    # Get customer database
    customer_db = await get_customer_db()

    # Perform readiness check
    readiness = await health_checker.check_readiness(
        kyrodb_router=kyrodb_router,
        customer_db=customer_db,
        embedding_service=embedding_service,
    )

    # Update Prometheus health metrics
    for component in readiness.components:
        if component.name == "kyrodb":
            text_healthy = component.metadata.get("text_healthy", False)
            image_healthy = component.metadata.get("image_healthy", False)
            set_kyrodb_health("text", text_healthy)
            set_kyrodb_health("image", image_healthy)

    # Set HTTP status code based on readiness
    if not readiness.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    logger.debug(
        "Readiness probe completed",
        ready=readiness.ready,
        status=readiness.status.value,
        num_components=len(readiness.components),
    )

    return readiness


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["Health"],
    summary="Comprehensive health check",
    description="Detailed health status of all components. Not for Kubernetes probes (too slow).",
)
async def comprehensive_health_check():
    """
    Comprehensive health check with all components.

    This endpoint provides detailed health information for:
    - KyroDB connections (text and image)
    - Customer database
    - Embedding service
    - Reflection service (LLM)

    Use for:
    - Debugging
    - Monitoring dashboards
    - Manual health verification

    NOT recommended for:
    - Kubernetes probes (use /health/liveness or /health/readiness instead)
    - High-frequency polling (response is cached for 5 seconds)

    Performance: <200ms (with caching)

    Returns:
        HealthCheckResponse: Detailed health status with all components
    """
    health_checker = get_health_checker()

    # Get customer database
    customer_db = await get_customer_db()

    # Perform comprehensive health check
    health = await health_checker.check_health(
        kyrodb_router=kyrodb_router,
        customer_db=customer_db,
        embedding_service=embedding_service,
        reflection_service=reflection_service,
    )

    # Update Prometheus health metrics
    for component in health.components:
        if component.name == "kyrodb":
            text_healthy = component.metadata.get("text_healthy", False)
            image_healthy = component.metadata.get("image_healthy", False)
            set_kyrodb_health("text", text_healthy)
            set_kyrodb_health("image", image_healthy)

    logger.info(
        "Comprehensive health check completed",
        status=health.status.value,
        uptime_seconds=health.uptime_seconds,
        num_components=len(health.components),
    )

    return health


# Statistics endpoint
class StatsResponse(BaseModel):
    """Service statistics response."""

    ingestion_stats: dict
    search_stats: dict


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """
    Get service statistics.

    Returns ingestion and search pipeline metrics.
    """
    ingestion_stats = {}
    search_stats = {}

    if ingestion_pipeline:
        ingestion_stats = ingestion_pipeline.get_stats()

    if search_pipeline:
        search_stats = search_pipeline.get_stats()

    return StatsResponse(
        ingestion_stats=ingestion_stats,
        search_stats=search_stats,
    )


# Prometheus metrics endpoint (Phase 2 Week 5)
@app.get("/metrics", tags=["System"])
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus exposition format for scraping.

    Metrics include:
    - HTTP request latency (histogram)
    - HTTP request count (counter)
    - Active HTTP requests (gauge)
    - API key cache hit/miss rate
    - KyroDB operation latency
    - Episode ingestion counts
    - Search request counts
    - Credit usage by customer
    - Error rates by type

    Configuration:
        Prometheus scrape_interval: 15s (recommended)
        Retention: 90 days

    Example Prometheus config:
        scrape_configs:
          - job_name: 'episodic-memory'
            scrape_interval: 15s
            static_configs:
              - targets: ['localhost:8000']
    """
    metrics_data, content_type = generate_metrics()
    return Response(content=metrics_data, media_type=content_type)


# Ingestion endpoint
class CaptureResponse(BaseModel):
    """Episode capture response."""

    episode_id: int
    collection: str
    ingestion_latency_ms: float
    text_stored: bool
    image_stored: bool
    reflection_queued: bool


class SuccessValidationRequest(BaseModel):
    """
    Request to validate whether a suggested fix worked.

    Security:
    - episode_id validated against customer namespace
    - outcome restricted to valid values
    - notes sanitized
    """

    episode_id: int = Field(..., gt=0, description="Episode ID to validate")
    outcome: str = Field(
        ...,
        pattern="^(success|still_failed)$",
        description="Whether fix worked: 'success' or 'still_failed'"
    )
    applied_suggestion: bool = Field(
        ...,
        description="Whether the agent actually applied the suggested fix"
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional notes about the fix application"
    )


class SuccessValidationResponse(BaseModel):
    """Response from fix validation."""

    episode_id: int
    new_success_rate: float
    total_applications: int
    promoted_to_skill: bool = False
    skill_id: Optional[int] = None


@app.post(
    "/api/v1/capture",
    response_model=CaptureResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Ingestion"],
)
@limiter.limit("100/minute")  # Conservative rate limit (will be tier-based in Phase 4)
async def capture_episode(
    request: Request,  # Required by slowapi
    episode_data: EpisodeCreate,
    customer: Customer = Depends(get_authenticated_customer),
    customer_id: str = Depends(get_customer_id_from_request),
    db: CustomerDatabase = Depends(get_customer_db),
    generate_reflection: bool = True,
    tier: Optional[str] = None,  # NEW: Optional tier override (cheap/cached/premium)
):
    """
    Capture and store an episode.

    Pipeline:
    1. API key authentication (customer_id extraction)
    2. PII redaction
    3. ID generation
    4. Multi-modal embedding
    5. KyroDB storage
    6. Async reflection generation (optional)
    7. Usage tracking (credit deduction)

    Args:
        episode_data: Episode creation data (customer_id will be overridden)
        customer: Authenticated customer (injected from API key)
        customer_id: Customer ID from validated API key (injected)
        db: Customer database (for usage tracking)
        generate_reflection: Whether to generate LLM reflection (default: True)
        tier: Optional tier override (cheap/cached/premium), auto-selects if None

    Returns:
        CaptureResponse: Capture result with episode ID and metadata

    Raises:
        HTTPException 401: Invalid/missing API key
        HTTPException 403: Customer inactive or quota exceeded
        HTTPException 503: Ingestion pipeline not initialized
        HTTPException 500: Ingestion failure

    Security:
        ✓ customer_id extracted from validated API key (cannot be spoofed)
        ✓ User-provided customer_id in request body is IGNORED
        ✓ API key validated with bcrypt
        ✓ Quota enforcement (soft limit with warning)

    Usage Billing:
        - Base cost: 1 credit per episode
        - With image: +0.2 credits
        - With reflection: +0.5 credits
    """
    if not ingestion_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ingestion pipeline not initialized",
        )

    # SECURITY: Override user-provided customer_id with authenticated value
    # This prevents customer_id spoofing attacks
    episode_data.customer_id = customer_id

    # NEW: Parse and validate tier override if provided
    tier_enum = None
    if tier:
        try:
            from src.models.episode import ReflectionTier
            tier_enum = ReflectionTier(tier.lower())
            logger.info(f"Tier override requested: {tier_enum.value}")
        except ValueError:
            logger.warning(f"Invalid tier requested: {tier}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid tier: {tier}. Must be one of: cheap, cached, premium"
            )

    start_time = time.perf_counter()

    try:
        # Capture episode with optional tier override
        episode = await ingestion_pipeline.capture_episode(
            episode_data=episode_data,
            generate_reflection=generate_reflection and reflection_service is not None,
            tier_override=tier_enum,  # NEW: Pass tier override
        )

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Calculate usage credits
        credits_used = 1.0  # Base cost
        if episode_data.screenshot_path:
            credits_used += 0.2  # Image embedding cost
        if generate_reflection and reflection_service:
            credits_used += 0.5  # LLM reflection cost

        # Track usage (async, non-blocking)
        try:
            await db.increment_usage(customer_id, int(credits_used))
            logger.info(
                f"Usage tracked: customer={customer_id}, "
                f"episode={episode.episode_id}, credits={credits_used}"
            )
        except Exception as e:
            # Log but don't fail the request if usage tracking fails
            logger.error(f"Usage tracking failed: {e}", exc_info=True)

        # Track business metrics (Phase 2 Week 5)
        track_ingestion_credits(
            customer_id=customer_id,
            customer_tier=customer.subscription_tier.value,
            credits_used=credits_used,
            has_image=episode_data.screenshot_path is not None,
            has_reflection=generate_reflection and reflection_service is not None,
        )

        # Update customer quota gauge
        update_customer_quota_usage(
            customer_id=customer_id,
            customer_tier=customer.subscription_tier.value,
            credits_used=customer.credits_used_current_month + int(credits_used),
            monthly_limit=customer.monthly_credit_limit,
        )

        return CaptureResponse(
            episode_id=episode.episode_id,
            collection="failures",  # Only failures collection is supported
            ingestion_latency_ms=latency_ms,
            text_stored=True,  # Always stored in text instance
            image_stored=episode_data.screenshot_path is not None,
            reflection_queued=generate_reflection and reflection_service is not None,
        )

    except Exception as e:
        logger.error(f"Episode capture failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Episode capture failed: {str(e)}",
        )


@app.post(
    "/api/v1/validate_fix",
    response_model=SuccessValidationResponse,
    tags=["Validation"],
)
@limiter.limit("100/minute")
async def validate_fix(
    request: Request,
    validation: SuccessValidationRequest,
    customer: Customer = Depends(get_authenticated_customer),
    customer_id: str = Depends(get_customer_id_from_request),
):
    """
    Validate whether a suggested fix worked.

    This endpoint is called by agents AFTER they apply a retrieved fix.
    It updates the episode's usage statistics and may promote the episode
    to a skill if promotion criteria are met.

    Pipeline:
    1. Fetch episode from KyroDB
    2. Update usage_stats (fix_applied_count, success/failure counts)
    3. Re-insert episode with updated stats
    4. Check if episode should be promoted to skill
    5. Return updated success rate and promotion status

    Args:
        validation: Validation request with outcome
        customer: Authenticated customer
        customer_id: Customer ID from validated API key

    Returns:
        SuccessValidationResponse: Updated stats and promotion status

    Raises:
        HTTPException 401: Invalid/missing API key
        HTTPException 404: Episode not found
        HTTPException 403: Episode belongs to different customer
        HTTPException 500: Validation failure

    Security:
        - Customer namespace isolation enforced
        - Episode ownership validated
        - Stats validated for consistency
    """
    from src.skills.promotion import SkillPromotionService
    from src.kyrodb.router import get_namespaced_collection

    logger.info(
        f"Fix validation request for episode {validation.episode_id} "
        f"(customer: {customer_id}, outcome: {validation.outcome})"
    )

    collection = "failures"
    namespaced_collection = get_namespaced_collection(customer_id, collection)

    try:
        # Step 1: Fetch existing episode
        existing = await kyrodb_router.text_client.query(
            doc_id=validation.episode_id,
            namespace=namespaced_collection,
            include_embedding=True,
        )

        if not existing.found:
            logger.error(
                f"Episode {validation.episode_id} not found for validation "
                f"(customer: {customer_id})"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Episode {validation.episode_id} not found",
            )

        # Security: Verify customer ID
        existing_customer = existing.metadata.get("customer_id")
        if existing_customer != customer_id:
            logger.error(
                f"Customer ID mismatch: episode {validation.episode_id} belongs to "
                f"{existing_customer}, not {customer_id}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Episode belongs to different customer",
            )

        # Step 2: Deserialize episode and update stats
        from src.models.episode import Episode

        episode = Episode.from_metadata_dict(
            validation.episode_id, dict(existing.metadata)
        )

        # Update usage stats
        episode.usage_stats.total_retrievals += 1

        if validation.applied_suggestion:
            episode.usage_stats.fix_applied_count += 1

            if validation.outcome == "success":
                episode.usage_stats.fix_success_count += 1
            else:  # still_failed
                episode.usage_stats.fix_failure_count += 1

        # Step 3: Re-insert episode with updated stats
        updated_metadata = episode.to_metadata_dict()

        response = await kyrodb_router.text_client.insert(
            doc_id=validation.episode_id,
            embedding=list(existing.embedding),
            namespace=namespaced_collection,
            metadata=updated_metadata,
        )

        if not response.success:
            logger.error(f"Failed to update episode stats: {response.error}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update episode statistics",
            )

        logger.info(
            f"Updated episode {validation.episode_id} stats: "
            f"success_rate={episode.usage_stats.fix_success_rate:.2f}, "
            f"applications={episode.usage_stats.fix_applied_count}"
        )

        # Track fix validation metrics
        from src.observability.metrics import track_fix_validation

        track_fix_validation(
            customer_tier=customer.subscription_tier.value,
            outcome=validation.outcome,
        )

        # Step 4: Check if should promote to skill
        promoted = False
        skill_id = None

        if (
            validation.outcome == "success"
            and episode.usage_stats.fix_success_rate >= 0.9
            and episode.usage_stats.fix_applied_count >= 3
        ):
            logger.info(
                f"Episode {validation.episode_id} meets promotion criteria, "
                f"checking for skill promotion..."
            )

            try:
                import time
                from src.observability.metrics import track_skill_promotion

                start_time = time.perf_counter()

                promotion_service = SkillPromotionService(
                    kyrodb_router=kyrodb_router,
                    embedding_service=embedding_service,
                )

                skill = await promotion_service.check_and_promote(
                    episode_id=validation.episode_id,
                    customer_id=customer_id,
                )

                if skill:
                    promoted = True
                    skill_id = skill.skill_id

                    # Track promotion metrics
                    promotion_duration = time.perf_counter() - start_time
                    track_skill_promotion(
                        customer_tier=customer.subscription_tier.value,
                        error_class=episode.create_data.error_class.value,
                        duration_seconds=promotion_duration,
                    )

                    logger.info(
                        f"Episode {validation.episode_id} promoted to "
                        f"skill {skill_id}: {skill.name} "
                        f"(promotion took {promotion_duration:.2f}s)"
                    )

            except Exception as e:
                logger.error(
                    f"Skill promotion check failed for episode {validation.episode_id}: {e}",
                    exc_info=True,
                )

        return SuccessValidationResponse(
            episode_id=validation.episode_id,
            new_success_rate=episode.usage_stats.fix_success_rate,
            total_applications=episode.usage_stats.fix_applied_count,
            promoted_to_skill=promoted,
            skill_id=skill_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Fix validation failed for episode {validation.episode_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}",
        )


# Search endpoint
@app.post(
    "/api/v1/search",
    response_model=SearchResponse,
    tags=["Retrieval"],
)
@limiter.limit("200/minute")  # Higher limit for search (will be tier-based in Phase 4)
async def search_episodes(
    request: Request,  # Required by slowapi
    search_request: SearchRequest,  # Renamed to avoid conflict with Request parameter
    customer: Customer = Depends(get_authenticated_customer),
    customer_id: str = Depends(get_customer_id_from_request),
    db: CustomerDatabase = Depends(get_customer_db),
):
    """
    Search for relevant episodes.

    Pipeline:
    1. API key authentication (customer_id extraction)
    2. Query embedding generation
    3. KyroDB vector search (k×5 candidates)
    4. Metadata filtering
    5. Precondition matching
    6. Weighted ranking
    7. Top-k selection
    8. Usage tracking (credit deduction)

    Args:
        request: Search request with query and parameters (customer_id will be overridden)
        customer: Authenticated customer (injected from API key)
        customer_id: Customer ID from validated API key (injected)
        db: Customer database (for usage tracking)

    Returns:
        SearchResponse: Ranked search results with latency breakdown

    Raises:
        HTTPException 401: Invalid/missing API key
        HTTPException 403: Customer inactive or quota exceeded
        HTTPException 503: Search pipeline not initialized
        HTTPException 422: Validation error
        HTTPException 500: Search failure

    Security:
        ✓ customer_id extracted from validated API key (cannot be spoofed)
        ✓ User-provided customer_id in request body is IGNORED
        ✓ API key validated with bcrypt
        ✓ Namespace isolation (only searches customer's episodes)

    Usage Billing:
        - Base cost: 0.1 credits per search
        - With image search: +0.2 credits (Phase 2)
    """
    if not search_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search pipeline not initialized",
        )

    # SECURITY: Override user-provided customer_id with authenticated value
    # This prevents customer_id spoofing and cross-customer data access
    search_request.customer_id = customer_id

    try:
        # Execute search
        response = await search_pipeline.search(search_request)

        # Log slow queries
        if response.search_latency_ms > 100:
            logger.warning(
                f"Slow query detected: {response.search_latency_ms:.2f}ms "
                f"(goal: {search_request.goal[:50]}...)"
            )

        # Track usage (async, non-blocking)
        # Credit cost is fractional to encourage search over re-ingestion
        credits_used = 0.1  # Base search cost
        try:
            await db.increment_usage(customer_id, 1)  # Round to 1 credit
            logger.debug(
                f"Usage tracked: customer={customer_id}, search, "
                f"credits={credits_used}, results={response.total_returned}"
            )
        except Exception as e:
            # Log but don't fail the request if usage tracking fails
            logger.error(f"Usage tracking failed: {e}", exc_info=True)

        # Track business metrics (Phase 2 Week 5)
        track_search_credits(
            customer_id=customer_id,
            customer_tier=customer.subscription_tier.value,
            credits_used=credits_used,
            results_returned=response.total_returned,
        )

        # Update customer quota gauge
        update_customer_quota_usage(
            customer_id=customer_id,
            customer_tier=customer.subscription_tier.value,
            credits_used=customer.credits_used_current_month + 1,  # Rounded to 1
            monthly_limit=customer.monthly_credit_limit,
        )

        return response

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


# Pre-action gating endpoint (Phase 3)
@app.post(
    "/api/v1/reflect",
    response_model=ReflectResponse,
    tags=["Reflection"],
)
@limiter.limit("100/minute")
async def reflect_before_action(
    request: Request,
    reflect_request: ReflectRequest,
    customer: Customer = Depends(get_authenticated_customer),
    customer_id: str = Depends(get_customer_id_from_request),
    db: CustomerDatabase = Depends(get_customer_db),
):
    """
    Reflect before executing action.

    This is the core pre-action gating system. Agents call this
    BEFORE executing potentially risky actions.

    Returns:
        - BLOCK: High confidence this will fail (similarity > 0.9, preconditions match)
        - REWRITE: Likely to fail, suggest alternative
        - HINT: Might fail, show hints
        - PROCEED: No known issues

    Usage billing:
        - Base cost: 0.2 credits (2x search cost, since this prevents failures)
    """
    if not gating_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gating service not initialized",
        )

    try:
        response = await gating_service.reflect_before_action(reflect_request, customer_id)

        # Track usage (async, non-blocking)
        credits_used = 0.2
        try:
            await db.increment_usage(customer_id, 1)  # Round to 1 credit
            logger.debug(
                f"Usage tracked: customer={customer_id}, reflect, "
                f"credits={credits_used}, recommendation={response.recommendation}"
            )
        except Exception as e:
            logger.error(f"Usage tracking failed: {e}", exc_info=True)

        # TODO: Track business metrics for reflection

        return response

    except Exception as e:
        logger.error(f"Reflection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reflection failed: {str(e)}",
        )


# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "service": "Episodic Memory API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "stats": "/stats",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # For development
        log_level="info",
    )
