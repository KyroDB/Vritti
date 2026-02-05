"""
FastAPI application for Episodic Memory service.

Provides REST API for:
- Episode ingestion (failures, successes, skills)
- Semantic search with precondition matching
- Health monitoring and statistics

Designed for <50ms P99 latency.
"""

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from enum import Enum
from typing import Any

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
    require_admin_access,
)
from src.config import get_settings
from src.gating.service import GatingService
from src.ingestion.capture import IngestionPipeline
from src.ingestion.embedding import EmbeddingService
from src.ingestion.tiered_reflection import (
    TieredReflectionService,
    get_tiered_reflection_service,
)
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
from src.observability.middleware import ErrorTrackingMiddleware
from src.observability.request_limits import RequestSizeLimitMiddleware
from src.rate_limits import (
    CAPTURE_RATE_LIMIT,
    REFLECT_RATE_LIMIT,
    SEARCH_RATE_LIMIT,
    SKILLS_RATE_LIMIT,
    log_rate_limit_exceeded,
)
from src.retrieval.search import SearchPipeline
from src.routers import customers_router
from src.storage.database import CustomerDatabase, get_customer_db

# Initialize structured logging
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
        return str(request.state.customer.customer_id)
    else:
        # Unauthenticated request - rate limit by IP
        return get_remote_address(request)


limiter = Limiter(key_func=get_customer_id_for_rate_limit)


# Global service instances (initialized in lifespan)
kyrodb_router: KyroDBRouter | None = None
embedding_service: EmbeddingService | None = None
reflection_service: TieredReflectionService | None = None
ingestion_pipeline: IngestionPipeline | None = None
search_pipeline: SearchPipeline | None = None
gating_service: GatingService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
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
        # Best-effort cleanup for known multiprocessing semaphore leaks (e.g., PyTorch).
        try:
            from src.utils.resource_tracker import install_resource_tracker_cleanup

            install_resource_tracker_cleanup()
        except Exception as e:
            logger.warning("Resource tracker cleanup unavailable: %s", e)

        # Initialize KyroDB router
        logger.info("Initializing KyroDB router...")
        kyrodb_router = KyroDBRouter(config=settings.kyrodb)
        await kyrodb_router.connect()
        logger.info("[OK] KyroDB router connected")

        # Initialize embedding service
        logger.info("Initializing embedding service...")
        embedding_service = EmbeddingService(config=settings.embedding)
        logger.info("[OK] Embedding service initialized")

        # Fail fast in offline/air-gapped mode if required models are not preloaded.
        embedding_service.validate_offline_model_preflight()
        logger.info("[OK] Embedding model preflight validated")

        # Warm up embedding models (prevents cold start on first request)
        logger.info("Warming up embedding models...")
        embedding_service.warmup()
        logger.info("[OK] Embedding models warmed up")

        # Initialize tiered reflection service
        if settings.llm.has_any_api_key:
            logger.info("Initializing tiered LLM reflection service...")
            reflection_service = get_tiered_reflection_service(
                config=settings.llm,
                kyrodb_router=kyrodb_router,
                embedding_service=embedding_service,
            )
            logger.info(
                f"[OK] Tiered reflection service initialized "
                f"(providers: {settings.llm.enabled_providers})"
            )
        else:
            msg = "No LLM API keys configured. " "Set LLM_OPENROUTER_API_KEY environment variable."
            if settings.service.require_llm_reflection:
                raise RuntimeError(
                    f"{msg} SERVICE_REQUIRE_LLM_REFLECTION=true requires reflection service."
                )
            logger.warning(
                f"{msg} Reflection generation disabled (SERVICE_REQUIRE_LLM_REFLECTION=false)."
            )
            reflection_service = None

        # Initialize ingestion pipeline
        logger.info("Initializing ingestion pipeline...")
        ingestion_pipeline = IngestionPipeline(
            kyrodb_router=kyrodb_router,
            embedding_service=embedding_service,
            reflection_service=reflection_service,
        )
        logger.info("[OK] Ingestion pipeline ready")

        # Initialize search pipeline
        logger.info("Initializing search pipeline...")
        search_pipeline = SearchPipeline(
            kyrodb_router=kyrodb_router,
            embedding_service=embedding_service,
        )
        logger.info("[OK] Search pipeline ready")

        # Initialize gating service
        logger.info("Initializing gating service...")
        gating_service = GatingService(
            search_pipeline=search_pipeline,
            kyrodb_router=kyrodb_router,
        )
        logger.info("[OK] Gating service ready")

        logger.info("=== Service Ready ===")

        yield  # Application runs here

    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise

    finally:
        # Shutdown
        logger.info("=== Shutting down ===")

        # Wait for pending reflection tasks to complete
        if ingestion_pipeline:
            await ingestion_pipeline.shutdown(timeout=30.0)
            logger.info("[OK] Pending reflections completed")

        if kyrodb_router:
            await kyrodb_router.close()
            logger.info("[OK] KyroDB connections closed")

        logger.info("=== Shutdown complete ===")


def _require_kyrodb_router() -> KyroDBRouter:
    if kyrodb_router is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KyroDB router not initialized",
        )
    return kyrodb_router


def _require_embedding_service() -> EmbeddingService:
    if embedding_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service not initialized",
        )
    return embedding_service


def _require_reflection_service() -> TieredReflectionService:
    if reflection_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reflection service not initialized",
        )
    return reflection_service


def _require_ingestion_pipeline() -> IngestionPipeline:
    if ingestion_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ingestion pipeline not initialized",
        )
    return ingestion_pipeline


def _require_search_pipeline() -> SearchPipeline:
    if search_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search pipeline not initialized",
        )
    return search_pipeline


def _require_gating_service() -> GatingService:
    if gating_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gating service not initialized",
        )
    return gating_service


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


# Custom rate limit exceeded handler with tier-aware logging
async def custom_rate_limit_handler(request: Request, exc: Exception) -> Response:
    """Handle rate limit exceeded with tier-aware logging and metrics."""
    # Extract customer info for logging
    customer_id = "unknown"
    tier = None

    if hasattr(request.state, "customer"):
        customer = request.state.customer
        customer_id = customer.customer_id
        tier = customer.subscription_tier

        # Log and track metrics
        endpoint_type = _get_endpoint_type(request.url.path)
        log_rate_limit_exceeded(
            customer_id=customer_id,
            tier=tier,
            endpoint_type=endpoint_type,
        )
    else:
        logger.warning(f"Rate limit exceeded for unauthenticated request: {request.url.path}")

    # Use default handler for response
    if not isinstance(exc, RateLimitExceeded):
        raise exc
    return _rate_limit_exceeded_handler(request, exc)


def _get_endpoint_type(path: str) -> str:
    """Extract endpoint type from path for rate limit logging."""
    if "/capture" in path:
        return "capture"
    elif "/search" in path:
        return "search"
    elif "/reflect" in path:
        return "reflect"
    elif "/skills" in path:
        return "skills"
    elif "/admin" in path:
        return "admin"
    return "default"


# Rate limit exceeded handler
app.add_exception_handler(RateLimitExceeded, custom_rate_limit_handler)

# CORS middleware (configured via environment variables)
# Production: Set CORS_ALLOWED_ORIGINS="https://app.example.com,https://api.example.com"
# Development: Defaults to "*" (all origins)
settings = get_settings()
cors_origins = settings.cors.origins_list

# Security warning if wildcard is used
if "*" in cors_origins:
    logger.warning(
        "[SECURITY] CORS allows ALL origins (*) - configure CORS_ALLOWED_ORIGINS for production!"
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

# Observability and security middleware
# Order matters (processed in reverse order of registration):
# 1. RequestSizeLimitMiddleware (innermost) - Rejects oversized requests first
# 2. ErrorTrackingMiddleware - Classifies errors
# 3. SlowRequestLogger - Logs slow requests
# 4. StructuredLoggingMiddleware (outermost) - Sets request context
app.add_middleware(StructuredLoggingMiddleware)
app.add_middleware(
    SlowRequestLogger,
    warning_threshold_ms=settings.logging.slow_request_warning_ms,
    error_threshold_ms=settings.logging.slow_request_error_ms,
)
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
        "SlowRequestLogger",
        "StructuredLoggingMiddleware",
    ],
)

# Include routers
app.include_router(customers_router)


# Exception handler for KyroDB errors
@app.exception_handler(KyroDBError)
async def kyrodb_error_handler(request: Request, exc: KyroDBError) -> JSONResponse:
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
async def validation_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handle validation errors."""
    logger.warning(f"Validation error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation failed", "error": str(exc)},
    )


# Health check endpoints
# Kubernetes-ready liveness, readiness, and comprehensive health checks


@app.get(
    "/health/liveness",
    response_model=LivenessResponse,
    tags=["Health"],
    summary="Liveness probe for Kubernetes",
    description="Minimal health check to verify service is alive. Always returns 200 unless service is dead.",
)
async def liveness_probe() -> LivenessResponse:
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
async def readiness_probe(response: Response) -> ReadinessResponse:
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
async def comprehensive_health_check() -> HealthCheckResponse:
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
async def get_stats() -> StatsResponse:
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
        description="Whether fix worked: 'success' or 'still_failed'",
    )
    applied_suggestion: bool = Field(
        ..., description="Whether the agent actually applied the suggested fix"
    )
    notes: str | None = Field(
        default=None, max_length=1000, description="Optional notes about the fix application"
    )


class SuccessValidationResponse(BaseModel):
    """Response from fix validation."""

    episode_id: int
    new_success_rate: float
    total_applications: int
    promoted_to_skill: bool = False
    skill_id: int | None = None


@app.post(
    "/api/v1/capture",
    response_model=CaptureResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Ingestion"],
)
@limiter.limit(
    CAPTURE_RATE_LIMIT
)  # Tier-based: FREE=10/min, STARTER=100/min, PRO=500/min, ENTERPRISE=2000/min
async def capture_episode(
    request: Request,  # Required by slowapi
    episode_data: EpisodeCreate,
    customer: Customer = Depends(get_authenticated_customer),
    customer_id: str = Depends(get_customer_id_from_request),
    db: CustomerDatabase = Depends(get_customer_db),
    tier: str | None = None,  # NEW: Optional tier override (cheap/cached/premium)
) -> CaptureResponse:
    """
    Capture and store an episode.

    Pipeline:
    1. API key authentication (customer_id extraction)
    2. PII redaction
    3. ID generation
    4. Multi-modal embedding
    5. KyroDB storage
    6. Async reflection generation
    7. Usage tracking (credit deduction)

    Args:
        episode_data: Episode creation data (customer_id will be overridden)
        customer: Authenticated customer (injected from API key)
        customer_id: Customer ID from validated API key (injected)
        db: Customer database (for usage tracking)
        tier: Optional tier override (cheap/cached/premium), auto-selects if None

    Returns:
        CaptureResponse: Capture result with episode ID and metadata

    Raises:
        HTTPException 401: Invalid/missing API key
        HTTPException 403: Customer inactive or quota exceeded
        HTTPException 503: Ingestion pipeline not initialized
        HTTPException 500: Ingestion failure

    Security:
        - customer_id extracted from validated API key (cannot be spoofed)
        - User-provided customer_id in request body is IGNORED
        - API key validated via key_id lookup + adaptive hash verification
        - Quota enforcement (soft limit with warning)

    Usage Tracking:
        - Base cost: 1 credit per episode
        - With image: +0.2 credits
        - With reflection: +0.5 credits
    """
    pipeline = _require_ingestion_pipeline()

    enable_reflection = reflection_service is not None

    # SECURITY: Override user-provided customer_id with authenticated value
    # This prevents customer_id spoofing attacks
    episode_data.customer_id = customer_id

    # NEW: Parse and validate tier override if provided
    tier_enum = None
    if tier:
        if not enable_reflection:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tier override requires reflection service to be enabled",
            )
        try:
            from src.models.episode import ReflectionTier

            tier_enum = ReflectionTier(tier.lower())
            logger.info(f"Tier override requested: {tier_enum.value}")
        except ValueError:
            logger.warning(f"Invalid tier requested: {tier}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid tier: {tier}. Must be one of: cheap, cached, premium",
            )

    start_time = time.perf_counter()

    try:
        # Capture episode with optional tier override
        episode = await pipeline.capture_episode(
            episode_data=episode_data,
            generate_reflection=enable_reflection,
            tier_override=tier_enum,  # NEW: Pass tier override
        )

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Calculate usage credits
        credits_used = 1.0  # Base cost
        if episode.image_embedding_id is not None:
            credits_used += 0.2  # Image embedding cost
        if enable_reflection:
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

        return CaptureResponse(
            episode_id=episode.episode_id,
            collection="failures",  # Only failures collection is supported
            ingestion_latency_ms=latency_ms,
            text_stored=True,  # Always stored in text instance
            image_stored=episode.image_embedding_id is not None,
            reflection_queued=enable_reflection,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
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
@limiter.limit(CAPTURE_RATE_LIMIT)  # Same limits as capture (validation is part of ingestion flow)
async def validate_fix(
    request: Request,
    validation: SuccessValidationRequest,
    customer: Customer = Depends(get_authenticated_customer),
    customer_id: str = Depends(get_customer_id_from_request),
) -> SuccessValidationResponse:
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
    from src.kyrodb.router import get_namespaced_collection
    from src.skills.promotion import SkillPromotionService

    logger.info(
        f"Fix validation request for episode {validation.episode_id} "
        f"(customer: {customer_id}, outcome: {validation.outcome})"
    )

    collection = "failures"
    namespaced_collection = get_namespaced_collection(customer_id, collection)

    try:
        # Step 1: Fetch existing episode
        router = _require_kyrodb_router()
        existing = await router.text_client.query(
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

        episode = Episode.from_metadata_dict(validation.episode_id, dict(existing.metadata))

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

        response = await router.text_client.insert(
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

                start_time = time.perf_counter()

                promotion_service = SkillPromotionService(
                    kyrodb_router=router,
                    embedding_service=_require_embedding_service(),
                )

                skill = await promotion_service.check_and_promote(
                    episode_id=validation.episode_id,
                    customer_id=customer_id,
                )

                if skill:
                    promoted = True
                    skill_id = skill.skill_id

                    promotion_duration = time.perf_counter() - start_time
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
@limiter.limit(
    SEARCH_RATE_LIMIT
)  # Tier-based: FREE=20/min, STARTER=200/min, PRO=1000/min, ENTERPRISE=5000/min
async def search_episodes(
    request: Request,  # Required by slowapi
    search_request: SearchRequest,  # Renamed to avoid conflict with Request parameter
    customer: Customer = Depends(get_authenticated_customer),
    customer_id: str = Depends(get_customer_id_from_request),
    db: CustomerDatabase = Depends(get_customer_db),
) -> SearchResponse:
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
        - customer_id extracted from validated API key (cannot be spoofed)
        - User-provided customer_id in request body is IGNORED
        - API key validated via key_id lookup + adaptive hash verification
        - Namespace isolation (only searches customer's episodes)

    Usage Tracking:
        - Base cost: 0.1 credits per search
        - With image search: +0.2 credits
    """
    pipeline = _require_search_pipeline()

    # SECURITY: Override user-provided customer_id with authenticated value
    # This prevents customer_id spoofing and cross-customer data access
    search_request.customer_id = customer_id

    try:
        # Execute search
        response = await pipeline.search(search_request)

        # Log slow queries
        if response.search_latency_ms > 100:
            logger.warning(
                f"Slow query detected: {response.search_latency_ms:.2f}ms "
                f"(goal: {search_request.goal[:50]}...)"
            )

        # Track usage (async, non-blocking)
        # Credit cost is fractional to encourage search over re-ingestion
        credits_used = 0.1  # Base search cost
        if search_request.image_base64:
            credits_used += 0.2  # Image search cost
        try:
            await db.increment_usage(customer_id, max(1, int(round(credits_used))))
            logger.debug(
                f"Usage tracked: customer={customer_id}, search, "
                f"credits={credits_used}, results={response.total_returned}"
            )
        except Exception as e:
            # Log but don't fail the request if usage tracking fails
            logger.error(f"Usage tracking failed: {e}", exc_info=True)

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


# Pre-action gating endpoint
@app.post(
    "/api/v1/reflect",
    response_model=ReflectResponse,
    tags=["Reflection"],
)
@limiter.limit(
    REFLECT_RATE_LIMIT
)  # Tier-based: FREE=10/min, STARTER=100/min, PRO=500/min, ENTERPRISE=2000/min
async def reflect_before_action(
    request: Request,
    reflect_request: ReflectRequest,
    customer: Customer = Depends(get_authenticated_customer),
    customer_id: str = Depends(get_customer_id_from_request),
    db: CustomerDatabase = Depends(get_customer_db),
) -> ReflectResponse:
    """
    Reflect before executing action.

    This is the core pre-action gating system. Agents call this
    BEFORE executing potentially risky actions.

    Returns:
        - block: High confidence this will fail (similarity > 0.9, preconditions match)
        - rewrite: Likely to fail, suggest alternative
        - hint: Might fail, show hints
        - proceed: No known issues

    Usage tracking:
        - Base cost: 0.2 credits (2x search cost, since this prevents failures)
    """
    service = _require_gating_service()

    try:
        response = await service.reflect_before_action(reflect_request, customer_id)

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

        return response

    except Exception as e:
        logger.error(f"Reflection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reflection failed: {str(e)}",
        )


# ============================================================================
# SKILLS ENDPOINTS
# ============================================================================


class SkillFeedbackRequest(BaseModel):
    """
    Request to provide feedback on a skill application.

    Security:
    - skill_id validated against customer namespace
    - outcome restricted to valid values
    - notes sanitized for storage
    """

    outcome: str = Field(
        ...,
        pattern="^(success|failure)$",
        description="Whether skill application worked: 'success' or 'failure'",
    )
    notes: str | None = Field(
        default=None, max_length=1000, description="Optional notes about the skill application"
    )


class SkillFeedbackResponse(BaseModel):
    """Response from skill feedback."""

    skill_id: int
    new_success_rate: float
    total_usages: int
    success_count: int
    failure_count: int


class SkillSummary(BaseModel):
    """Summary of a skill for listing."""

    skill_id: int
    name: str
    docstring: str
    error_class: str
    success_rate: float
    usage_count: int
    source_episodes: int
    created_at: str


class SkillsListResponse(BaseModel):
    """Response for skills listing."""

    skills: list[SkillSummary]
    total_count: int


@app.post(
    "/api/v1/skills/{skill_id}/feedback",
    response_model=SkillFeedbackResponse,
    tags=["Skills"],
)
@limiter.limit(
    SKILLS_RATE_LIMIT
)  # Tier-based: FREE=10/min, STARTER=50/min, PRO=200/min, ENTERPRISE=1000/min
async def skill_feedback(
    request: Request,
    skill_id: int,
    feedback: SkillFeedbackRequest,
    customer: Customer = Depends(get_authenticated_customer),
    customer_id: str = Depends(get_customer_id_from_request),
) -> SkillFeedbackResponse:
    """
    Provide feedback on a skill application.

    This endpoint is called by agents AFTER they apply a skill from the gating
    system. It updates the skill's usage statistics to improve future
    recommendations.

    Pipeline:
    1. Validate skill exists and belongs to customer
    2. Update usage_count, success_count, or failure_count
    3. Re-insert skill with updated stats
    4. Return updated success rate

    Args:
        skill_id: ID of the skill that was applied
        feedback: Feedback request with outcome
        customer: Authenticated customer
        customer_id: Customer ID from validated API key

    Returns:
        SkillFeedbackResponse: Updated skill stats

    Raises:
        HTTPException 401: Invalid/missing API key
        HTTPException 404: Skill not found
        HTTPException 403: Skill belongs to different customer
        HTTPException 500: Feedback update failure

    Security:
        - Customer namespace isolation enforced
        - Skill ownership validated before update
        - Stats validated for consistency
    """
    logger.info(
        f"Skill feedback for skill {skill_id} "
        f"(customer: {customer_id}, outcome: {feedback.outcome})"
    )

    try:
        # Update skill stats via router (returns updated skill to avoid race conditions)
        router = _require_kyrodb_router()
        skill = await router.update_skill_stats(
            skill_id=skill_id,
            customer_id=customer_id,
            success=(feedback.outcome == "success"),
        )

        if skill is None:
            # Skill not found or customer mismatch
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Skill {skill_id} not found or access denied",
            )

        logger.info(
            f"Skill {skill_id} feedback recorded: "
            f"outcome={feedback.outcome}, new_success_rate={skill.success_rate:.2f}"
        )

        return SkillFeedbackResponse(
            skill_id=skill_id,
            new_success_rate=skill.success_rate,
            total_usages=skill.usage_count,
            success_count=skill.success_count,
            failure_count=skill.failure_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Skill feedback failed for skill {skill_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Skill feedback failed: {str(e)}",
        )


@app.get(
    "/api/v1/skills",
    response_model=SkillsListResponse,
    tags=["Skills"],
)
@limiter.limit(SKILLS_RATE_LIMIT)  # Tier-based skills rate limit
async def list_skills(
    request: Request,
    customer: Customer = Depends(get_authenticated_customer),
    customer_id: str = Depends(get_customer_id_from_request),
    limit: int = 50,
    min_success_rate: float = 0.0,
) -> SkillsListResponse:
    """
    List skills for the authenticated customer.

    Returns a list of all promoted skills for the customer, sorted by
    success rate descending.

    Args:
        customer: Authenticated customer
        customer_id: Customer ID from validated API key
        limit: Maximum number of skills to return (default: 50)
        min_success_rate: Minimum success rate filter (default: 0.0)

    Returns:
        SkillsListResponse: List of skill summaries

    Security:
        - Customer namespace isolation enforced
        - Only returns skills belonging to authenticated customer
    """

    logger.info(f"Listing skills for customer {customer_id} (limit: {limit})")

    try:
        # Use a neutral embedding to fetch all skills (search by customer namespace)
        # This is a workaround - ideally we'd have a list/scan endpoint
        embed_service = _require_embedding_service()
        router = _require_kyrodb_router()
        neutral_embedding = [0.0] * embed_service.config.text_dimension

        results = await router.search_skills(
            query_embedding=neutral_embedding,
            customer_id=customer_id,
            k=limit,
            min_score=0.0,  # Get all skills regardless of similarity
        )

        # Convert to summaries and filter by success rate
        skills = []
        for skill, _ in results:
            if skill.success_rate >= min_success_rate:
                skills.append(
                    SkillSummary(
                        skill_id=skill.skill_id,
                        name=skill.name,
                        docstring=(
                            skill.docstring[:197] + "..."
                            if len(skill.docstring) > 200
                            else skill.docstring
                        ),
                        error_class=skill.error_class,
                        success_rate=skill.success_rate,
                        usage_count=skill.usage_count,
                        source_episodes=len(skill.source_episodes),
                        created_at=skill.created_at.isoformat(),
                    )
                )

        # Sort by success rate descending
        skills.sort(key=lambda s: s.success_rate, reverse=True)

        logger.info(f"Found {len(skills)} skills for customer {customer_id}")

        return SkillsListResponse(
            skills=skills,
            total_count=len(skills),
        )

    except Exception as e:
        logger.error(
            f"Skills listing failed for customer {customer_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Skills listing failed: {str(e)}",
        )


@app.get(
    "/api/v1/skills/{skill_id}",
    tags=["Skills"],
)
@limiter.limit(SKILLS_RATE_LIMIT)  # Tier-based skills rate limit
async def get_skill(
    request: Request,
    skill_id: int,
    customer: Customer = Depends(get_authenticated_customer),
    customer_id: str = Depends(get_customer_id_from_request),
) -> dict[str, Any]:
    """
    Get a specific skill by ID.

    Args:
        skill_id: ID of the skill to retrieve
        customer: Authenticated customer
        customer_id: Customer ID from validated API key

    Returns:
        Full skill object with all metadata

    Security:
        - Customer namespace isolation enforced
        - Only returns skill if it belongs to authenticated customer
    """
    from src.kyrodb.router import get_namespaced_collection
    from src.models.skill import Skill

    logger.debug(f"Getting skill {skill_id} for customer {customer_id}")

    try:
        namespaced_collection = get_namespaced_collection(customer_id, "skills")
        router = _require_kyrodb_router()
        skill_data = await router.text_client.query(
            doc_id=skill_id,
            namespace=namespaced_collection,
            include_embedding=False,
        )

        if not skill_data.found:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Skill {skill_id} not found",
            )

        skill = Skill.from_metadata_dict(skill_id, dict(skill_data.metadata))

        # Verify customer ownership
        if skill.customer_id != customer_id:
            logger.error(
                f"Customer mismatch for skill {skill_id}: "
                f"expected {customer_id}, got {skill.customer_id}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Skill belongs to different customer",
            )

        return skill.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Get skill failed for skill {skill_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Get skill failed: {str(e)}",
        )


# Admin endpoints


def _enum_key_to_str(key: Any) -> str:
    """
    Convert dictionary key to string, handling Enum types properly.

    Uses isinstance(key, Enum) instead of hasattr(key, 'value') for robust
    enum detection that won't match arbitrary objects with 'value' attribute.

    Args:
        key: Dictionary key (may be Enum, string, or other type)

    Returns:
        String representation of the key
    """
    if isinstance(key, Enum):
        return str(key.value)
    return str(key)


def _convert_dict_enum_keys(d: dict[Any, Any]) -> dict[str, Any]:
    """
    Convert all enum keys in a dictionary to their string values.

    Args:
        d: Dictionary with potentially enum keys

    Returns:
        Dictionary with all keys converted to strings
    """
    return {_enum_key_to_str(k): v for k, v in d.items()}


class BudgetResponse(BaseModel):
    """Daily budget status response."""

    date: str = Field(..., description="Date for budget (YYYY-MM-DD)")
    daily_cost_usd: float = Field(..., description="Total cost today in USD")
    warning_threshold_usd: float = Field(..., description="Warning threshold ($10)")
    limit_threshold_usd: float = Field(..., description="Hard limit threshold ($50)")
    warning_triggered: bool = Field(..., description="Whether warning has been logged")
    limit_exceeded: bool = Field(..., description="Whether limit has been exceeded")
    budget_remaining_usd: float = Field(..., description="Remaining budget today")
    premium_tier_blocked: bool = Field(..., description="Whether premium tier is blocked")
    cost_by_tier: dict = Field(default_factory=dict, description="Cost breakdown by tier")
    count_by_tier: dict = Field(default_factory=dict, description="Reflection count by tier")


@app.get(
    "/admin/budget",
    response_model=BudgetResponse,
    tags=["Admin"],
    summary="Check daily LLM budget status",
    description="Returns current daily cost, budget remaining, and tier blocking status. Requires X-Admin-API-Key header if ADMIN_API_KEY is configured.",
)
async def get_budget_status(_: None = Depends(require_admin_access)) -> BudgetResponse:
    """
    Check daily LLM budget status.

    Returns current daily spending, budget thresholds, and whether
    premium tier is blocked due to budget exhaustion.

    This endpoint is for monitoring and debugging LLM costs.
    Premium tier is automatically blocked when daily spend >= $50.

    Security:
        - Requires X-Admin-API-Key header if ADMIN_API_KEY environment variable is set.
        - If ADMIN_API_KEY is not configured, endpoint is unprotected (warning logged).

    Returns:
        BudgetResponse: Current budget status
    """
    if not reflection_service:
        return BudgetResponse(
            date=str(datetime.now(UTC).date()),
            daily_cost_usd=0.0,
            warning_threshold_usd=10.0,
            limit_threshold_usd=50.0,
            warning_triggered=False,
            limit_exceeded=False,
            budget_remaining_usd=50.0,
            premium_tier_blocked=False,
            cost_by_tier={},
            count_by_tier={},
        )

    stats = reflection_service.get_stats()
    daily = stats.get("daily_cost", {})

    return BudgetResponse(
        date=daily.get("date", str(datetime.now(UTC).date())),
        daily_cost_usd=daily.get("daily_cost_usd", 0.0),
        warning_threshold_usd=daily.get("warning_threshold_usd", 10.0),
        limit_threshold_usd=daily.get("limit_threshold_usd", 50.0),
        warning_triggered=daily.get("warning_triggered", False),
        limit_exceeded=daily.get("limit_exceeded", False),
        budget_remaining_usd=daily.get("budget_remaining_usd", 50.0),
        premium_tier_blocked=daily.get("limit_exceeded", False),
        cost_by_tier=_convert_dict_enum_keys(stats.get("cost_by_tier", {})),
        count_by_tier=_convert_dict_enum_keys(stats.get("count_by_tier", {})),
    )


class ReflectionStatsResponse(BaseModel):
    """Reflection service statistics response."""

    total_cost_usd: float
    total_reflections: int
    average_cost_per_reflection: float
    cost_savings_usd: float
    cost_savings_percentage: float
    daily_cost: dict
    cost_by_tier: dict
    count_by_tier: dict
    percentage_by_tier: dict


@app.get(
    "/admin/reflection/stats",
    response_model=ReflectionStatsResponse,
    tags=["Admin"],
    summary="Get reflection generation statistics",
    description="Returns detailed statistics about reflection generation including cost savings. Requires X-Admin-API-Key header if ADMIN_API_KEY is configured.",
)
async def get_reflection_stats(_: None = Depends(require_admin_access)) -> ReflectionStatsResponse:
    """
    Get reflection generation statistics.

    Returns comprehensive statistics about reflection generation:
    - Total cost and reflection count
    - Cost breakdown by tier (cheap/cached/premium)
    - Cost savings vs all-premium baseline
    - Daily cost tracking

    Security:
        - Requires X-Admin-API-Key header if ADMIN_API_KEY environment variable is set.
        - If ADMIN_API_KEY is not configured, endpoint is unprotected (warning logged).

    Returns:
        ReflectionStatsResponse: Reflection statistics
    """
    if not reflection_service:
        return ReflectionStatsResponse(
            total_cost_usd=0.0,
            total_reflections=0,
            average_cost_per_reflection=0.0,
            cost_savings_usd=0.0,
            cost_savings_percentage=0.0,
            daily_cost={},
            cost_by_tier={},
            count_by_tier={},
            percentage_by_tier={},
        )

    stats = reflection_service.get_stats()

    return ReflectionStatsResponse(
        total_cost_usd=stats.get("total_cost_usd", 0.0),
        total_reflections=stats.get("total_reflections", 0),
        average_cost_per_reflection=stats.get("average_cost_per_reflection", 0.0),
        cost_savings_usd=stats.get("cost_savings_usd", 0.0),
        cost_savings_percentage=stats.get("cost_savings_percentage", 0.0),
        daily_cost=stats.get("daily_cost", {}),
        cost_by_tier=_convert_dict_enum_keys(stats.get("cost_by_tier", {})),
        count_by_tier=_convert_dict_enum_keys(stats.get("count_by_tier", {})),
        percentage_by_tier=_convert_dict_enum_keys(stats.get("percentage_by_tier", {})),
    )


# Root endpoint
@app.get("/", tags=["System"])
async def root() -> dict[str, Any]:
    """
    Root endpoint with API information.
    """
    return {
        "service": "Episodic Memory API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "stats": "/stats",
        "admin": {
            "budget": "/admin/budget",
            "reflection_stats": "/admin/reflection/stats",
        },
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
