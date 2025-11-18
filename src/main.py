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
from typing import Optional

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
    require_active_customer,
)
from src.config import get_settings
from src.ingestion.capture import IngestionPipeline
from src.ingestion.embedding import EmbeddingService
from src.ingestion.reflection import ReflectionService
from src.kyrodb.client import KyroDBError
from src.kyrodb.router import KyroDBRouter
from src.models.customer import Customer
from src.models.episode import EpisodeCreate
from src.models.search import SearchRequest, SearchResponse
from src.observability.metrics import (
    generate_metrics,
    track_ingestion_credits,
    track_search_credits,
    update_customer_quota_usage,
    set_kyrodb_health,
)
from src.observability.middleware import PrometheusMiddleware, ErrorTrackingMiddleware
from src.observability.logging import configure_logging, get_logger, RequestContext
from src.observability.logging_middleware import (
    StructuredLoggingMiddleware,
    SlowRequestLogger,
)
from src.retrieval.search import SearchPipeline
from src.routers import customers_router
from src.storage.database import CustomerDatabase, get_customer_db

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
reflection_service: Optional[ReflectionService] = None
ingestion_pipeline: Optional[IngestionPipeline] = None
search_pipeline: Optional[SearchPipeline] = None


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
    global ingestion_pipeline, search_pipeline

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

        # Initialize reflection service (optional)
        if settings.llm.api_key:
            logger.info("Initializing LLM reflection service...")
            reflection_service = ReflectionService(config=settings.llm)
            logger.info("✓ Reflection service initialized")
        else:
            logger.warning("No LLM API key - reflection generation disabled")
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

# Observability middleware (Phase 2 Week 5-6)
# Order matters:
# 1. StructuredLoggingMiddleware (outermost) - Sets request context
# 2. SlowRequestLogger - Logs slow requests
# 3. PrometheusMiddleware - Tracks metrics (needs request context)
# 4. ErrorTrackingMiddleware - Classifies errors
app.add_middleware(ErrorTrackingMiddleware)
app.add_middleware(PrometheusMiddleware)
app.add_middleware(
    SlowRequestLogger,
    warning_threshold_ms=settings.logging.slow_request_warning_ms,
    error_threshold_ms=settings.logging.slow_request_error_ms,
)
app.add_middleware(StructuredLoggingMiddleware)
logger.info(
    "Observability middleware registered",
    middlewares=[
        "StructuredLoggingMiddleware",
        "SlowRequestLogger",
        "PrometheusMiddleware",
        "ErrorTrackingMiddleware",
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


# Health check endpoint
class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    kyrodb_connected: bool
    embedding_service_ready: bool
    reflection_service_ready: bool


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.

    Returns service status and component availability.
    Also updates Prometheus health metrics for monitoring.
    """
    kyrodb_connected = False
    text_healthy = False
    image_healthy = False

    if kyrodb_router:
        try:
            # Check both text and image instance health
            health = await kyrodb_router.health_check()
            text_healthy = health.get("text", False)
            image_healthy = health.get("image", False)
            kyrodb_connected = text_healthy and image_healthy

            # Update Prometheus health metrics
            set_kyrodb_health("text", text_healthy)
            set_kyrodb_health("image", image_healthy)
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            set_kyrodb_health("text", False)
            set_kyrodb_health("image", False)

    return HealthResponse(
        status="healthy" if kyrodb_connected else "degraded",
        kyrodb_connected=kyrodb_connected,
        embedding_service_ready=embedding_service is not None,
        reflection_service_ready=reflection_service is not None,
    )


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

    start_time = time.perf_counter()

    try:
        # Capture episode
        episode = await ingestion_pipeline.capture_episode(
            episode_data=episode_data,
            generate_reflection=generate_reflection and reflection_service is not None,
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
