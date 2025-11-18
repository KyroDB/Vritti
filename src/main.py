"""
FastAPI application for Episodic Memory service.

Provides REST API for:
- Episode ingestion (failures, successes, skills)
- Semantic search with precondition matching
- Health monitoring and statistics

Designed for <50ms P99 latency.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import get_settings
from src.ingestion.capture import IngestionPipeline
from src.ingestion.embedding import EmbeddingService
from src.ingestion.reflection import ReflectionService
from src.kyrodb.client import KyroDBError
from src.kyrodb.router import KyroDBRouter
from src.models.episode import EpisodeCreate
from src.models.search import SearchRequest, SearchResponse
from src.retrieval.search import SearchPipeline
from src.routers import customers_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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

# CORS middleware
# TODO Phase 1 Week 4: Restrict origins for production (currently wildcard for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(customers_router)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with latency tracking."""
    start_time = time.perf_counter()

    # Process request
    response = await call_next(request)

    # Log request details
    latency_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Latency: {latency_ms:.2f}ms"
    )

    return response


# Exception handler for KyroDB errors
@app.exception_handler(KyroDBError)
async def kyrodb_error_handler(request: Request, exc: KyroDBError):
    """Handle KyroDB connection/operation errors."""
    logger.error(f"KyroDB error on {request.url.path}: {exc}")
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
    """
    kyrodb_connected = False
    if kyrodb_router:
        try:
            # Simple connectivity check via health endpoint
            health = await kyrodb_router.health_check()
            kyrodb_connected = health.get("text", False)
        except Exception as e:
            logger.warning(f"Health check failed: {e}")

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
async def capture_episode(
    episode_data: EpisodeCreate,
    generate_reflection: bool = True,
):
    """
    Capture and store an episode.

    Pipeline:
    1. PII redaction
    2. ID generation
    3. Multi-modal embedding
    4. KyroDB storage
    5. Async reflection generation (optional)

    Args:
        episode_data: Episode creation data
        generate_reflection: Whether to generate LLM reflection (default: True)

    Returns:
        CaptureResponse: Capture result with episode ID and metadata

    Raises:
        HTTPException: On ingestion failure

    Security:
        TODO Phase 1 Week 3: Add API key authentication middleware.
        Currently customer_id is accepted from request body (SECURITY GAP).
        Must be extracted from validated API key to prevent cross-customer access.
    """
    if not ingestion_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ingestion pipeline not initialized",
        )

    start_time = time.perf_counter()

    try:
        # Capture episode
        episode = await ingestion_pipeline.capture_episode(
            episode_data=episode_data,
            generate_reflection=generate_reflection and reflection_service is not None,
        )

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

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
async def search_episodes(request: SearchRequest):
    """
    Search for relevant episodes.

    Pipeline:
    1. Query embedding generation
    2. KyroDB vector search (k×5 candidates)
    3. Metadata filtering
    4. Precondition matching
    5. Weighted ranking
    6. Top-k selection

    Args:
        request: Search request with query and parameters

    Returns:
        SearchResponse: Ranked search results with latency breakdown

    Raises:
        HTTPException: On search failure

    Security:
        TODO Phase 1 Week 3: Add API key authentication middleware.
        Currently customer_id is accepted from request body (SECURITY GAP).
        Must be extracted from validated API key to prevent cross-customer access.
    """
    if not search_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search pipeline not initialized",
        )

    try:
        # Execute search
        response = await search_pipeline.search(request)

        # Log slow queries
        if response.search_latency_ms > 100:
            logger.warning(
                f"Slow query detected: {response.search_latency_ms:.2f}ms "
                f"(goal: {request.goal[:50]}...)"
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
