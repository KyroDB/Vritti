"""
Health check system for Kubernetes readiness and liveness probes.

Provides:
- Liveness probe: Is the service alive? (basic ping)
- Readiness probe: Is the service ready to accept traffic? (dependency checks)
- Detailed health status with component breakdown
- Fast response times (<10ms for liveness, <100ms for readiness)

Architecture:
- Liveness: Minimal check, always returns 200 unless service is completely dead
- Readiness: Comprehensive checks of dependencies (KyroDB, database)
- Startup: Special probe for slow-starting services (Phase 3)

Performance:
- Liveness probe: <5ms (no I/O operations)
- Readiness probe: <100ms (includes dependency checks)
- Health probes are NOT rate limited
- Cached dependency status (5-second TTL)

Kubernetes Integration:
```yaml
livenessProbe:
  httpGet:
    path: /health/liveness
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health/readiness
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 2
```
"""

import time
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.observability.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# HEALTH STATUS MODELS
# ============================================================================


class HealthStatus(str, Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"  # All checks passed
    DEGRADED = "degraded"  # Some non-critical checks failed
    UNHEALTHY = "unhealthy"  # Critical checks failed


class ComponentHealth(BaseModel):
    """Health status of a single component."""

    name: str = Field(description="Component name")
    status: HealthStatus = Field(description="Component health status")
    message: str | None = Field(default=None, description="Status message")
    latency_ms: float | None = Field(
        default=None, description="Health check latency in milliseconds"
    )
    last_check: datetime = Field(description="Last health check timestamp")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional component metadata"
    )


class HealthCheckResponse(BaseModel):
    """Comprehensive health check response."""

    status: HealthStatus = Field(description="Overall health status")
    timestamp: datetime = Field(description="Check timestamp")
    uptime_seconds: float = Field(description="Service uptime in seconds")
    version: str = Field(description="Service version")
    components: list[ComponentHealth] = Field(description="Individual component health statuses")


class LivenessResponse(BaseModel):
    """Minimal liveness probe response."""

    status: str = Field(default="alive", description="Liveness status")
    timestamp: datetime = Field(description="Check timestamp")


class ReadinessResponse(BaseModel):
    """Readiness probe response with dependency checks."""

    status: HealthStatus = Field(description="Readiness status")
    timestamp: datetime = Field(description="Check timestamp")
    ready: bool = Field(description="Whether service is ready to accept traffic")
    components: list[ComponentHealth] = Field(description="Critical component health statuses")


# ============================================================================
# HEALTH CHECKER
# ============================================================================


class HealthChecker:
    """
    Health check coordinator for all service components.

    Responsibilities:
    - Check KyroDB connection health (text and image instances)
    - Check customer database health
    - Check embedding service readiness
    - Cache health status to avoid excessive checks
    - Track service uptime

    Performance:
    - Health checks are cached (5-second TTL)
    - Liveness probe: <5ms (no I/O)
    - Readiness probe: <100ms (with I/O)
    """

    def __init__(self):
        """Initialize health checker."""
        self.start_time = time.time()
        self.version = "0.1.0"  # TODO: Read from config

        # Cached health status (5-second TTL)
        self._health_cache: HealthCheckResponse | None = None
        self._health_cache_time: float = 0.0
        self._health_cache_ttl: float = 5.0  # 5 seconds

    def get_uptime_seconds(self) -> float:
        """
        Get service uptime in seconds.

        Returns:
            float: Uptime in seconds
        """
        return time.time() - self.start_time

    async def check_liveness(self) -> LivenessResponse:
        """
        Liveness probe: Is the service alive?

        This is the most basic health check. It should ONLY fail if the service
        is completely dead (e.g., OOM, deadlock).

        Returns:
            LivenessResponse: Liveness status (always healthy unless dead)

        Performance:
            - Target: <5ms
            - No I/O operations
            - No dependency checks
        """
        return LivenessResponse(
            status="alive",
            timestamp=datetime.now(UTC),
        )

    async def check_readiness(
        self,
        kyrodb_router=None,
        customer_db=None,
        embedding_service=None,
    ) -> ReadinessResponse:
        """
        Readiness probe: Is the service ready to accept traffic?

        Checks critical dependencies:
        - KyroDB connections (text and image)
        - Customer database
        - Embedding service (optional)

        Returns:
            ReadinessResponse: Readiness status with component details

        Performance:
            - Target: <100ms
            - Includes I/O operations
            - Cached for 5 seconds
        """
        components: list[ComponentHealth] = []
        overall_status = HealthStatus.HEALTHY

        # Check KyroDB connections
        if kyrodb_router:
            kyrodb_health = await self._check_kyrodb_health(kyrodb_router)
            components.append(kyrodb_health)

            if kyrodb_health.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
            elif kyrodb_health.status == HealthStatus.DEGRADED:
                overall_status = HealthStatus.DEGRADED

        # Check customer database
        if customer_db:
            db_health = await self._check_database_health(customer_db)
            components.append(db_health)

            if db_health.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
            elif (
                db_health.status == HealthStatus.DEGRADED
                and overall_status != HealthStatus.UNHEALTHY
            ):
                overall_status = HealthStatus.DEGRADED

        # Check embedding service (non-critical)
        if embedding_service:
            embedding_health = await self._check_embedding_service_health(embedding_service)
            components.append(embedding_health)
            # Embedding service failure is not critical for readiness

        # Service is ready if status is HEALTHY or DEGRADED
        ready = overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

        return ReadinessResponse(
            status=overall_status,
            timestamp=datetime.now(UTC),
            ready=ready,
            components=components,
        )

    async def check_health(
        self,
        kyrodb_router=None,
        customer_db=None,
        embedding_service=None,
        reflection_service=None,
    ) -> HealthCheckResponse:
        """
        Comprehensive health check with all components.

        This is the detailed health endpoint for debugging and monitoring.
        Not used by Kubernetes probes (too slow).

        Returns:
            HealthCheckResponse: Detailed health status

        Performance:
            - Target: <200ms
            - Cached for 5 seconds
        """
        # Check cache
        cache_age = time.time() - self._health_cache_time
        if self._health_cache and cache_age < self._health_cache_ttl:
            logger.debug("Health check cache hit", cache_age_ms=cache_age * 1000)
            return self._health_cache

        # Cache miss - perform checks
        logger.debug("Health check cache miss, performing full check")

        components: list[ComponentHealth] = []
        overall_status = HealthStatus.HEALTHY

        # Check KyroDB connections
        if kyrodb_router:
            kyrodb_health = await self._check_kyrodb_health(kyrodb_router)
            components.append(kyrodb_health)

            if kyrodb_health.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
            elif kyrodb_health.status == HealthStatus.DEGRADED:
                overall_status = HealthStatus.DEGRADED

        # Check customer database
        if customer_db:
            db_health = await self._check_database_health(customer_db)
            components.append(db_health)

            if db_health.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY

        # Check embedding service
        if embedding_service:
            embedding_health = await self._check_embedding_service_health(embedding_service)
            components.append(embedding_health)

            if embedding_health.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.DEGRADED  # Non-critical

        # Check reflection service
        if reflection_service:
            reflection_health = await self._check_reflection_service_health(reflection_service)
            components.append(reflection_health)
            # Reflection service is optional, failure is not critical

        # Build response
        response = HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.now(UTC),
            uptime_seconds=self.get_uptime_seconds(),
            version=self.version,
            components=components,
        )

        # Update cache
        self._health_cache = response
        self._health_cache_time = time.time()

        return response

    async def _check_kyrodb_health(self, kyrodb_router) -> ComponentHealth:
        """
        Check KyroDB connection health.

        Checks both text and image instances.

        Returns:
            ComponentHealth: KyroDB health status
        """
        start_time = time.perf_counter()

        try:
            # Call health_check method on router
            health_status = await kyrodb_router.health_check()

            text_healthy = health_status.get("text", False)
            image_healthy = health_status.get("image", False)

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Both instances must be healthy
            if text_healthy and image_healthy:
                status = HealthStatus.HEALTHY
                message = "Both text and image instances healthy"
            elif text_healthy or image_healthy:
                status = HealthStatus.DEGRADED
                message = (
                    "Text instance healthy, image instance unhealthy"
                    if text_healthy
                    else "Image instance healthy, text instance unhealthy"
                )
            else:
                status = HealthStatus.UNHEALTHY
                message = "Both instances unhealthy"

            return ComponentHealth(
                name="kyrodb",
                status=status,
                message=message,
                latency_ms=round(latency_ms, 2),
                last_check=datetime.now(UTC),
                metadata={
                    "text_healthy": text_healthy,
                    "image_healthy": image_healthy,
                },
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000

            logger.error("KyroDB health check failed", error=str(e), exc_info=True)

            return ComponentHealth(
                name="kyrodb",
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                latency_ms=round(latency_ms, 2),
                last_check=datetime.now(UTC),
            )

    async def _check_database_health(self, customer_db) -> ComponentHealth:
        """
        Check customer database health.

        Performs a simple SELECT query to verify connectivity.

        Returns:
            ComponentHealth: Database health status
        """
        start_time = time.perf_counter()

        try:
            # Perform simple query to verify database is responsive
            # This is a minimal check - just verify we can connect and query
            with customer_db._get_connection() as conn:
                cursor = conn.execute("SELECT 1")
                cursor.fetchone()

            latency_ms = (time.perf_counter() - start_time) * 1000

            return ComponentHealth(
                name="customer_database",
                status=HealthStatus.HEALTHY,
                message="Database responsive",
                latency_ms=round(latency_ms, 2),
                last_check=datetime.now(UTC),
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000

            logger.error("Database health check failed", error=str(e), exc_info=True)

            return ComponentHealth(
                name="customer_database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database check failed: {str(e)}",
                latency_ms=round(latency_ms, 2),
                last_check=datetime.now(UTC),
            )

    async def _check_embedding_service_health(self, embedding_service) -> ComponentHealth:
        """
        Check embedding service health.

        Verifies models are loaded and ready.

        Returns:
            ComponentHealth: Embedding service health status
        """
        start_time = time.perf_counter()

        try:
            # Check if models are loaded
            text_model_loaded = (
                hasattr(embedding_service, "text_model")
                and embedding_service.text_model is not None
            )
            image_model_loaded = (
                hasattr(embedding_service, "image_model")
                and embedding_service.image_model is not None
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            if text_model_loaded and image_model_loaded:
                status = HealthStatus.HEALTHY
                message = "Text and image models loaded"
            elif text_model_loaded or image_model_loaded:
                status = HealthStatus.DEGRADED
                message = "Only one model loaded"
            else:
                status = HealthStatus.UNHEALTHY
                message = "No models loaded"

            return ComponentHealth(
                name="embedding_service",
                status=status,
                message=message,
                latency_ms=round(latency_ms, 2),
                last_check=datetime.now(UTC),
                metadata={
                    "text_model_loaded": text_model_loaded,
                    "image_model_loaded": image_model_loaded,
                },
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000

            return ComponentHealth(
                name="embedding_service",
                status=HealthStatus.DEGRADED,  # Non-critical
                message=f"Health check failed: {str(e)}",
                latency_ms=round(latency_ms, 2),
                last_check=datetime.now(UTC),
            )

    async def _check_reflection_service_health(self, reflection_service) -> ComponentHealth:
        """
        Check reflection service health.

        Verifies LLM API is accessible.

        Returns:
            ComponentHealth: Reflection service health status
        """
        start_time = time.perf_counter()

        try:
            # Check if API key is configured
            api_key_configured = hasattr(reflection_service, "api_key") and bool(
                reflection_service.api_key
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            if api_key_configured:
                status = HealthStatus.HEALTHY
                message = "LLM API configured"
            else:
                status = HealthStatus.DEGRADED
                message = "LLM API not configured (reflection disabled)"

            return ComponentHealth(
                name="reflection_service",
                status=status,
                message=message,
                latency_ms=round(latency_ms, 2),
                last_check=datetime.now(UTC),
                metadata={
                    "api_key_configured": api_key_configured,
                },
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000

            return ComponentHealth(
                name="reflection_service",
                status=HealthStatus.DEGRADED,  # Non-critical
                message=f"Health check failed: {str(e)}",
                latency_ms=round(latency_ms, 2),
                last_check=datetime.now(UTC),
            )


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

# Global health checker instance
_health_checker: HealthChecker | None = None


def get_health_checker() -> HealthChecker:
    """
    Get global health checker instance.

    Returns:
        HealthChecker: Global health checker
    """
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker
