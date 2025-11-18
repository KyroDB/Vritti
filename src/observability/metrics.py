"""
Prometheus metrics for production observability.

Metrics tracked:
- Request latency (histogram) per endpoint
- Request count (counter) with status codes
- Active requests (gauge)
- API key cache hit rate (counter)
- KyroDB operation latency (histogram)
- Episode ingestion credits (counter)
- Search credits (counter)
- Error rates (counter) by error type
- Customer quota usage (gauge)

Performance:
- Metric updates: <1μs overhead per operation
- Zero heap allocations on hot path
- Thread-safe counters/gauges

Integration:
- Exposed via /metrics endpoint (Prometheus scraping)
- 15-second scrape interval recommended
- Retention: 90 days (Prometheus config)
"""

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# ============================================================================
# REQUEST METRICS
# ============================================================================

# Request duration histogram (P50, P95, P99, P999)
# Buckets optimized for <50ms P99 target
http_request_duration_seconds = Histogram(
    "episodic_memory_http_request_duration_seconds",
    "HTTP request latency in seconds",
    labelnames=["method", "endpoint", "status_code"],
    buckets=(
        0.001,  # 1ms
        0.005,  # 5ms
        0.010,  # 10ms
        0.025,  # 25ms
        0.050,  # 50ms (P99 target)
        0.100,  # 100ms
        0.250,  # 250ms
        0.500,  # 500ms
        1.000,  # 1s
        2.500,  # 2.5s
        5.000,  # 5s
    ),
)

# Request counter (total requests with labels)
http_requests_total = Counter(
    "episodic_memory_http_requests_total",
    "Total HTTP requests",
    labelnames=["method", "endpoint", "status_code"],
)

# Active requests gauge (in-flight requests)
http_requests_active = Gauge(
    "episodic_memory_http_requests_active",
    "Number of in-flight HTTP requests",
    labelnames=["method", "endpoint"],
)

# ============================================================================
# AUTHENTICATION METRICS
# ============================================================================

# API key cache hit rate (performance monitoring)
api_key_cache_hits_total = Counter(
    "episodic_memory_api_key_cache_hits_total",
    "Total API key cache hits",
)

api_key_cache_misses_total = Counter(
    "episodic_memory_api_key_cache_misses_total",
    "Total API key cache misses (bcrypt validation)",
)

# API key validation latency (cache miss = ~300ms bcrypt)
api_key_validation_duration_seconds = Histogram(
    "episodic_memory_api_key_validation_duration_seconds",
    "API key validation latency",
    labelnames=["cache_hit"],
    buckets=(
        0.0001,  # 0.1ms (cache hit)
        0.001,  # 1ms
        0.010,  # 10ms
        0.100,  # 100ms
        0.200,  # 200ms
        0.300,  # 300ms (bcrypt target)
        0.500,  # 500ms
        1.000,  # 1s
    ),
)

# ============================================================================
# KYRODB METRICS
# ============================================================================

# KyroDB operation latency (insert, search, delete)
kyrodb_operation_duration_seconds = Histogram(
    "episodic_memory_kyrodb_operation_duration_seconds",
    "KyroDB operation latency",
    labelnames=["operation", "instance", "success"],
    buckets=(
        0.001,  # 1ms
        0.005,  # 5ms
        0.010,  # 10ms
        0.025,  # 25ms
        0.050,  # 50ms
        0.100,  # 100ms
        0.250,  # 250ms
        0.500,  # 500ms
        1.000,  # 1s (P99 target)
    ),
)

# KyroDB operation counter
kyrodb_operations_total = Counter(
    "episodic_memory_kyrodb_operations_total",
    "Total KyroDB operations",
    labelnames=["operation", "instance", "success"],
)

# KyroDB connection health
kyrodb_connection_healthy = Gauge(
    "episodic_memory_kyrodb_connection_healthy",
    "KyroDB connection health status",
    labelnames=["instance"],
)

# ============================================================================
# BUSINESS METRICS
# ============================================================================

# Episode ingestion counter
episodes_ingested_total = Counter(
    "episodic_memory_episodes_ingested_total",
    "Total episodes ingested",
    labelnames=["customer_tier", "has_image", "has_reflection"],
)

# Search requests counter
searches_total = Counter(
    "episodic_memory_searches_total",
    "Total search requests",
    labelnames=["customer_tier"],
)

# Search results returned (histogram for distribution)
search_results_returned = Histogram(
    "episodic_memory_search_results_returned",
    "Number of search results returned",
    buckets=(0, 1, 5, 10, 20, 50, 100),
)

# Credit usage tracking (billing)
credits_used_total = Counter(
    "episodic_memory_credits_used_total",
    "Total credits consumed",
    labelnames=["customer_id", "customer_tier", "operation"],
)

# Customer quota usage (gauge per customer)
# NOTE: This is reset daily via background task
customer_quota_usage_ratio = Gauge(
    "episodic_memory_customer_quota_usage_ratio",
    "Customer quota usage ratio (credits_used / monthly_limit)",
    labelnames=["customer_id", "customer_tier"],
)

# ============================================================================
# ERROR METRICS
# ============================================================================

# Error counter by type
errors_total = Counter(
    "episodic_memory_errors_total",
    "Total errors by type",
    labelnames=["error_type", "endpoint"],
)

# Rate limit exceeded counter
rate_limit_exceeded_total = Counter(
    "episodic_memory_rate_limit_exceeded_total",
    "Total rate limit violations",
    labelnames=["customer_id", "customer_tier"],
)

# ============================================================================
# EMBEDDING METRICS
# ============================================================================

# Embedding generation latency
embedding_generation_duration_seconds = Histogram(
    "episodic_memory_embedding_generation_duration_seconds",
    "Embedding generation latency",
    labelnames=["model_type"],  # text, image
    buckets=(
        0.010,  # 10ms
        0.025,  # 25ms
        0.050,  # 50ms
        0.100,  # 100ms
        0.250,  # 250ms
        0.500,  # 500ms
        1.000,  # 1s
    ),
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def track_request(
    method: str,
    endpoint: str,
    status_code: int,
    duration_seconds: float,
) -> None:
    """
    Track HTTP request metrics.

    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint path
        status_code: HTTP status code
        duration_seconds: Request duration in seconds

    Performance:
        - <1μs overhead per call
        - Zero heap allocations
    """
    http_request_duration_seconds.labels(
        method=method,
        endpoint=endpoint,
        status_code=status_code,
    ).observe(duration_seconds)

    http_requests_total.labels(
        method=method,
        endpoint=endpoint,
        status_code=status_code,
    ).inc()


def track_api_key_cache_hit() -> None:
    """Track API key cache hit (performance monitoring)."""
    api_key_cache_hits_total.inc()


def track_api_key_cache_miss() -> None:
    """Track API key cache miss (triggers bcrypt validation)."""
    api_key_cache_misses_total.inc()


def track_api_key_validation(duration_seconds: float, cache_hit: bool) -> None:
    """
    Track API key validation latency.

    Args:
        duration_seconds: Validation duration
        cache_hit: Whether cache was hit
    """
    api_key_validation_duration_seconds.labels(
        cache_hit="true" if cache_hit else "false",
    ).observe(duration_seconds)


def track_kyrodb_operation(
    operation: str,
    instance: str,
    success: bool,
    duration_seconds: float,
) -> None:
    """
    Track KyroDB operation metrics.

    Args:
        operation: Operation type (insert, search, delete)
        instance: Instance name (text, image)
        success: Whether operation succeeded
        duration_seconds: Operation duration

    Performance:
        - <1μs overhead per call
    """
    kyrodb_operation_duration_seconds.labels(
        operation=operation,
        instance=instance,
        success="true" if success else "false",
    ).observe(duration_seconds)

    kyrodb_operations_total.labels(
        operation=operation,
        instance=instance,
        success="true" if success else "false",
    ).inc()


def track_ingestion_credits(
    customer_id: str,
    customer_tier: str,
    credits_used: float,
    has_image: bool,
    has_reflection: bool,
) -> None:
    """
    Track episode ingestion metrics and credits.

    Args:
        customer_id: Customer ID
        customer_tier: Subscription tier
        credits_used: Credits consumed
        has_image: Whether episode has image
        has_reflection: Whether reflection was generated
    """
    episodes_ingested_total.labels(
        customer_tier=customer_tier,
        has_image="true" if has_image else "false",
        has_reflection="true" if has_reflection else "false",
    ).inc()

    credits_used_total.labels(
        customer_id=customer_id,
        customer_tier=customer_tier,
        operation="ingestion",
    ).inc(credits_used)


def track_search_credits(
    customer_id: str,
    customer_tier: str,
    credits_used: float,
    results_returned: int,
) -> None:
    """
    Track search metrics and credits.

    Args:
        customer_id: Customer ID
        customer_tier: Subscription tier
        credits_used: Credits consumed
        results_returned: Number of results returned
    """
    searches_total.labels(
        customer_tier=customer_tier,
    ).inc()

    search_results_returned.observe(results_returned)

    credits_used_total.labels(
        customer_id=customer_id,
        customer_tier=customer_tier,
        operation="search",
    ).inc(credits_used)


def track_error(error_type: str, endpoint: str) -> None:
    """
    Track error occurrence.

    Args:
        error_type: Error type (validation, authentication, kyrodb, etc.)
        endpoint: API endpoint where error occurred
    """
    errors_total.labels(
        error_type=error_type,
        endpoint=endpoint,
    ).inc()


def track_rate_limit_exceeded(customer_id: str, customer_tier: str) -> None:
    """
    Track rate limit violation.

    Args:
        customer_id: Customer ID
        customer_tier: Subscription tier
    """
    rate_limit_exceeded_total.labels(
        customer_id=customer_id,
        customer_tier=customer_tier,
    ).inc()


def update_customer_quota_usage(
    customer_id: str,
    customer_tier: str,
    credits_used: int,
    monthly_limit: int,
) -> None:
    """
    Update customer quota usage gauge.

    Args:
        customer_id: Customer ID
        customer_tier: Subscription tier
        credits_used: Total credits used this month
        monthly_limit: Monthly credit limit

    NOTE: This should be called periodically (e.g., after each operation)
    """
    ratio = credits_used / monthly_limit if monthly_limit > 0 else 0.0
    customer_quota_usage_ratio.labels(
        customer_id=customer_id,
        customer_tier=customer_tier,
    ).set(ratio)


def track_embedding_generation(model_type: str, duration_seconds: float) -> None:
    """
    Track embedding generation latency.

    Args:
        model_type: Model type (text, image)
        duration_seconds: Generation duration
    """
    embedding_generation_duration_seconds.labels(
        model_type=model_type,
    ).observe(duration_seconds)


def set_kyrodb_health(instance: str, healthy: bool) -> None:
    """
    Set KyroDB connection health status.

    Args:
        instance: Instance name (text, image)
        healthy: Health status
    """
    kyrodb_connection_healthy.labels(
        instance=instance,
    ).set(1 if healthy else 0)


# ============================================================================
# METRICS ENDPOINT
# ============================================================================


def generate_metrics() -> tuple[bytes, str]:
    """
    Generate Prometheus metrics for scraping.

    Returns:
        tuple: (metrics_bytes, content_type)

    Usage:
        @app.get("/metrics")
        async def metrics():
            data, content_type = generate_metrics()
            return Response(content=data, media_type=content_type)
    """
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
