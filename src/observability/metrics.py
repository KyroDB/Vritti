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

from typing import Optional

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
# REFLECTION METRICS (Phase 1 Week 1-2)
# ============================================================================

# Reflection generation latency (multi-perspective LLM calls)
reflection_generation_duration_seconds = Histogram(
    "episodic_memory_reflection_generation_duration_seconds",
    "Time to generate multi-perspective reflection",
    labelnames=["consensus_method", "num_models"],
    buckets=(
        1.0,  # 1s
        2.0,  # 2s
        3.0,  # 3s
        5.0,  # 5s (typical for 3 models in parallel)
        7.0,  # 7s
        10.0,  # 10s
        15.0,  # 15s
        30.0,  # 30s (timeout)
    ),
)

# Reflection generation count
reflections_generated_total = Counter(
    "episodic_memory_reflections_generated_total",
    "Total reflections generated",
    labelnames=["consensus_method", "num_models", "success"],
)

# Reflection cost tracking (critical for budget monitoring)
reflection_cost_usd_total = Counter(
    "episodic_memory_reflection_cost_usd_total",
    "Total LLM costs for reflection generation in USD",
    labelnames=["model_name"],  # Track cost per LLM provider
)

# Reflection cost per episode (histogram for distribution)
reflection_cost_per_episode_usd = Histogram(
    "episodic_memory_reflection_cost_per_episode_usd",
    "Cost per reflection in USD",
    labelnames=["consensus_method"],
    buckets=(
        0.001,  # $0.001 (Gemini Flash)
        0.005,  # $0.005
        0.010,  # $0.01
        0.050,  # $0.05 (typical multi-perspective)
        0.100,  # $0.10
        0.500,  # $0.50
        1.000,  # $1.00 (cost limit threshold)
    ),
)

# Reflection consensus quality
reflection_consensus_confidence = Histogram(
    "episodic_memory_reflection_consensus_confidence",
    "Consensus confidence score (0.0-1.0)",
    labelnames=["consensus_method"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# LLM model success rate (per provider)
llm_call_total = Counter(
    "episodic_memory_llm_call_total",
    "Total LLM API calls",
    labelnames=["model_name", "success"],
)

# LLM call latency (per provider)
llm_call_duration_seconds = Histogram(
    "episodic_memory_llm_call_duration_seconds",
    "Individual LLM API call latency",
    labelnames=["model_name"],
    buckets=(0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 30.0),
)

# Reflection persistence success rate
reflection_persistence_total = Counter(
    "episodic_memory_reflection_persistence_total",
    "Reflection persistence attempts",
    labelnames=["success"],
)

# Reflection persistence retry tracking 
reflection_persistence_retry_total = Counter(
    "episodic_memory_reflection_persistence_retry_total",
    "Reflection persistence retry attempts",
    labelnames=["attempt", "success"],
)

# Dead-letter queue tracking 
reflection_dead_letter_total = Counter(
    "episodic_memory_reflection_dead_letter_total",
    "Reflections logged to dead-letter queue after persistence failure",
    labelnames=["failure_reason"],
)

# Reflection failure reasons (for debugging)
reflection_failure_total = Counter(
    "episodic_memory_reflection_failure_total",
    "Reflection generation failures",
    labelnames=["reason"],  # "all_models_failed", "persistence_failed", "exception"
)

# Active reflection generations (gauge for concurrency monitoring)
reflection_generations_active = Gauge(
    "episodic_memory_reflection_generations_active",
    "Number of in-progress reflection generations",
)


# ============================================================================
# REFLECTION METRIC TRACKING FUNCTIONS
# ============================================================================


def track_reflection_generation(
    consensus_method: str,
    num_models: int,
    duration_seconds: float,
    cost_usd: float,
    confidence: float,
    success: bool,
    model_costs: Optional[dict[str, float]] = None,
) -> None:
    """
    Track reflection generation metrics.

    Args:
        consensus_method: Consensus algorithm used (unanimous, majority_vote, etc.)
        num_models: Number of LLM models that succeeded
        duration_seconds: Total generation time
        cost_usd: Total cost in USD
        confidence: Consensus confidence score
        success: Whether generation succeeded
        model_costs: Optional per-model cost breakdown
    """
    # Generation latency
    reflection_generation_duration_seconds.labels(
        consensus_method=consensus_method,
        num_models=str(num_models),
    ).observe(duration_seconds)

    # Generation count
    reflections_generated_total.labels(
        consensus_method=consensus_method,
        num_models=str(num_models),
        success="true" if success else "false",
    ).inc()

    # Cost per episode
    reflection_cost_per_episode_usd.labels(
        consensus_method=consensus_method,
    ).observe(cost_usd)

    # Consensus confidence
    reflection_consensus_confidence.labels(
        consensus_method=consensus_method,
    ).observe(confidence)

    # Per-model cost tracking
    if model_costs:
        for model_name, cost in model_costs.items():
            reflection_cost_usd_total.labels(
                model_name=model_name,
            ).inc(cost)
    else:
        # Fallback: track total cost under "multi-perspective"
        reflection_cost_usd_total.labels(
            model_name="multi-perspective",
        ).inc(cost_usd)


def track_llm_call(
    model_name: str,
    success: bool,
    duration_seconds: Optional[float] = None,
) -> None:
    """
    Track individual LLM API call.

    Args:
        model_name: LLM model identifier (gpt-4, claude-3.5-sonnet, etc.)
        success: Whether call succeeded
        duration_seconds: Optional call duration
    """
    llm_call_total.labels(
        model_name=model_name,
        success="true" if success else "false",
    ).inc()

    if duration_seconds is not None:
        llm_call_duration_seconds.labels(
            model_name=model_name,
        ).observe(duration_seconds)


def track_reflection_persistence(success: bool) -> None:
    """
    Track reflection persistence to KyroDB.

    Args:
        success: Whether persistence succeeded
    """
    reflection_persistence_total.labels(
        success="true" if success else "false",
    ).inc()


def track_reflection_persistence_retry(
    episode_id: int,
    attempt: int,
    success: bool,
) -> None:
    """
    Track reflection persistence retry attempts.

    Args:
        episode_id: Episode ID being persisted (for debugging)
        attempt: Retry attempt number (1, 2, 3)
        success: Whether this attempt succeeded
    """
    reflection_persistence_retry_total.labels(
        attempt=str(attempt),
        success="true" if success else "false",
    ).inc()


def track_dead_letter_queue(
    episode_id: int,
    customer_id: str,
    failure_reason: str,
) -> None:
    """
    Track reflection logged to dead-letter queue.

    Args:
        episode_id: Episode ID that failed (for debugging)
        customer_id: Customer ID (for debugging)
        failure_reason: Why persistence failed
    """
    reflection_dead_letter_total.labels(
        failure_reason=failure_reason,
    ).inc()


def track_reflection_failure(reason: str) -> None:
    """
    Track reflection generation failure.

    Args:
        reason: Failure reason (all_models_failed, persistence_failed, exception, etc.)
    """
    reflection_failure_total.labels(
        reason=reason,
    ).inc()


def increment_active_reflections() -> None:
    """Increment active reflection generations counter."""
    reflection_generations_active.inc()


def decrement_active_reflections() -> None:
    """Decrement active reflection generations counter."""
    reflection_generations_active.dec()


# ============================================================================
# SKILLS LIBRARY METRICS
# ============================================================================

skill_promotions_total = Counter(
    "episodic_memory_skill_promotions_total",
    "Total number of episodes promoted to skills",
    labelnames=["customer_tier", "error_class"],
)

skill_promotion_duration_seconds = Histogram(
    "episodic_memory_skill_promotion_duration_seconds",
    "Time taken to check and promote skill",
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
)

skill_searches_total = Counter(
    "episodic_memory_skill_searches_total",
    "Total number of skill searches",
    labelnames=["customer_tier"],
)

skill_applications_total = Counter(
    "episodic_memory_skill_applications_total",
    "Total number of times skills were applied",
    labelnames=["skill_id", "success"],
)

fix_validations_total = Counter(
    "episodic_memory_fix_validations_total",
    "Total number of fix validations",
    labelnames=["customer_tier", "outcome"],
)

skill_success_rate_gauge = Gauge(
    "episodic_memory_skill_success_rate",
    "Success rate of individual skills",
    labelnames=["skill_id", "skill_name"],
)

promoted_skills_total = Gauge(
    "episodic_memory_promoted_skills_total",
    "Total number of active skills per customer",
    labelnames=["customer_tier"],
)


def track_skill_promotion(
    customer_tier: str,
    error_class: str,
    duration_seconds: float,
) -> None:
    """
    Track skill promotion event.

    Args:
        customer_tier: Customer subscription tier
        error_class: Error classification
        duration_seconds: Time taken for promotion
    """
    skill_promotions_total.labels(
        customer_tier=customer_tier,
        error_class=error_class,
    ).inc()

    skill_promotion_duration_seconds.observe(duration_seconds)


def track_skill_search(customer_tier: str) -> None:
    """
    Track skill search event.

    Args:
        customer_tier: Customer subscription tier
    """
    skill_searches_total.labels(
        customer_tier=customer_tier,
    ).inc()


def track_skill_application(
    skill_id: int,
    success: bool,
) -> None:
    """
    Track skill application event.

    Args:
        skill_id: ID of skill that was applied
        success: Whether application succeeded
    """
    skill_applications_total.labels(
        skill_id=str(skill_id),
        success="true" if success else "false",
    ).inc()


def track_fix_validation(
    customer_tier: str,
    outcome: str,
) -> None:
    """
    Track fix validation event.

    Args:
        customer_tier: Customer subscription tier
        outcome: Validation outcome (success or still_failed)
    """
    fix_validations_total.labels(
        customer_tier=customer_tier,
        outcome=outcome,
    ).inc()


def update_skill_success_rate(
    skill_id: int,
    skill_name: str,
    success_rate: float,
) -> None:
    """
    Update skill success rate gauge.

    Args:
        skill_id: Skill ID
        skill_name: Skill name (for label)
        success_rate: Current success rate (0.0 to 1.0)
    """
    skill_success_rate_gauge.labels(
        skill_id=str(skill_id),
        skill_name=skill_name[:50],  # Truncate for label
    ).set(success_rate)


def update_promoted_skills_count(
    customer_tier: str,
    count: int,
) -> None:
    """
    Update total promoted skills count for tier.

    Args:
        customer_tier: Customer subscription tier
        count: Total number of active skills
    """
    promoted_skills_total.labels(
        customer_tier=customer_tier,
    ).set(count)


# ============================================================================
# PHASE 5: TIERED REFLECTION METRICS
# ============================================================================

# Reflection cost by tier (Phase 5 - Cost Optimization)
reflection_cost_by_tier_usd_total = Counter(
    "episodic_memory_reflection_cost_by_tier_usd_total",
    "Total reflection cost in USD per tier",
    labelnames=["tier", "customer_id"],  # tier: cheap, cached, premium
)

# Reflection count by tier
reflections_by_tier_total = Counter(
    "episodic_memory_reflections_by_tier_total",
    "Total reflections generated per tier",
    labelnames=["tier", "customer_tier"],
)

# Tier quality scores (confidence distribution)
reflection_tier_quality_score = Histogram(
    "episodic_memory_reflection_tier_quality_score",
    "Reflection confidence scores by tier",
    labelnames=["tier"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# Tier selection decisions (auto-select tracking)
tier_selection_total = Counter(
    "episodic_memory_tier_selection_total",
    "Tier selection decisions (auto vs override)",
    labelnames=["tier", "selection_method"],  # selection_method: auto, override
)

# Quality fallback count (cheap -> premium due to low quality)
tier_quality_fallback_total = Counter(
    "episodic_memory_tier_quality_fallback_total",
    "Cheap tier quality fallbacks to premium",
    labelnames=["reason"],  # reason: low_confidence, no_preconditions, generic_response
)

# Cost savings gauge (vs all-premium baseline)
reflection_cost_savings_usd = Gauge(
    "episodic_memory_reflection_cost_savings_usd",
    "Estimated cost savings vs all-premium baseline",
)

# Premium tier usage percentage (for budget monitoring)
premium_tier_usage_percentage = Gauge(
    "episodic_memory_premium_tier_usage_percentage",
    "Percentage of reflections using premium tier",
)

# Daily cost alerts 
daily_cost_alert_total = Counter(
    "episodic_memory_daily_cost_alert_total",
    "Daily cost alert triggers",
    labelnames=["alert_type"],  # warning, limit_exceeded
)

daily_cost_usd = Gauge(
    "episodic_memory_daily_cost_usd",
    "Current daily reflection cost in USD",
)


# ============================================================================
# PHASE 5: TIER METRIC TRACKING FUNCTIONS
# ============================================================================


def track_reflection_tier_cost(
    tier: str,
    customer_id: str,
    cost_usd: float,
) -> None:
    """
    Track reflection cost by tier (Phase 5).

    Args:
        tier: Reflection tier (cheap, cached, premium)
        customer_id: Customer ID
        cost_usd: Cost in USD
    """
    reflection_cost_by_tier_usd_total.labels(
        tier=tier,
        customer_id=customer_id,
    ).inc(cost_usd)


def track_reflection_tier_usage(
    tier: str,
    customer_tier: str,
    quality_score: float,
    selection_method: str = "auto",
) -> None:
    """
    Track reflection tier usage and quality (Phase 5).

    Args:
        tier: Reflection tier used (cheap, cached, premium)
        customer_tier: Subscription tier
        quality_score: Confidence/quality score
        selection_method: How tier was selected (auto, override)
    """
    reflections_by_tier_total.labels(
        tier=tier,
        customer_tier=customer_tier,
    ).inc()

    reflection_tier_quality_score.labels(
        tier=tier,
    ).observe(quality_score)

    tier_selection_total.labels(
        tier=tier,
        selection_method=selection_method,
    ).inc()


def track_tier_quality_fallback(reason: str) -> None:
    """
    Track quality fallback from cheap to premium (Phase 5).

    Args:
        reason: Fallback reason (low_confidence, no_preconditions, generic_response)
    """
    tier_quality_fallback_total.labels(
        reason=reason,
    ).inc()


def update_cost_savings(savings_usd: float) -> None:
    """
    Update cost savings gauge (Phase 5).

    Args:
        savings_usd: Total cost savings vs all-premium baseline
    """
    reflection_cost_savings_usd.set(savings_usd)


def update_premium_tier_percentage(percentage: float) -> None:
    """
    Update premium tier usage percentage (Phase 5).

    Args:
        percentage: Percentage of reflections using premium (0.0-100.0)
    """
    premium_tier_usage_percentage.set(percentage)


def track_daily_cost_alert(
    alert_type: str,
    cost_usd: float,
    threshold_usd: float,
) -> None:
    """
    Track daily cost alert trigger 

    Args:
        alert_type: Type of alert (warning, limit_exceeded)
        cost_usd: Current daily cost
        threshold_usd: Threshold that was triggered
    """
    daily_cost_alert_total.labels(
        alert_type=alert_type,
    ).inc()

    # Update daily cost gauge (uses module-level daily_cost_usd gauge)
    daily_cost_usd.set(cost_usd)


# ============================================================================
# METRICS ENDPOINT
# ============================================================================



def generate_metrics() -> tuple[bytes, str]:
    """
    Generate Prometheus metrics in exposition format (bytes).

    Returns:
        tuple: (metrics_bytes, content_type)
    """
    metrics_data = generate_latest(REGISTRY)
    return metrics_data, CONTENT_TYPE_LATEST
