"""
Tests for Prometheus metrics observability (Phase 2 Week 5).

Tests:
- Metrics endpoint returns valid Prometheus format
- Request metrics are tracked correctly
- API key cache hit/miss tracking
- Business metrics tracking
- Error tracking
"""

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.observability.metrics import (
    set_kyrodb_health,
    track_api_key_cache_hit,
    track_api_key_cache_miss,
    track_error,
    track_ingestion_credits,
    track_request,
    track_search_credits,
    update_customer_quota_usage,
)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_metrics_endpoint_exists(client):
    """Test that /metrics endpoint exists and returns data."""
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert len(response.content) > 0


def test_metrics_prometheus_format(client):
    """Test that metrics are in valid Prometheus format."""
    response = client.get("/metrics")

    content = response.text

    # Check for required Prometheus metric types
    assert "# TYPE" in content  # Metric type declarations
    assert "# HELP" in content  # Metric help text

    # Check for our custom metrics
    assert "episodic_memory_http_request_duration_seconds" in content
    assert "episodic_memory_http_requests_total" in content
    assert "episodic_memory_api_key_cache_hits_total" in content
    assert "episodic_memory_api_key_cache_misses_total" in content


def test_metrics_track_requests():
    """Test that request metrics are tracked correctly."""
    # Track a sample request
    track_request(
        method="POST",
        endpoint="/api/v1/capture",
        status_code=201,
        duration_seconds=0.05,
    )

    # Metrics should be recorded (we can't easily verify values without
    # accessing internal Prometheus registry, but we can verify no errors)
    # If metrics tracking failed, it would raise an exception


def test_metrics_track_api_key_cache():
    """Test API key cache hit/miss tracking."""
    track_api_key_cache_hit()
    track_api_key_cache_miss()

    # Should not raise any exceptions


def test_metrics_track_ingestion_credits():
    """Test ingestion credit tracking."""
    track_ingestion_credits(
        customer_id="test-customer",
        customer_tier="pro",
        credits_used=1.7,
        has_image=True,
        has_reflection=True,
    )

    # Should not raise any exceptions


def test_metrics_track_search_credits():
    """Test search credit tracking."""
    track_search_credits(
        customer_id="test-customer",
        customer_tier="pro",
        credits_used=0.1,
        results_returned=5,
    )

    # Should not raise any exceptions


def test_metrics_track_errors():
    """Test error tracking."""
    track_error(error_type="validation", endpoint="/api/v1/capture")
    track_error(error_type="authentication", endpoint="/api/v1/search")
    track_error(error_type="kyrodb", endpoint="/api/v1/capture")

    # Should not raise any exceptions


def test_metrics_customer_quota_gauge():
    """Test customer quota usage gauge."""
    update_customer_quota_usage(
        customer_id="test-customer",
        customer_tier="pro",
        credits_used=5000,
        monthly_limit=100000,
    )

    # Should not raise any exceptions


def test_metrics_kyrodb_health_gauge():
    """Test KyroDB health status gauge."""
    set_kyrodb_health("text", True)
    set_kyrodb_health("image", False)

    # Should not raise any exceptions


def test_health_endpoint_updates_metrics(client):
    """Test that /health endpoint updates KyroDB health metrics."""
    response = client.get("/health")

    assert response.status_code == 200

    # Check that health metrics were updated (visible in /metrics)
    metrics_response = client.get("/metrics")
    content = metrics_response.text

    # Should contain KyroDB health metric
    assert "episodic_memory_kyrodb_connection_healthy" in content


def test_middleware_tracks_requests(client):
    """Test that PrometheusMiddleware tracks all requests."""
    # Make a request to any endpoint
    response = client.get("/health")

    assert response.status_code == 200

    # Check that request was tracked in metrics
    metrics_response = client.get("/metrics")
    content = metrics_response.text

    # Should see request counter increment
    assert "episodic_memory_http_requests_total" in content
    assert "episodic_memory_http_request_duration_seconds" in content


def test_middleware_normalizes_endpoint_paths(client):
    """Test that PrometheusMiddleware normalizes dynamic path segments."""
    # The middleware should normalize paths like /customers/{id} to prevent
    # unbounded metric cardinality

    # Make request to /health (static path)
    response1 = client.get("/health")
    assert response1.status_code == 200

    # Make request to /stats (static path)
    response2 = client.get("/stats")
    assert response2.status_code == 200

    # Metrics should be tracked with normalized paths
    metrics_response = client.get("/metrics")
    content = metrics_response.text

    # Both requests should be tracked
    assert 'endpoint="/health"' in content
    assert 'endpoint="/stats"' in content


def test_metrics_endpoint_not_rate_limited(client):
    """Test that /metrics endpoint is not rate limited."""
    # Metrics endpoint should be accessible without authentication
    # and without rate limiting (for Prometheus scraping)

    for _ in range(10):
        response = client.get("/metrics")
        assert response.status_code == 200


def test_metrics_content_type_header(client):
    """Test that /metrics endpoint returns correct content type."""
    response = client.get("/metrics")

    # Prometheus expects specific content type
    assert "content-type" in response.headers
    assert "text/plain" in response.headers["content-type"]
    # Prometheus format includes version (varies by prometheus_client library)
    assert "version=" in response.headers["content-type"]


def test_metrics_include_histogram_buckets(client):
    """Test that histogram metrics include bucket definitions."""
    response = client.get("/metrics")
    content = response.text

    # Histograms should have bucket labels
    assert "_bucket{" in content
    assert "le=" in content  # "less than or equal" bucket label

    # Check for our custom histogram buckets
    assert "episodic_memory_http_request_duration_seconds_bucket" in content


def test_metrics_include_counter_totals(client):
    """Test that counter metrics include _total suffix."""
    response = client.get("/metrics")
    content = response.text

    # Counters should have _total suffix (Prometheus convention)
    assert "episodic_memory_http_requests_total" in content
    assert "episodic_memory_api_key_cache_hits_total" in content
    assert "episodic_memory_credits_used_total" in content


def test_metrics_include_gauge_values(client):
    """Test that gauge metrics include current values."""
    # Set some gauge values
    set_kyrodb_health("text", True)
    set_kyrodb_health("image", False)

    response = client.get("/metrics")
    content = response.text

    # Gauges should show current state
    assert "episodic_memory_kyrodb_connection_healthy" in content
    assert 'instance="text"' in content
    assert 'instance="image"' in content
