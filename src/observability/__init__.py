"""
Observability infrastructure for production monitoring.

Components:
- metrics.py: Prometheus metrics (counters, histograms, gauges)
- logging.py: Structured JSON logging with request context
- tracing.py: Distributed tracing (Phase 2 Week 7)
"""

from src.observability.metrics import (
    track_api_key_cache_hit,
    track_api_key_cache_miss,
    track_ingestion_credits,
    track_kyrodb_operation,
    track_request,
    track_search_credits,
)

__all__ = [
    "track_request",
    "track_api_key_cache_hit",
    "track_api_key_cache_miss",
    "track_kyrodb_operation",
    "track_ingestion_credits",
    "track_search_credits",
]
