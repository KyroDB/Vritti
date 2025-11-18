"""
Demo script for structured logging with JSON output.

Shows:
- JSON log output for production
- Request context propagation
- Correlation IDs
- Operation timing
- PII redaction

Usage:
    # JSON output (production)
    LOGGING_JSON_OUTPUT=true python3 examples/structured_logging_demo.py

    # Console output (development)
    LOGGING_JSON_OUTPUT=false python3 examples/structured_logging_demo.py
"""

import time
import asyncio
from src.observability.logging import (
    configure_logging,
    get_logger,
    RequestContext,
    OperationContext,
)

# Configure logging (reads from environment or uses defaults)
configure_logging(
    log_level="INFO",
    json_output=True,  # Set to False for human-readable console output
    colorized=False,
)

logger = get_logger(__name__)


def simulate_episode_ingestion():
    """Simulate episode ingestion with structured logging."""
    logger.info("=== Starting Episode Ingestion Demo ===")

    # Simulate authenticated request with customer context
    with RequestContext(customer_id="demo-customer"):
        logger.info(
            "Received episode ingestion request",
            goal="Deploy application to production",
            has_image=True,
            has_reflection=True,
        )

        # Simulate embedding generation
        with OperationContext("text_embedding", model="all-MiniLM-L6-v2"):
            time.sleep(0.01)  # Simulate embedding generation
            logger.debug("Text embedding generated", dimensions=384)

        # Simulate KyroDB insertion
        with OperationContext("kyrodb_insert", collection="failures"):
            time.sleep(0.015)  # Simulate database operation
            logger.info("Episode stored in vector database", episode_id=12345)

        # Log final success
        logger.info(
            "Episode ingestion complete",
            episode_id=12345,
            total_latency_ms=25.5,
            credits_used=1.7,
        )


def simulate_search_query():
    """Simulate search query with structured logging."""
    logger.info("=== Starting Search Query Demo ===")

    with RequestContext(customer_id="demo-customer"):
        logger.info(
            "Received search request",
            goal="Unable to connect to database",
            k=10,
        )

        # Simulate semantic search
        with OperationContext("semantic_search", k=10):
            time.sleep(0.008)  # Simulate search
            logger.info("Search completed", results_found=5)

        # Log results
        logger.info(
            "Search query complete",
            total_results=5,
            search_latency_ms=8.2,
            credits_used=0.1,
        )


def simulate_error_handling():
    """Demonstrate error logging with exception info."""
    logger.info("=== Starting Error Handling Demo ===")

    with RequestContext(customer_id="demo-customer"):
        try:
            # Simulate error
            raise ValueError("KyroDB connection timeout after 30s")
        except Exception as e:
            logger.error(
                "Episode ingestion failed",
                error=str(e),
                exc_info=True,  # Includes full traceback
            )


def simulate_pii_redaction():
    """Demonstrate PII redaction in logs."""
    logger.info("=== Starting PII Redaction Demo ===")

    # Log with sensitive data (will be redacted)
    logger.info(
        "User authenticated",
        email="user@example.com",  # Redacted to ***@example.com
        api_key="em_live_abcdefgh123456789",  # Shows prefix only
        customer_id="demo-customer",  # Not PII, shown fully
    )


def simulate_slow_request_warning():
    """Demonstrate slow request logging."""
    logger.info("=== Starting Slow Request Demo ===")

    with RequestContext(customer_id="demo-customer"):
        # Simulate slow operation
        logger.warning(
            "Slow request detected",
            method="POST",
            path="/api/v1/capture",
            latency_ms=150.5,
            threshold_ms=100.0,
        )


def simulate_distributed_tracing():
    """Demonstrate distributed tracing with trace IDs."""
    logger.info("=== Starting Distributed Tracing Demo ===")

    # Simulate request from upstream service with trace ID
    with RequestContext(
        customer_id="demo-customer",
        trace_id="trace_from_upstream_service",  # Passed from upstream
    ):
        logger.info("Processing request from upstream service")

        # All logs in this context will have the same trace_id
        # This allows correlation across multiple services
        with OperationContext("call_downstream_service"):
            logger.info("Calling downstream service", trace_propagation=True)


async def simulate_async_operations():
    """Demonstrate context propagation in async code."""
    logger.info("=== Starting Async Operations Demo ===")

    async def async_task(task_id: int):
        # Context is preserved across async boundaries
        logger.info(f"Async task {task_id} started")
        await asyncio.sleep(0.01)
        logger.info(f"Async task {task_id} completed")

    # Context propagates to async tasks
    with RequestContext(customer_id="demo-customer"):
        tasks = [async_task(i) for i in range(3)]
        await asyncio.gather(*tasks)


def main():
    """Run all demo scenarios."""
    print("Structured Logging Demo")
    print("=" * 60)
    print("All logs below demonstrate structured logging with:")
    print("- JSON output (for log aggregation)")
    print("- Request context (customer_id, request_id, trace_id)")
    print("- Operation timing")
    print("- PII redaction")
    print("=" * 60)
    print()

    # Run demos
    simulate_episode_ingestion()
    print()

    simulate_search_query()
    print()

    simulate_error_handling()
    print()

    simulate_pii_redaction()
    print()

    simulate_slow_request_warning()
    print()

    simulate_distributed_tracing()
    print()

    # Async demo
    asyncio.run(simulate_async_operations())
    print()

    logger.info("=== Demo Complete ===")
    print()
    print("=" * 60)
    print("In production, these logs would be sent to:")
    print("- ELK Stack (Elasticsearch + Logstash + Kibana)")
    print("- Grafana Loki")
    print("- AWS CloudWatch Logs")
    print("- Google Cloud Logging")
    print()
    print("Query examples:")
    print('- ELK: customer_id:"demo-customer" AND latency_ms:>100')
    print('- Loki: {service="episodic-memory", customer_id="demo-customer"}')
    print("=" * 60)


if __name__ == "__main__":
    main()
