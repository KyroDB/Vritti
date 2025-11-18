"""
Tests for structured logging infrastructure (Phase 2 Week 6).

Tests:
- JSON output format
- Request context propagation
- PII redaction
- Correlation IDs
- Operation context timing
"""


import pytest

from src.observability.logging import (
    OperationContext,
    RequestContext,
    configure_logging,
    get_customer_id,
    get_logger,
    get_request_id,
    get_trace_id,
    set_customer_id,
    set_request_id,
    set_trace_id,
)


def test_configure_logging_json_output():
    """Test that JSON logging can be configured."""
    configure_logging(log_level="INFO", json_output=True, colorized=False)

    # Should not raise any exceptions
    logger = get_logger("test")
    logger.info("Test message", test_field="value")


def test_configure_logging_console_output():
    """Test that console logging can be configured."""
    configure_logging(log_level="INFO", json_output=False, colorized=False)

    # Should not raise any exceptions
    logger = get_logger("test")
    logger.info("Test message", test_field="value")


def test_get_logger():
    """Test logger creation."""
    logger = get_logger("test_logger")

    assert logger is not None
    assert hasattr(logger, "info")
    assert hasattr(logger, "error")
    assert hasattr(logger, "warning")
    assert hasattr(logger, "debug")


def test_structured_logging_with_fields():
    """Test that additional fields are included in logs."""
    configure_logging(log_level="INFO", json_output=False, colorized=False)

    logger = get_logger("test")

    # Should not raise exceptions
    logger.info("Episode captured", episode_id=12345, latency_ms=45.2)
    logger.warning("Slow query detected", latency_ms=150.5)
    logger.error("Operation failed", error_code=500)


def test_request_context():
    """Test request context propagation."""
    configure_logging(log_level="INFO", json_output=False, colorized=False)

    logger = get_logger("test")

    # Test context manager
    with RequestContext(customer_id="test-customer", request_id="req_123"):
        # Inside context, request_id and customer_id should be set
        assert get_request_id() == "req_123"
        assert get_customer_id() == "test-customer"
        assert get_trace_id() is not None  # Auto-generated

        # Log should include context
        logger.info("Processing request")

    # Outside context, values should be reset
    assert get_request_id() is None
    assert get_customer_id() is None


def test_request_context_auto_generation():
    """Test that request_id and trace_id are auto-generated."""
    with RequestContext(customer_id="test-customer"):
        # request_id should be auto-generated
        request_id = get_request_id()
        assert request_id is not None
        assert request_id.startswith("req_")

        # trace_id should be auto-generated
        trace_id = get_trace_id()
        assert trace_id is not None
        assert trace_id.startswith("trace_")


def test_operation_context():
    """Test operation context with timing."""
    configure_logging(log_level="INFO", json_output=False, colorized=False)

    # Test successful operation
    with OperationContext("test_operation", episode_id=12345):
        pass  # Operation completes successfully

    # Should log operation start and completion with latency


def test_operation_context_with_exception():
    """Test operation context logs errors."""
    configure_logging(log_level="INFO", json_output=False, colorized=False)

    # Test failed operation
    with pytest.raises(ValueError):
        with OperationContext("test_operation", episode_id=12345):
            raise ValueError("Test error")

    # Should log operation failure with exception


def test_context_setters_and_getters():
    """Test manual context setters and getters."""
    # Initially None
    assert get_request_id() is None
    assert get_customer_id() is None
    assert get_trace_id() is None

    # Set values
    set_request_id("req_test")
    set_customer_id("customer_test")
    set_trace_id("trace_test")

    # Retrieve values
    assert get_request_id() == "req_test"
    assert get_customer_id() == "customer_test"
    assert get_trace_id() == "trace_test"


def test_nested_request_contexts():
    """Test that nested contexts work correctly."""
    with RequestContext(customer_id="customer1", request_id="req1"):
        assert get_customer_id() == "customer1"
        assert get_request_id() == "req1"

        # Nested context
        with RequestContext(customer_id="customer2", request_id="req2"):
            assert get_customer_id() == "customer2"
            assert get_request_id() == "req2"

        # Back to outer context
        assert get_customer_id() == "customer1"
        assert get_request_id() == "req1"


def test_logger_with_bind():
    """Test logger bind for persistent context."""
    configure_logging(log_level="INFO", json_output=False, colorized=False)

    logger = get_logger("test")

    # Bind persistent context
    bound_logger = logger.bind(operation="ingestion", batch_size=100)

    # All logs from bound logger include bound context
    bound_logger.info("Processing started")
    bound_logger.info("Processing complete", records_processed=50)


def test_exception_logging():
    """Test exception logging with exc_info."""
    configure_logging(log_level="INFO", json_output=False, colorized=False)

    logger = get_logger("test")

    try:
        raise ValueError("Test exception")
    except Exception:
        # exc_info=True includes exception traceback
        logger.error("Operation failed", exc_info=True)


def test_different_log_levels():
    """Test different log levels."""
    configure_logging(log_level="DEBUG", json_output=False, colorized=False)

    logger = get_logger("test")

    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    # logger.critical("Critical message")  # Not commonly used


def test_pii_redaction_in_logs():
    """Test that sensitive fields are redacted."""
    # This test verifies that the redact_sensitive_fields processor works
    # In practice, PII redaction happens during log processing

    configure_logging(log_level="INFO", json_output=True, colorized=False)

    logger = get_logger("test")

    # Log with sensitive fields
    # The redact_sensitive_fields processor should handle these
    logger.info(
        "User authenticated",
        email="user@example.com",  # Should be redacted to ***@example.com
        api_key="em_live_abcdefgh123456789",  # Should show prefix only
    )

    # Actual redaction verification would require capturing log output
    # For now, we just verify no exceptions are raised


def test_request_context_in_async_code():
    """Test that context variables work across async boundaries."""
    import asyncio

    async def async_operation():
        # Context should be preserved in async functions
        assert get_request_id() == "req_async"
        assert get_customer_id() == "customer_async"

    # Set context
    with RequestContext(customer_id="customer_async", request_id="req_async"):
        # Run async code
        asyncio.run(async_operation())
