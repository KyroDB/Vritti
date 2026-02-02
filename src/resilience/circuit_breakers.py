"""
Circuit breakers for external dependencies.

Prevents cascade failures when KyroDB experiences outages.

Circuit breaker states:
- CLOSED: Normal operation, requests pass through
- OPEN: Failure threshold exceeded, requests fail immediately
- HALF_OPEN: Testing if service recovered, limited requests allowed

Configuration:
- fail_max: Number of failures before opening circuit (default: 5)
- timeout: Seconds circuit stays open before trying half-open (default: 60)
- expected_exception: Exception types that trigger circuit breaker
"""

import functools
import logging

from pybreaker import CircuitBreaker, CircuitBreakerError, CircuitBreakerListener
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class KyroDBCircuitBreakerError(Exception):
    """Circuit breaker open for KyroDB operations."""

    pass


class LoggingCircuitBreakerListener(CircuitBreakerListener):
    """
    Circuit breaker listener that logs state changes.
    
    Implements the proper CircuitBreakerListener interface from pybreaker.
    """
    
    def state_change(self, cb: CircuitBreaker, old_state, new_state) -> None:
        """Called when circuit breaker state changes."""
        if new_state.name == "open":
            logger.error(
                f"Circuit breaker OPENED: {cb.name}",
                extra={
                    "breaker_name": cb.name,
                    "fail_count": cb.fail_counter,
                    "fail_max": cb.fail_max,
                    "state": "OPEN",
                },
            )
        elif new_state.name == "closed":
            logger.info(
                f"Circuit breaker CLOSED: {cb.name} (service recovered)",
                extra={
                    "breaker_name": cb.name,
                    "state": "CLOSED",
                },
            )
        elif new_state.name == "half-open":
            logger.warning(
                f"Circuit breaker HALF-OPEN: {cb.name} (testing recovery)",
                extra={
                    "breaker_name": cb.name,
                    "state": "HALF_OPEN",
                },
            )


# Shared listener instance for all circuit breakers
_logging_listener = LoggingCircuitBreakerListener()


# KyroDB circuit breaker
# Opens after 5 consecutive failures, stays open for 60 seconds
kyrodb_breaker = CircuitBreaker(
    fail_max=5,
    reset_timeout=60,
    name="KyroDB",
    listeners=[_logging_listener],
)


def get_kyrodb_breaker() -> CircuitBreaker:
    """
    Get KyroDB circuit breaker instance.

    Returns:
        CircuitBreaker: Configured for KyroDB operations

    Usage:
        breaker = get_kyrodb_breaker()

        @breaker
        async def search_kyrodb(...):
            return await client.search(...)
    """
    return kyrodb_breaker


def reset_all_breakers() -> None:
    """
    Reset all circuit breakers to CLOSED state.

    Use for testing or manual recovery.
    """
    kyrodb_breaker.close()
    logger.info("All circuit breakers reset to CLOSED state")


def with_kyrodb_circuit_breaker(func):
    """
    Decorator to wrap KyroDB operations with circuit breaker.

    Args:
        func: Async function to wrap

    Returns:
        Wrapped function with circuit breaker protection

    Raises:
        KyroDBCircuitBreakerError: If circuit is open

    Usage:
        @with_kyrodb_circuit_breaker
        async def search_episodes(...):
            return await self.kyrodb_client.search(...)
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Check if circuit is open before calling
        state = kyrodb_breaker.current_state
        if state == "open":
            logger.warning(
                "KyroDB circuit breaker OPEN - failing fast",
                extra={
                    "function": func.__name__,
                    "state": state,
                },
            )
            raise KyroDBCircuitBreakerError(
                f"KyroDB service unavailable (circuit breaker open). "
                f"Retry after {kyrodb_breaker.reset_timeout} seconds."
            )

        try:
            # Execute the async function directly
            result = await func(*args, **kwargs)
            # On success, reset state appropriately
            # pybreaker resets fail_counter when a call succeeds through call()
            # For async, we manually manage state since we bypass call()
            if state == "half_open":
                # Successful call in half-open state closes the circuit
                kyrodb_breaker.close()
            elif state == "closed":
                # Reset fail counter on success in closed state
                # This prevents accumulation of failures across successful calls
                kyrodb_breaker._state._fail_counter = 0
            return result
        except Exception as e:
            # Register failure with the circuit breaker by calling sync path
            # with a function that raises the same exception type
            exc = e

            def raise_original() -> None:
                raise exc

            try:
                kyrodb_breaker.call(raise_original)
            except (type(e), CircuitBreakerError):
                # Expected - we just want to register the failure
                # CircuitBreakerError occurs if circuit just opened
                pass
            raise

    return wrapper


def with_retry(
    max_attempts: int = 3,
    min_wait: int = 1,
    max_wait: int = 10,
    exceptions: tuple[type[Exception], ...] = (Exception,),
):
    """
    Retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time in seconds
        max_wait: Maximum wait time in seconds
        exceptions: Exception types to retry on

    Returns:
        Retry decorator

    Usage:
        @with_retry(max_attempts=3, exceptions=(ConnectionError,))
        async def fetch_data():
            return await client.get(...)
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(exceptions),
        reraise=True,
    )
