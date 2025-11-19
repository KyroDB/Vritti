"""
Circuit breakers for external dependencies.

Prevents cascade failures when KyroDB or Stripe experience outages.

Circuit breaker states:
- CLOSED: Normal operation, requests pass through
- OPEN: Failure threshold exceeded, requests fail immediately
- HALF_OPEN: Testing if service recovered, limited requests allowed

Configuration:
- fail_max: Number of failures before opening circuit (default: 5)
- timeout: Seconds circuit stays open before trying half-open (default: 60)
- expected_exception: Exception types that trigger circuit breaker
"""

import logging

from pybreaker import CircuitBreaker, CircuitBreakerError
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


class StripeCircuitBreakerError(Exception):
    """Circuit breaker open for Stripe operations."""

    pass


def _on_circuit_open(breaker: CircuitBreaker) -> None:
    """
    Callback when circuit breaker opens.

    Args:
        breaker: Circuit breaker instance
    """
    logger.error(
        f"Circuit breaker OPENED: {breaker.name}",
        extra={
            "breaker_name": breaker.name,
            "fail_count": breaker.fail_counter,
            "fail_max": breaker.fail_max,
            "state": "OPEN",
        },
    )


def _on_circuit_close(breaker: CircuitBreaker) -> None:
    """
    Callback when circuit breaker closes (recovery).

    Args:
        breaker: Circuit breaker instance
    """
    logger.info(
        f"Circuit breaker CLOSED: {breaker.name} (service recovered)",
        extra={
            "breaker_name": breaker.name,
            "state": "CLOSED",
        },
    )


def _on_circuit_half_open(breaker: CircuitBreaker) -> None:
    """
    Callback when circuit breaker enters half-open state.

    Args:
        breaker: Circuit breaker instance
    """
    logger.warning(
        f"Circuit breaker HALF-OPEN: {breaker.name} (testing recovery)",
        extra={
            "breaker_name": breaker.name,
            "state": "HALF_OPEN",
        },
    )


# KyroDB circuit breaker
# Opens after 5 consecutive failures, stays open for 60 seconds
kyrodb_breaker = CircuitBreaker(
    fail_max=5,
    timeout_duration=60,
    name="KyroDB",
    listeners=[_on_circuit_open, _on_circuit_close, _on_circuit_half_open],
)


# Stripe circuit breaker
# Opens after 3 consecutive failures, stays open for 30 seconds (faster recovery for billing)
stripe_breaker = CircuitBreaker(
    fail_max=3,
    timeout_duration=30,
    name="Stripe",
    listeners=[_on_circuit_open, _on_circuit_close, _on_circuit_half_open],
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


def get_stripe_breaker() -> CircuitBreaker:
    """
    Get Stripe circuit breaker instance.

    Returns:
        CircuitBreaker: Configured for Stripe API calls

    Usage:
        breaker = get_stripe_breaker()

        @breaker
        def create_subscription(...):
            return stripe.Subscription.create(...)
    """
    return stripe_breaker


def reset_all_breakers() -> None:
    """
    Reset all circuit breakers to CLOSED state.

    Use for testing or manual recovery.
    """
    kyrodb_breaker.close()
    stripe_breaker.close()
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

    async def wrapper(*args, **kwargs):
        try:
            # Call function through circuit breaker
            result = await kyrodb_breaker.call_async(func, *args, **kwargs)
            return result
        except CircuitBreakerError as e:
            # Circuit is open - fail fast
            logger.warning(
                "KyroDB circuit breaker OPEN - failing fast",
                extra={
                    "function": func.__name__,
                    "state": kyrodb_breaker.current_state,
                },
            )
            raise KyroDBCircuitBreakerError(
                f"KyroDB service unavailable (circuit breaker open). "
                f"Retry after {kyrodb_breaker.timeout_duration} seconds."
            ) from e

    return wrapper


def with_stripe_circuit_breaker(func):
    """
    Decorator to wrap Stripe operations with circuit breaker.

    Args:
        func: Function to wrap (sync or async)

    Returns:
        Wrapped function with circuit breaker protection

    Raises:
        StripeCircuitBreakerError: If circuit is open

    Usage:
        @with_stripe_circuit_breaker
        def create_customer(...):
            return stripe.Customer.create(...)
    """

    def wrapper(*args, **kwargs):
        try:
            # Call function through circuit breaker
            result = stripe_breaker.call(func, *args, **kwargs)
            return result
        except CircuitBreakerError as e:
            # Circuit is open - fail fast
            logger.warning(
                "Stripe circuit breaker OPEN - failing fast",
                extra={
                    "function": func.__name__,
                    "state": stripe_breaker.current_state,
                },
            )
            raise StripeCircuitBreakerError(
                f"Stripe service unavailable (circuit breaker open). "
                f"Retry after {stripe_breaker.timeout_duration} seconds."
            ) from e

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
