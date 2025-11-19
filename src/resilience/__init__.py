"""
Resilience patterns for external dependencies.

Circuit breakers prevent cascade failures when dependencies fail.
"""

from src.resilience.circuit_breakers import (
    get_kyrodb_breaker,
    get_stripe_breaker,
    reset_all_breakers,
)

__all__ = [
    "get_kyrodb_breaker",
    "get_stripe_breaker",
    "reset_all_breakers",
]
