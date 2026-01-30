"""src.observability.middleware

Minimal observability middleware.

We intentionally keep this lightweight and dependency-free:
- Structured logging is handled by src.observability.logging_middleware
- Health endpoints are handled by src.observability.health
- This middleware only classifies errors for consistent logging
"""

import logging
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class ErrorTrackingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for error classification and tracking.

    Classifies errors into categories:
    - validation: Pydantic validation errors
    - authentication: API key validation failures
    - authorization: Permission denied errors
    - kyrodb: Vector database errors
    - internal: Unhandled exceptions

    Performance:
    - <10μs overhead per request (only on error path)
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Track errors with classification.

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response: Handler response
        """
        try:
            response = await call_next(request)
            return response

        except Exception as exc:
            # Classify error
            error_type = self._classify_error(exc)

            endpoint = self._normalize_endpoint(request.url.path)
            logger.warning(
                "Request failed",
                extra={
                    "error_type": error_type,
                    "endpoint": endpoint,
                    "exception": type(exc).__name__,
                },
            )

            # Re-raise for FastAPI exception handlers
            raise

    def _classify_error(self, exc: Exception) -> str:
        """
        Classify error into category.

        Args:
            exc: Exception instance

        Returns:
            str: Error category
        """
        exc_name = type(exc).__name__

        # Validation errors
        if "ValidationError" in exc_name or "ValueError" in exc_name:
            return "validation"

        # Authentication errors
        if "Unauthorized" in exc_name or "InvalidAPIKey" in exc_name:
            return "authentication"

        # Authorization errors
        if "Forbidden" in exc_name or "PermissionDenied" in exc_name:
            return "authorization"

        # KyroDB errors
        if "KyroDBError" in exc_name or "ConnectionError" in exc_name:
            return "kyrodb"

        # Rate limiting
        if "RateLimitExceeded" in exc_name:
            return "rate_limit"

        # Default: internal error
        return "internal"

    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path to avoid log-cardinality explosions."""
        import re

        path = re.sub(r"/customers/[a-z0-9-]+", "/customers/{customer_id}", path)
        path = re.sub(r"/api-keys/ak_[a-z0-9]+", "/api-keys/{api_key_id}", path)
        path = re.sub(r"/episodes/\d+", "/episodes/{episode_id}", path)

        return path
