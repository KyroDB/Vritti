"""
Request size limiting middleware for DoS protection.

Prevents memory exhaustion attacks from large payloads.
"""

import logging
from collections.abc import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce request body size limits.

    Protects against:
    - Memory exhaustion from large JSON payloads
    - File upload flooding
    - DoS attacks via resource exhaustion

    Configuration:
        max_body_size: Maximum request body size in bytes (default: 10MB)
    """

    def __init__(self, app, max_body_size: int = 10 * 1024 * 1024):
        """
        Initialize request size limit middleware.

        Args:
            app: FastAPI application
            max_body_size: Maximum request body size in bytes
        """
        super().__init__(app)
        self.max_body_size = max_body_size

        logger.info(
            f"Request size limit middleware enabled (max: {max_body_size / 1024 / 1024:.1f}MB)"
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Check request body size before processing.

        Args:
            request: Incoming request
            call_next: Next middleware in chain

        Returns:
            Response: Either error response or result from next middleware
        """
        # Check Content-Length header
        content_length = request.headers.get("content-length")

        if content_length:
            content_length_int = int(content_length)

            if content_length_int > self.max_body_size:
                size_mb = content_length_int / 1024 / 1024
                limit_mb = self.max_body_size / 1024 / 1024

                logger.warning(
                    f"Request body too large: {size_mb:.2f}MB (limit: {limit_mb:.1f}MB)",
                    extra={
                        "path": request.url.path,
                        "method": request.method,
                        "content_length": content_length_int,
                        "max_allowed": self.max_body_size,
                    },
                )

                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={
                        "detail": f"Request body too large. Maximum allowed: {limit_mb:.1f}MB, "
                        f"received: {size_mb:.2f}MB",
                        "max_size_bytes": self.max_body_size,
                        "received_size_bytes": content_length_int,
                    },
                )

        # Process request
        response = await call_next(request)
        return response


def validate_file_size(file_bytes: bytes, max_size: int, filename: str = "file") -> None:
    """
    Validate file upload size.

    Args:
        file_bytes: File content as bytes
        max_size: Maximum allowed size in bytes
        filename: Filename for error messages

    Raises:
        ValueError: If file exceeds size limit
    """
    file_size = len(file_bytes)

    if file_size > max_size:
        size_mb = file_size / 1024 / 1024
        limit_mb = max_size / 1024 / 1024

        logger.warning(
            f"File upload too large: {filename} ({size_mb:.2f}MB, limit: {limit_mb:.1f}MB)"
        )

        raise ValueError(
            f"File '{filename}' is too large. "
            f"Maximum allowed: {limit_mb:.1f}MB, received: {size_mb:.2f}MB"
        )
