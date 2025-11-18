"""
Quota enforcement middleware.

Checks customer quota before allowing requests and tracks usage after successful operations.
"""

import logging
from collections.abc import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.billing.usage_tracking import UsageTracker, UsageType
from src.models.customer import Customer, CustomerStatus

logger = logging.getLogger(__name__)


class QuotaEnforcementMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce customer quota limits.

    Checks quota before expensive operations (ingestion) and returns 429 if exceeded.
    Allows free operations (health checks, metrics) to pass through.
    """

    # Endpoints that consume credits
    METERED_ENDPOINTS = {
        "/api/v1/capture": UsageType.EPISODE_INGESTION,
        "/api/v1/search": UsageType.EPISODE_SEARCH,
        "/api/v1/reflect": UsageType.REFLECTION_GENERATION,
    }

    # Endpoints that are always allowed (no quota check)
    EXEMPT_ENDPOINTS = {
        "/health",
        "/health/liveness",
        "/health/readiness",
        "/metrics",
        "/docs",
        "/openapi.json",
        "/api/v1/usage",  # Allow checking usage even if quota exceeded
    }

    def __init__(self, app, usage_tracker: UsageTracker):
        """
        Initialize quota enforcement middleware.

        Args:
            app: FastAPI application
            usage_tracker: Usage tracker service
        """
        super().__init__(app)
        self.usage_tracker = usage_tracker

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Check quota before request and track usage after.

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response: API response or 429 if quota exceeded
        """
        path = request.url.path

        # Skip exempt endpoints
        if path in self.EXEMPT_ENDPOINTS or path.startswith("/static"):
            return await call_next(request)

        # Skip non-metered endpoints
        if path not in self.METERED_ENDPOINTS:
            return await call_next(request)

        # Get customer from request state (set by auth middleware)
        customer: Customer | None = getattr(request.state, "customer", None)

        if not customer:
            # No customer authenticated, let auth middleware handle it
            return await call_next(request)

        # Check customer status
        if customer.status == CustomerStatus.SUSPENDED:
            logger.warning(
                "Request denied: customer suspended",
                extra={"customer_id": customer.customer_id, "path": path},
            )
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": "account_suspended",
                    "message": "Your account is suspended. Please contact support or update your payment method.",
                    "customer_id": customer.customer_id,
                },
            )

        # Get usage type for endpoint
        usage_type = self.METERED_ENDPOINTS.get(path)
        if not usage_type:
            # Not a metered endpoint, allow
            return await call_next(request)

        # Check quota (soft check - warn but allow)
        can_use = await self.usage_tracker.check_quota(customer, usage_type, quantity=1)

        if not can_use:
            # Hard limit exceeded
            logger.warning(
                "Request denied: quota exceeded",
                extra={
                    "customer_id": customer.customer_id,
                    "path": path,
                    "credits_used": customer.credits_used_current_month,
                    "quota": customer.monthly_credit_limit,
                },
            )

            # Return 429 Too Many Requests
            usage_summary = await self.usage_tracker.get_usage_summary(customer)

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "quota_exceeded",
                    "message": f"Monthly quota exceeded. Used {customer.credits_used_current_month}/{customer.monthly_credit_limit} credits.",
                    "usage": usage_summary,
                    "upgrade_url": "/api/v1/billing/upgrade",
                },
                headers={
                    "Retry-After": "3600",  # Retry after 1 hour
                    "X-RateLimit-Limit": str(customer.monthly_credit_limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": (
                        customer.billing_cycle_end.isoformat() if customer.billing_cycle_end else ""
                    ),
                },
            )

        # Quota OK, proceed with request
        response = await call_next(request)

        # Track usage only if request succeeded (2xx status code)
        if 200 <= response.status_code < 300:
            try:
                usage_result = await self.usage_tracker.track_usage(
                    customer, usage_type, quantity=1
                )

                # Add usage headers to response
                response.headers["X-Credits-Used"] = str(usage_result["credits_used"])
                response.headers["X-Credits-Remaining"] = str(usage_result["quota_remaining"])
                response.headers["X-Credits-Total"] = str(usage_result["total_credits_used"])

                # Warn if approaching quota
                if usage_result["quota_exceeded"]:
                    response.headers[
                        "X-Quota-Warning"
                    ] = "Quota exceeded. Upgrade your plan to continue."
                elif usage_result["quota_remaining"] < customer.monthly_credit_limit * 0.1:
                    response.headers[
                        "X-Quota-Warning"
                    ] = "Less than 10% quota remaining. Consider upgrading."

            except Exception as e:
                logger.error(
                    "Failed to track usage",
                    extra={
                        "customer_id": customer.customer_id,
                        "path": path,
                        "error": str(e),
                    },
                )

        return response


def get_quota_middleware(usage_tracker: UsageTracker) -> QuotaEnforcementMiddleware:
    """
    Create quota enforcement middleware.

    Args:
        usage_tracker: Usage tracker service

    Returns:
        QuotaEnforcementMiddleware: Configured middleware
    """
    return QuotaEnforcementMiddleware
