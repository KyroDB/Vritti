"""
Tier-based rate limiting configuration.

Defines rate limits per subscription tier and endpoint type.
Uses slowapi with Redis backend (falls back to in-memory for development).

Rate Limit Tiers:
    FREE:       10 req/min, 100 req/hour
    STARTER:    100 req/min, 1000 req/hour  
    PRO:        500 req/min, 5000 req/hour
    ENTERPRISE: 2000 req/min, 50000 req/hour

Endpoint Types:
    - capture: Episode ingestion (compute-intensive)
    - search: Episode search (moderate)
    - reflect: Pre-action gating (moderate)
    - admin: Admin endpoints (strict)
"""

import logging
from dataclasses import dataclass

from src.models.customer import SubscriptionTier

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration for an endpoint."""
    
    per_minute: int
    per_hour: int
    burst_limit: int  # Maximum concurrent requests
    
    def to_slowapi_string(self, period: str = "minute") -> str:
        """Convert to slowapi rate limit string format."""
        if period == "minute":
            return f"{self.per_minute}/minute"
        elif period == "hour":
            return f"{self.per_hour}/hour"
        else:
            return f"{self.per_minute}/minute"


# Rate limits by tier and endpoint type
TIER_RATE_LIMITS: dict[SubscriptionTier, dict[str, RateLimitConfig]] = {
    SubscriptionTier.FREE: {
        "capture": RateLimitConfig(per_minute=10, per_hour=100, burst_limit=5),
        "search": RateLimitConfig(per_minute=20, per_hour=200, burst_limit=10),
        "reflect": RateLimitConfig(per_minute=10, per_hour=100, burst_limit=5),
        "skills": RateLimitConfig(per_minute=10, per_hour=100, burst_limit=5),
        "admin": RateLimitConfig(per_minute=5, per_hour=50, burst_limit=2),
        "default": RateLimitConfig(per_minute=30, per_hour=300, burst_limit=10),
    },
    SubscriptionTier.STARTER: {
        "capture": RateLimitConfig(per_minute=100, per_hour=1000, burst_limit=20),
        "search": RateLimitConfig(per_minute=200, per_hour=2000, burst_limit=50),
        "reflect": RateLimitConfig(per_minute=100, per_hour=1000, burst_limit=20),
        "skills": RateLimitConfig(per_minute=50, per_hour=500, burst_limit=20),
        "admin": RateLimitConfig(per_minute=10, per_hour=100, burst_limit=5),
        "default": RateLimitConfig(per_minute=200, per_hour=2000, burst_limit=50),
    },
    SubscriptionTier.PRO: {
        "capture": RateLimitConfig(per_minute=500, per_hour=5000, burst_limit=100),
        "search": RateLimitConfig(per_minute=1000, per_hour=10000, burst_limit=200),
        "reflect": RateLimitConfig(per_minute=500, per_hour=5000, burst_limit=100),
        "skills": RateLimitConfig(per_minute=200, per_hour=2000, burst_limit=50),
        "admin": RateLimitConfig(per_minute=50, per_hour=500, burst_limit=10),
        "default": RateLimitConfig(per_minute=1000, per_hour=10000, burst_limit=200),
    },
    SubscriptionTier.ENTERPRISE: {
        "capture": RateLimitConfig(per_minute=2000, per_hour=50000, burst_limit=500),
        "search": RateLimitConfig(per_minute=5000, per_hour=100000, burst_limit=1000),
        "reflect": RateLimitConfig(per_minute=2000, per_hour=50000, burst_limit=500),
        "skills": RateLimitConfig(per_minute=1000, per_hour=20000, burst_limit=200),
        "admin": RateLimitConfig(per_minute=200, per_hour=2000, burst_limit=50),
        "default": RateLimitConfig(per_minute=5000, per_hour=100000, burst_limit=1000),
    },
}


def get_rate_limit_for_tier(
    tier: SubscriptionTier,
    endpoint_type: str = "default",
) -> RateLimitConfig:
    """
    Get rate limit configuration for a subscription tier and endpoint type.
    
    Args:
        tier: Customer subscription tier
        endpoint_type: Type of endpoint (capture, search, reflect, skills, admin, default)
        
    Returns:
        RateLimitConfig for the tier/endpoint combination
    """
    tier_limits = TIER_RATE_LIMITS.get(tier, TIER_RATE_LIMITS[SubscriptionTier.FREE])
    return tier_limits.get(endpoint_type, tier_limits["default"])


def get_rate_limit_string(
    tier: SubscriptionTier,
    endpoint_type: str = "default",
    period: str = "minute",
) -> str:
    """
    Get rate limit string for slowapi decorator.
    
    Args:
        tier: Customer subscription tier
        endpoint_type: Type of endpoint
        period: Time period (minute or hour)
        
    Returns:
        Rate limit string (e.g., "100/minute")
    """
    config = get_rate_limit_for_tier(tier, endpoint_type)
    return config.to_slowapi_string(period)


# Default rate limits for unauthenticated requests
UNAUTHENTICATED_LIMITS = RateLimitConfig(
    per_minute=10,
    per_hour=50,
    burst_limit=3,
)


def get_dynamic_rate_limit(endpoint_type: str = "default"):
    """
    Create a dynamic rate limit function for slowapi.
    
    This function returns a callable that determines the rate limit
    based on the rate limit key (customer_id or IP address).
    
    NOTE: slowapi calls this callable with the key extracted by key_func.
    The key is customer_id if authenticated, or IP address if not.
    Since we don't have tier info from just the key, we use a default
    tier (STARTER) for rate limiting. Actual quota enforcement happens
    via the quota tracking system.
    
    For true tier-based rate limiting, a middleware approach is recommended:
    1. AuthenticationMiddleware attaches customer to request.state
    2. Custom rate limiter checks request.state.customer for tier
    
    Usage:
        @app.post("/api/v1/capture")
        @limiter.limit(get_dynamic_rate_limit("capture"))
        async def capture_episode(request: Request, ...):
            ...
    
    Args:
        endpoint_type: Type of endpoint for limit lookup
        
    Returns:
        Callable that returns rate limit string (slowapi calls with key parameter)
    """
    def rate_limit_value(key: str) -> str:
        """
        Return rate limit based on endpoint type.
        
        Note: We use STARTER tier limits as the default since we don't have
        access to customer tier information at this point. The key is just
        the customer_id or IP address.
        
        Real tier-based enforcement happens via quota tracking.
        """
        # Use STARTER tier as default - provides reasonable limits
        # FREE tier is too restrictive, ENTERPRISE would allow abuse
        default_config = TIER_RATE_LIMITS[SubscriptionTier.STARTER].get(
            endpoint_type,
            TIER_RATE_LIMITS[SubscriptionTier.STARTER]["default"]
        )
        
        logger.debug(
            f"Rate limit for key={key[:8]}... on {endpoint_type}: "
            f"{default_config.per_minute}/min (STARTER tier default)"
        )
        
        return default_config.to_slowapi_string("minute")
    
    return rate_limit_value


# Pre-built dynamic limiters for common endpoints
CAPTURE_RATE_LIMIT = get_dynamic_rate_limit("capture")
SEARCH_RATE_LIMIT = get_dynamic_rate_limit("search")
REFLECT_RATE_LIMIT = get_dynamic_rate_limit("reflect")
SKILLS_RATE_LIMIT = get_dynamic_rate_limit("skills")
ADMIN_RATE_LIMIT = get_dynamic_rate_limit("admin")


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded."""
    
    def __init__(
        self,
        customer_id: str,
        tier: SubscriptionTier,
        endpoint_type: str,
        limit: int,
        retry_after_seconds: int = 60,
    ):
        self.customer_id = customer_id
        self.tier = tier
        self.endpoint_type = endpoint_type
        self.limit = limit
        self.retry_after_seconds = retry_after_seconds
        
        super().__init__(
            f"Rate limit exceeded for {customer_id} ({tier.value}): "
            f"{limit}/min on {endpoint_type}. Retry after {retry_after_seconds}s."
        )


def log_rate_limit_exceeded(
    customer_id: str,
    tier: SubscriptionTier,
    endpoint_type: str,
) -> None:
    """Log rate limit exceeded event for monitoring."""
    from src.observability.metrics import track_rate_limit_exceeded
    
    logger.warning(
        f"Rate limit exceeded: customer={customer_id}, tier={tier.value}, "
        f"endpoint={endpoint_type}"
    )
    
    track_rate_limit_exceeded(
        customer_id=customer_id,
        customer_tier=tier.value,
    )
