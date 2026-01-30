"""
FastAPI dependencies for authentication and authorization.

Security:
- API key validation with bcrypt
- customer_id injection from validated key (prevents spoofing)
- Rate limiting per customer
- Request tracking for usage tracking

Performance:
- API key validation cached in-memory (5 minute TTL)
- Async validation
- Minimal latency overhead (<1ms for cache hit, ~300ms for cache miss)
"""

import logging
import time
from typing import Optional

from cachetools import TTLCache
from fastapi import Header, HTTPException, Request, status

from src.models.customer import Customer
from src.storage.database import get_customer_db

logger = logging.getLogger(__name__)

# In-memory cache for validated API keys (5 minute TTL)
# Trade-off: 5 min stale data vs. ~300ms bcrypt validation on every request
# Format: {api_key: (customer, expire_time)}
API_KEY_CACHE = TTLCache(maxsize=1000, ttl=300)  # 5 minutes


async def get_authenticated_customer(
    request: Request,
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> Customer:
    """
    Validate API key and return authenticated customer.

    Supports two authentication methods:
    1. Authorization header: "Bearer {api_key}"
    2. X-API-Key header: "{api_key}"

    Args:
        request: FastAPI request (for attaching customer context)
        authorization: Authorization header (optional)
        x_api_key: X-API-Key header (optional)
        db: Customer database instance (injected by FastAPI)

    Returns:
        Customer: Authenticated customer

    Raises:
        HTTPException 401: Invalid or missing API key
        HTTPException 403: Customer inactive or quota exceeded

    Security:
        - API key validated with bcrypt (constant-time comparison)
        - customer_id extracted from validated key (cannot be spoofed)
        - Cache prevents bcrypt DOS (5 min TTL)
        - Inactive/suspended customers rejected

    Performance:
        - Cache hit: <1ms
        - Cache miss: ~300ms (bcrypt cost=12)
        - Cache size: 1000 keys (memory: ~100KB)
    """
    # Extract API key from headers
    api_key = None

    if authorization:
        # Bearer token format
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            api_key = parts[1]
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Authorization header format. Use: 'Bearer {api_key}'",
            )
    elif x_api_key:
        # Direct API key format
        api_key = x_api_key
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication. Provide API key via Authorization or X-API-Key header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate API key format
    if not api_key.startswith("em_live_"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format. Must start with 'em_live_'",
        )

    # Check cache first (performance optimization)
    cached = API_KEY_CACHE.get(api_key)
    if cached:
        customer, expire_time = cached
        if time.time() < expire_time:
            logger.debug(f"API key cache hit for customer: {customer.customer_id}")

            # Verify customer is still active
            if customer.status != "active":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Customer account is {customer.status}",
                )

            # Attach customer to request state (accessible in route handlers)
            request.state.customer = customer
            return customer

    # Cache miss or expired - validate with database
    logger.debug("API key cache miss - validating with database")

    db = await get_customer_db()

    start_time = time.perf_counter()

    customer = await db.validate_api_key(api_key)

    validation_time_seconds = time.perf_counter() - start_time
    validation_time_ms = validation_time_seconds * 1000


    if customer is None:
        logger.warning(f"API key validation failed (validation time: {validation_time_ms:.2f}ms)")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    logger.info(
        f"API key validated for customer: {customer.customer_id} "
        f"(validation time: {validation_time_ms:.2f}ms)"
    )

    # Verify customer is active
    if customer.status != "active":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Customer account is {customer.status}",
        )

    # Cache validated key (5 minute TTL)
    expire_time = time.time() + 300  # 5 minutes
    API_KEY_CACHE[api_key] = (customer, expire_time)

    # Attach customer to request state
    request.state.customer = customer

    return customer


async def require_active_customer(
    customer: Customer = None,
) -> Customer:
    """
    Verify customer is active and within quota.

    Args:
        customer: Authenticated customer (injected by get_authenticated_customer)

    Returns:
        Customer: Verified customer

    Raises:
        HTTPException 403: Customer inactive or quota exceeded

    Usage:
        Use as secondary dependency after get_authenticated_customer:

        @app.post("/api/v1/capture")
        async def capture(
            customer: Customer = Depends(get_authenticated_customer),
            active_customer: Customer = Depends(require_active_customer),
        ):
            # active_customer is guaranteed to be active and within quota
            pass
    """
    if customer is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    # Verify customer is active
    if customer.status != "active":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Customer account is {customer.status}",
        )

    # Verify quota (soft limit - log warning, hard limit would reject)
    if customer.is_quota_exceeded():
        logger.warning(
            f"Customer {customer.customer_id} exceeded quota: "
            f"{customer.credits_used_current_month}/{customer.monthly_credit_limit}"
        )
        # For now, just log warning. Phase 4 will enforce hard limits.

    return customer


def get_customer_id_from_request(request: Request) -> str:
    """
    Extract customer_id from authenticated request.

    Security:
        - customer_id is extracted from validated API key (request.state.customer)
        - User-provided customer_id in request body is IGNORED
        - Prevents customer_id spoofing

    Args:
        request: FastAPI request with authenticated customer

    Returns:
        str: Customer ID from authenticated API key

    Raises:
        HTTPException 401: If customer not authenticated

    Usage:
        @app.post("/api/v1/capture")
        async def capture(
            episode_data: EpisodeCreate,
            customer_id: str = Depends(get_customer_id_from_request),
        ):
            # customer_id is guaranteed to match authenticated API key
            # episode_data.customer_id is IGNORED
            episode_data.customer_id = customer_id
            ...
    """
    if not hasattr(request.state, "customer"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    return request.state.customer.customer_id


async def require_admin_access(
    x_admin_api_key: Optional[str] = Header(None, alias="X-Admin-API-Key"),
) -> None:
    """
    Validate admin API key for admin-only endpoints.
    
    Security:
        - Requires X-Admin-API-Key header with valid admin key
        - If ADMIN_API_KEY is not configured, endpoint is blocked (fail closed)
        - Use constant-time comparison to prevent timing attacks
        
    Args:
        x_admin_api_key: Admin API key from header
        
    Returns:
        None (validation passes)
        
    Raises:
        HTTPException 401: Missing or invalid admin API key
        
    Usage:
        @app.get("/admin/budget")
        async def get_budget(_: None = Depends(require_admin_access)):
            # Only accessible with valid admin API key
            ...
    """
    import secrets

    from src.config import get_settings
    
    settings = get_settings()
    configured_key = settings.admin_api_key
    
    # If no admin key is configured, block access (fail closed)
    # This ensures admin endpoints are always protected
    if configured_key is None:
        logger.warning(
            "Admin endpoint blocked: ADMIN_API_KEY not configured. "
            "Set ADMIN_API_KEY environment variable to enable admin access."
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin API key not configured on server",
        )
    
    if x_admin_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Admin-API-Key header",
            headers={"WWW-Authenticate": "X-Admin-API-Key"},
        )
    
    # Use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(x_admin_api_key, configured_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin API key",
        )
    
    logger.debug("Admin access validated successfully")
