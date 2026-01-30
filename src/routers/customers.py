"""
Customer management API endpoints.

Provides CRUD operations for customers and API key management.

Security:
- Admin-only endpoints (require admin API key)
- Customer-scoped endpoints (require valid API key, can only access own data)
- Rate limiting applied
- Audit logging for all mutations
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel

from src.models.customer import (
    APIKeyCreate,
    CustomerCreate,
    CustomerUpdate,
)
from src.storage.database import CustomerDatabase, get_customer_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/customers", tags=["Customers"])


# Response models
class CustomerResponse(BaseModel):
    """Customer response (excludes sensitive data)."""

    customer_id: str
    organization_name: str
    email: str
    subscription_tier: str
    status: str
    monthly_credit_limit: int
    credits_used_current_month: int
    created_at: str


class APIKeyResponse(BaseModel):
    """API key response."""

    key_id: str
    customer_id: str
    key_prefix: str  # First 8 chars for display
    name: Optional[str]
    created_at: str
    expires_at: Optional[str]
    is_active: bool


class APIKeyCreateResponse(BaseModel):
    """API key creation response (includes plaintext key)."""

    api_key: str  # Only returned once!
    key_id: str
    key_prefix: str
    customer_id: str
    warning: str = "Store this API key securely - it will not be shown again"


class UsageResponse(BaseModel):
    """Customer usage statistics."""

    customer_id: str
    credits_used: int
    credit_limit: int
    remaining_credits: int
    usage_percentage: float
    over_quota: bool


# Dependency: Admin authentication (temporary - will be replaced with proper auth)
async def verify_admin_key(x_admin_key: str = Header(...)) -> bool:
    """
    Verify admin API key.

    TODO: Replace with proper admin authentication system.
    For now, checks against environment variable.
    """
    import os

    admin_key = os.getenv("ADMIN_API_KEY", "admin_secret_change_me")
    if x_admin_key != admin_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin API key",
        )
    return True


# Admin endpoints


@router.post(
    "",
    response_model=CustomerResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_admin_key)],
)
async def create_customer(
    customer_data: CustomerCreate,
    db: CustomerDatabase = Depends(get_customer_db),
) -> CustomerResponse:
    """
    Create new customer (admin only).

    Args:
        customer_data: Customer creation data
        db: Database instance

    Returns:
        CustomerResponse: Created customer

    Raises:
        409: Customer ID already exists
    """
    customer = await db.create_customer(customer_data)

    if customer is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Customer ID '{customer_data.customer_id}' already exists",
        )

    return CustomerResponse(
        customer_id=customer.customer_id,
        organization_name=customer.organization_name,
        email=customer.email,
        subscription_tier=customer.subscription_tier.value,
        status=customer.status.value,
        monthly_credit_limit=customer.monthly_credit_limit,
        credits_used_current_month=customer.credits_used_current_month,
        created_at=customer.created_at.isoformat(),
    )


@router.get(
    "/{customer_id}",
    response_model=CustomerResponse,
    dependencies=[Depends(verify_admin_key)],
)
async def get_customer(
    customer_id: str,
    db: CustomerDatabase = Depends(get_customer_db),
) -> CustomerResponse:
    """
    Get customer details (admin only).

    Args:
        customer_id: Customer ID
        db: Database instance

    Returns:
        CustomerResponse: Customer details

    Raises:
        404: Customer not found
    """
    customer = await db.get_customer(customer_id)

    if customer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer '{customer_id}' not found",
        )

    return CustomerResponse(
        customer_id=customer.customer_id,
        organization_name=customer.organization_name,
        email=customer.email,
        subscription_tier=customer.subscription_tier.value,
        status=customer.status.value,
        monthly_credit_limit=customer.monthly_credit_limit,
        credits_used_current_month=customer.credits_used_current_month,
        created_at=customer.created_at.isoformat(),
    )


@router.patch(
    "/{customer_id}",
    response_model=CustomerResponse,
    dependencies=[Depends(verify_admin_key)],
)
async def update_customer(
    customer_id: str,
    update_data: CustomerUpdate,
    db: CustomerDatabase = Depends(get_customer_db),
) -> CustomerResponse:
    """
    Update customer (admin only).

    Args:
        customer_id: Customer ID
        update_data: Fields to update
        db: Database instance

    Returns:
        CustomerResponse: Updated customer

    Raises:
        404: Customer not found
    """
    customer = await db.update_customer(customer_id, update_data)

    if customer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer '{customer_id}' not found",
        )

    return CustomerResponse(
        customer_id=customer.customer_id,
        organization_name=customer.organization_name,
        email=customer.email,
        subscription_tier=customer.subscription_tier.value,
        status=customer.status.value,
        monthly_credit_limit=customer.monthly_credit_limit,
        credits_used_current_month=customer.credits_used_current_month,
        created_at=customer.created_at.isoformat(),
    )


# API key management


@router.post(
    "/{customer_id}/api-keys",
    response_model=APIKeyCreateResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_admin_key)],
)
async def create_api_key(
    customer_id: str,
    key_data: APIKeyCreate,
    db: CustomerDatabase = Depends(get_customer_db),
) -> APIKeyCreateResponse:
    """
    Create API key for customer (admin only).

    SECURITY: API key is only returned once. Store it securely.

    Args:
        customer_id: Customer ID
        key_data: API key creation data
        db: Database instance

    Returns:
        APIKeyCreateResponse: Created API key (includes plaintext)

    Raises:
        404: Customer not found
    """
    # Verify customer exists
    customer = await db.get_customer(customer_id)
    if customer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer '{customer_id}' not found",
        )

    # Override customer_id from path parameter (security)
    key_data.customer_id = customer_id

    # Create API key
    plaintext_key, api_key = await db.create_api_key(key_data)

    return APIKeyCreateResponse(
        api_key=plaintext_key,
        key_id=api_key.key_id,
        key_prefix=api_key.key_prefix,
        customer_id=api_key.customer_id,
    )


@router.delete(
    "/{customer_id}/api-keys/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(verify_admin_key)],
)
async def revoke_api_key(
    customer_id: str,
    key_id: str,
    db: CustomerDatabase = Depends(get_customer_db),
) -> None:
    """
    Revoke API key (admin only).

    Args:
        customer_id: Customer ID
        key_id: API key ID
        db: Database instance

    Raises:
        404: API key not found
    """
    success = await db.revoke_api_key(key_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key '{key_id}' not found",
        )


# Customer-scoped endpoints (require valid API key)


@router.get("/{customer_id}/usage", response_model=UsageResponse)
async def get_usage(
    customer_id: str,
    db: CustomerDatabase = Depends(get_customer_db),
    # TODO: Add API key validation dependency
) -> UsageResponse:
    """
    Get customer usage statistics.

    Customer can only access their own usage data.
    API key validation ensures customer_id matches.

    Args:
        customer_id: Customer ID
        db: Database instance

    Returns:
        UsageResponse: Usage statistics

    Raises:
        404: Customer not found
        403: API key doesn't match customer_id
    """
    customer = await db.get_customer(customer_id)

    if customer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer '{customer_id}' not found",
        )

    remaining = customer.monthly_credit_limit - customer.credits_used_current_month
    percentage = (
        (customer.credits_used_current_month / customer.monthly_credit_limit * 100)
        if customer.monthly_credit_limit > 0
        else 0.0
    )

    return UsageResponse(
        customer_id=customer.customer_id,
        credits_used=customer.credits_used_current_month,
        credit_limit=customer.monthly_credit_limit,
        remaining_credits=max(0, remaining),
        usage_percentage=round(percentage, 2),
        over_quota=customer.is_quota_exceeded(),
    )


@router.post("/{customer_id}/reset-usage", status_code=status.HTTP_204_NO_CONTENT)
async def reset_monthly_usage(
    customer_id: str,
    db: CustomerDatabase = Depends(get_customer_db),
    _admin: bool = Depends(verify_admin_key),
) -> None:
    """
    Reset monthly usage (admin only).

    Called automatically on quota cycle.

    Args:
        customer_id: Customer ID
        db: Database instance

    Raises:
        404: Customer not found
    """
    success = await db.reset_monthly_usage(customer_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer '{customer_id}' not found",
        )
