"""
Customer/tenant data models for multi-tenancy support.

Each customer represents an organization/workspace using the Vritti service.
Customer IDs are used for namespace isolation in KyroDB and quota enforcement.
"""

import re
from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class SubscriptionTier(str, Enum):
    """Subscription tier for quota management."""

    FREE = "free"  # 1K credits/month
    STARTER = "starter"  # 10K credits/month
    PRO = "pro"  # 100K credits/month
    ENTERPRISE = "enterprise"  # Unlimited


class CustomerStatus(str, Enum):
    """Customer account status."""

    ACTIVE = "active"
    SUSPENDED = "suspended"  # Quota abuse
    DELETED = "deleted"  # Soft delete - data retained for 30 days


class Customer(BaseModel):
    """
    Customer (tenant) in multi-tenant system.

    Each customer has isolated data namespace and usage quotas.
    """

    # Identity
    customer_id: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Unique customer identifier (slug format: lowercase, alphanumeric, hyphens)",
    )
    organization_name: str = Field(..., min_length=1, max_length=200)
    email: str = Field(..., description="Primary contact email")

    # Subscription
    subscription_tier: SubscriptionTier = Field(default=SubscriptionTier.FREE)
    status: CustomerStatus = Field(default=CustomerStatus.ACTIVE)

    # Quotas
    monthly_credit_limit: int = Field(default=1000, ge=0, description="Credit quota per month")
    credits_used_current_month: int = Field(default=0, ge=0)

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_api_call_at: datetime | None = Field(default=None)

    @field_validator("customer_id")
    @classmethod
    def validate_customer_id_format(cls, v: str) -> str:
        """
        Validate customer_id is slug format.

        Must be lowercase, alphanumeric, hyphens only (safe for namespaces).
        """
        if not re.match(r"^[a-z0-9-]+$", v):
            raise ValueError("customer_id must be lowercase alphanumeric with hyphens only")
        if v.startswith("-") or v.endswith("-"):
            raise ValueError("customer_id cannot start or end with hyphen")
        if "--" in v:
            raise ValueError("customer_id cannot contain consecutive hyphens")
        return v

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        """Basic email validation."""
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("Invalid email format")
        return v.lower()

    def is_quota_exceeded(self) -> bool:
        """Check if customer has exceeded monthly credit quota."""
        return self.credits_used_current_month >= self.monthly_credit_limit

    def can_use_credits(self, credits: int) -> bool:
        """Check if customer can use specified credits without exceeding quota."""
        return (
            self.status == CustomerStatus.ACTIVE
            and self.credits_used_current_month + credits <= self.monthly_credit_limit
        )

    def increment_usage(self, credits: int) -> None:
        """Increment credits used for current month."""
        self.credits_used_current_month += credits
        self.last_api_call_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

    def reset_monthly_usage(self) -> None:
        """Reset monthly credit usage (called at start of new month)."""
        self.credits_used_current_month = 0
        self.updated_at = datetime.now(UTC)

    def should_suspend(self) -> bool:
        """Check if customer should be suspended due to quota abuse."""
        if self.credits_used_current_month > self.monthly_credit_limit * 1.2:
            return True
        return False


class CustomerCreate(BaseModel):
    """Schema for creating a new customer."""

    customer_id: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Desired customer ID (must be unique)",
    )
    organization_name: str = Field(..., min_length=1, max_length=200)
    email: str = Field(..., description="Primary contact email")
    subscription_tier: SubscriptionTier = Field(default=SubscriptionTier.FREE)

    @field_validator("customer_id")
    @classmethod
    def validate_customer_id_format(cls, v: str) -> str:
        """Validate customer_id is slug format."""
        if not re.match(r"^[a-z0-9-]+$", v):
            raise ValueError("customer_id must be lowercase alphanumeric with hyphens only")
        if v.startswith("-") or v.endswith("-"):
            raise ValueError("customer_id cannot start or end with hyphen")
        if "--" in v:
            raise ValueError("customer_id cannot contain consecutive hyphens")
        return v


class CustomerUpdate(BaseModel):
    """Schema for updating customer details."""

    organization_name: str | None = Field(default=None, min_length=1, max_length=200)
    email: str | None = Field(default=None)
    subscription_tier: SubscriptionTier | None = Field(default=None)
    status: CustomerStatus | None = Field(default=None)
    monthly_credit_limit: int | None = Field(default=None, ge=0)


class APIKey(BaseModel):
    """
    API key for customer authentication.

    API keys are stored as SHA-256 digest (never plaintext).
    """

    key_id: str = Field(..., description="Unique key identifier (UUID)")
    customer_id: str = Field(..., description="Owner customer ID")
    key_hash_sha256: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 digest of API key",
    )
    key_prefix: str = Field(
        ..., min_length=8, max_length=8, description="First 8 chars for display"
    )
    name: str | None = Field(default=None, description="User-assigned key name")

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_used_at: datetime | None = Field(default=None)
    expires_at: datetime | None = Field(default=None)
    is_active: bool = Field(default=True)

    def is_valid(self) -> bool:
        """Check if API key is valid (active and not expired)."""
        if not self.is_active:
            return False
        if self.expires_at and datetime.now(UTC) > self.expires_at:
            return False
        return True


class APIKeyCreate(BaseModel):
    """Schema for creating new API key."""

    customer_id: str
    name: str | None = Field(default=None, max_length=100)
    expires_in_days: int | None = Field(
        default=None, ge=1, le=365, description="Key expiration (1-365 days)"
    )
