"""
Usage tracking and metering service.

Tracks customer usage (credits) and reports to Stripe for metered billing.

Credit costs:
- Episode ingestion: 1 credit
- Episode search: 0.1 credits (10 searches = 1 credit)
- Reflection generation: 2 credits
"""

import logging
from datetime import UTC, datetime
from enum import Enum

from src.billing.stripe_service import StripeService
from src.models.customer import Customer
from src.storage.database import CustomerDatabase

logger = logging.getLogger(__name__)


class UsageType(str, Enum):
    """Types of billable usage."""

    EPISODE_INGESTION = "episode_ingestion"  # 1 credit
    EPISODE_SEARCH = "episode_search"  # 0.1 credits
    REFLECTION_GENERATION = "reflection_generation"  # 2 credits


class UsageTracker:
    """
    Track and meter customer usage.

    Responsibilities:
    - Calculate credits for operations
    - Update customer usage in database
    - Report usage to Stripe (metered billing)
    - Enforce quota limits
    """

    # Credit costs per operation type
    CREDIT_COSTS = {
        UsageType.EPISODE_INGESTION: 1.0,
        UsageType.EPISODE_SEARCH: 0.1,
        UsageType.REFLECTION_GENERATION: 2.0,
    }

    def __init__(self, customer_db: CustomerDatabase, stripe_service: StripeService | None = None):
        """
        Initialize usage tracker.

        Args:
            customer_db: Customer database for usage updates
            stripe_service: Stripe service for metered billing (optional)
        """
        self.customer_db = customer_db
        self.stripe_service = stripe_service

    async def track_usage(
        self, customer: Customer, usage_type: UsageType, quantity: int = 1
    ) -> dict[str, any]:
        """
        Track usage and update customer credits.

        Args:
            customer: Customer who used the service
            usage_type: Type of usage (ingestion, search, reflection)
            quantity: Quantity of operations (default: 1)

        Returns:
            dict with:
                - credits_used: Credits consumed for this operation
                - total_credits_used: Total credits used this month
                - quota_remaining: Credits remaining in quota
                - quota_exceeded: True if quota exceeded

        Raises:
            QuotaExceededError: If quota exceeded and hard limit enforced
        """
        # Calculate credits
        credit_cost = self.CREDIT_COSTS.get(usage_type, 0.0)
        credits_used = credit_cost * quantity

        # Update customer usage
        success = await self.customer_db.increment_usage(customer.customer_id, int(credits_used))

        if not success:
            logger.error(
                "Failed to update customer usage",
                extra={
                    "customer_id": customer.customer_id,
                    "usage_type": usage_type.value,
                    "credits": credits_used,
                },
            )

        # Get updated customer
        updated_customer = await self.customer_db.get_customer(customer.customer_id)
        if not updated_customer:
            updated_customer = customer

        # Report to Stripe (metered billing)
        if self.stripe_service:
            try:
                await self.stripe_service.report_usage(
                    updated_customer, int(credits_used), datetime.now(UTC)
                )
            except Exception as e:
                logger.error(
                    "Failed to report usage to Stripe",
                    extra={
                        "customer_id": customer.customer_id,
                        "error": str(e),
                    },
                )

        # Calculate quota status
        quota_remaining = max(
            0,
            updated_customer.monthly_credit_limit - updated_customer.credits_used_current_month,
        )
        quota_exceeded = updated_customer.is_quota_exceeded()

        logger.info(
            "Usage tracked",
            extra={
                "customer_id": customer.customer_id,
                "usage_type": usage_type.value,
                "quantity": quantity,
                "credits_used": credits_used,
                "total_credits_used": updated_customer.credits_used_current_month,
                "quota_remaining": quota_remaining,
                "quota_exceeded": quota_exceeded,
            },
        )

        return {
            "credits_used": credits_used,
            "total_credits_used": updated_customer.credits_used_current_month,
            "quota_remaining": quota_remaining,
            "quota_exceeded": quota_exceeded,
        }

    async def check_quota(
        self, customer: Customer, usage_type: UsageType, quantity: int = 1
    ) -> bool:
        """
        Check if customer has enough quota for operation.

        Args:
            customer: Customer to check
            usage_type: Type of operation
            quantity: Quantity of operations

        Returns:
            bool: True if customer can use credits, False if quota exceeded
        """
        credit_cost = self.CREDIT_COSTS.get(usage_type, 0.0)
        credits_needed = int(credit_cost * quantity)

        return customer.can_use_credits(credits_needed)

    async def get_usage_summary(self, customer: Customer) -> dict[str, any]:
        """
        Get usage summary for customer.

        Args:
            customer: Customer to summarize

        Returns:
            dict with usage statistics
        """
        quota_remaining = max(
            0, customer.monthly_credit_limit - customer.credits_used_current_month
        )

        usage_percentage = 0.0
        if customer.monthly_credit_limit > 0:
            usage_percentage = (
                customer.credits_used_current_month / customer.monthly_credit_limit
            ) * 100

        return {
            "customer_id": customer.customer_id,
            "subscription_tier": customer.subscription_tier.value,
            "credits_used": customer.credits_used_current_month,
            "monthly_limit": customer.monthly_credit_limit,
            "quota_remaining": quota_remaining,
            "usage_percentage": round(usage_percentage, 2),
            "quota_exceeded": customer.is_quota_exceeded(),
            "status": customer.status.value,
            "in_trial": customer.is_in_trial(),
            "trial_end_date": (
                customer.trial_end_date.isoformat() if customer.trial_end_date else None
            ),
        }


class QuotaExceededError(Exception):
    """Raised when customer quota is exceeded."""

    pass


# Global usage tracker instance
_usage_tracker: UsageTracker | None = None


def get_usage_tracker(
    customer_db: CustomerDatabase, stripe_service: StripeService | None = None
) -> UsageTracker:
    """
    Get global usage tracker instance (singleton).

    Args:
        customer_db: Customer database
        stripe_service: Stripe service (optional)

    Returns:
        UsageTracker: Usage tracker instance
    """
    global _usage_tracker
    if _usage_tracker is None:
        _usage_tracker = UsageTracker(customer_db, stripe_service)
    return _usage_tracker
