"""
Stripe integration service for subscription and payment management.

Features:
- Create and manage subscriptions
- Handle payment methods
- Usage-based metering (credits)
- Invoice generation
- Webhook event processing
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import stripe

from src.config import StripeConfig
from src.models.customer import (
    Customer,
    CustomerStatus,
    SubscriptionTier,
)
from src.storage.database import CustomerDatabase

logger = logging.getLogger(__name__)


class StripeError(Exception):
    """Base exception for Stripe-related errors."""

    pass


class SubscriptionError(StripeError):
    """Subscription management error."""

    pass


class PaymentError(StripeError):
    """Payment processing error."""

    pass


class StripeService:
    """
    Stripe integration service.

    Handles:
    - Subscription lifecycle (create, update, cancel)
    - Payment method management
    - Usage-based metering
    - Invoice generation
    - Webhook processing
    """

    def __init__(self, config: StripeConfig, customer_db: CustomerDatabase):
        """
        Initialize Stripe service.

        Args:
            config: Stripe configuration
            customer_db: Customer database for updates
        """
        self.config = config
        self.customer_db = customer_db

        if config.api_key:
            stripe.api_key = config.api_key
            logger.info("Stripe service initialized")
        else:
            logger.warning("Stripe API key not configured - billing disabled")

    @property
    def is_enabled(self) -> bool:
        """Check if Stripe is properly configured."""
        return self.config.is_configured

    async def create_customer(self, customer: Customer) -> str:
        """
        Create Stripe customer.

        Args:
            customer: Customer to create in Stripe

        Returns:
            Stripe customer ID (cus_xxx)

        Raises:
            StripeError: If creation fails
        """
        if not self.is_enabled:
            raise StripeError("Stripe not configured")

        try:
            stripe_customer = stripe.Customer.create(
                email=customer.email,
                name=customer.organization_name,
                metadata={
                    "customer_id": customer.customer_id,
                    "subscription_tier": customer.subscription_tier.value,
                },
            )

            logger.info(
                "Created Stripe customer",
                extra={
                    "customer_id": customer.customer_id,
                    "stripe_customer_id": stripe_customer.id,
                },
            )

            return stripe_customer.id

        except stripe.StripeError as e:
            logger.error(
                "Failed to create Stripe customer",
                extra={"customer_id": customer.customer_id, "error": str(e)},
            )
            raise StripeError(f"Failed to create Stripe customer: {e}") from e

    async def create_subscription(
        self,
        customer: Customer,
        tier: SubscriptionTier,
        payment_method_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Create subscription for customer.

        Args:
            customer: Customer to subscribe
            tier: Subscription tier (STARTER, PRO, ENTERPRISE)
            payment_method_id: Stripe payment method ID (required for paid tiers)

        Returns:
            Subscription details including subscription_id, status, trial_end

        Raises:
            SubscriptionError: If creation fails
        """
        if not self.is_enabled:
            raise SubscriptionError("Stripe not configured")

        if tier == SubscriptionTier.FREE:
            raise SubscriptionError("FREE tier does not require Stripe subscription")

        # Get price ID for tier
        price_id = self._get_price_id_for_tier(tier)
        if not price_id:
            raise SubscriptionError(f"Price ID not configured for tier: {tier}")

        # Ensure Stripe customer exists
        if not customer.stripe_customer_id:
            stripe_customer_id = await self.create_customer(customer)
            await self.customer_db.update_customer_stripe_id(
                customer.customer_id, stripe_customer_id
            )
        else:
            stripe_customer_id = customer.stripe_customer_id

        try:
            # Attach payment method if provided
            if payment_method_id:
                await self._attach_payment_method(stripe_customer_id, payment_method_id)

            # Create subscription
            subscription = stripe.Subscription.create(
                customer=stripe_customer_id,
                items=[{"price": price_id}],
                trial_period_days=(self.config.trial_period_days if not payment_method_id else 0),
                metadata={
                    "customer_id": customer.customer_id,
                    "tier": tier.value,
                },
            )

            # Update customer record
            trial_end = None
            if subscription.trial_end:
                trial_end = datetime.fromtimestamp(subscription.trial_end, UTC)

            billing_cycle_start = datetime.fromtimestamp(subscription.current_period_start, UTC)
            billing_cycle_end = datetime.fromtimestamp(subscription.current_period_end, UTC)

            logger.info(
                "Created subscription",
                extra={
                    "customer_id": customer.customer_id,
                    "subscription_id": subscription.id,
                    "tier": tier.value,
                    "status": subscription.status,
                    "trial_end": trial_end,
                },
            )

            return {
                "subscription_id": subscription.id,
                "status": subscription.status,
                "trial_end": trial_end,
                "billing_cycle_start": billing_cycle_start,
                "billing_cycle_end": billing_cycle_end,
                "payment_method_id": payment_method_id,
            }

        except stripe.StripeError as e:
            logger.error(
                "Failed to create subscription",
                extra={
                    "customer_id": customer.customer_id,
                    "tier": tier.value,
                    "error": str(e),
                },
            )
            raise SubscriptionError(f"Failed to create subscription: {e}") from e

    async def update_subscription(
        self, customer: Customer, new_tier: SubscriptionTier
    ) -> dict[str, Any]:
        """
        Update subscription tier (upgrade/downgrade).

        Args:
            customer: Customer to update
            new_tier: New subscription tier

        Returns:
            Updated subscription details

        Raises:
            SubscriptionError: If update fails
        """
        if not self.is_enabled:
            raise SubscriptionError("Stripe not configured")

        if not customer.stripe_subscription_id:
            raise SubscriptionError("Customer does not have active subscription")

        price_id = self._get_price_id_for_tier(new_tier)
        if not price_id:
            raise SubscriptionError(f"Price ID not configured for tier: {new_tier}")

        try:
            # Get current subscription
            subscription = stripe.Subscription.retrieve(customer.stripe_subscription_id)

            # Update subscription item
            stripe.Subscription.modify(
                customer.stripe_subscription_id,
                items=[{"id": subscription["items"]["data"][0].id, "price": price_id}],
                proration_behavior="create_prorations",
                metadata={"tier": new_tier.value},
            )

            logger.info(
                "Updated subscription",
                extra={
                    "customer_id": customer.customer_id,
                    "subscription_id": customer.stripe_subscription_id,
                    "old_tier": customer.subscription_tier.value,
                    "new_tier": new_tier.value,
                },
            )

            return {
                "subscription_id": customer.stripe_subscription_id,
                "status": subscription.status,
                "tier": new_tier.value,
            }

        except stripe.StripeError as e:
            logger.error(
                "Failed to update subscription",
                extra={
                    "customer_id": customer.customer_id,
                    "new_tier": new_tier.value,
                    "error": str(e),
                },
            )
            raise SubscriptionError(f"Failed to update subscription: {e}") from e

    async def cancel_subscription(self, customer: Customer, immediate: bool = False) -> None:
        """
        Cancel subscription.

        Args:
            customer: Customer to cancel
            immediate: If True, cancel immediately. If False, cancel at period end.

        Raises:
            SubscriptionError: If cancellation fails
        """
        if not self.is_enabled:
            raise SubscriptionError("Stripe not configured")

        if not customer.stripe_subscription_id:
            logger.warning(
                "Customer does not have active subscription",
                extra={"customer_id": customer.customer_id},
            )
            return

        try:
            if immediate:
                stripe.Subscription.delete(customer.stripe_subscription_id)
            else:
                stripe.Subscription.modify(
                    customer.stripe_subscription_id, cancel_at_period_end=True
                )

            logger.info(
                "Cancelled subscription",
                extra={
                    "customer_id": customer.customer_id,
                    "subscription_id": customer.stripe_subscription_id,
                    "immediate": immediate,
                },
            )

        except stripe.StripeError as e:
            logger.error(
                "Failed to cancel subscription",
                extra={
                    "customer_id": customer.customer_id,
                    "error": str(e),
                },
            )
            raise SubscriptionError(f"Failed to cancel subscription: {e}") from e

    async def report_usage(
        self, customer: Customer, credits_used: int, timestamp: datetime | None = None
    ) -> None:
        """
        Report usage to Stripe for metered billing.

        Args:
            customer: Customer who used credits
            credits_used: Number of credits consumed
            timestamp: Usage timestamp (default: now)

        Raises:
            StripeError: If reporting fails
        """
        if not self.is_enabled or not self.config.enable_metered_billing:
            logger.debug("Metered billing disabled, skipping usage report")
            return

        if not customer.stripe_subscription_id or not self.config.metered_price_id:
            logger.warning(
                "Cannot report usage: missing subscription or metered price ID",
                extra={"customer_id": customer.customer_id},
            )
            return

        try:
            # Get subscription item for metered price
            subscription = stripe.Subscription.retrieve(customer.stripe_subscription_id)

            metered_item = None
            for item in subscription["items"]["data"]:
                if item.price.id == self.config.metered_price_id:
                    metered_item = item
                    break

            if not metered_item:
                logger.warning(
                    "Metered price not found in subscription",
                    extra={"customer_id": customer.customer_id},
                )
                return

            # Report usage
            usage_timestamp = int((timestamp or datetime.now(UTC)).timestamp())

            stripe.SubscriptionItem.create_usage_record(
                metered_item.id,
                quantity=credits_used,
                timestamp=usage_timestamp,
                action="increment",
            )

            logger.debug(
                "Reported usage to Stripe",
                extra={
                    "customer_id": customer.customer_id,
                    "credits": credits_used,
                    "timestamp": timestamp,
                },
            )

        except stripe.StripeError as e:
            logger.error(
                "Failed to report usage",
                extra={
                    "customer_id": customer.customer_id,
                    "credits": credits_used,
                    "error": str(e),
                },
            )

    async def handle_payment_failed(self, customer: Customer) -> None:
        """
        Handle payment failure.

        Args:
            customer: Customer with failed payment

        Actions:
        - Mark customer as payment_failed
        - Suspend after grace period
        """
        now = datetime.now(UTC)

        # Mark payment as failed
        await self.customer_db.update_customer_payment_failed(customer.customer_id, True, now)

        logger.warning(
            "Payment failed",
            extra={
                "customer_id": customer.customer_id,
                "grace_period_days": self.config.payment_grace_period_days,
            },
        )

        # Check if grace period expired
        if customer.payment_failed_at:
            grace_period_end = customer.payment_failed_at + timedelta(
                days=self.config.payment_grace_period_days
            )

            if now > grace_period_end:
                # Suspend customer
                await self.customer_db.update_customer_status(
                    customer.customer_id, CustomerStatus.SUSPENDED
                )

                logger.error(
                    "Customer suspended due to payment failure",
                    extra={"customer_id": customer.customer_id},
                )

    async def handle_payment_succeeded(self, customer: Customer) -> None:
        """
        Handle successful payment.

        Args:
            customer: Customer with successful payment

        Actions:
        - Clear payment_failed flag
        - Reactivate if suspended due to payment
        """
        await self.customer_db.update_customer_payment_failed(customer.customer_id, False, None)

        # Reactivate if suspended
        if customer.status == CustomerStatus.SUSPENDED:
            await self.customer_db.update_customer_status(
                customer.customer_id, CustomerStatus.ACTIVE
            )

            logger.info(
                "Customer reactivated after successful payment",
                extra={"customer_id": customer.customer_id},
            )

    async def _attach_payment_method(self, stripe_customer_id: str, payment_method_id: str) -> None:
        """Attach payment method to customer and set as default."""
        try:
            stripe.PaymentMethod.attach(payment_method_id, customer=stripe_customer_id)

            stripe.Customer.modify(
                stripe_customer_id,
                invoice_settings={"default_payment_method": payment_method_id},
            )

        except stripe.StripeError as e:
            logger.error(
                "Failed to attach payment method",
                extra={
                    "stripe_customer_id": stripe_customer_id,
                    "payment_method_id": payment_method_id,
                    "error": str(e),
                },
            )
            raise PaymentError(f"Failed to attach payment method: {e}") from e

    def _get_price_id_for_tier(self, tier: SubscriptionTier) -> str | None:
        """Get Stripe price ID for subscription tier."""
        mapping = {
            SubscriptionTier.STARTER: self.config.price_id_starter,
            SubscriptionTier.PRO: self.config.price_id_pro,
            SubscriptionTier.ENTERPRISE: self.config.price_id_enterprise,
        }
        return mapping.get(tier)


# Global Stripe service instance
_stripe_service: StripeService | None = None


def get_stripe_service(config: StripeConfig, customer_db: CustomerDatabase) -> StripeService:
    """
    Get global Stripe service instance (singleton).

    Args:
        config: Stripe configuration
        customer_db: Customer database

    Returns:
        StripeService: Stripe service instance
    """
    global _stripe_service
    if _stripe_service is None:
        _stripe_service = StripeService(config, customer_db)
    return _stripe_service
