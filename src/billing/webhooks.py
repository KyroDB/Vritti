"""
Stripe webhook event handlers.

Handles Stripe webhook events for payment processing:
- invoice.payment_succeeded
- invoice.payment_failed
- customer.subscription.updated
- customer.subscription.deleted
"""

import logging
from datetime import UTC, datetime

import stripe

from src.billing.stripe_service import StripeService
from src.config import StripeConfig
from src.models.customer import SubscriptionTier
from src.storage.database import CustomerDatabase

logger = logging.getLogger(__name__)


class WebhookError(Exception):
    """Base exception for webhook processing errors."""

    pass


class StripeWebhookHandler:
    """
    Handle Stripe webhook events.

    Processes payment events and updates customer records accordingly.
    """

    def __init__(
        self,
        config: StripeConfig,
        customer_db: CustomerDatabase,
        stripe_service: StripeService,
    ):
        """
        Initialize webhook handler.

        Args:
            config: Stripe configuration (for webhook secret)
            customer_db: Customer database
            stripe_service: Stripe service
        """
        self.config = config
        self.customer_db = customer_db
        self.stripe_service = stripe_service

    async def handle_event(self, payload: bytes, signature: str) -> dict[str, any]:
        """
        Process Stripe webhook event.

        Args:
            payload: Raw webhook payload
            signature: Stripe signature header (Stripe-Signature)

        Returns:
            dict: Processing result with status and message

        Raises:
            WebhookError: If event processing fails
        """
        if not self.config.webhook_secret:
            raise WebhookError("Webhook secret not configured")

        try:
            # Verify webhook signature
            event = stripe.Webhook.construct_event(payload, signature, self.config.webhook_secret)

        except ValueError:
            raise WebhookError("Invalid payload")

        except stripe.SignatureVerificationError:
            raise WebhookError("Invalid signature")

        # Route event to handler
        event_type = event["type"]
        event_data = event["data"]["object"]

        logger.info(
            "Processing Stripe webhook event",
            extra={"event_type": event_type, "event_id": event["id"]},
        )

        handlers = {
            "invoice.payment_succeeded": self._handle_payment_succeeded,
            "invoice.payment_failed": self._handle_payment_failed,
            "customer.subscription.created": self._handle_subscription_created,
            "customer.subscription.updated": self._handle_subscription_updated,
            "customer.subscription.deleted": self._handle_subscription_deleted,
            "customer.subscription.trial_will_end": self._handle_trial_ending,
        }

        handler = handlers.get(event_type)
        if handler:
            try:
                result = await handler(event_data)
                logger.info(
                    "Webhook event processed",
                    extra={"event_type": event_type, "result": result},
                )
                return {"status": "success", "message": result}

            except Exception as e:
                logger.error(
                    "Webhook event processing failed",
                    extra={"event_type": event_type, "error": str(e)},
                )
                raise WebhookError(f"Event processing failed: {e}") from e
        else:
            logger.warning(
                "Unhandled webhook event type",
                extra={"event_type": event_type},
            )
            return {"status": "ignored", "message": f"Unhandled event: {event_type}"}

    async def _handle_payment_succeeded(self, invoice: dict) -> str:
        """Handle successful payment."""
        customer_id = await self._get_customer_id_from_stripe_id(invoice["customer"])

        if not customer_id:
            logger.warning(
                "Payment succeeded for unknown customer",
                extra={"stripe_customer_id": invoice["customer"]},
            )
            return "Unknown customer"

        customer = await self.customer_db.get_customer(customer_id)
        if not customer:
            return "Customer not found"

        # Clear payment failure flag
        await self.stripe_service.handle_payment_succeeded(customer)

        # Reset monthly usage on new billing cycle
        await self.customer_db.reset_monthly_usage(customer_id)

        # Update billing cycle dates
        if invoice.get("period_start") and invoice.get("period_end"):
            billing_start = datetime.fromtimestamp(invoice["period_start"], UTC)
            billing_end = datetime.fromtimestamp(invoice["period_end"], UTC)

            await self.customer_db.update_billing_cycle(
                customer_id, billing_start, billing_end, billing_end
            )

        logger.info(
            "Payment succeeded, customer reactivated",
            extra={"customer_id": customer_id, "amount": invoice.get("amount_paid")},
        )

        return f"Payment succeeded for {customer_id}"

    async def _handle_payment_failed(self, invoice: dict) -> str:
        """Handle failed payment."""
        customer_id = await self._get_customer_id_from_stripe_id(invoice["customer"])

        if not customer_id:
            logger.warning(
                "Payment failed for unknown customer",
                extra={"stripe_customer_id": invoice["customer"]},
            )
            return "Unknown customer"

        customer = await self.customer_db.get_customer(customer_id)
        if not customer:
            return "Customer not found"

        # Mark payment as failed
        await self.stripe_service.handle_payment_failed(customer)

        logger.error(
            "Payment failed for customer",
            extra={
                "customer_id": customer_id,
                "amount": invoice.get("amount_due"),
                "attempt_count": invoice.get("attempt_count"),
            },
        )

        return f"Payment failed for {customer_id}"

    async def _handle_subscription_created(self, subscription: dict) -> str:
        """Handle subscription creation."""
        customer_id = await self._get_customer_id_from_stripe_id(subscription["customer"])

        if not customer_id:
            return "Unknown customer"

        # Update subscription ID
        await self.customer_db.update_customer_stripe_id(
            customer_id,
            subscription["customer"],
            subscription["id"],
        )

        # Update billing cycle
        billing_start = datetime.fromtimestamp(subscription["current_period_start"], UTC)
        billing_end = datetime.fromtimestamp(subscription["current_period_end"], UTC)

        await self.customer_db.update_billing_cycle(
            customer_id, billing_start, billing_end, billing_end
        )

        logger.info(
            "Subscription created",
            extra={
                "customer_id": customer_id,
                "subscription_id": subscription["id"],
                "status": subscription["status"],
            },
        )

        return f"Subscription created for {customer_id}"

    async def _handle_subscription_updated(self, subscription: dict) -> str:
        """Handle subscription update."""
        customer_id = await self._get_customer_id_from_stripe_id(subscription["customer"])

        if not customer_id:
            return "Unknown customer"

        customer = await self.customer_db.get_customer(customer_id)
        if not customer:
            return "Customer not found"

        # Check for cancellation
        if subscription.get("cancel_at_period_end"):
            logger.warning(
                "Subscription scheduled for cancellation",
                extra={
                    "customer_id": customer_id,
                    "cancel_at": subscription.get("cancel_at"),
                },
            )

        # Update billing cycle if changed
        billing_start = datetime.fromtimestamp(subscription["current_period_start"], UTC)
        billing_end = datetime.fromtimestamp(subscription["current_period_end"], UTC)

        await self.customer_db.update_billing_cycle(
            customer_id, billing_start, billing_end, billing_end
        )

        logger.info(
            "Subscription updated",
            extra={
                "customer_id": customer_id,
                "subscription_id": subscription["id"],
                "status": subscription["status"],
            },
        )

        return f"Subscription updated for {customer_id}"

    async def _handle_subscription_deleted(self, subscription: dict) -> str:
        """Handle subscription cancellation."""
        customer_id = await self._get_customer_id_from_stripe_id(subscription["customer"])

        if not customer_id:
            return "Unknown customer"

        customer = await self.customer_db.get_customer(customer_id)
        if not customer:
            return "Customer not found"

        # Downgrade to FREE tier
        from src.models.customer import CustomerUpdate

        await self.customer_db.update_customer(
            customer_id,
            CustomerUpdate(subscription_tier=SubscriptionTier.FREE, monthly_credit_limit=1000),
        )

        logger.warning(
            "Subscription cancelled, downgraded to FREE",
            extra={"customer_id": customer_id},
        )

        return f"Subscription cancelled for {customer_id}, downgraded to FREE"

    async def _handle_trial_ending(self, subscription: dict) -> str:
        """Handle trial ending soon (3 days before)."""
        customer_id = await self._get_customer_id_from_stripe_id(subscription["customer"])

        if not customer_id:
            return "Unknown customer"

        trial_end = subscription.get("trial_end")
        if trial_end:
            trial_end_date = datetime.fromtimestamp(trial_end, UTC)

            logger.info(
                "Trial ending soon",
                extra={
                    "customer_id": customer_id,
                    "trial_end_date": trial_end_date.isoformat(),
                },
            )

        return f"Trial ending notification for {customer_id}"

    async def _get_customer_id_from_stripe_id(self, stripe_customer_id: str) -> str | None:
        """
        Get internal customer_id from Stripe customer ID.

        Args:
            stripe_customer_id: Stripe customer ID (cus_xxx)

        Returns:
            Internal customer_id or None if not found
        """
        # Query database for customer with this Stripe ID
        conn = self.customer_db._get_connection()
        row = conn.execute(
            "SELECT customer_id FROM customers WHERE stripe_customer_id = ?",
            (stripe_customer_id,),
        ).fetchone()

        if row:
            return row["customer_id"]

        # Try to get from Stripe metadata
        try:
            stripe_customer = stripe.Customer.retrieve(stripe_customer_id)
            metadata_customer_id = stripe_customer.get("metadata", {}).get("customer_id")

            if metadata_customer_id:
                return metadata_customer_id

        except stripe.StripeError as e:
            logger.error(
                "Failed to retrieve Stripe customer",
                extra={"stripe_customer_id": stripe_customer_id, "error": str(e)},
            )

        return None


# Global webhook handler instance
_webhook_handler: StripeWebhookHandler | None = None


def get_webhook_handler(
    config: StripeConfig,
    customer_db: CustomerDatabase,
    stripe_service: StripeService,
) -> StripeWebhookHandler:
    """
    Get global webhook handler instance (singleton).

    Args:
        config: Stripe configuration
        customer_db: Customer database
        stripe_service: Stripe service

    Returns:
        StripeWebhookHandler: Webhook handler instance
    """
    global _webhook_handler
    if _webhook_handler is None:
        _webhook_handler = StripeWebhookHandler(config, customer_db, stripe_service)
    return _webhook_handler
