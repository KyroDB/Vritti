# Billing Integration

Complete documentation for Stripe billing integration in EpisodicMemory.

---

## Overview

EpisodicMemory uses Stripe for subscription management and usage-based billing. The system supports:

- **Subscription tiers**: FREE, STARTER, PRO, ENTERPRISE
- **Usage tracking**: Credit-based metering for operations
- **Quota enforcement**: Hard limits with grace periods
- **Webhook processing**: Automated payment event handling

---

## Subscription Tiers

| Tier | Monthly Limit | Price | Features |
|------|--------------|-------|----------|
| **FREE** | 100 episodes | $0 | Basic features, community support |
| **STARTER** | 10,000 episodes | TBD | All features, email support |
| **PRO** | 100,000 episodes | $29/month | All features, priority support |
| **ENTERPRISE** | Unlimited | Custom | Dedicated infrastructure, SLA, custom contracts |

---

## Credit System

Each API operation consumes credits:

| Operation | Credit Cost |
|-----------|-------------|
| Episode ingestion | 1.0 credit |
| Episode search | 0.1 credits |
| Reflection generation | 2.0 credits |

---

## Configuration

### Environment Variables

```bash
# Stripe API Keys
STRIPE_API_KEY=sk_live_xxx              # Secret key (required)
STRIPE_WEBHOOK_SECRET=whsec_xxx         # Webhook secret (required)
STRIPE_PUBLISHABLE_KEY=pk_live_xxx      # Publishable key (optional)

# Price IDs (created in Stripe Dashboard)
STRIPE_PRICE_ID_STARTER=price_xxx       # Starter tier price ID
STRIPE_PRICE_ID_PRO=price_xxx           # Pro tier price ID
STRIPE_PRICE_ID_ENTERPRISE=price_xxx    # Enterprise tier price ID

# Billing Behavior
STRIPE_TRIAL_PERIOD_DAYS=14             # Free trial period (default: 14)
STRIPE_PAYMENT_GRACE_PERIOD_DAYS=3      # Grace period after payment failure (default: 3)

# Metered Billing (optional)
STRIPE_ENABLE_METERED_BILLING=false     # Enable usage-based billing
STRIPE_METERED_PRICE_ID=price_xxx       # Metered usage price ID
```

### Stripe Dashboard Setup

1. **Create Products**: Create products for each tier (Starter, Pro, Enterprise)
2. **Create Prices**: Create recurring prices for each product
3. **Get Price IDs**: Copy price IDs (price_xxx) and set in environment
4. **Create Webhook**: Add webhook endpoint at `/api/v1/webhooks/stripe`
5. **Get Webhook Secret**: Copy webhook signing secret (whsec_xxx)

---

## Usage Tracking

### How It Works

1. **Pre-Check**: Before request, check customer quota
2. **Request Processing**: Execute operation if quota available
3. **Post-Track**: After successful operation, increment usage
4. **Stripe Reporting**: Report usage to Stripe for metered billing

### Example

```python
from src.billing import UsageTracker, UsageType

# Track episode ingestion
usage = await usage_tracker.track_usage(
    customer=customer,
    usage_type=UsageType.EPISODE_INGESTION,
    quantity=1
)

# Result:
{
    "credits_used": 1.0,
    "total_credits_used": 42,
    "quota_remaining": 958,
    "quota_exceeded": False
}
```

---

## Quota Enforcement

### Middleware

The quota enforcement middleware checks quota before each metered request:

1. **Exempt Endpoints**: Health checks, metrics, docs (always allowed)
2. **Metered Endpoints**: `/api/v1/capture`, `/api/v1/search`, `/api/v1/reflect`
3. **Quota Check**: Verify customer has available credits
4. **429 Response**: Return "Too Many Requests" if quota exceeded

### Response Headers

```
X-Credits-Used: 1.0
X-Credits-Remaining: 958
X-Credits-Total: 42
X-Quota-Warning: Less than 10% quota remaining. Consider upgrading.
```

### 429 Response

```json
{
    "error": "quota_exceeded",
    "message": "Monthly quota exceeded. Used 1000/1000 credits.",
    "usage": {
        "customer_id": "acme-corp",
        "subscription_tier": "free",
        "credits_used": 1000,
        "monthly_limit": 1000,
        "quota_remaining": 0,
        "usage_percentage": 100.0
    },
    "upgrade_url": "/api/v1/billing/upgrade"
}
```

---

## Stripe Webhooks

### Supported Events

| Event | Action |
|-------|--------|
| `invoice.payment_succeeded` | Clear payment failure, reactivate customer, reset monthly usage |
| `invoice.payment_failed` | Mark payment failed, suspend after grace period |
| `customer.subscription.created` | Update subscription ID and billing cycle |
| `customer.subscription.updated` | Update billing cycle dates |
| `customer.subscription.deleted` | Downgrade to FREE tier |
| `customer.subscription.trial_will_end` | Send notification (3 days before) |

### Webhook Endpoint

```bash
POST /api/v1/webhooks/stripe
Content-Type: application/json
Stripe-Signature: xxx
```

### Verification

Webhooks are verified using Stripe signature:

```python
event = stripe.Webhook.construct_event(
    payload=request.body,
    sig_header=request.headers["Stripe-Signature"],
    secret=config.webhook_secret
)
```

---

## Payment Failure Handling

### Flow

1. **Payment Fails**: Stripe sends `invoice.payment_failed` webhook
2. **Mark Failed**: Set `payment_failed=True`, `payment_failed_at=now`
3. **Grace Period**: Allow usage for 3 days (configurable)
4. **Suspend**: If not resolved after grace period, suspend customer (`status=SUSPENDED`)
5. **Block Access**: Suspended customers receive 403 Forbidden

### Recovery

1. **Customer Updates Payment**: Update payment method in Stripe
2. **Payment Succeeds**: Stripe sends `invoice.payment_succeeded` webhook
3. **Reactivate**: Clear `payment_failed`, set `status=ACTIVE`
4. **Reset Usage**: Reset monthly credits on new billing cycle

---

## Testing

### Stripe Test Mode

Use Stripe test keys for development:

```bash
STRIPE_API_KEY=sk_test_xxx
STRIPE_WEBHOOK_SECRET=whsec_test_xxx
```

### Test Cards

```
Success: 4242 4242 4242 4242
Decline: 4000 0000 0000 0002
Insufficient Funds: 4000 0000 0000 9995
```

### Webhook Testing

Use Stripe CLI to forward webhooks:

```bash
stripe listen --forward-to localhost:8000/api/v1/webhooks/stripe
stripe trigger invoice.payment_succeeded
```

---

## Database Schema

### Customer Table (Billing Fields)

```sql
-- Stripe Integration
stripe_customer_id TEXT UNIQUE,         -- cus_xxx
stripe_subscription_id TEXT,            -- sub_xxx
stripe_payment_method_id TEXT,          -- pm_xxx

-- Billing Cycle
billing_cycle_start TEXT,               -- ISO 8601 datetime
billing_cycle_end TEXT,                 -- ISO 8601 datetime
next_invoice_date TEXT,                 -- ISO 8601 datetime

-- Payment Status
payment_failed INTEGER DEFAULT 0,       -- Boolean (0/1)
payment_failed_at TEXT,                 -- ISO 8601 datetime

-- Trial
trial_end_date TEXT                     -- ISO 8601 datetime
```

---

## API Reference

### Get Usage Summary

```bash
GET /api/v1/usage
X-API-Key: xxx

Response:
{
    "customer_id": "acme-corp",
    "subscription_tier": "pro",
    "credits_used": 4250,
    "monthly_limit": 100000,
    "quota_remaining": 95750,
    "usage_percentage": 4.25,
    "quota_exceeded": false,
    "status": "active",
    "in_trial": false,
    "trial_end_date": null
}
```

### Create Subscription

```bash
POST /api/v1/billing/subscribe
X-API-Key: xxx
Content-Type: application/json

{
    "tier": "pro",
    "payment_method_id": "pm_xxx"  // Optional for trial
}

Response:
{
    "subscription_id": "sub_xxx",
    "status": "trialing",
    "trial_end": "2025-12-03T10:30:00Z",
    "billing_cycle_start": "2025-11-19T10:30:00Z",
    "billing_cycle_end": "2025-12-19T10:30:00Z"
}
```

### Upgrade/Downgrade

```bash
POST /api/v1/billing/upgrade
X-API-Key: xxx
Content-Type: application/json

{
    "new_tier": "enterprise"
}

Response:
{
    "subscription_id": "sub_xxx",
    "status": "active",
    "tier": "enterprise"
}
```

### Cancel Subscription

```bash
POST /api/v1/billing/cancel
X-API-Key: xxx
Content-Type: application/json

{
    "immediate": false  // If false, cancel at period end
}

Response:
{
    "status": "cancelled",
    "cancel_at": "2025-12-19T10:30:00Z"
}
```

---

## Security

### API Key Protection

- API keys are bcrypt hashed (cost factor 12)
- 5-minute TTL cache for performance
- Never logged or exposed in responses

### Webhook Verification

- All webhooks verified using Stripe signature
- Invalid signatures rejected with 400 Bad Request
- Prevents replay attacks and unauthorized access

### PII Handling

- Credit card details never stored (Stripe hosted)
- Customer data encrypted in transit (TLS)
- Audit logging for compliance

---

## Monitoring

### Metrics

```
# Subscription metrics
stripe_subscriptions_total{tier="pro"} 150
stripe_subscriptions_total{tier="free"} 1250

# Payment metrics
stripe_payments_succeeded_total 1450
stripe_payments_failed_total 12

# Usage metrics
credits_used_total{customer_id="acme-corp"} 4250
quota_exceeded_total{tier="free"} 45
```

### Alerts

```yaml
- alert: HighPaymentFailureRate
  expr: rate(stripe_payments_failed_total[5m]) > 0.05
  for: 10m

- alert: FreeTierQuotaExceeded
  expr: quota_exceeded_total{tier="free"} > 100
  for: 5m
```

---

## Troubleshooting

### Payment Webhook Not Received

1. Check webhook endpoint is accessible: `https://yourdomain.com/api/v1/webhooks/stripe`
2. Verify webhook secret matches Stripe dashboard
3. Check Stripe dashboard → Developers → Webhooks → Event logs
4. Ensure Stripe CLI forwarding is running (development)

### Customer Suspended Incorrectly

1. Check payment status in database: `SELECT payment_failed, payment_failed_at FROM customers`
2. Check Stripe subscription status: `stripe subscriptions retrieve sub_xxx`
3. Manually reactivate: `UPDATE customers SET status='active', payment_failed=0 WHERE customer_id='xxx'`

### Usage Not Tracking

1. Check middleware is registered in main.py
2. Verify customer authenticated (request.state.customer set)
3. Check logs for "Usage tracked" or errors
4. Ensure Stripe metered billing is configured (if enabled)

---

## Production Checklist

- [ ] Stripe account verified and activated
- [ ] Live API keys configured (sk_live_xxx, not sk_test_xxx)
- [ ] Products and prices created in Stripe dashboard
- [ ] Price IDs configured in environment variables
- [ ] Webhook endpoint publicly accessible (HTTPS)
- [ ] Webhook secret configured
- [ ] Webhook events tested (use Stripe CLI)
- [ ] Payment failure handling tested
- [ ] Quota enforcement tested
- [ ] Database backups enabled
- [ ] Monitoring and alerts configured

---


