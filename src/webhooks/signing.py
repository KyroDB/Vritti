"""
Webhook request signing for secure event delivery.

Security:
- HMAC-SHA256 signatures prevent tampering
- Timestamp verification prevents replay attacks
- Signature verification at customer endpoint

Use case:
- Notify customers of async events (reflection complete, quota exceeded, etc.)
- Customers verify webhook authenticity using shared secret
- Protection against man-in-the-middle and replay attacks
"""

import hashlib
import hmac
import logging
import time

logger = logging.getLogger(__name__)


class WebhookSigner:
    """
    Signs webhook requests with HMAC-SHA256.

    Security features:
    - HMAC-SHA256: Prevents tampering (customer can verify integrity)
    - Timestamp: Prevents replay attacks (rejects old signatures)
    - Versioned signatures: Allows signature algorithm upgrades

    Integration:
        1. Customer provides webhook URL + secret during onboarding
        2. EpisodicMemory signs webhook payload with secret
        3. Customer verifies signature before processing
    """

    SIGNATURE_VERSION = "v1"  # Allow future algorithm changes
    TIMESTAMP_TOLERANCE_SECONDS = 300  # 5 minutes (prevents replay attacks)

    def __init__(self, secret: str):
        """
        Initialize webhook signer.

        Args:
            secret: Shared secret for HMAC signing (customer-specific)

        Security:
            - Secret must be cryptographically random (>= 32 bytes)
            - Store securely (environment variable, secrets manager)
            - Rotate periodically (e.g., every 90 days)
        """
        if not secret or len(secret) < 32:
            raise ValueError("Webhook secret must be at least 32 characters for security")

        self.secret = secret.encode("utf-8")

    def sign_payload(
        self,
        payload: str,
        timestamp: int | None = None,
    ) -> tuple[str, int]:
        """
        Sign webhook payload with HMAC-SHA256.

        Args:
            payload: JSON webhook payload (as string)
            timestamp: Unix timestamp (None = use current time)

        Returns:
            tuple: (signature, timestamp)

        Example signature format:
            "v1=a1b2c3d4..."

        Security:
            - Timestamp included in signature (prevents replay attacks)
            - HMAC-SHA256: Industry standard, FIPS 140-2 approved
            - Constant-time comparison required for verification
        """
        if timestamp is None:
            timestamp = int(time.time())

        # Create signed payload: timestamp.payload
        # This binds the signature to a specific time window
        signed_payload = f"{timestamp}.{payload}"

        # Compute HMAC-SHA256 signature
        signature_bytes = hmac.new(
            self.secret,
            signed_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        # Version-prefixed signature (allows algorithm upgrades)
        signature = f"{self.SIGNATURE_VERSION}={signature_bytes}"

        return signature, timestamp

    def verify_signature(
        self,
        payload: str,
        signature: str,
        timestamp: int,
        tolerance_seconds: int | None = None,
    ) -> bool:
        """
        Verify webhook signature.

        Args:
            payload: JSON webhook payload (as string)
            signature: Signature from X-Webhook-Signature header
            timestamp: Timestamp from X-Webhook-Timestamp header
            tolerance_seconds: Max age of signature (None = use default 5 min)

        Returns:
            bool: True if signature is valid and not expired

        Raises:
            ValueError: If signature format is invalid

        Security checks:
            1. Signature format validation (v1=...)
            2. Timestamp freshness (prevents replay attacks)
            3. HMAC verification (constant-time comparison)
        """
        if tolerance_seconds is None:
            tolerance_seconds = self.TIMESTAMP_TOLERANCE_SECONDS

        # Parse signature version
        if "=" not in signature:
            raise ValueError("Invalid signature format (expected 'v1=...')")

        version, signature_bytes = signature.split("=", 1)

        if version != self.SIGNATURE_VERSION:
            logger.warning(f"Unsupported signature version: {version}")
            return False

        # Check timestamp freshness (replay attack prevention)
        current_time = int(time.time())
        age_seconds = current_time - timestamp

        if age_seconds > tolerance_seconds:
            logger.warning(f"Signature expired: age={age_seconds}s, tolerance={tolerance_seconds}s")
            return False

        if age_seconds < -tolerance_seconds:
            # Timestamp is in the future (clock skew or attack)
            logger.warning(f"Signature timestamp in future: skew={abs(age_seconds)}s")
            return False

        # Recompute signature for comparison
        expected_signature, _ = self.sign_payload(payload, timestamp)
        expected_bytes = expected_signature.split("=", 1)[1]

        # Constant-time comparison (prevents timing attacks)
        return hmac.compare_digest(signature_bytes, expected_bytes)

    def create_headers(
        self,
        payload: str,
        timestamp: int | None = None,
    ) -> dict[str, str]:
        """
        Create webhook headers with signature and timestamp.

        Args:
            payload: JSON webhook payload (as string)
            timestamp: Unix timestamp (None = use current time)

        Returns:
            dict: Headers to include in webhook POST request

        Example:
            {
                "X-Webhook-Signature": "v1=a1b2c3d4...",
                "X-Webhook-Timestamp": "1700000000",
                "Content-Type": "application/json",
            }

        Usage:
            headers = signer.create_headers(json.dumps(event_data))
            response = requests.post(webhook_url, data=payload, headers=headers)
        """
        signature, ts = self.sign_payload(payload, timestamp)

        return {
            "X-Webhook-Signature": signature,
            "X-Webhook-Timestamp": str(ts),
            "Content-Type": "application/json",
            "User-Agent": "EpisodicMemory-Webhook/1.0",
        }


def generate_webhook_secret() -> str:
    """
    Generate cryptographically secure webhook secret.

    Returns:
        str: 64-character hex string (256-bit entropy)

    Usage:
        secret = generate_webhook_secret()
        # Store in customer record
        # Share with customer during onboarding
    """
    import secrets

    return secrets.token_hex(32)  # 32 bytes = 64 hex chars


# Example verification code for customer documentation
VERIFICATION_EXAMPLE = '''
# Customer webhook endpoint (example in Python)
from flask import Flask, request, abort
import hmac
import hashlib
import time

app = Flask(__name__)
WEBHOOK_SECRET = "your-webhook-secret-from-episodicmemory"

@app.route("/webhooks/episodicmemory", methods=["POST"])
def handle_webhook():
    """Verify and process EpisodicMemory webhook."""

    # Extract headers
    signature = request.headers.get("X-Webhook-Signature")
    timestamp = request.headers.get("X-Webhook-Timestamp")

    if not signature or not timestamp:
        abort(400, "Missing webhook signature headers")

    # Get payload
    payload = request.get_data(as_text=True)

    # Verify signature
    if not verify_webhook_signature(payload, signature, int(timestamp)):
        abort(401, "Invalid webhook signature")

    # Process event
    event_data = request.get_json()
    print(f"Received webhook: {event_data['event_type']}")

    return {"status": "received"}, 200

def verify_webhook_signature(payload, signature, timestamp):
    """Verify webhook signature from EpisodicMemory."""

    # Check timestamp (prevent replay attacks)
    current_time = int(time.time())
    if abs(current_time - timestamp) > 300:  # 5 minutes
        return False

    # Recompute signature
    signed_payload = f"{timestamp}.{payload}"
    expected_sig = hmac.new(
        WEBHOOK_SECRET.encode(),
        signed_payload.encode(),
        hashlib.sha256
    ).hexdigest()

    # Extract signature bytes (remove "v1=" prefix)
    if "=" not in signature:
        return False
    version, sig_bytes = signature.split("=", 1)

    # Constant-time comparison
    return hmac.compare_digest(sig_bytes, expected_sig)
'''
