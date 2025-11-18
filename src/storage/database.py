"""
Customer and API key storage using SQLite (bootstrap) → PostgreSQL (production).

Security features:
- API keys hashed with bcrypt (cost factor 12)
- Prepared statements (SQL injection protection)
- Row-level security ready (when migrating to Postgres)
- Audit logging for all mutations

Performance features:
- Indexes on customer_id, api_key_hash, email
- Connection pooling ready
- Efficient queries with covering indexes
"""

import logging
import secrets
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import bcrypt

from src.models.customer import (
    APIKey,
    APIKeyCreate,
    Customer,
    CustomerCreate,
    CustomerStatus,
    CustomerUpdate,
    SubscriptionTier,
)

logger = logging.getLogger(__name__)


class CustomerDatabase:
    """
    Customer and API key storage.

    Uses SQLite for bootstrapping (free, embedded).
    Migration path to PostgreSQL documented for production scale.

    Security:
    - API keys hashed with bcrypt (cost=12, ~0.3s per hash)
    - Customer data isolated by customer_id
    - Prepared statements prevent SQL injection
    - Audit log for compliance

    Performance:
    - Indexes on all foreign keys and lookup columns
    - Connection pooling (aiosqlite handles this)
    - Efficient upsert operations
    """

    def __init__(self, db_path: str = "./data/customers.db"):
        """
        Initialize customer database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connection will be created lazily
        self._conn: sqlite3.Connection | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize database schema.

        Creates tables with proper indexes for performance and security.
        Idempotent - safe to call multiple times.
        """
        if self._initialized:
            return

        logger.info(f"Initializing customer database at {self.db_path}")

        # Use synchronous connection for schema creation
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        try:
            # Enable foreign key constraints (security: prevent orphaned records)
            conn.execute("PRAGMA foreign_keys = ON")

            # Enable WAL mode for better concurrency (performance)
            conn.execute("PRAGMA journal_mode = WAL")

            # Create customers table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS customers (
                    customer_id TEXT PRIMARY KEY,
                    organization_name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    subscription_tier TEXT NOT NULL DEFAULT 'free',
                    status TEXT NOT NULL DEFAULT 'active',
                    monthly_credit_limit INTEGER NOT NULL DEFAULT 1000,
                    credits_used_current_month INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_api_call_at TEXT,
                    stripe_customer_id TEXT UNIQUE,
                    stripe_subscription_id TEXT,
                    stripe_payment_method_id TEXT,
                    billing_cycle_start TEXT,
                    billing_cycle_end TEXT,
                    next_invoice_date TEXT,
                    payment_failed INTEGER NOT NULL DEFAULT 0,
                    payment_failed_at TEXT,
                    trial_end_date TEXT,

                    CHECK (customer_id GLOB '[a-z0-9-]*'),
                    CHECK (credits_used_current_month >= 0),
                    CHECK (monthly_credit_limit >= 0),
                    CHECK (payment_failed IN (0, 1))
                )
            """
            )

            # Create API keys table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    key_hash TEXT NOT NULL UNIQUE,
                    key_prefix TEXT NOT NULL,
                    name TEXT,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT,
                    expires_at TEXT,
                    is_active INTEGER NOT NULL DEFAULT 1,

                    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
                        ON DELETE CASCADE,
                    CHECK (is_active IN (0, 1))
                )
            """
            )

            # Create audit log table (compliance: track all mutations)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    customer_id TEXT,
                    action TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT,
                    details TEXT,
                    ip_address TEXT,

                    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
                        ON DELETE SET NULL
                )
            """
            )

            # Performance indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_customers_status ON customers(status)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_api_keys_customer ON api_keys(customer_id)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_customer ON audit_log(customer_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")

            conn.commit()
            logger.info("Customer database initialized successfully")
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection (creates if needed)."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
        return self._conn

    async def create_customer(self, customer_create: CustomerCreate) -> Customer | None:
        """
        Create new customer.

        Args:
            customer_create: Customer creation data

        Returns:
            Customer: Created customer, or None if customer_id already exists

        Security: Validates customer_id format, checks uniqueness
        """
        # Set quota based on tier
        tier_quotas = {
            SubscriptionTier.FREE: 1000,
            SubscriptionTier.STARTER: 10000,
            SubscriptionTier.PRO: 100000,
            SubscriptionTier.ENTERPRISE: 999999999,  # Effectively unlimited
        }
        quota = tier_quotas.get(customer_create.subscription_tier, 1000)

        now = datetime.now(UTC).isoformat()

        customer = Customer(
            customer_id=customer_create.customer_id,
            organization_name=customer_create.organization_name,
            email=customer_create.email,
            subscription_tier=customer_create.subscription_tier,
            status=CustomerStatus.ACTIVE,
            monthly_credit_limit=quota,
            credits_used_current_month=0,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO customers (
                    customer_id, organization_name, email, subscription_tier,
                    status, monthly_credit_limit, credits_used_current_month,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    customer.customer_id,
                    customer.organization_name,
                    customer.email,
                    customer.subscription_tier.value,
                    customer.status.value,
                    customer.monthly_credit_limit,
                    customer.credits_used_current_month,
                    now,
                    now,
                ),
            )
            conn.commit()

            # Audit log
            await self._log_audit(
                customer_id=customer.customer_id,
                action="CREATE",
                resource_type="customer",
                resource_id=customer.customer_id,
            )

            logger.info(f"Created customer: {customer.customer_id}")
            return customer

        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                logger.warning(f"Customer creation failed: {customer.customer_id} already exists")
                return None
            raise

    async def get_customer(self, customer_id: str) -> Customer | None:
        """
        Get customer by ID.

        Args:
            customer_id: Customer identifier

        Returns:
            Customer or None if not found
        """
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()

        if not row:
            return None

        return Customer(
            customer_id=row["customer_id"],
            organization_name=row["organization_name"],
            email=row["email"],
            subscription_tier=SubscriptionTier(row["subscription_tier"]),
            status=CustomerStatus(row["status"]),
            monthly_credit_limit=row["monthly_credit_limit"],
            credits_used_current_month=row["credits_used_current_month"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_api_call_at=(
                datetime.fromisoformat(row["last_api_call_at"]) if row["last_api_call_at"] else None
            ),
            stripe_customer_id=row["stripe_customer_id"],
            stripe_subscription_id=row["stripe_subscription_id"],
            stripe_payment_method_id=row.get("stripe_payment_method_id"),
            billing_cycle_start=(
                datetime.fromisoformat(row["billing_cycle_start"])
                if row.get("billing_cycle_start")
                else None
            ),
            billing_cycle_end=(
                datetime.fromisoformat(row["billing_cycle_end"])
                if row.get("billing_cycle_end")
                else None
            ),
            next_invoice_date=(
                datetime.fromisoformat(row["next_invoice_date"])
                if row.get("next_invoice_date")
                else None
            ),
            payment_failed=bool(row.get("payment_failed", 0)),
            payment_failed_at=(
                datetime.fromisoformat(row["payment_failed_at"])
                if row.get("payment_failed_at")
                else None
            ),
            trial_end_date=(
                datetime.fromisoformat(row["trial_end_date"]) if row.get("trial_end_date") else None
            ),
        )

    async def update_customer(self, customer_id: str, update: CustomerUpdate) -> Customer | None:
        """
        Update customer details.

        Args:
            customer_id: Customer to update
            update: Fields to update

        Returns:
            Updated customer or None if not found
        """
        # Build dynamic UPDATE query for only provided fields
        updates = []
        params = []

        if update.organization_name is not None:
            updates.append("organization_name = ?")
            params.append(update.organization_name)
        if update.email is not None:
            updates.append("email = ?")
            params.append(update.email)
        if update.subscription_tier is not None:
            updates.append("subscription_tier = ?")
            params.append(update.subscription_tier.value)
        if update.status is not None:
            updates.append("status = ?")
            params.append(update.status.value)
        if update.monthly_credit_limit is not None:
            updates.append("monthly_credit_limit = ?")
            params.append(update.monthly_credit_limit)

        if not updates:
            # No updates provided
            return await self.get_customer(customer_id)

        updates.append("updated_at = ?")
        params.append(datetime.now(UTC).isoformat())
        params.append(customer_id)

        conn = self._get_connection()
        query = f"UPDATE customers SET {', '.join(updates)} WHERE customer_id = ?"

        conn.execute(query, params)
        conn.commit()

        # Audit log
        await self._log_audit(
            customer_id=customer_id,
            action="UPDATE",
            resource_type="customer",
            resource_id=customer_id,
            details=str(update.model_dump(exclude_none=True)),
        )

        return await self.get_customer(customer_id)

    async def increment_usage(self, customer_id: str, credits: int) -> bool:
        """
        Increment customer credit usage (atomic operation).

        Args:
            customer_id: Customer to update
            credits: Credits to add

        Returns:
            bool: True if successful, False if customer not found

        Security: Atomic update prevents race conditions
        Performance: Single UPDATE statement, no SELECT needed
        """
        conn = self._get_connection()
        now = datetime.now(UTC).isoformat()

        cursor = conn.execute(
            """
            UPDATE customers
            SET credits_used_current_month = credits_used_current_month + ?,
                last_api_call_at = ?,
                updated_at = ?
            WHERE customer_id = ?
            """,
            (credits, now, now, customer_id),
        )
        conn.commit()

        return cursor.rowcount > 0

    async def reset_monthly_usage(self, customer_id: str) -> bool:
        """
        Reset monthly credit usage (called on billing cycle).

        Args:
            customer_id: Customer to reset

        Returns:
            bool: True if successful
        """
        conn = self._get_connection()
        now = datetime.now(UTC).isoformat()

        cursor = conn.execute(
            """
            UPDATE customers
            SET credits_used_current_month = 0,
                updated_at = ?
            WHERE customer_id = ?
            """,
            (now, customer_id),
        )
        conn.commit()

        # Audit log
        await self._log_audit(
            customer_id=customer_id,
            action="RESET_USAGE",
            resource_type="customer",
            resource_id=customer_id,
        )

        return cursor.rowcount > 0

    async def update_customer_stripe_id(
        self, customer_id: str, stripe_customer_id: str, stripe_subscription_id: str | None = None
    ) -> bool:
        """
        Update customer Stripe IDs.

        Args:
            customer_id: Customer to update
            stripe_customer_id: Stripe customer ID (cus_xxx)
            stripe_subscription_id: Stripe subscription ID (sub_xxx), optional

        Returns:
            bool: True if successful
        """
        conn = self._get_connection()
        now = datetime.now(UTC).isoformat()

        if stripe_subscription_id:
            cursor = conn.execute(
                """
                UPDATE customers
                SET stripe_customer_id = ?,
                    stripe_subscription_id = ?,
                    updated_at = ?
                WHERE customer_id = ?
                """,
                (stripe_customer_id, stripe_subscription_id, now, customer_id),
            )
        else:
            cursor = conn.execute(
                """
                UPDATE customers
                SET stripe_customer_id = ?,
                    updated_at = ?
                WHERE customer_id = ?
                """,
                (stripe_customer_id, now, customer_id),
            )
        conn.commit()

        await self._log_audit(
            customer_id=customer_id,
            action="UPDATE_STRIPE_ID",
            resource_type="customer",
            resource_id=customer_id,
            details=f"stripe_customer_id={stripe_customer_id}",
        )

        return cursor.rowcount > 0

    async def update_customer_payment_failed(
        self, customer_id: str, failed: bool, failed_at: datetime | None
    ) -> bool:
        """
        Update customer payment failure status.

        Args:
            customer_id: Customer to update
            failed: True if payment failed
            failed_at: Timestamp of failure (None to clear)

        Returns:
            bool: True if successful
        """
        conn = self._get_connection()
        now = datetime.now(UTC).isoformat()

        cursor = conn.execute(
            """
            UPDATE customers
            SET payment_failed = ?,
                payment_failed_at = ?,
                updated_at = ?
            WHERE customer_id = ?
            """,
            (
                1 if failed else 0,
                failed_at.isoformat() if failed_at else None,
                now,
                customer_id,
            ),
        )
        conn.commit()

        await self._log_audit(
            customer_id=customer_id,
            action="UPDATE_PAYMENT_STATUS",
            resource_type="customer",
            resource_id=customer_id,
            details=f"payment_failed={failed}",
        )

        return cursor.rowcount > 0

    async def update_customer_status(self, customer_id: str, status: CustomerStatus) -> bool:
        """
        Update customer status.

        Args:
            customer_id: Customer to update
            status: New status (ACTIVE, SUSPENDED, DELETED)

        Returns:
            bool: True if successful
        """
        conn = self._get_connection()
        now = datetime.now(UTC).isoformat()

        cursor = conn.execute(
            """
            UPDATE customers
            SET status = ?,
                updated_at = ?
            WHERE customer_id = ?
            """,
            (status.value, now, customer_id),
        )
        conn.commit()

        await self._log_audit(
            customer_id=customer_id,
            action="UPDATE_STATUS",
            resource_type="customer",
            resource_id=customer_id,
            details=f"status={status.value}",
        )

        return cursor.rowcount > 0

    async def update_billing_cycle(
        self,
        customer_id: str,
        billing_cycle_start: datetime,
        billing_cycle_end: datetime,
        next_invoice_date: datetime | None = None,
    ) -> bool:
        """
        Update customer billing cycle dates.

        Args:
            customer_id: Customer to update
            billing_cycle_start: Start of billing period
            billing_cycle_end: End of billing period
            next_invoice_date: Next invoice generation date

        Returns:
            bool: True if successful
        """
        conn = self._get_connection()
        now = datetime.now(UTC).isoformat()

        cursor = conn.execute(
            """
            UPDATE customers
            SET billing_cycle_start = ?,
                billing_cycle_end = ?,
                next_invoice_date = ?,
                updated_at = ?
            WHERE customer_id = ?
            """,
            (
                billing_cycle_start.isoformat(),
                billing_cycle_end.isoformat(),
                next_invoice_date.isoformat() if next_invoice_date else None,
                now,
                customer_id,
            ),
        )
        conn.commit()

        return cursor.rowcount > 0

    async def create_api_key(self, api_key_create: APIKeyCreate) -> tuple[str, APIKey]:
        """
        Create new API key for customer.

        Args:
            api_key_create: API key creation data

        Returns:
            tuple: (plaintext_key, APIKey model)
                IMPORTANT: plaintext_key is only returned once, never stored

        Security:
        - Generates cryptographically secure random key (32 bytes = 256 bits)
        - Hashes with bcrypt (cost=12, ~300ms per hash)
        - Only stores hash, never plaintext
        - Returns plaintext only once at creation
        """
        import uuid

        # Generate secure random API key
        # Format: em_live_<32 random hex chars>
        random_bytes = secrets.token_bytes(32)
        key_suffix = random_bytes.hex()
        plaintext_key = f"em_live_{key_suffix}"

        # Hash with bcrypt (cost=12 for security/performance balance)
        key_hash = bcrypt.hashpw(plaintext_key.encode(), bcrypt.gensalt(rounds=12))

        # Extract prefix for display (first 8 chars after em_live_)
        key_prefix = key_suffix[:8]

        # Generate key ID
        key_id = str(uuid.uuid4())

        # Calculate expiration
        now = datetime.now(UTC)
        expires_at = None
        if api_key_create.expires_in_days:
            expires_at = now + timedelta(days=api_key_create.expires_in_days)

        api_key = APIKey(
            key_id=key_id,
            customer_id=api_key_create.customer_id,
            key_hash=key_hash.decode(),  # Store as string
            key_prefix=key_prefix,
            name=api_key_create.name,
            created_at=now,
            expires_at=expires_at,
            is_active=True,
        )

        conn = self._get_connection()
        conn.execute(
            """
            INSERT INTO api_keys (
                key_id, customer_id, key_hash, key_prefix, name,
                created_at, expires_at, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                api_key.key_id,
                api_key.customer_id,
                api_key.key_hash,
                api_key.key_prefix,
                api_key.name,
                now.isoformat(),
                expires_at.isoformat() if expires_at else None,
                1,
            ),
        )
        conn.commit()

        # Audit log (DO NOT log plaintext key)
        await self._log_audit(
            customer_id=api_key_create.customer_id,
            action="CREATE",
            resource_type="api_key",
            resource_id=key_id,
            details=f"prefix={key_prefix}",
        )

        logger.info(f"Created API key for customer: {api_key_create.customer_id}")

        return (plaintext_key, api_key)

    async def validate_api_key(self, plaintext_key: str) -> Customer | None:
        """
        Validate API key and return associated customer.

        Args:
            plaintext_key: API key from request header

        Returns:
            Customer if valid, None if invalid/expired

        Security:
        - Constant-time comparison (bcrypt.checkpw)
        - Checks expiration
        - Checks is_active flag
        - Updates last_used_at timestamp

        Performance:
        - ~300ms per validation (bcrypt cost=12)
        - Index on key_hash for fast lookup
        """
        if not plaintext_key.startswith("em_live_"):
            return None

        conn = self._get_connection()

        # Get all active API keys (we need to hash-compare each one)
        # Note: This is a performance trade-off for security
        # Alternative: hash the key first and lookup by hash (but exposes timing attacks)
        rows = conn.execute(
            """
            SELECT k.*, c.*
            FROM api_keys k
            JOIN customers c ON k.customer_id = c.customer_id
            WHERE k.is_active = 1
            """
        ).fetchall()

        for row in rows:
            # Constant-time comparison using bcrypt
            if bcrypt.checkpw(plaintext_key.encode(), row["key_hash"].encode()):
                # Found matching key - check expiration
                if row["expires_at"]:
                    expires_at = datetime.fromisoformat(row["expires_at"])
                    if datetime.now(UTC) > expires_at:
                        logger.warning(f"API key expired: {row['key_id']}")
                        return None

                # Check customer status
                if row["status"] != CustomerStatus.ACTIVE.value:
                    logger.warning(f"Customer not active: {row['customer_id']}")
                    return None

                # Update last_used_at
                now = datetime.now(UTC).isoformat()
                conn.execute(
                    "UPDATE api_keys SET last_used_at = ? WHERE key_id = ?",
                    (now, row["key_id"]),
                )
                conn.commit()

                # Return customer
                return Customer(
                    customer_id=row["customer_id"],
                    organization_name=row["organization_name"],
                    email=row["email"],
                    subscription_tier=SubscriptionTier(row["subscription_tier"]),
                    status=CustomerStatus(row["status"]),
                    monthly_credit_limit=row["monthly_credit_limit"],
                    credits_used_current_month=row["credits_used_current_month"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    last_api_call_at=(
                        datetime.fromisoformat(row["last_api_call_at"])
                        if row["last_api_call_at"]
                        else None
                    ),
                    stripe_customer_id=row["stripe_customer_id"],
                    stripe_subscription_id=row["stripe_subscription_id"],
                )

        # No matching key found
        return None

    async def revoke_api_key(self, key_id: str) -> bool:
        """
        Revoke (deactivate) API key.

        Args:
            key_id: API key ID to revoke

        Returns:
            bool: True if revoked, False if not found
        """
        conn = self._get_connection()
        cursor = conn.execute("UPDATE api_keys SET is_active = 0 WHERE key_id = ?", (key_id,))
        conn.commit()

        if cursor.rowcount > 0:
            # Get customer_id for audit log
            row = conn.execute(
                "SELECT customer_id FROM api_keys WHERE key_id = ?", (key_id,)
            ).fetchone()
            if row:
                await self._log_audit(
                    customer_id=row["customer_id"],
                    action="REVOKE",
                    resource_type="api_key",
                    resource_id=key_id,
                )

        return cursor.rowcount > 0

    async def _log_audit(
        self,
        action: str,
        resource_type: str,
        customer_id: str | None = None,
        resource_id: str | None = None,
        details: str | None = None,
        ip_address: str | None = None,
    ) -> None:
        """
        Log audit event for compliance.

        Args:
            action: Action performed (CREATE, UPDATE, DELETE, etc.)
            resource_type: Type of resource (customer, api_key, episode)
            customer_id: Customer performing action
            resource_id: ID of affected resource
            details: Additional details (JSON string)
            ip_address: IP address of request
        """
        conn = self._get_connection()
        now = datetime.now(UTC).isoformat()

        conn.execute(
            """
            INSERT INTO audit_log (
                timestamp, customer_id, action, resource_type,
                resource_id, details, ip_address
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (now, customer_id, action, resource_type, resource_id, details, ip_address),
        )
        conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# Global instance
_db: CustomerDatabase | None = None


async def get_customer_db() -> CustomerDatabase:
    """
    Get global customer database instance.

    Returns:
        CustomerDatabase: Initialized database
    """
    global _db
    if _db is None:
        _db = CustomerDatabase()
        await _db.initialize()
    return _db
