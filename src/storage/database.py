"""
Customer and API key storage using SQLite .

Security features:
- API keys are never stored in plaintext (only SHA-256 digest is stored)
- Prepared statements (SQL injection protection)
- Row-level security ready (when migrating to Postgres)
- Audit logging for all mutations

Performance features:
- Async operations with aiosqlite (non-blocking)
- O(1) API key validation via SHA-256 indexed lookup
- Indexes on customer_id, key_hash_sha256, email
- Connection pooling ready
- Efficient queries with covering indexes
"""

import hashlib
import logging
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite

from src.config import get_settings
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
    - API keys stored as SHA-256 digest (keys are 256-bit random; digest is not reversible)
    - Customer data isolated by customer_id
    - Prepared statements prevent SQL injection
    - Audit log for compliance

    Performance:
    - Async operations with aiosqlite (non-blocking)
    - Indexes on all foreign keys and lookup columns
    - O(1) API key validation via indexed digest lookup
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
        self._conn: aiosqlite.Connection | None = None
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

        # Use async connection for schema creation
        async with aiosqlite.connect(str(self.db_path)) as conn:
            conn.row_factory = aiosqlite.Row

            try:
                # Enable foreign key constraints (security: prevent orphaned records)
                await conn.execute("PRAGMA foreign_keys = ON")

                # Enable WAL mode for better concurrency (performance)
                await conn.execute("PRAGMA journal_mode = WAL")

                # Create customers table
                await conn.execute(
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

                        CHECK (customer_id GLOB '[a-z0-9-]*'),
                        CHECK (credits_used_current_month >= 0),
                        CHECK (monthly_credit_limit >= 0)
                    )
                """
                )

                # Create API keys table
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS api_keys (
                        key_id TEXT PRIMARY KEY,
                        customer_id TEXT NOT NULL,
                        key_hash_sha256 TEXT NOT NULL UNIQUE,
                        key_prefix TEXT NOT NULL,
                        name TEXT,
                        created_at TEXT NOT NULL,
                        last_used_at TEXT,
                        expires_at TEXT,
                        is_active INTEGER NOT NULL DEFAULT 1,

                        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
                            ON DELETE CASCADE,
                        CHECK (is_active IN (0, 1)),
                        CHECK (length(key_hash_sha256) = 64),
                        CHECK (key_hash_sha256 NOT GLOB '*[^0-9a-f]*')
                    )
                """
                )

                # Schema guardrail: we don't support legacy API key schemas in code.
                cursor = await conn.execute("PRAGMA table_info(api_keys)")
                api_key_columns = {row["name"] for row in await cursor.fetchall()}
                if "key_hash" in api_key_columns or "key_hash_sha256" not in api_key_columns:
                    raise RuntimeError(
                        "Unsupported api_keys schema detected. "
                        "Delete the customer DB file and re-initialize it."
                    )

                # Create audit log table (compliance: track all mutations)
                await conn.execute(
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
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_customers_status ON customers(status)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_api_keys_customer ON api_keys(customer_id)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_api_keys_hash_sha256 ON api_keys(key_hash_sha256)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_customer ON audit_log(customer_id)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)"
                )

                await conn.commit()
                logger.info("Customer database initialized successfully")
                self._initialized = True

            except Exception as e:
                logger.error(f"Failed to initialize database: {e}")
                raise

    async def _get_connection(self) -> aiosqlite.Connection:
        """Get database connection (creates if needed)."""
        if self._conn is None:
            self._conn = await aiosqlite.connect(str(self.db_path))
            self._conn.row_factory = aiosqlite.Row
            await self._conn.execute("PRAGMA foreign_keys = ON")
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

        conn = await self._get_connection()
        try:
            await conn.execute(
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
            await conn.commit()

            # Audit log
            await self._log_audit(
                customer_id=customer.customer_id,
                action="CREATE",
                resource_type="customer",
                resource_id=customer.customer_id,
            )

            logger.info(f"Created customer: {customer.customer_id}")
            return customer

        except aiosqlite.IntegrityError as e:
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
        conn = await self._get_connection()
        cursor = await conn.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,))
        row = await cursor.fetchone()

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

        conn = await self._get_connection()
        query = f"UPDATE customers SET {', '.join(updates)} WHERE customer_id = ?"

        await conn.execute(query, params)
        await conn.commit()

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
        conn = await self._get_connection()
        now = datetime.now(UTC).isoformat()

        cursor = await conn.execute(
            """
            UPDATE customers
            SET credits_used_current_month = credits_used_current_month + ?,
                last_api_call_at = ?,
                updated_at = ?
            WHERE customer_id = ?
            """,
            (credits, now, now, customer_id),
        )
        await conn.commit()

        return cursor.rowcount > 0

    async def reset_monthly_usage(self, customer_id: str) -> bool:
        """
        Reset monthly credit usage (called on quota cycle).

        Args:
            customer_id: Customer to reset

        Returns:
            bool: True if successful
        """
        conn = await self._get_connection()
        now = datetime.now(UTC).isoformat()

        cursor = await conn.execute(
            """
            UPDATE customers
            SET credits_used_current_month = 0,
                updated_at = ?
            WHERE customer_id = ?
            """,
            (now, customer_id),
        )
        await conn.commit()

        # Audit log
        await self._log_audit(
            customer_id=customer_id,
            action="RESET_USAGE",
            resource_type="customer",
            resource_id=customer_id,
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
        conn = await self._get_connection()
        now = datetime.now(UTC).isoformat()

        cursor = await conn.execute(
            """
            UPDATE customers
            SET status = ?,
                updated_at = ?
            WHERE customer_id = ?
            """,
            (status.value, now, customer_id),
        )
        await conn.commit()

        await self._log_audit(
            customer_id=customer_id,
            action="UPDATE_STATUS",
            resource_type="customer",
            resource_id=customer_id,
            details=f"status={status.value}",
        )

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
        - Stores only SHA-256 digest (never plaintext)
        - Returns plaintext only once at creation
        """
        import uuid

        # Generate secure random API key
        # Format: em_live_<64 hex chars> (32 bytes)
        random_bytes = secrets.token_bytes(32)
        key_suffix = random_bytes.hex()
        plaintext_key = f"em_live_{key_suffix}"

        key_hash_sha256 = hashlib.sha256(plaintext_key.encode()).hexdigest()

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
            key_hash_sha256=key_hash_sha256,
            key_prefix=key_prefix,
            name=api_key_create.name,
            created_at=now,
            expires_at=expires_at,
            is_active=True,
        )

        conn = await self._get_connection()
        await conn.execute(
            """
            INSERT INTO api_keys (
                key_id, customer_id, key_hash_sha256, key_prefix, name,
                created_at, expires_at, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                api_key.key_id,
                api_key.customer_id,
                key_hash_sha256,
                api_key.key_prefix,
                api_key.name,
                now.isoformat(),
                expires_at.isoformat() if expires_at else None,
                1,
            ),
        )
        await conn.commit()

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
        - Checks expiration
        - Checks is_active flag
        - Updates last_used_at timestamp

        Performance:
        - SHA-256 indexed lookup (O(1))
        """
        if not plaintext_key.startswith("em_live_"):
            return None
        # Defensive: prevent unusually large headers from wasting CPU/memory.
        # Valid keys are ~72 chars (em_live_ + 64 hex chars).
        if len(plaintext_key) > 256:
            return None

        conn = await self._get_connection()

        # Fast path: SHA-256 lookup narrows to a single key, then bcrypt verify
        key_hash_sha256 = hashlib.sha256(plaintext_key.encode()).hexdigest()
        cursor = await conn.execute(
            """
            SELECT k.*, c.*
            FROM api_keys k
            JOIN customers c ON k.customer_id = c.customer_id
            WHERE k.is_active = 1 AND k.key_hash_sha256 = ?
            """
            ,
            (key_hash_sha256,),
        )
        row = await cursor.fetchone()

        if row:
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
            await conn.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE key_id = ?",
                (now, row["key_id"]),
            )
            await conn.commit()

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
        conn = await self._get_connection()
        cursor = await conn.execute("UPDATE api_keys SET is_active = 0 WHERE key_id = ?", (key_id,))
        await conn.commit()

        if cursor.rowcount > 0:
            # Get customer_id for audit log
            cursor2 = await conn.execute(
                "SELECT customer_id FROM api_keys WHERE key_id = ?", (key_id,)
            )
            row = await cursor2.fetchone()
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
        conn = await self._get_connection()
        now = datetime.now(UTC).isoformat()

        await conn.execute(
            """
            INSERT INTO audit_log (
                timestamp, customer_id, action, resource_type,
                resource_id, details, ip_address
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (now, customer_id, action, resource_type, resource_id, details, ip_address),
        )
        await conn.commit()

    async def close(self) -> None:
        """Close database connection."""
        if self._conn:
            await self._conn.close()
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
        settings = get_settings()
        _db = CustomerDatabase(db_path=settings.storage.customer_db_path)
        await _db.initialize()
    return _db
