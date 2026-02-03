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

import asyncio
import hashlib
import logging
import secrets
import sqlite3
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
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

                # Guardrail: refuse to start with a legacy api_keys schema.
                # This codebase is intentionally "clean-slate" (no legacy compatibility).
                cursor = await conn.execute("PRAGMA table_info(api_keys)")
                api_key_columns = {row["name"] for row in await cursor.fetchall()}
                await cursor.close()
                if "key_hash_sha256" not in api_key_columns:
                    raise RuntimeError(
                        "Unsupported api_keys schema detected (missing 'key_hash_sha256'). "
                        "Delete the customer DB file and re-create API keys."
                    )
                if "key_hash" in api_key_columns:
                    raise RuntimeError(
                        "Legacy api_keys schema detected (column 'key_hash'). "
                        "Delete the customer DB file and re-create API keys."
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

                # Episode index (required for Phase 6 clustering / hygiene jobs).
                #
                # KyroDB does not provide a server-side scan/list API, so Vritti
                # maintains a minimal authoritative index of episode IDs per customer.
                #
                # We intentionally store only small, non-sensitive fields:
                # - episode_id for enumeration
                # - collection and timestamps for batching
                # - archived/deleted flags for hygiene
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS episodes_index (
                        episode_id INTEGER PRIMARY KEY,
                        customer_id TEXT NOT NULL,
                        collection TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        has_image INTEGER NOT NULL DEFAULT 0,
                        archived INTEGER NOT NULL DEFAULT 0,
                        archived_at TEXT,
                        deleted INTEGER NOT NULL DEFAULT 0,
                        deleted_at TEXT,
                        last_updated_at TEXT NOT NULL,

                        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
                            ON DELETE CASCADE,
                        CHECK (has_image IN (0, 1)),
                        CHECK (archived IN (0, 1)),
                        CHECK (deleted IN (0, 1))
                    )
                    """
                )

                # Skills index (enumeration + reliable per-customer search fallback).
                #
                # KyroDB does not provide a server-side scan/list API. For skills (which are
                # expected to remain a relatively small set per customer), we maintain a minimal
                # index to enumerate IDs and, when needed, compute similarity in-process.
                #
                # This table is intentionally decoupled from `customers` via FK to keep indexing
                # best-effort in local/test flows; API-layer auth still enforces tenant isolation.
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS skills_index (
                        skill_id INTEGER PRIMARY KEY,
                        customer_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        deleted INTEGER NOT NULL DEFAULT 0,
                        deleted_at TEXT,
                        last_updated_at TEXT NOT NULL,

                        CHECK (deleted IN (0, 1))
                    )
                    """
                )

                # Doc ID allocator (KyroDB doc_id must be dense/small).
                #
                # KyroDB's current HNSW backend uses doc_id as an array index; very large or sparse
                # doc_ids can trigger pathological allocations and crash the DB process. In
                # addition, KyroDB's `namespace` is metadata-only: doc_id must be globally unique
                # across *all* namespaces and document types within an instance.
                #
                # We therefore allocate monotonically increasing global doc_ids via a single-row
                # counter keyed by a scope string (future-proofing for additional KyroDB instances).
                cursor = await conn.execute("PRAGMA table_info(doc_id_counters)")
                doc_id_columns = [row["name"] for row in await cursor.fetchall()]
                now = datetime.now(UTC).isoformat()

                if not doc_id_columns:
                    await conn.execute(
                        """
                        CREATE TABLE doc_id_counters (
                            scope TEXT PRIMARY KEY,
                            next_doc_id INTEGER NOT NULL,
                            updated_at TEXT NOT NULL,

                            CHECK (next_doc_id >= 1)
                        )
                        """
                    )
                    await conn.execute(
                        """
                        INSERT INTO doc_id_counters (scope, next_doc_id, updated_at)
                        VALUES ('kyrodb_text', 1, ?)
                        ON CONFLICT(scope) DO NOTHING
                        """,
                        (now,),
                    )
                elif "scope" in doc_id_columns and "next_doc_id" in doc_id_columns:
                    # Current schema already present.
                    pass
                else:
                    # Legacy schema: (customer_id, collection, next_doc_id, updated_at)
                    try:
                        await conn.execute("BEGIN IMMEDIATE")
                        row = await (
                            await conn.execute(
                                "SELECT MAX(next_doc_id) AS max_next FROM doc_id_counters"
                            )
                        ).fetchone()
                        max_next = int(row["max_next"] or 1) if row else 1

                        await conn.execute(
                            "DROP INDEX IF EXISTS idx_doc_id_counters_customer_collection"
                        )
                        await conn.execute(
                            "ALTER TABLE doc_id_counters RENAME TO doc_id_counters_old"
                        )
                        await conn.execute(
                            """
                            CREATE TABLE doc_id_counters (
                                scope TEXT PRIMARY KEY,
                                next_doc_id INTEGER NOT NULL,
                                updated_at TEXT NOT NULL,

                                CHECK (next_doc_id >= 1)
                            )
                            """
                        )
                        await conn.execute(
                            """
                            INSERT INTO doc_id_counters (scope, next_doc_id, updated_at)
                            VALUES ('kyrodb_text', ?, ?)
                            """,
                            (max_next, now),
                        )
                        await conn.execute("DROP TABLE doc_id_counters_old")
                        await conn.execute("COMMIT")
                    except Exception:
                        await conn.execute("ROLLBACK")
                        raise

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
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_episodes_customer_collection ON episodes_index(customer_id, collection)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_episodes_active ON episodes_index(customer_id, collection, archived, deleted, created_at)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_skills_active ON skills_index(customer_id, deleted, created_at)"
                )

                await conn.commit()
                logger.info("Customer database initialized successfully")
                self._initialized = True

            except Exception as e:
                logger.error(f"Failed to initialize database: {e}")
                raise

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        """
        Open a new connection for a single operation.

        We intentionally do not share a single connection across concurrent async tasks or
        across event loops (pytest-asyncio can create multiple loops). A connection-per-operation
        model avoids aiosqlite concurrency hazards and reduces "database is locked" errors under
        load when combined with WAL + busy_timeout.
        """
        conn = await aiosqlite.connect(str(self.db_path))
        conn.row_factory = aiosqlite.Row
        try:
            await conn.execute("PRAGMA foreign_keys = ON")
            await conn.execute("PRAGMA journal_mode = WAL")
            await conn.execute("PRAGMA busy_timeout = 5000")
            yield conn
        finally:
            await conn.close()

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

        async with self._connect() as conn:
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
                    logger.warning(
                        f"Customer creation failed: {customer.customer_id} already exists"
                    )
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
        async with self._connect() as conn:
            cursor = await conn.execute(
                "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
            )
            row = await cursor.fetchone()
            await cursor.close()

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

        async with self._connect() as conn:
            query = f"UPDATE customers SET {', '.join(updates)} WHERE customer_id = ?"

            cursor = await conn.execute(query, params)
            await conn.commit()
            await cursor.close()

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
        async with self._connect() as conn:
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
            await cursor.close()

            return cursor.rowcount > 0

    async def allocate_doc_id(
        self,
        *,
        scope: str = "kyrodb_text",
    ) -> int:
        """
        Allocate a new dense, globally unique doc_id for KyroDB.

        KyroDB's current HNSW backend uses doc_id as an array index. Large or sparse doc_ids
        can trigger catastrophic allocations inside KyroDB.

        Important: KyroDB's `namespace` is metadata-only; doc_id is the actual primary key.
        doc_ids must therefore be globally unique across all namespaces and document types
        within the same KyroDB instance.

        This method is concurrency-safe across processes as long as they share the same DB file.
        """
        scope = (scope or "").strip()
        if not scope:
            raise ValueError("scope is required")
        if len(scope) > 100:
            raise ValueError("Invalid scope")

        now = datetime.now(UTC).isoformat()

        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                async with self._connect() as conn:
                    cursor = await conn.execute(
                        """
                        INSERT INTO doc_id_counters (scope, next_doc_id, updated_at)
                        VALUES (?, 2, ?)
                        ON CONFLICT(scope) DO UPDATE SET
                            next_doc_id = doc_id_counters.next_doc_id + 1,
                            updated_at = excluded.updated_at
                        RETURNING (next_doc_id - 1) AS allocated
                        """,
                        (scope, now),
                    )
                    row = await cursor.fetchone()
                    await cursor.close()
                    await conn.commit()

                allocated = int(row["allocated"]) if row else 0
                if allocated < 1:
                    raise RuntimeError("Allocated doc_id is invalid")
                return allocated
            except (aiosqlite.OperationalError, sqlite3.OperationalError) as e:
                message = str(e).lower()
                if "database is locked" not in message and "database is busy" not in message:
                    raise
                if attempt >= max_attempts - 1:
                    raise
                await asyncio.sleep(0.05 * (2**attempt))

    async def index_episode(
        self,
        *,
        episode_id: int,
        customer_id: str,
        collection: str,
        created_at: datetime,
        has_image: bool,
    ) -> None:
        """
        Insert/update the episode index entry for enumeration.

        This index is required because KyroDB does not provide server-side scans
        of all documents in a namespace. Offline jobs (clustering/decay) rely on it.

        Security:
        - Stores only non-sensitive metadata (no goal/error_trace/etc.)
        - Enforces customer_id via FK constraint

        Reliability:
        - Idempotent upsert on episode_id
        """
        if not customer_id or len(customer_id) > 100:
            raise ValueError("Invalid customer_id for episode indexing")
        if not collection or len(collection) > 100:
            raise ValueError("Invalid collection for episode indexing")
        if episode_id <= 0:
            raise ValueError("episode_id must be > 0 for episode indexing")

        now = datetime.now(UTC).isoformat()

        async with self._connect() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO episodes_index (
                    episode_id, customer_id, collection, created_at,
                    has_image, archived, archived_at, deleted, deleted_at,
                    last_updated_at
                ) VALUES (?, ?, ?, ?, ?, 0, NULL, 0, NULL, ?)
                ON CONFLICT(episode_id) DO UPDATE SET
                    customer_id = excluded.customer_id,
                    collection = excluded.collection,
                    created_at = excluded.created_at,
                    has_image = excluded.has_image,
                    last_updated_at = excluded.last_updated_at
                """,
                (
                    int(episode_id),
                    customer_id,
                    collection,
                    created_at.isoformat(),
                    1 if has_image else 0,
                    now,
                ),
            )
            await conn.commit()
            await cursor.close()

    async def list_episode_ids(
        self,
        *,
        customer_id: str,
        collection: str,
        include_archived: bool = False,
        include_deleted: bool = False,
        limit: int | None = None,
    ) -> list[int]:
        """
        Enumerate episode IDs for a customer/collection.

        Returns newest-first.
        """
        if not customer_id or len(customer_id) > 100:
            raise ValueError("Invalid customer_id for episode listing")
        if not collection or len(collection) > 100:
            raise ValueError("Invalid collection for episode listing")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be > 0 when provided")

        where = ["customer_id = ?", "collection = ?"]
        params: list[object] = [customer_id, collection]
        if not include_archived:
            where.append("archived = 0")
        if not include_deleted:
            where.append("deleted = 0")

        query = (
            "SELECT episode_id FROM episodes_index "
            f"WHERE {' AND '.join(where)} "
            "ORDER BY created_at DESC"
        )
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))

        async with self._connect() as conn:
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()
            await cursor.close()
            return [int(row["episode_id"]) for row in rows]

    async def index_skill(
        self,
        *,
        skill_id: int,
        customer_id: str,
        created_at: datetime,
    ) -> None:
        """
        Insert/update the skills index entry for enumeration and safe fallbacks.

        This index is required because KyroDB does not provide server-side scans by namespace.
        """
        if skill_id <= 0:
            raise ValueError("skill_id must be > 0 for skill indexing")
        if not customer_id or len(customer_id) > 100:
            raise ValueError("Invalid customer_id for skill indexing")

        now = datetime.now(UTC).isoformat()

        async with self._connect() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO skills_index (
                    skill_id, customer_id, created_at,
                    deleted, deleted_at, last_updated_at
                ) VALUES (?, ?, ?, 0, NULL, ?)
                ON CONFLICT(skill_id) DO UPDATE SET
                    customer_id = excluded.customer_id,
                    created_at = excluded.created_at,
                    deleted = 0,
                    deleted_at = NULL,
                    last_updated_at = excluded.last_updated_at
                """,
                (
                    int(skill_id),
                    customer_id,
                    created_at.isoformat(),
                    now,
                ),
            )
            await conn.commit()
            await cursor.close()

    async def list_skill_ids(
        self,
        *,
        customer_id: str,
        include_deleted: bool = False,
        limit: int | None = None,
    ) -> list[int]:
        """Enumerate skill IDs for a customer (newest-first)."""
        if not customer_id or len(customer_id) > 100:
            raise ValueError("Invalid customer_id for skill listing")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be > 0 when provided")

        where = ["customer_id = ?"]
        params: list[object] = [customer_id]
        if not include_deleted:
            where.append("deleted = 0")

        query = (
            "SELECT skill_id FROM skills_index "
            f"WHERE {' AND '.join(where)} "
            "ORDER BY created_at DESC"
        )
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))

        async with self._connect() as conn:
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()
            await cursor.close()
            return [int(row["skill_id"]) for row in rows]

    async def mark_episode_archived(
        self,
        *,
        customer_id: str,
        episode_id: int,
        archived: bool,
    ) -> bool:
        """
        Mark an episode as archived/unarchived in the local index.

        Note: The authoritative archived flag for retrieval is KyroDB metadata.
        This index flag is for hygiene job enumeration.
        """
        if episode_id <= 0:
            raise ValueError("episode_id must be > 0")
        if not customer_id or len(customer_id) > 100:
            raise ValueError("Invalid customer_id")

        now = datetime.now(UTC).isoformat()
        async with self._connect() as conn:
            cursor = await conn.execute(
                """
                UPDATE episodes_index
                SET archived = ?,
                    archived_at = ?,
                    last_updated_at = ?
                WHERE customer_id = ? AND episode_id = ?
                """,
                (
                    1 if archived else 0,
                    now if archived else None,
                    now,
                    customer_id,
                    int(episode_id),
                ),
            )
            await conn.commit()
            await cursor.close()
            return cursor.rowcount > 0

    async def mark_episode_deleted(
        self,
        *,
        customer_id: str,
        episode_id: int,
    ) -> bool:
        """
        Mark an episode as deleted in the local index.

        Used when KyroDB episode deletion succeeds.
        """
        if episode_id <= 0:
            raise ValueError("episode_id must be > 0")
        if not customer_id or len(customer_id) > 100:
            raise ValueError("Invalid customer_id")

        now = datetime.now(UTC).isoformat()
        async with self._connect() as conn:
            cursor = await conn.execute(
                """
                UPDATE episodes_index
                SET deleted = 1,
                    deleted_at = ?,
                    last_updated_at = ?
                WHERE customer_id = ? AND episode_id = ?
                """,
                (now, now, customer_id, int(episode_id)),
            )
            await conn.commit()
            await cursor.close()
            return cursor.rowcount > 0

    async def reset_monthly_usage(self, customer_id: str) -> bool:
        """
        Reset monthly credit usage (called on quota cycle).

        Args:
            customer_id: Customer to reset

        Returns:
            bool: True if successful
        """
        async with self._connect() as conn:
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
            await cursor.close()

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
        async with self._connect() as conn:
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
            await cursor.close()

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

        async with self._connect() as conn:
            cursor = await conn.execute(
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
            await cursor.close()

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

        async with self._connect() as conn:
            # SHA-256 indexed lookup narrows to a single row.
            key_hash_sha256 = hashlib.sha256(plaintext_key.encode()).hexdigest()
            cursor = await conn.execute(
                """
                SELECT k.*, c.*
                FROM api_keys k
                JOIN customers c ON k.customer_id = c.customer_id
                WHERE k.is_active = 1 AND k.key_hash_sha256 = ?
                """,
                (key_hash_sha256,),
            )
            row = await cursor.fetchone()
            await cursor.close()

            if not row:
                return None

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
            cursor = await conn.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE key_id = ?",
                (now, row["key_id"]),
            )
            await conn.commit()
            await cursor.close()

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

    async def revoke_api_key(self, key_id: str) -> bool:
        """
        Revoke (deactivate) API key.

        Args:
            key_id: API key ID to revoke

        Returns:
            bool: True if revoked, False if not found
        """
        async with self._connect() as conn:
            cursor = await conn.execute(
                "UPDATE api_keys SET is_active = 0 WHERE key_id = ?", (key_id,)
            )
            await conn.commit()
            rowcount = cursor.rowcount
            await cursor.close()

            if rowcount <= 0:
                return False

            cursor2 = await conn.execute(
                "SELECT customer_id FROM api_keys WHERE key_id = ?", (key_id,)
            )
            row = await cursor2.fetchone()
            await cursor2.close()

        if row:
            await self._log_audit(
                customer_id=row["customer_id"],
                action="REVOKE",
                resource_type="api_key",
                resource_id=key_id,
            )

        return rowcount > 0

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
        async with self._connect() as conn:
            now = datetime.now(UTC).isoformat()

            cursor = await conn.execute(
                """
                INSERT INTO audit_log (
                    timestamp, customer_id, action, resource_type,
                    resource_id, details, ip_address
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (now, customer_id, action, resource_type, resource_id, details, ip_address),
            )
            await conn.commit()
            await cursor.close()

    async def close(self) -> None:
        """No-op (connections are per-operation)."""
        return None


# Global instance
_db: CustomerDatabase | None = None
_DB_CREATE_LOCK = threading.Lock()


async def get_customer_db() -> CustomerDatabase:
    """
    Get global customer database instance.

    CustomerDatabase uses connection-per-operation, so the instance is safe to share across
    event loops without leaking asyncio primitives across loops.
    """
    global _db

    if _db is None:
        with _DB_CREATE_LOCK:
            if _db is None:
                settings = get_settings()
                _db = CustomerDatabase(db_path=settings.storage.customer_db_path)

    await _db.initialize()
    return _db
