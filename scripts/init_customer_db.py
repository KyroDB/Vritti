#!/usr/bin/env python3
"""
Database initialization script for customer management.

Creates SQLite database with customer and API key tables.

Usage:
    python scripts/init_customer_db.py [--db-path PATH]

Options:
    --db-path PATH    Path to SQLite database file (default: ./data/customers.db)

This script is idempotent - safe to run multiple times.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.database import CustomerDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def init_database(db_path: str) -> bool:
    """
    Initialize customer database schema.

    Args:
        db_path: Path to SQLite database file

    Returns:
        bool: True if initialization succeeded
    """
    try:
        # Ensure data directory exists
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing customer database at {db_path}")

        # Create database instance
        db = CustomerDatabase(db_path=db_path)

        # Initialize schema (creates tables)
        await db.initialize()

        # Verify connection (use regular context manager, not async)
        with db._get_connection() as conn:
            # Check tables exist
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]

            expected_tables = {"customers", "api_keys", "audit_log"}
            missing = expected_tables - set(tables)

            if missing:
                logger.error(f"Missing tables: {missing}")
                return False

            logger.info(f"✓ Found tables: {', '.join(sorted(tables))}")

            # Check row counts
            for table in ["customers", "api_keys", "audit_log"]:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                logger.info(f"  {table}: {count} rows")

        logger.info("✓ Database initialization complete")
        return True

    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        return False


async def create_demo_customer(db_path: str) -> bool:
    """
    Create a demo customer for testing (optional).

    Args:
        db_path: Path to SQLite database file

    Returns:
        bool: True if creation succeeded
    """
    try:
        from src.models.customer import CustomerCreate, APIKeyCreate

        db = CustomerDatabase(db_path=db_path)

        logger.info("Creating demo customer 'demo-customer'...")

        # Create demo customer
        customer_data = CustomerCreate(
            customer_id="demo-customer",
            organization_name="Demo Organization",
            email="demo@example.com",
            subscription_tier="free",
        )

        customer = await db.create_customer(customer_data)

        if customer:
            logger.info(
                f"✓ Demo customer created: {customer.customer_id} "
                f"(email: {customer.email})"
            )

            # Create API key for demo customer
            key_data = APIKeyCreate(
                customer_id="demo-customer",
                name="Demo API Key",
            )

            plaintext_key, api_key = await db.create_api_key(key_data)

            logger.info(f"✓ Demo API key created:")
            logger.info(f"  Key ID: {api_key.key_id}")
            logger.info(f"  Key Prefix: {api_key.key_prefix}")
            logger.info(f"  Full Key: {plaintext_key}")
            logger.warning(
                f"  ⚠️  SAVE THIS KEY - it will not be shown again!"
            )

            return True
        else:
            logger.warning("Demo customer already exists - skipping")
            return True

    except Exception as e:
        logger.error(f"Demo customer creation failed: {e}", exc_info=True)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Initialize customer database schema"
    )
    parser.add_argument(
        "--db-path",
        default="./data/customers.db",
        help="Path to SQLite database file (default: ./data/customers.db)",
    )
    parser.add_argument(
        "--create-demo",
        action="store_true",
        help="Create demo customer with API key for testing",
    )

    args = parser.parse_args()

    # Initialize database
    success = asyncio.run(init_database(args.db_path))

    if not success:
        logger.error("❌ Database initialization failed")
        sys.exit(1)

    # Optionally create demo customer
    if args.create_demo:
        logger.info("")
        demo_success = asyncio.run(create_demo_customer(args.db_path))

        if not demo_success:
            logger.error("❌ Demo customer creation failed")
            sys.exit(1)

    logger.info("")
    logger.info("=== Database Ready ===")
    logger.info(f"Database path: {Path(args.db_path).absolute()}")
    logger.info("")


if __name__ == "__main__":
    main()
