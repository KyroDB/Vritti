"""
Storage layer for customer and API key management.

Uses SQLite for bootstrapping (free, embedded).
Migration path to PostgreSQL for production scale.
"""

from src.storage.database import CustomerDatabase, get_customer_db

__all__ = ["CustomerDatabase", "get_customer_db"]
