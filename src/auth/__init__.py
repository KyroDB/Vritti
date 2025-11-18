"""
Authentication and authorization for EpisodicMemory API.

Security:
- API key validation with bcrypt
- customer_id injection from validated key (prevents spoofing)
- Rate limiting per customer
"""

from src.auth.dependencies import (
    get_authenticated_customer,
    get_customer_id_from_request,
    require_active_customer,
)

__all__ = [
    "get_authenticated_customer",
    "require_active_customer",
    "get_customer_id_from_request",
]
