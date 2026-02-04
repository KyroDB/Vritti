"""
Authentication and authorization for EpisodicMemory API.

Security:
- API key validation via key_id lookup + adaptive hash verify
- customer_id injection from validated key (prevents spoofing)
- Rate limiting per customer
- Admin API key for admin endpoints
"""

from src.auth.dependencies import (
    get_authenticated_customer,
    get_customer_id_from_request,
    require_active_customer,
    require_admin_access,
)

__all__ = [
    "get_authenticated_customer",
    "require_active_customer",
    "get_customer_id_from_request",
    "require_admin_access",
]
