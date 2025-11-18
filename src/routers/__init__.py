"""
API routers for EpisodicMemory service.

Routers:
- customers: Customer and API key management
"""

from src.routers.customers import router as customers_router

__all__ = ["customers_router"]
