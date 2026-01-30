"""src.observability

Minimal observability surface.

We keep:
- Structured logging (src.observability.logging, logging_middleware)
- Health endpoints (src.observability.health)
- Lightweight error classification middleware (src.observability.middleware)

We intentionally do not ship a bundled metrics/monitoring stack in the core product.
"""

__all__: list[str] = []
