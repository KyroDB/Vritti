"""
Hashing utilities for deduplication and matching.

Important:
- KyroDB `doc_id` values must be dense/small integers in the current KyroDB backend.
  Vritti allocates doc_ids via `CustomerDatabase.allocate_doc_id(...)` and does not
  generate them here.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def hash_environment(environment_info: dict[str, Any]) -> str:
    """
    Generate deterministic hash of environment configuration.

    Used for deduplication: episodes with identical environments and errors
    can be clustered together.
    """
    normalized = json.dumps(environment_info, sort_keys=True, separators=(",", ":"))
    hasher = hashlib.sha256()
    hasher.update(normalized.encode("utf-8"))
    return hasher.hexdigest()


def hash_error_signature(error_class: str, tool: str, environment_hash: str) -> str:
    """Generate signature for error deduplication."""
    signature = f"{error_class}|{tool}|{environment_hash}"
    hasher = hashlib.sha256()
    hasher.update(signature.encode("utf-8"))
    return hasher.hexdigest()


def hash_text_content(text: str) -> str:
    """Generate SHA-256 hash of text content for deduplication."""
    hasher = hashlib.sha256()
    hasher.update(text.strip().encode("utf-8"))
    return hasher.hexdigest()
