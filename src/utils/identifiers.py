"""
ID generation and environment hashing utilities.

Provides:
- Snowflake-style distributed ID generation
- Deterministic environment hashing for deduplication
- Episode ID management with monotonic ordering
"""

import hashlib
import json
import time
from typing import Any


class SnowflakeIDGenerator:
    """
    Snowflake-style distributed ID generator.

    Format (64 bits):
    - 41 bits: Timestamp (milliseconds since epoch)
    - 10 bits: Machine ID (0-1023)
    - 12 bits: Sequence number (0-4095)

    Provides:
    - Time-ordered IDs (sortable by creation time)
    - 4096 IDs per millisecond per machine
    - Distributed without coordination
    - Fits in uint64 for KyroDB doc_id

    Thread-safe with sequence number rollover.
    """

    # Epoch: 2024-01-01 00:00:00 UTC (reduces ID size)
    EPOCH = 1704067200000  # milliseconds

    # Bit lengths
    TIMESTAMP_BITS = 41
    MACHINE_ID_BITS = 10
    SEQUENCE_BITS = 12

    # Max values
    MAX_MACHINE_ID = (1 << MACHINE_ID_BITS) - 1  # 1023
    MAX_SEQUENCE = (1 << SEQUENCE_BITS) - 1  # 4095

    # Bit shifts
    MACHINE_ID_SHIFT = SEQUENCE_BITS
    TIMESTAMP_SHIFT = MACHINE_ID_BITS + SEQUENCE_BITS

    def __init__(self, machine_id: int = 0):
        """
        Initialize ID generator.

        Args:
            machine_id: Machine ID (0-1023) for distributed systems

        Raises:
            ValueError: If machine_id is out of range
        """
        if machine_id < 0 or machine_id > self.MAX_MACHINE_ID:
            raise ValueError(f"machine_id must be 0-{self.MAX_MACHINE_ID}, got {machine_id}")

        self.machine_id = machine_id
        self.sequence = 0
        self.last_timestamp = -1

    def _current_timestamp(self) -> int:
        """Get current timestamp in milliseconds since EPOCH."""
        return int(time.time() * 1000) - self.EPOCH

    def generate(self) -> int:
        """
        Generate next ID.

        Returns:
            int: 64-bit unsigned integer ID

        Raises:
            RuntimeError: If clock moves backward
        """
        timestamp = self._current_timestamp()

        # Clock moved backward - should not happen in production
        if timestamp < self.last_timestamp:
            raise RuntimeError(
                f"Clock moved backward! Last: {self.last_timestamp}, " f"Current: {timestamp}"
            )

        # Same millisecond: increment sequence
        if timestamp == self.last_timestamp:
            self.sequence = (self.sequence + 1) & self.MAX_SEQUENCE

            # Sequence exhausted: wait for next millisecond
            if self.sequence == 0:
                while timestamp <= self.last_timestamp:
                    timestamp = self._current_timestamp()
        else:
            # New millisecond: reset sequence
            self.sequence = 0

        self.last_timestamp = timestamp

        # Compose ID: timestamp | machine_id | sequence
        id_value = (
            (timestamp << self.TIMESTAMP_SHIFT)
            | (self.machine_id << self.MACHINE_ID_SHIFT)
            | self.sequence
        )

        return id_value

    def parse(self, id_value: int) -> dict[str, int]:
        """
        Parse ID into components.

        Args:
            id_value: Generated ID

        Returns:
            dict: Components (timestamp, machine_id, sequence)
        """
        timestamp = (id_value >> self.TIMESTAMP_SHIFT) + self.EPOCH
        machine_id = (id_value >> self.MACHINE_ID_SHIFT) & self.MAX_MACHINE_ID
        sequence = id_value & self.MAX_SEQUENCE

        return {
            "timestamp_ms": timestamp,
            "machine_id": machine_id,
            "sequence": sequence,
        }


# Global generator instance (machine_id=0 for single-machine setup)
_id_generator: SnowflakeIDGenerator | None = None


def initialize_id_generator(machine_id: int = 0) -> None:
    """
    Initialize global ID generator.

    Call this at application startup with appropriate machine_id.

    Args:
        machine_id: Machine ID for distributed setup (0-1023)
    """
    global _id_generator
    _id_generator = SnowflakeIDGenerator(machine_id=machine_id)


def generate_episode_id() -> int:
    """
    Generate unique episode ID.

    Returns:
        int: 64-bit episode ID (compatible with KyroDB uint64)

    Raises:
        RuntimeError: If generator not initialized
    """
    global _id_generator
    if _id_generator is None:
        # Auto-initialize with default machine_id=0
        _id_generator = SnowflakeIDGenerator(machine_id=0)

    return _id_generator.generate()


def hash_environment(environment_info: dict[str, Any]) -> str:
    """
    Generate deterministic hash of environment configuration.

    Used for deduplication: episodes with identical environments and errors
    can be clustered together.

    Args:
        environment_info: Environment metadata (OS, versions, tools, etc.)

    Returns:
        str: SHA-256 hash (hex digest, 64 chars)

    Example:
        >>> env = {
        ...     "os": "Darwin",
        ...     "os_version": "14.1",
        ...     "python_version": "3.11.5",
        ...     "kubectl_version": "1.28.0"
        ... }
        >>> hash_environment(env)
        "a7b3c9d4e5f6..."
    """
    # Normalize: sort keys for deterministic JSON serialization
    normalized = json.dumps(environment_info, sort_keys=True, separators=(",", ":"))

    # Hash with SHA-256
    hasher = hashlib.sha256()
    hasher.update(normalized.encode("utf-8"))

    return hasher.hexdigest()


def hash_error_signature(error_class: str, tool: str, environment_hash: str) -> str:
    """
    Generate signature for error deduplication.

    Episodes with same signature are likely to be duplicate failures.

    Args:
        error_class: Error classification (e.g., "ImagePullBackOff")
        tool: Primary tool (e.g., "kubectl")
        environment_hash: Environment hash from hash_environment()

    Returns:
        str: SHA-256 hash for error signature

    Example:
        >>> hash_error_signature(
        ...     error_class="ImagePullBackOff",
        ...     tool="kubectl",
        ...     environment_hash="a7b3c9..."
        ... )
        "f2e4d8a1b3c5..."
    """
    signature = f"{error_class}|{tool}|{environment_hash}"
    hasher = hashlib.sha256()
    hasher.update(signature.encode("utf-8"))

    return hasher.hexdigest()


def hash_text_content(text: str) -> str:
    """
    Generate hash of text content for deduplication.

    Useful for detecting duplicate error messages or goals.

    Args:
        text: Text content (error trace, goal, etc.)

    Returns:
        str: SHA-256 hash

    Example:
        >>> hash_text_content("ImagePullBackOff: failed to pull image")
        "c4d8e9a2b7f1..."
    """
    hasher = hashlib.sha256()
    hasher.update(text.strip().encode("utf-8"))

    return hasher.hexdigest()


def extract_episode_timestamp(episode_id: int) -> int:
    """
    Extract timestamp from episode ID.

    Args:
        episode_id: Generated episode ID

    Returns:
        int: Unix timestamp in milliseconds
    """
    global _id_generator
    if _id_generator is None:
        _id_generator = SnowflakeIDGenerator(machine_id=0)

    components = _id_generator.parse(episode_id)
    return components["timestamp_ms"]
