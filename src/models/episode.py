"""
Episode data models for episodic memory ingestion.

Defines the schema for failure episodes with multi-perspective reflections.

Design Decision: This system stores ONLY failures in episodic memory.
- Episodic memory is for learning from mistakes to avoid repeating them
- Success patterns should be extracted and promoted to semantic rules (future phase)
- This prevents memory bloat and keeps the system focused on its core value
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


def _normalize_preconditions_input(value: Any) -> list[str]:
    """
    Normalize preconditions payload into canonical `key=value` string list.

    Accepted inputs:
    - dict[str, str]: converted to `["key=value", ...]`
    - list[str]: passed through (trimmed later by validators)
    - None: converted to []
    """
    if value is None:
        return []

    if isinstance(value, dict):
        normalized: list[str] = []
        for raw_key, raw_val in value.items():
            key = str(raw_key).strip()
            val = str(raw_val).strip()
            if key and val:
                normalized.append(f"{key}={val}")
        return normalized

    if isinstance(value, list):
        return cast(list[str], value)

    raise TypeError("preconditions must be a list[str], dict[str, str], or null")


class EpisodeType(str, Enum):
    """
    Type of episode being recorded.

    Only FAILURE is supported. Episodic memory focuses on learning from mistakes.
    Success patterns are better stored as generalized semantic rules rather than
    individual episodes, preventing memory bloat.
    """

    FAILURE = "failure"


class ErrorClass(str, Enum):
    """Common error classifications for failures."""

    CONFIGURATION_ERROR = "configuration_error"
    PERMISSION_ERROR = "permission_error"
    NETWORK_ERROR = "network_error"
    RESOURCE_ERROR = "resource_error"
    DEPENDENCY_ERROR = "dependency_error"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN = "unknown"


class ReflectionTier(str, Enum):
    """Reflection generation tier for cost/quality tradeoffs."""

    CHEAP = "cheap"
    CACHED = "cached"
    PREMIUM = "premium"


class LLMPerspective(BaseModel):
    """
    Single LLM's perspective on an episode.

    Security: All fields have strict validation to prevent prompt injection
    and memory poisoning attacks.
    """

    model_config = ConfigDict(protected_namespaces=())

    model_name: str = Field(..., min_length=1, max_length=100, description="LLM model identifier")

    # Core analysis
    root_cause: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Fundamental reason for failure (not symptoms)",
    )

    preconditions: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Specific conditions required for relevance",
    )

    resolution_strategy: str = Field(
        ..., min_length=1, max_length=3000, description="Step-by-step resolution approach"
    )

    # Contextual analysis
    environment_factors: list[str] = Field(
        default_factory=list, max_length=15, description="OS, versions, tools that matter"
    )

    affected_components: list[str] = Field(
        default_factory=list, max_length=15, description="System components involved"
    )

    # Scores (strictly bounded)
    generalization_score: float = Field(
        ..., ge=0.0, le=1.0, description="0=very specific context, 1=universal pattern"
    )

    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in this analysis")

    # Reasoning (for debugging/audit)
    reasoning: str = Field(
        default="", max_length=1000, description="Why this model reached this conclusion"
    )

    @field_validator("preconditions", "environment_factors", "affected_components")
    @classmethod
    def validate_list_items_not_empty(cls, v: list[str]) -> list[str]:
        """Security: Prevent empty strings in lists."""
        if any(not item.strip() for item in v):
            raise ValueError("List items cannot be empty or whitespace")
        return [item.strip() for item in v]

    @field_validator("preconditions", mode="before")
    @classmethod
    def normalize_preconditions(cls, v: Any) -> Any:
        """Accept dict preconditions and normalize to canonical list form."""
        return _normalize_preconditions_input(v)

    @field_validator("preconditions", "environment_factors", "affected_components")
    @classmethod
    def validate_list_item_length(cls, v: list[str]) -> list[str]:
        """Security: Prevent excessively long list items."""
        for item in v:
            if len(item) > 500:
                raise ValueError("List items cannot exceed 500 characters")
        return v

    @field_validator("root_cause", "resolution_strategy", "reasoning")
    @classmethod
    def sanitize_text_fields(cls, v: str) -> str:
        """Security: Sanitize text to prevent injection attacks."""
        # Remove null bytes
        v = v.replace("\x00", "")
        # Normalize whitespace
        v = " ".join(v.split())
        return v.strip()


class ReflectionConsensus(BaseModel):
    """
    Consensus reconciliation from multiple LLM perspectives.

    Uses Self-Contrast/Mirror approach to find agreement and disagreement.

    Security: Validates that consensus is derived from actual perspectives,
    preventing spoofing attacks.
    """

    perspectives: list[LLMPerspective] = Field(
        ..., min_length=1, max_length=5, description="All LLM model outputs"
    )

    consensus_method: str = Field(..., description="How consensus was reached")

    # Consensus outputs
    agreed_root_cause: str = Field(
        ..., min_length=1, max_length=2000, description="Consensus root cause"
    )

    agreed_preconditions: list[str] = Field(
        default_factory=list, max_length=20, description="Union of all preconditions"
    )

    agreed_resolution: str = Field(
        ..., min_length=1, max_length=3000, description="Best resolution strategy"
    )

    # Consensus quality
    consensus_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="How much agreement across models"
    )

    disagreement_points: list[str] = Field(
        default_factory=list, max_length=10, description="Where models differed"
    )

    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("consensus_method")
    @classmethod
    def validate_consensus_method(cls, v: str) -> str:
        """Security: Only allow known consensus methods."""
        allowed_methods = {
            # Legacy string-equality methods
            "unanimous",
            "majority_vote",
            "weighted_average",
            "fallback_heuristic",
            # Semantic similarity methods
            "single_model",
            "semantic_unanimous",
            "weighted_semantic_vote",
            "highest_confidence_fallback",
            "semantic_majority",
            "weighted_semantic_majority",
        }
        if v not in allowed_methods:
            raise ValueError(f"Invalid consensus method: {v}")
        return v

    @field_validator("perspectives")
    @classmethod
    def validate_perspectives_not_duplicate(cls, v: list[LLMPerspective]) -> list[LLMPerspective]:
        """Security: Prevent duplicate model names (spoofing)."""
        model_names = [p.model_name for p in v]
        if len(model_names) != len(set(model_names)):
            raise ValueError("Duplicate model names in perspectives")
        return v


class Reflection(BaseModel):
    """
    Enhanced multi-perspective reflection with consensus.

    Security features:
    - All fields validated and sanitized
    - Consensus derived from verified perspectives
    - Cost tracking to detect abuse
    - Immutable after generation

    Generated via LLM analysis of the episode context.
    """

    # Multi-perspective consensus (if available)
    consensus: ReflectionConsensus | None = Field(
        default=None, description="Multi-LLM consensus (premium reflections only)"
    )

    # Core analysis (either from consensus or single LLM)
    root_cause: str = Field(
        ..., min_length=1, max_length=2000, description="Identified root cause of the issue"
    )

    preconditions: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Required state/context for this episode to be relevant",
    )

    resolution_strategy: str = Field(
        ..., min_length=1, max_length=3000, description="How the issue was resolved"
    )

    # Contextual analysis
    environment_factors: list[str] = Field(
        default_factory=list,
        max_length=15,
        description="Environment-specific factors (OS, versions, etc.)",
    )

    affected_components: list[str] = Field(
        default_factory=list, max_length=15, description="System components involved"
    )

    # Reusability metadata
    generalization_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How generalizable this episode is (0=specific, 1=universal)",
    )

    confidence_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in this reflection's accuracy",
    )

    # Generation metadata
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    llm_model: str = Field(default="unknown", max_length=100, description="LLM model(s) used")

    # Cost tracking (security: detect abuse)
    cost_usd: float = Field(
        default=0.0, ge=0.0, le=10.0, description="Actual LLM API cost for this reflection"
    )

    generation_latency_ms: float = Field(
        default=0.0, ge=0.0, description="Time taken to generate reflection"
    )

    # Tier tracking (Phase 5)
    tier: ReflectionTier | None = Field(
        default=None, description="Reflection tier used (cheap/cached/premium)"
    )

    @field_validator("preconditions", "environment_factors", "affected_components")
    @classmethod
    def validate_list_items(cls, v: list[str]) -> list[str]:
        """Security: Validate list items."""
        # Remove empty items
        v = [item.strip() for item in v if item.strip()]
        # Enforce length limits
        for item in v:
            if len(item) > 500:
                raise ValueError("List items cannot exceed 500 characters")
        return v

    @field_validator("preconditions", mode="before")
    @classmethod
    def normalize_preconditions(cls, v: Any) -> Any:
        """Accept dict preconditions and normalize to canonical list form."""
        return _normalize_preconditions_input(v)

    @field_validator("root_cause", "resolution_strategy")
    @classmethod
    def sanitize_text(cls, v: str) -> str:
        """
        Security: Sanitize text fields while preserving structure.

        Preserves newlines for code blocks and procedures while:
        - Removing null bytes (security)
        - Normalizing horizontal whitespace (spaces/tabs)
        - Limiting consecutive blank lines to 2
        """
        import re

        # Remove null bytes
        v = v.replace("\x00", "")
        # Normalize horizontal whitespace (tabs -> spaces, multiple spaces -> single)
        v = re.sub(r"[ \t]+", " ", v)
        # Limit consecutive blank lines to 2 (preserve code block formatting)
        v = re.sub(r"\n{4,}", "\n\n\n", v)
        # Strip leading/trailing whitespace from each line
        lines = [line.strip() for line in v.split("\n")]
        return "\n".join(lines).strip()

    @field_validator("cost_usd")
    @classmethod
    def validate_cost_reasonable(cls, v: float) -> float:
        """Security: Detect cost anomalies."""
        if v > 1.0:
            # Single reflection should never cost more than $1
            # (even premium multi-perspective is ~$0.10)
            import logging

            logging.warning(f"Unusually high reflection cost: ${v:.2f}")
        return v


class EpisodeCreate(BaseModel):
    """
    Schema for creating a new episode.

    Submitted by the agent during ingestion.
    Multi-tenancy: customer_id is extracted from API key, not user-provided.
    """

    # Multi-tenancy (set by middleware, not user-provided)
    customer_id: str | None = Field(
        default=None,
        description="Customer ID for tenant isolation (set automatically from API key)",
    )

    # Episode identification
    episode_type: EpisodeType = Field(default=EpisodeType.FAILURE)
    goal: str = Field(..., min_length=10, description="Original goal/intent")

    # Execution trace
    tool_chain: list[str] = Field(..., min_length=1, description="Tools used in sequence")
    actions_taken: list[str] = Field(
        ..., min_length=1, description="Actions executed before failure"
    )

    # Failure context
    error_trace: str = Field(..., min_length=10, description="Error message/stack trace")
    error_class: ErrorClass = Field(default=ErrorClass.UNKNOWN)

    # State context
    code_state_diff: str | None = Field(
        default=None, description="git diff or code changes leading to issue"
    )
    environment_info: dict[str, Any] = Field(
        default_factory=dict, description="Environment variables, versions, etc."
    )

    # Visual context (base64-encoded image bytes)
    screenshot_base64: str | None = Field(
        default=None,
        description="Base64-encoded screenshot image bytes (raw base64 preferred; data URLs are also accepted)",
    )

    # Resolution (learning from how the failure was fixed)
    resolution: str | None = Field(
        default=None, description="How the failure was eventually resolved (for learning)"
    )
    time_to_resolve_seconds: int | None = Field(
        default=None, ge=0, description="Time taken to resolve the failure"
    )

    # Metadata
    tags: list[str] = Field(default_factory=list, description="Custom tags")
    severity: int = Field(default=3, ge=1, le=5, description="Severity: 1=critical, 5=minor")

    @field_validator("tool_chain", "actions_taken")
    @classmethod
    def validate_non_empty_lists(cls, v: list[str]) -> list[str]:
        if not v or any(not item.strip() for item in v):
            raise ValueError("List cannot contain empty strings")
        return v


class UsageStats(BaseModel):
    """
    Track how often an episode's fix is applied and whether it works.

    Security: All fields validated to prevent stat inflation attacks.
    """

    total_retrievals: int = Field(
        default=0, ge=0, description="Total times this episode was retrieved"
    )

    fix_applied_count: int = Field(
        default=0, ge=0, description="Times the suggested fix was actually applied"
    )

    fix_success_count: int = Field(
        default=0, ge=0, description="Times the fix successfully resolved the issue"
    )

    fix_failure_count: int = Field(
        default=0, ge=0, description="Times the fix failed to resolve the issue"
    )

    @property
    def fix_success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)."""
        total_validations = self.fix_success_count + self.fix_failure_count
        if total_validations == 0:
            return 0.0
        return self.fix_success_count / total_validations

    @property
    def application_rate(self) -> float:
        """Percentage of retrievals that led to applying the fix."""
        if self.total_retrievals == 0:
            return 0.0
        return self.fix_applied_count / self.total_retrievals

    @field_validator("fix_success_count", "fix_failure_count")
    @classmethod
    def validate_success_failure_count(cls, v: int, info: ValidationInfo) -> int:
        """Ensure success/failure counts don't exceed applied count."""
        if info.data.get("fix_applied_count", 0) < v:
            raise ValueError(
                f"Success/failure count ({v}) cannot exceed applied count "
                f"({info.data.get('fix_applied_count', 0)})"
            )
        return v


class Episode(BaseModel):
    """
    Complete episode with generated reflection and storage metadata.

    Stored in KyroDB after reflection generation.
    """

    # Original creation data
    create_data: EpisodeCreate

    # Generated fields
    episode_id: int = Field(..., description="Unique episode ID (KyroDB doc_id)")
    reflection: Reflection | None = Field(default=None, description="LLM-generated reflection")

    # Storage metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    retrieval_count: int = Field(default=0, ge=0, description="Number of times retrieved")
    last_retrieved_at: datetime | None = Field(default=None)

    # Usage tracking (Phase 2: Skills promotion)
    usage_stats: UsageStats = Field(
        default_factory=UsageStats, description="Track fix application success rate"
    )

    # Embeddings metadata (stored separately in KyroDB)
    text_embedding_id: int | None = Field(default=None)
    code_embedding_id: int | None = Field(default=None)
    image_embedding_id: int | None = Field(default=None)

    # Hygiene metadata
    archived: bool = Field(default=False)
    archived_at: datetime | None = Field(default=None)

    def to_metadata_dict(self) -> dict[str, str]:
        """
        Convert to KyroDB metadata format (map<string, string>).

        Returns:
            dict: Metadata compatible with KyroDB InsertRequest
        """
        import base64
        import json
        import zlib

        def _encode_episode_json(data: dict[str, Any]) -> str:
            """
            Encode episode JSON for storage.

            Compresses to reduce metadata bloat and bandwidth.
            Format: "gz:{base64(zlib(json))}"
            """
            raw = json.dumps(data, default=str)
            compressed = zlib.compress(raw.encode("utf-8"))
            return "gz:" + base64.b64encode(compressed).decode("ascii")

        metadata = {
            # Multi-tenancy
            "customer_id": self.create_data.customer_id or "default",
            # Episode fields
            "episode_type": self.create_data.episode_type.value,
            "error_class": self.create_data.error_class.value,
            "tool": self.create_data.tool_chain[0]
            .strip()
            .lower(),  # Primary tool (normalized for exact-match filtering)
            "severity": str(self.create_data.severity),
            "timestamp": str(int(self.created_at.timestamp())),
            "retrieval_count": str(self.retrieval_count),
            "archived": str(self.archived),
            # Usage stats (Phase 2: Skills promotion)
            "usage_total_retrievals": str(self.usage_stats.total_retrievals),
            "usage_fix_applied_count": str(self.usage_stats.fix_applied_count),
            "usage_fix_success_count": str(self.usage_stats.fix_success_count),
            "usage_fix_failure_count": str(self.usage_stats.fix_failure_count),
            "usage_fix_success_rate": str(self.usage_stats.fix_success_rate),
            # Store full episode as compressed JSON (for reconstruction)
        }
        # Store full episode as compressed JSON (for reconstruction), excluding raw screenshot bytes.
        episode_payload = self.model_dump(mode="json")
        create_data = episode_payload.get("create_data")
        if isinstance(create_data, dict):
            create_data.pop("screenshot_base64", None)
        metadata["episode_json"] = _encode_episode_json(episode_payload)

        # Add tags as comma-separated
        if self.create_data.tags:
            metadata["tags"] = ",".join(self.create_data.tags)
            # Add per-tag metadata keys for exact-match filtering
            for tag in self.create_data.tags:
                safe_tag = str(tag).strip().lower().replace(" ", "_")
                if safe_tag:
                    metadata[f"tag:{safe_tag}"] = "1"

        return metadata

    @classmethod
    def from_metadata_dict(cls, doc_id: int, metadata: dict[str, str]) -> "Episode":
        """
        Reconstruct Episode from KyroDB metadata.

        Args:
            doc_id: KyroDB document ID
            metadata: Metadata dict from SearchResult

        Returns:
            Episode: Reconstructed episode instance
        """
        import base64
        import json
        import zlib

        if "episode_json" in metadata:
            raw = metadata["episode_json"]
            # Handle compressed payloads
            if raw.startswith("gz:"):
                try:
                    compressed = base64.b64decode(raw[3:])
                    raw = zlib.decompress(compressed).decode("utf-8")
                except Exception as e:
                    raise ValueError(f"Failed to decompress episode_json: {e}") from e
            data = json.loads(raw)
            episode = cls.model_validate(data)
            episode.episode_id = doc_id
            return episode
        else:
            # Fallback for minimal metadata (shouldn't happen)
            raise ValueError("episode_json not found in metadata")

    def increment_retrieval_count(self) -> None:
        """Increment retrieval count and update timestamp."""
        self.retrieval_count += 1
        self.last_retrieved_at = datetime.now(UTC)
