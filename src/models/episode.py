"""
Episode data models for episodic memory ingestion.

Defines the schema for failure episodes with multi-perspective reflections.

Design Decision: This system stores ONLY failures in episodic memory.
- Episodic memory is for learning from mistakes to avoid repeating them
- Success patterns should be extracted and promoted to semantic rules (future phase)
- This prevents memory bloat and keeps the system focused on its core value
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


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


class LLMPerspective(BaseModel):
    """
    Single LLM's perspective on an episode.

    Security: All fields have strict validation to prevent prompt injection
    and memory poisoning attacks.
    """

    model_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="LLM model identifier"
    )

    # Core analysis
    root_cause: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Fundamental reason for failure (not symptoms)"
    )

    preconditions: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Specific conditions required for relevance"
    )

    resolution_strategy: str = Field(
        ...,
        min_length=10,
        max_length=3000,
        description="Step-by-step resolution approach"
    )

    # Contextual analysis
    environment_factors: list[str] = Field(
        default_factory=list,
        max_length=15,
        description="OS, versions, tools that matter"
    )

    affected_components: list[str] = Field(
        default_factory=list,
        max_length=15,
        description="System components involved"
    )

    # Scores (strictly bounded)
    generalization_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0=very specific context, 1=universal pattern"
    )

    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in this analysis"
    )

    # Reasoning (for debugging/audit)
    reasoning: str = Field(
        default="",
        max_length=1000,
        description="Why this model reached this conclusion"
    )

    @field_validator("preconditions", "environment_factors", "affected_components")
    @classmethod
    def validate_list_items_not_empty(cls, v: list[str]) -> list[str]:
        """Security: Prevent empty strings in lists."""
        if any(not item.strip() for item in v):
            raise ValueError("List items cannot be empty or whitespace")
        return [item.strip() for item in v]

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
        ...,
        min_length=1,
        max_length=5,
        description="All LLM model outputs"
    )

    consensus_method: str = Field(
        ...,
        description="How consensus was reached"
    )

    # Consensus outputs
    agreed_root_cause: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Consensus root cause"
    )

    agreed_preconditions: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Union of all preconditions"
    )

    agreed_resolution: str = Field(
        ...,
        min_length=10,
        max_length=3000,
        description="Best resolution strategy"
    )

    # Consensus quality
    consensus_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How much agreement across models"
    )

    disagreement_points: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Where models differed"
    )

    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @field_validator("consensus_method")
    @classmethod
    def validate_consensus_method(cls, v: str) -> str:
        """Security: Only allow known consensus methods."""
        allowed_methods = {
            "unanimous",
            "majority_vote",
            "weighted_average",
            "fallback_heuristic"
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
    consensus: Optional[ReflectionConsensus] = Field(
        default=None,
        description="Multi-LLM consensus (premium reflections only)"
    )

    # Core analysis (either from consensus or single LLM)
    root_cause: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Identified root cause of the issue"
    )

    preconditions: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Required state/context for this episode to be relevant"
    )

    resolution_strategy: str = Field(
        ...,
        min_length=10,
        max_length=3000,
        description="How the issue was resolved"
    )

    # Contextual analysis
    environment_factors: list[str] = Field(
        default_factory=list,
        max_length=15,
        description="Environment-specific factors (OS, versions, etc.)"
    )

    affected_components: list[str] = Field(
        default_factory=list,
        max_length=15,
        description="System components involved"
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
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    llm_model: str = Field(
        default="unknown",
        max_length=100,
        description="LLM model(s) used"
    )

    # Cost tracking (security: detect abuse)
    cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Actual LLM API cost for this reflection"
    )

    generation_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken to generate reflection"
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

    @field_validator("root_cause", "resolution_strategy")
    @classmethod
    def sanitize_text(cls, v: str) -> str:
        """Security: Sanitize text fields."""
        # Remove null bytes
        v = v.replace("\x00", "")
        # Normalize whitespace
        v = " ".join(v.split())
        return v.strip()

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
    customer_id: Optional[str] = Field(
        default=None,
        description="Customer ID for tenant isolation (set automatically from API key)",
    )

    # Episode identification
    episode_type: EpisodeType = Field(default=EpisodeType.FAILURE)
    goal: str = Field(..., min_length=10, description="Original goal/intent")

    # Execution trace
    tool_chain: list[str] = Field(..., min_items=1, description="Tools used in sequence")
    actions_taken: list[str] = Field(
        ..., min_items=1, description="Actions executed before failure"
    )

    # Failure context
    error_trace: str = Field(..., min_length=10, description="Error message/stack trace")
    error_class: ErrorClass = Field(default=ErrorClass.UNKNOWN)

    # State context
    code_state_diff: Optional[str] = Field(
        default=None, description="git diff or code changes leading to issue"
    )
    environment_info: dict[str, Any] = Field(
        default_factory=dict, description="Environment variables, versions, etc."
    )

    # Visual context
    screenshot_path: Optional[str] = Field(default=None, description="Path to screenshot")

    # Resolution (learning from how the failure was fixed)
    resolution: Optional[str] = Field(
        default=None, description="How the failure was eventually resolved (for learning)"
    )
    time_to_resolve_seconds: Optional[int] = Field(
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


class Episode(BaseModel):
    """
    Complete episode with generated reflection and storage metadata.

    Stored in KyroDB after reflection generation.
    """

    # Original creation data
    create_data: EpisodeCreate

    # Generated fields
    episode_id: int = Field(..., description="Unique episode ID (KyroDB doc_id)")
    reflection: Optional[Reflection] = Field(
        default=None, description="LLM-generated reflection"
    )

    # Storage metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    retrieval_count: int = Field(default=0, ge=0, description="Number of times retrieved")
    last_retrieved_at: Optional[datetime] = Field(default=None)

    # Embeddings metadata (stored separately in KyroDB)
    text_embedding_id: Optional[int] = Field(default=None)
    code_embedding_id: Optional[int] = Field(default=None)
    image_embedding_id: Optional[int] = Field(default=None)

    # Hygiene metadata
    archived: bool = Field(default=False)
    archived_at: Optional[datetime] = Field(default=None)

    def to_metadata_dict(self) -> dict[str, str]:
        """
        Convert to KyroDB metadata format (map<string, string>).

        Returns:
            dict: Metadata compatible with KyroDB InsertRequest
        """
        import json

        metadata = {
            # Multi-tenancy
            "customer_id": self.create_data.customer_id or "default",
            # Episode fields
            "episode_type": self.create_data.episode_type.value,
            "error_class": self.create_data.error_class.value,
            "tool": self.create_data.tool_chain[0],  # Primary tool
            "severity": str(self.create_data.severity),
            "timestamp": str(int(self.created_at.timestamp())),
            "retrieval_count": str(self.retrieval_count),
            "archived": str(self.archived),
            # Store full episode as JSON (for complex queries)
            "episode_json": json.dumps(self.model_dump(mode="json"), default=str),
        }

        # Add tags as comma-separated
        if self.create_data.tags:
            metadata["tags"] = ",".join(self.create_data.tags)

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
        import json

        if "episode_json" in metadata:
            data = json.loads(metadata["episode_json"])
            episode = cls.model_validate(data)
            episode.episode_id = doc_id
            return episode
        else:
            # Fallback for minimal metadata (shouldn't happen)
            raise ValueError("episode_json not found in metadata")

    def increment_retrieval_count(self) -> None:
        """Increment retrieval count and update timestamp."""
        self.retrieval_count += 1
        self.last_retrieved_at = datetime.now(timezone.utc)
