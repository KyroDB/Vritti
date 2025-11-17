"""
Episode data models for episodic memory ingestion.

Defines the schema for failure/success episodes with multi-perspective reflections.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class EpisodeType(str, Enum):
    """Type of episode being recorded."""

    FAILURE = "failure"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"


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


class Reflection(BaseModel):
    """
    Multi-perspective reflection on an episode.

    Generated via LLM analysis of the episode context.
    """

    # Core analysis
    root_cause: str = Field(..., description="Identified root cause of the issue")
    preconditions: list[str] = Field(
        default_factory=list,
        description="Required state/context for this episode to be relevant",
    )
    resolution_strategy: str = Field(..., description="How the issue was resolved")

    # Contextual analysis
    environment_factors: list[str] = Field(
        default_factory=list, description="Environment-specific factors (OS, versions, etc.)"
    )
    affected_components: list[str] = Field(
        default_factory=list, description="System components involved"
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
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    llm_model: Optional[str] = Field(default=None, description="LLM model used")


class EpisodeCreate(BaseModel):
    """
    Schema for creating a new episode.

    Submitted by the agent during ingestion.
    """

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

    # Resolution (for success/partial_success)
    resolution: Optional[str] = Field(
        default=None, description="How the issue was resolved"
    )
    time_to_resolve_seconds: Optional[int] = Field(default=None, ge=0)

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
    created_at: datetime = Field(default_factory=datetime.utcnow)
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
        self.last_retrieved_at = datetime.utcnow()
