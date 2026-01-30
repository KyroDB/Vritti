"""
Search request/response models for episodic memory retrieval.

Defines the API schema for searching failure episodes.

Design Decision: Only "failures" collection is supported. This system does not
store success episodes to prevent memory bloat and maintain focus on learning
from mistakes.
"""

from datetime import timezone, datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from src.models.episode import Episode


class RankingWeights(BaseModel):
    """Configurable weights for result ranking."""

    similarity_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    precondition_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    recency_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    usage_weight: float = Field(default=0.1, ge=0.0, le=1.0)

    @field_validator("*")
    @classmethod
    def validate_weights_sum(cls, v: float, info) -> float:
        """Weights should sum to ~1.0 for interpretability."""
        # This is just a warning - we'll normalize in the ranker
        return v


class SearchRequest(BaseModel):
    """
    Search request for finding relevant failure episodes.

    Supports multi-modal queries and filtering.
    Multi-tenancy: customer_id is extracted from API key, not user-provided.
    """

    # Multi-tenancy (set by middleware, not user-provided)
    customer_id: Optional[str] = Field(
        default=None,
        description="Customer ID for tenant isolation (set automatically from API key)",
    )

    # Query intent
    goal: str = Field(..., min_length=5, description="Goal/intent to match")
    current_state: dict[str, Any] = Field(
        default_factory=dict, description="Current execution context"
    )

    # Filtering
    collection: Literal["failures"] = Field(
        default="failures",
        description="Collection to search (only 'failures' supported)",
    )
    tool_filter: Optional[str] = Field(default=None, description="Filter by primary tool")
    min_timestamp: Optional[int] = Field(
        default=None, ge=0, description="Unix timestamp - only return episodes after this"
    )
    max_timestamp: Optional[int] = Field(default=None, ge=0)
    tags: list[str] = Field(default_factory=list, description="Required tags")

    # Search parameters
    k: int = Field(default=5, ge=1, le=1000, description="Number of results to return")
    min_similarity: float = Field(
        default=0.6, ge=-1.0, le=1.0, description="Minimum cosine similarity"
    )
    precondition_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum precondition match score"
    )

    # Ranking
    ranking_weights: RankingWeights = Field(default_factory=RankingWeights)

    # Multi-modal
    include_image_search: bool = Field(
        default=False, description="Also search image embeddings"
    )
    image_weight: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Weight for image similarity if enabled"
    )

    @field_validator("max_timestamp")
    @classmethod
    def validate_timestamp_range(cls, v: Optional[int], info) -> Optional[int]:
        if v is not None:
            min_ts = info.data.get("min_timestamp")
            if min_ts is not None and v < min_ts:
                raise ValueError("max_timestamp must be >= min_timestamp")
        return v


class SearchResult(BaseModel):
    """
    Single search result with scoring breakdown.

    Contains the episode and relevance scores.
    """

    episode: Episode
    scores: dict[str, float] = Field(
        ...,
        description="Score breakdown: similarity, precondition, recency, usage, combined",
    )
    rank: int = Field(..., ge=1, description="Result rank (1 = best match)")

    # Matching details
    matched_preconditions: list[str] = Field(
        default_factory=list, description="Which preconditions matched"
    )
    similarity_explanation: Optional[str] = Field(
        default=None, description="Why this result is relevant"
    )


class SearchResponse(BaseModel):
    """
    Search response with ranked results and metadata.
    """

    results: list[SearchResult] = Field(..., description="Ranked search results")
    total_candidates: int = Field(..., ge=0, description="Total candidates before filtering")
    total_filtered: int = Field(..., ge=0, description="Results after filtering")
    total_returned: int = Field(..., ge=0, description="Results returned (limited by k)")

    # Performance metrics
    search_latency_ms: float = Field(..., ge=0.0)
    breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Latency breakdown: embedding, search, filtering, ranking",
    )

    # Search metadata
    collection: str
    query_embedding_dimension: int
    searched_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def avg_combined_score(self) -> float:
        """Average combined score of returned results."""
        if not self.results:
            return 0.0
        return sum(r.scores.get("combined", 0.0) for r in self.results) / len(
            self.results
        )


class PreconditionCheckResult(BaseModel):
    """
    Result of checking preconditions against current state.

    Used internally by the retrieval pipeline.
    """

    matched: bool
    match_score: float = Field(ge=0.0, le=1.0)
    matched_preconditions: list[str] = Field(default_factory=list)
    missing_preconditions: list[str] = Field(default_factory=list)
    explanation: Optional[str] = Field(default=None)
