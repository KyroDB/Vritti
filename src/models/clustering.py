"""
Clustering data models for memory hygiene.

Defines schemas for episode clusters, templates, and clustering metadata.
"""

import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class ClusterInfo(BaseModel):
    """
    Metadata about an episode cluster.
    
    Contains aggregate statistics and episode membership.
    """
    
    cluster_id: int = Field(
        ...,
        ge=1,
        description="Unique cluster identifier used as KyroDB doc_id for the cluster template (>=1)."
    )
    
    customer_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Customer who owns this cluster"
    )
    
    episode_ids: list[int] = Field(
        ...,
        min_length=1,
        description="Episodes belonging to this cluster"
    )
    
    centroid_embedding: list[float] = Field(
        ...,
        min_length=1,
        description="Cluster centroid for similarity matching"
    )
    
    avg_intra_cluster_similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average similarity between episodes in cluster"
    )
    
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    
    last_updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


class ClusterTemplate(BaseModel):
    """
    Template reflection for an episode cluster.
    
    Generated from the best episode in the cluster and reused
    for all new episodes matching this cluster.
    
    Security:
    - customer_id validated for multi-tenancy
    - template_reflection fully validated
    - usage tracking prevents abuse
    """
    
    cluster_id: int = Field(
        ...,
        gt=0,
        description="Cluster identifier (KyroDB doc_id for this template)"
    )
    
    customer_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Customer who owns this template"
    )
    
    # Import will be done at runtime to avoid circular dependency
    # template_reflection: Reflection
    template_reflection: dict = Field(
        ...,
        description="The canonical reflection for this cluster"
    )
    
    source_episode_id: int = Field(
        ...,
        description="Episode that generated this template"
    )
    
    episode_count: int = Field(
        ...,
        ge=1,
        description="Number of episodes in this cluster"
    )
    
    avg_similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average intra-cluster similarity"
    )

    match_similarity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Similarity between query and this template (set at retrieval time)",
    )
    
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    
    last_used_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    
    usage_count: int = Field(
        default=0,
        ge=0,
        description="How many times this template has been reused"
    )
    
    template_version: int = Field(
        default=1,
        ge=1,
        description="Template version for future updates"
    )

    def to_metadata_dict(self) -> dict[str, str]:
        """
        Serialize template for KyroDB metadata (string-string map).

        Note: match_similarity is intentionally excluded (query-specific, not persisted).
        """
        return {
            "cluster_id": str(self.cluster_id),
            "customer_id": self.customer_id,
            "source_episode_id": str(self.source_episode_id),
            "episode_count": str(self.episode_count),
            "avg_similarity": f"{self.avg_similarity:.4f}",
            "usage_count": str(self.usage_count),
            "created_at": self.created_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else "",
            "template_version": str(self.template_version),
            "template_reflection_json": json.dumps(self.template_reflection),
            "doc_type": "cluster_template",
        }

    @classmethod
    def from_metadata_dict(cls, metadata: dict[str, str]) -> "ClusterTemplate":
        """
        Deserialize ClusterTemplate from KyroDB metadata.
        """
        from src.models.episode import Reflection

        def _get_int(key: str, default: int = 0) -> int:
            try:
                return int(metadata.get(key, str(default)) or default)
            except Exception:
                return default

        def _get_float(key: str, default: float = 0.0) -> float:
            try:
                return float(metadata.get(key, str(default)) or default)
            except Exception:
                return default

        def _get_dt(key: str) -> datetime:
            raw = (metadata.get(key) or "").strip()
            if not raw:
                return datetime.now(UTC)
            try:
                return datetime.fromisoformat(raw)
            except Exception:
                return datetime.now(UTC)

        raw_reflection = (metadata.get("template_reflection_json") or "").strip()
        if not raw_reflection:
            raise ValueError("Missing template_reflection_json in cluster template metadata")

        try:
            template_payload: Any = json.loads(raw_reflection)
        except Exception as e:
            raise ValueError(f"Invalid template_reflection_json: {e}") from e

        # Validate template payload against Reflection schema for safety.
        reflection = Reflection.model_validate(template_payload)
        template_reflection = reflection.model_dump(mode="json")

        return cls(
            cluster_id=_get_int("cluster_id"),
            customer_id=(metadata.get("customer_id") or "").strip(),
            template_reflection=template_reflection,
            source_episode_id=_get_int("source_episode_id"),
            episode_count=_get_int("episode_count", default=1),
            avg_similarity=_get_float("avg_similarity", default=0.0),
            usage_count=_get_int("usage_count", default=0),
            created_at=_get_dt("created_at"),
            last_used_at=_get_dt("last_used_at"),
            template_version=_get_int("template_version", default=1),
        )


class ClusteringStats(BaseModel):
    """
    Statistics from a clustering run.
    
    Used for monitoring and optimization.
    """
    
    customer_id: str
    total_episodes: int = Field(ge=0)
    total_clusters: int = Field(ge=0)
    noise_episodes: int = Field(ge=0)
    clustered_episodes: int = Field(ge=0)
    avg_cluster_size: float = Field(ge=0.0)
    min_cluster_size: int = Field(ge=0)
    max_cluster_size: int = Field(ge=0)
    clustering_duration_seconds: float = Field(ge=0.0)
    templates_generated: int = Field(ge=0)
    
    @property
    def clustered_percentage(self) -> float:
        """Percentage of episodes successfully clustered."""
        if self.total_episodes == 0:
            return 0.0
        return (self.clustered_episodes / self.total_episodes) * 100
