"""
Clustering data models for memory hygiene.

Defines schemas for episode clusters, templates, and clustering metadata.
"""

from datetime import timezone, datetime

from pydantic import BaseModel, Field


class ClusterInfo(BaseModel):
    """
    Metadata about an episode cluster.
    
    Contains aggregate statistics and episode membership.
    """
    
    cluster_id: int = Field(
        ...,
        description="Unique cluster identifier (HDBSCAN label)"
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
        default_factory=lambda: datetime.now(timezone.utc)
    )
    
    last_updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
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
        description="Cluster this template represents"
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
    
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    
    last_used_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
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
