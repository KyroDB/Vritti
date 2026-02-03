"""
Unit tests for Phase 6 clustering logic.

Tests HDBSCAN clustering, cluster matching, and centroid caching.
"""

from unittest.mock import AsyncMock

import numpy as np
import pytest

from src.hygiene.clustering import EpisodeClusterer
from src.models.clustering import ClusterInfo, ClusterTemplate


class TestEpisodeClusterer:
    """Test suite for EpisodeClusterer."""
    
    @pytest.fixture
    def mock_kyrodb_router(self):
        """Mock KyroDB router."""
        router = AsyncMock()
        return router
    
    @pytest.fixture
    def clusterer(self, mock_kyrodb_router):
        """Create clusterer instance."""
        return EpisodeClusterer(
            kyrodb_router=mock_kyrodb_router,
            min_cluster_size=3,
            min_samples=2,
            metric='cosine'
        )
    
    def test_initialization(self, clusterer):
        """Test clusterer initialization."""
        assert clusterer.min_cluster_size == 3
        assert clusterer.min_samples == 2
        assert clusterer.metric == 'cosine'
        assert clusterer.cache_ttl == 3600
    
    def test_initialization_validation(self, mock_kyrodb_router):
        """Test that min_cluster_size < 3 raises error."""
        with pytest.raises(ValueError, match="min_cluster_size must be >= 3"):
            EpisodeClusterer(
                kyrodb_router=mock_kyrodb_router,
                min_cluster_size=2
            )
    
    def test_cosine_similarity(self, clusterer):
        """Test cosine similarity calculation."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        
        similarity = clusterer._cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(1.0)
        
        # Orthogonal vectors
        vec3 = np.array([0.0, 1.0, 0.0])
        similarity = clusterer._cosine_similarity(vec1, vec3)
        assert similarity == pytest.approx(0.0)
    
    def test_cosine_similarity_zero_vectors(self, clusterer):
        """Test cosine similarity with zero vectors."""
        vec1 = np.array([0.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        
        similarity = clusterer._cosine_similarity(vec1, vec2)
        assert similarity == 0.0
    
    def test_calculate_avg_similarity(self, clusterer):
        """Test average intra-cluster similarity calculation."""
        # Identical embeddings
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        
        avg_sim = clusterer._calculate_avg_similarity(embeddings)
        assert avg_sim == pytest.approx(1.0)
        
        # Diverse embeddings
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        avg_sim = clusterer._calculate_avg_similarity(embeddings)
        assert avg_sim == pytest.approx(0.0)
    
    def test_calculate_avg_similarity_single_episode(self, clusterer):
        """Test avg similarity with single episode."""
        embeddings = np.array([[1.0, 0.0, 0.0]])
        
        avg_sim = clusterer._calculate_avg_similarity(embeddings)
        assert avg_sim == 1.0  # Should return 1.0 for single episode
    
    def test_cache_operations(self, clusterer):
        """Test cluster cache update and retrieval."""
        customer_id = "test_customer"
        
        clusters = {
            1: ClusterInfo(
                cluster_id=1,
                customer_id=customer_id,
                episode_ids=[1, 2, 3],
                centroid_embedding=[0.5, 0.5, 0.0],
                avg_intra_cluster_similarity=0.85
            )
        }
        
        # Update cache
        clusterer._update_cache(customer_id, clusters)
        
        # Retrieve from cache
        centroids = clusterer._get_cached_centroids(customer_id)
        
        assert customer_id in clusterer._cluster_cache
        assert 1 in centroids
        assert isinstance(centroids[1], np.ndarray)
        assert np.array_equal(centroids[1], np.array([0.5, 0.5, 0.0]))
    
    def test_cache_expiration(self, clusterer):
        """Test that cache expires after TTL."""
        customer_id = "test_customer"
        
        # Set very short TTL for testing
        clusterer.cache_ttl = 0
        
        clusters = {
            1: ClusterInfo(
                cluster_id=1,
                customer_id=customer_id,
                episode_ids=[1, 2, 3],
                centroid_embedding=[0.5, 0.5, 0.0],
                avg_intra_cluster_similarity=0.85
            )
        }
        
        clusterer._update_cache(customer_id, clusters)
        
        # Cache should be expired immediately
        centroids = clusterer._get_cached_centroids(customer_id)
        assert len(centroids) == 0
    
    @pytest.mark.asyncio
    async def test_cluster_customer_episodes_insufficient(self, clusterer, mock_kyrodb_router):
        """Test clustering with insufficient episodes."""
        clusterer._fetch_active_episode_embeddings = AsyncMock(return_value=[])
        
        result = await clusterer.cluster_customer_episodes("test_customer")
        
        assert result == {}
    
    @pytest.mark.asyncio
    async def test_find_matching_cluster_no_cache(self, clusterer):
        """Test cluster matching with empty cache."""
        episode_embedding = [0.5, 0.5, 0.0]
        
        result = await clusterer.find_matching_cluster(
            episode_embedding=episode_embedding,
            customer_id="test_customer",
            similarity_threshold=0.85
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_find_matching_cluster_below_threshold(self, clusterer):
        """Test cluster matching below threshold."""
        customer_id = "test_customer"
        
        # Add cluster to cache
        clusters = {
            1: ClusterInfo(
                cluster_id=1,
                customer_id=customer_id,
                episode_ids=[1, 2, 3],
                centroid_embedding=[1.0, 0.0, 0.0],  # Different from query
                avg_intra_cluster_similarity=0.85
            )
        }
        clusterer._update_cache(customer_id, clusters)
        
        # Query with orthogonal embedding (similarity will be 0)
        episode_embedding = [0.0, 1.0, 0.0]
        
        result = await clusterer.find_matching_cluster(
            episode_embedding=episode_embedding,
            customer_id=customer_id,
            similarity_threshold=0.85
        )
        
        assert result is None


class TestClusterMatching:
    """Test cluster matching logic."""
    
    @pytest.fixture
    def clusterer(self):
        """Create clusterer for testing."""
        return EpisodeClusterer(
            kyrodb_router=AsyncMock(),
            min_cluster_size=3
        )
    
    @pytest.mark.asyncio
    async def test_match_similar_embedding(self, clusterer):
        """Test matching with similar embedding."""
        customer_id = "test_customer"
        
        # Create cluster with specific centroid
        clusters = {
            1: ClusterInfo(
                cluster_id=1,
                customer_id=customer_id,
                episode_ids=[1, 2, 3],
                centroid_embedding=[0.9, 0.1, 0.0],
                avg_intra_cluster_similarity=0.85
            )
        }
        clusterer._update_cache(customer_id, clusters)
        
        # Mock template retrieval
        mock_template = ClusterTemplate(
            cluster_id=1,
            customer_id=customer_id,
            template_reflection={"root_cause": "test"},
            source_episode_id=1,
            episode_count=3,
            avg_similarity=0.85
        )
        clusterer._get_cluster_template = AsyncMock(return_value=mock_template)
        
        # Query with very similar embedding
        episode_embedding = [0.95, 0.05, 0.0]
        
        result = await clusterer.find_matching_cluster(
            episode_embedding=episode_embedding,
            customer_id=customer_id,
            similarity_threshold=0.8
        )
        
        assert result is not None
        assert result.cluster_id == 1


@pytest.mark.asyncio
async def test_customer_id_validation():
    """Test customer ID validation."""
    clusterer = EpisodeClusterer(
        kyrodb_router=AsyncMock(),
        min_cluster_size=3
    )
    
    # Invalid customer IDs
    with pytest.raises(ValueError, match="Invalid customer_id"):
        await clusterer.cluster_customer_episodes("")
    
    with pytest.raises(ValueError, match="Invalid customer_id"):
        await clusterer.cluster_customer_episodes("a" * 101)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
