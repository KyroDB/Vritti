"""
Integration tests for cached tier workflow (Phase 6).

Tests end-to-end flow: clustering → template generation → cached reflections.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import LLMConfig
from src.ingestion.tiered_reflection import ReflectionTier, TieredReflectionService
from src.models.clustering import ClusterTemplate
from src.models.episode import EpisodeCreate, ErrorClass, Reflection


# TODO(ISSUE-CACHED-TIER): Re-enable after clustering/template architecture implementation
# Tracking: This class tests Phase 6 cached tier which depends on:
#   1. EpisodeClusterer - clustering similar episodes by embedding similarity
#   2. TemplateGenerator - generating reusable reflection templates from clusters
#   3. TieredReflectionService integration with CACHED tier
# Plan: Implement clustering infrastructure in Phase 6, then re-enable these tests
# Timeline: Target Phase 6 (after Phase 5 premium tier stabilization)
# See: docs/STORAGE_ARCHITECTURE.md for cached tier design
@pytest.mark.asyncio
class TestCachedTierIntegration:
    """Integration tests for cached reflection tier."""
    
    @pytest.fixture
    def mock_kyrodb_router(self):
        """Mock KyroDB router."""
        router = AsyncMock()
        router.bulk_fetch_episodes = AsyncMock(return_value=[])
        router.search_cluster_templates = AsyncMock(return_value=[])
        router.insert_cluster_template = AsyncMock(return_value=True)
        router.update_episode_metadata = AsyncMock(return_value=True)
        return router
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service."""
        service = MagicMock()
        # Return consistent embeddings for similar errors
        service.embed_text = MagicMock(return_value=[0.5, 0.5, 0.0] * 128)  # 384-dim
        return service
    
    @pytest.fixture
    def llm_config(self):
        """LLM configuration for testing."""
        return LLMConfig(
            openrouter_api_key="test_key_1234567890abcdef",  # Needs to be 20+ chars
        )
    
    @pytest.fixture
    def tiered_service(self, llm_config, mock_kyrodb_router, mock_embedding_service, monkeypatch):
        """Create tiered reflection service with clustering."""
        monkeypatch.setenv("CLUSTERING_ENABLED", "true")
        # Ensure Settings picks up the env var (Settings is cached as a singleton).
        import src.config as config_module
        config_module._settings = None

        service = TieredReflectionService(
            config=llm_config,
            kyrodb_router=mock_kyrodb_router,
            embedding_service=mock_embedding_service
        )
        yield service
    
    def create_test_episode(self, error_msg: str = "Connection timeout") -> EpisodeCreate:
        """Helper to create test episode."""
        return EpisodeCreate(
            goal="Deploy application to production",
            error_trace=f"Error: {error_msg}\\nStack trace here",
            error_class=ErrorClass.NETWORK_ERROR,
            context={"env": "production"},
            tags=["deployment"],
            tool_chain=["kubectl", "helm"],
            actions_taken=["kubectl apply -f deployment.yaml", "helm upgrade my-app"]
        )
    
    async def test_cached_tier_not_selected_without_clustering(self, tiered_service):
        """Test that CACHED tier is not selected when clustering not available."""
        # Create service without clustering
        service = TieredReflectionService(
            config=tiered_service.config,
            kyrodb_router=None,  # No KyroDB = no clustering
            embedding_service=None
        )
        
        episode = self.create_test_episode()
        tier = await service._select_tier(episode)
        
        # Should select CHEAP (not CACHED) without clustering
        assert tier != ReflectionTier.CACHED
        assert tier in [ReflectionTier.CHEAP, ReflectionTier.PREMIUM]
    
    async def test_cached_tier_selected_with_cluster_match(self, tiered_service):
        """Test CACHED tier selection when episode matches cluster."""
        # Mock cluster match
        mock_template = ClusterTemplate(
            cluster_id=1,
            customer_id="test_customer",
            template_reflection={
                "root_cause": "Network timeout in production deployment",
                "resolution_strategy": "Increase timeout settings"
            },
            source_episode_id=100,
            episode_count=10,
            avg_similarity=0.90
        )
        
        tiered_service.clusterer.find_matching_cluster = AsyncMock(return_value=mock_template)
        
        episode = self.create_test_episode()
        episode.customer_id = "test_customer"
        
        tier = await tiered_service._select_tier(episode)
        
        # Should select CACHED tier
        assert tier == ReflectionTier.CACHED
        assert hasattr(tiered_service._cached_template_local, 'cluster_template')
        assert tiered_service._cached_template_local.cluster_template == mock_template
    
    async def test_cached_reflection_generation(self, tiered_service):
        """Test generating reflection from cached template."""
        # Mock template
        mock_template = ClusterTemplate(
            cluster_id=1,
            customer_id="test_customer",
            template_reflection={
                "root_cause": "Network timeout",
                "resolution_strategy": "Increase timeout",
                "preconditions": ["production environment"],
                "confidence_score": 0.85
            },
            source_episode_id=100,
            episode_count=10,
            avg_similarity=0.90
        )
        
        # Mock template generator
        mock_reflection = Reflection(
            root_cause="Network timeout",
            resolution_strategy="Increase timeout",
            preconditions=["production environment"],
            confidence_score=0.85,
            generalization_score=0.75,
            environment_factors=["production"],
            affected_components=["network"],
            llm_model="cached_template",
            generated_at=datetime.now(UTC),
            cost_usd=0.0,  # Cached = $0
            generation_latency_ms=5.0,
            tier="cached"
        )
        
        tiered_service.template_generator.get_cached_reflection = AsyncMock(return_value=mock_reflection)
        tiered_service._cached_template_local.cluster_template = mock_template
        
        episode = self.create_test_episode()
        
        # Generate reflection with CACHED tier
        with patch.object(tiered_service, '_select_tier', return_value=ReflectionTier.CACHED):
            reflection = await tiered_service.generate_reflection(episode)
        
        assert reflection is not None
        assert reflection.tier == "cached"
        assert reflection.cost_usd == 0.0
        assert reflection.confidence_score >= 0.6
    
    async def test_cached_reflection_quality_fallback(self, tiered_service):
        """Test fallback to CHEAP when cached reflection quality is low."""
        # Mock low-quality template
        low_quality_reflection = Reflection(
            root_cause="Generic error",
            resolution_strategy="Try again",
            preconditions=[],
            confidence_score=0.4,  # Below threshold
            generalization_score=0.3,
            environment_factors=[],
            affected_components=[],
            llm_model="cached_template",
            generated_at=datetime.now(UTC),
            cost_usd=0.0,
            generation_latency_ms=5.0,
            tier="cached"
        )
        
        tiered_service.template_generator.get_cached_reflection = AsyncMock(return_value=low_quality_reflection)
        tiered_service._cached_template_local.cluster_template = MagicMock()
        
        # Mock cheap service to return good reflection
        cheap_reflection = Reflection(
            root_cause="Specific error root cause",
            resolution_strategy="Detailed resolution",
            preconditions=["condition1"],
            confidence_score=0.75,
            generalization_score=0.70,
            environment_factors=["env1"],
            affected_components=["comp1"],
            llm_model="openrouter-cheap-model",
            generated_at=datetime.now(UTC),
            cost_usd=0.0003,
            generation_latency_ms=500.0,
            tier="cheap"
        )
        tiered_service.cheap_service.generate_reflection = AsyncMock(return_value=cheap_reflection)
        
        episode = self.create_test_episode()
        
        # Generate with CACHED tier (should fallback to CHEAP)
        with patch.object(tiered_service, '_select_tier', return_value=ReflectionTier.CACHED):
            reflection = await tiered_service.generate_reflection(episode)
        
        # Should have fallen back to CHEAP
        assert reflection.tier == "cheap"
        assert reflection.cost_usd > 0.0
   
    async def test_cost_savings_with_cached_tier(self, tiered_service):
        """Test that cached tier achieves cost savings."""
        # Track costs
        costs = []
        
        # Mock premium service to return costly reflection
        premium_reflection = Reflection(
            root_cause="Complex error",
            resolution_strategy="Complex resolution",
            preconditions=["complex"],
            confidence_score=0.95,
            generalization_score=0.8,
            environment_factors=["complex"],
            affected_components=["complex"],
            llm_model="gpt-4",
            generated_at=datetime.now(UTC),
            cost_usd=0.096,
            generation_latency_ms=1000.0,
            tier="premium"
        )
        tiered_service.premium_service.generate_multi_perspective_reflection = AsyncMock(return_value=premium_reflection)

        # Generate 10 reflections (simulate cluster reuse)
        for i in range(10):
            episode = self.create_test_episode(f"Error {i}")
            
            if i == 0:
                # First one uses PREMIUM (generates template)
                tier = ReflectionTier.PREMIUM
            else:
                # Rest use CACHED (reuse template)
                tier = ReflectionTier.CACHED
                
                # Mock cached reflection
                cached_reflection = Reflection(
                    root_cause="Network timeout",
                    resolution_strategy="Increase timeout",
                    preconditions=["production"],
                    confidence_score=0.80,
                    generalization_score=0.75,
                    environment_factors=["production"],
                    affected_components=["network"],
                    llm_model="cached_template",
                    generated_at=datetime.now(UTC),
                    cost_usd=0.0,
                    generation_latency_ms=5.0,
                    tier="cached"
                )
                tiered_service.template_generator.get_cached_reflection = AsyncMock(return_value=cached_reflection)
                tiered_service._cached_template_local.cluster_template = MagicMock()
            
            with patch.object(tiered_service, '_select_tier', return_value=tier):
                reflection = await tiered_service.generate_reflection(episode)
                costs.append(reflection.cost_usd)
        
        # Verify cost savings
        total_cost = sum(costs)
        
        # With caching: 1 premium ($0.096) + 9 cached ($0) = $0.096
        # Without caching: 10 cheap ($0.0003 each) = $0.003
        # Or 10 premium ($0.096 each) = $0.96
        
        assert total_cost < 0.10, f"Cost with caching should be <$0.10, got ${total_cost}"
        assert costs[0] > 0.0, "First reflection should have cost (template generation)"
        assert all(c == 0.0 for c in costs[1:]), "Cached reflections should be $0"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
