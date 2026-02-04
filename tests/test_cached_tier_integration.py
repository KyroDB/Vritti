"""
Integration tests for cached tier workflow (Phase 6).

These tests validate the TieredReflectionService cached tier behavior:
- cluster match → selects CACHED tier
- cached reflection uses the matched template and the real episode_id
- low-quality cached reflections fall back to CHEAP
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import LLMConfig
from src.ingestion.tiered_reflection import (
    ReflectionTier,
    TieredReflectionService,
    _cached_cluster_template_var,
)
from src.models.clustering import ClusterTemplate
from src.models.episode import EpisodeCreate, ErrorClass, Reflection


@pytest.mark.asyncio
class TestCachedTierIntegration:
    @pytest.fixture
    def llm_config(self) -> LLMConfig:
        return LLMConfig(openrouter_api_key="test_key_1234567890abcdef")

    @pytest.fixture
    def mock_kyrodb_router(self):
        router = AsyncMock()
        return router

    @pytest.fixture
    def mock_embedding_service(self):
        service = MagicMock()
        # Deterministic embedding for clustering match path.
        service.embed_text = MagicMock(return_value=[0.5] * 384)
        service.embed_text_async = AsyncMock(return_value=[0.5] * 384)
        return service

    @pytest.fixture
    def tiered_service(self, llm_config, mock_kyrodb_router, mock_embedding_service, monkeypatch):
        # Enable clustering in settings for this test process.
        monkeypatch.setenv("CLUSTERING_ENABLED", "true")
        import src.config as config_module

        config_module._settings = None

        return TieredReflectionService(
            config=llm_config,
            kyrodb_router=mock_kyrodb_router,
            embedding_service=mock_embedding_service,
        )

    def _make_episode(self, *, customer_id: str) -> EpisodeCreate:
        return EpisodeCreate(
            customer_id=customer_id,
            goal="Deploy application to production",
            tool_chain=["kubectl", "helm"],
            actions_taken=["kubectl apply -f deployment.yaml"],
            error_trace="Error: connection timeout while contacting cluster API",
            error_class=ErrorClass.NETWORK_ERROR,
            environment_info={"env": "production"},
            tags=["deployment"],
        )

    async def test_cached_tier_not_selected_without_clustering(self, llm_config):
        service = TieredReflectionService(
            config=llm_config,
            kyrodb_router=None,
            embedding_service=None,
        )
        episode = self._make_episode(customer_id="test_customer")
        tier = await service._select_tier(episode)
        assert tier in {ReflectionTier.CHEAP, ReflectionTier.PREMIUM}
        assert tier != ReflectionTier.CACHED

    async def test_cached_tier_selected_with_cluster_match(self, tiered_service):
        mock_template = ClusterTemplate(
            cluster_id=123,
            customer_id="test_customer",
            template_reflection={
                "root_cause": "Network timeout in production deployment",
                "preconditions": ["env=production"],
                "resolution_strategy": "Increase timeout settings",
                "environment_factors": [],
                "affected_components": [],
                "generalization_score": 0.7,
                "confidence_score": 0.9,
                "llm_model": "cluster-template-123",
                "generated_at": datetime.now(UTC).isoformat(),
                "cost_usd": 0.0,
                "generation_latency_ms": 1.0,
                "tier": "cached",
            },
            source_episode_id=1,
            episode_count=10,
            avg_similarity=0.9,
        )
        tiered_service.clusterer.find_matching_cluster = AsyncMock(return_value=mock_template)

        episode = self._make_episode(customer_id="test_customer")
        tier = await tiered_service._select_tier(episode)

        assert tier == ReflectionTier.CACHED
        assert _cached_cluster_template_var.get() == mock_template

    async def test_cached_reflection_uses_real_episode_id(self, tiered_service):
        mock_template = ClusterTemplate(
            cluster_id=123,
            customer_id="test_customer",
            template_reflection={
                "root_cause": "Network timeout",
                "preconditions": ["env=production"],
                "resolution_strategy": "Increase timeout",
                "environment_factors": [],
                "affected_components": [],
                "generalization_score": 0.7,
                "confidence_score": 0.9,
                "llm_model": "cluster-template-123",
                "generated_at": datetime.now(UTC).isoformat(),
                "cost_usd": 0.0,
                "generation_latency_ms": 1.0,
                "tier": "cached",
            },
            source_episode_id=1,
            episode_count=10,
            avg_similarity=0.9,
        )
        _cached_cluster_template_var.set(mock_template)

        cached_reflection = Reflection(
            root_cause="Network timeout",
            preconditions=["env=production"],
            resolution_strategy="Increase timeout",
            confidence_score=0.8,
            generalization_score=0.7,
            environment_factors=[],
            affected_components=[],
            llm_model="cluster-template-123",
            generated_at=datetime.now(UTC),
            cost_usd=0.0,
            generation_latency_ms=2.0,
            tier=ReflectionTier.CACHED.value,
        )
        tiered_service.template_generator.get_cached_reflection = AsyncMock(
            return_value=cached_reflection
        )

        episode = self._make_episode(customer_id="test_customer")
        reflection = await tiered_service.generate_reflection(
            episode, episode_id=999, tier=ReflectionTier.CACHED
        )

        assert reflection.tier == ReflectionTier.CACHED.value
        tiered_service.template_generator.get_cached_reflection.assert_awaited_once()
        _, kwargs = tiered_service.template_generator.get_cached_reflection.await_args
        assert kwargs["episode_id"] == 999

    async def test_cached_quality_fallback_to_cheap(self, tiered_service):
        # Arrange: cached tier selected, template available, but cached reflection is low confidence.
        mock_template = ClusterTemplate(
            cluster_id=123,
            customer_id="test_customer",
            template_reflection={"root_cause": "x", "resolution_strategy": "y"},
            source_episode_id=1,
            episode_count=10,
            avg_similarity=0.9,
        )
        _cached_cluster_template_var.set(mock_template)

        low_quality_cached = Reflection(
            root_cause="Generic error",
            resolution_strategy="Try again",
            preconditions=[],
            confidence_score=0.4,
            generalization_score=0.3,
            environment_factors=[],
            affected_components=[],
            llm_model="cluster-template-123",
            generated_at=datetime.now(UTC),
            cost_usd=0.0,
            generation_latency_ms=5.0,
            tier=ReflectionTier.CACHED.value,
        )
        tiered_service.template_generator.get_cached_reflection = AsyncMock(
            return_value=low_quality_cached
        )

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
            tier=ReflectionTier.CHEAP.value,
        )
        tiered_service.cheap_service.generate_reflection = AsyncMock(return_value=cheap_reflection)

        episode = self._make_episode(customer_id="test_customer")
        reflection = await tiered_service.generate_reflection(
            episode, episode_id=100, tier=ReflectionTier.CACHED
        )

        assert reflection.tier == ReflectionTier.CHEAP.value
        assert reflection.cost_usd > 0.0

    async def test_cost_savings_with_cached_tier(self, tiered_service):
        costs: list[float] = []

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
            tier=ReflectionTier.PREMIUM.value,
        )
        tiered_service.premium_service.generate_multi_perspective_reflection = AsyncMock(
            return_value=premium_reflection
        )

        cached_reflection = Reflection(
            root_cause="Network timeout",
            resolution_strategy="Increase timeout",
            preconditions=["production"],
            confidence_score=0.80,
            generalization_score=0.75,
            environment_factors=["production"],
            affected_components=["network"],
            llm_model="cluster-template-123",
            generated_at=datetime.now(UTC),
            cost_usd=0.0,
            generation_latency_ms=5.0,
            tier=ReflectionTier.CACHED.value,
        )
        tiered_service.template_generator.get_cached_reflection = AsyncMock(
            return_value=cached_reflection
        )

        mock_template = ClusterTemplate(
            cluster_id=123,
            customer_id="test_customer",
            template_reflection={"root_cause": "x", "resolution_strategy": "y"},
            source_episode_id=1,
            episode_count=10,
            avg_similarity=0.9,
        )

        # First reflection uses PREMIUM, subsequent use CACHED (template reuse).
        for i in range(10):
            tier = ReflectionTier.PREMIUM if i == 0 else ReflectionTier.CACHED
            if tier == ReflectionTier.CACHED:
                _cached_cluster_template_var.set(mock_template)

            episode = self._make_episode(customer_id="test_customer")
            with patch.object(tiered_service, "_select_tier", return_value=tier):
                reflection = await tiered_service.generate_reflection(
                    episode, episode_id=1000 + i
                )
            costs.append(reflection.cost_usd)

        total_cost = sum(costs)

        assert total_cost < 0.10, f"Cost with caching should be <$0.10, got ${total_cost}"
        assert costs[0] > 0.0, "First reflection should have cost (template generation)"
        assert all(c == 0.0 for c in costs[1:]), "Cached reflections should be $0"
