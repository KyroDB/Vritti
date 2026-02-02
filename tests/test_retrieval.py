"""
Integration tests for episode retrieval pipeline.

Tests end-to-end retrieval flow:
- Query embedding generation
- KyroDB vector search
- Metadata filtering
- Precondition matching
- Weighted ranking
"""

from datetime import UTC, datetime, timedelta

import pytest

from src.ingestion.embedding import EmbeddingService
from src.kyrodb.router import KyroDBRouter
from src.models.episode import Episode, EpisodeCreate
from src.models.search import RankingWeights, SearchRequest
from src.retrieval.preconditions import PreconditionMatcher
from src.retrieval.ranking import EpisodeRanker
from src.retrieval.search import SearchPipeline


class TestPreconditionMatcher:
    """Test suite for precondition matching."""

    def test_exact_tool_match(self, sample_episode: Episode):
        """Test exact tool name matching."""
        matcher = PreconditionMatcher()

        current_state = {
            "tool": "kubectl",
            "error_class": "ImagePullBackOff",
            "environment": {"os": "Darwin"},
        }

        result = matcher.check_preconditions(
            episode=sample_episode,
            current_state=current_state,
            threshold=0.5,
        )

        assert result.matched is True
        assert result.match_score >= 0.5
        assert "Using tool: kubectl" in result.matched_preconditions

    def test_error_class_match(self, sample_episode: Episode):
        """Test error class matching."""
        matcher = PreconditionMatcher()

        current_state = {
            "tool": "kubectl",
            "error_class": "ImagePullBackOff",
        }

        result = matcher.check_preconditions(
            episode=sample_episode,
            current_state=current_state,
            threshold=0.5,
        )

        assert result.matched is True
        assert "Error class: ImagePullBackOff" in result.matched_preconditions

    def test_component_match(self, sample_episode: Episode):
        """Test component name matching."""
        matcher = PreconditionMatcher()

        current_state = {
            "components": ["kubernetes", "docker"],
        }

        result = matcher.check_preconditions(
            episode=sample_episode,
            current_state=current_state,
            threshold=0.5,
        )

        assert result.matched is True
        assert result.match_score > 0.0

    def test_no_preconditions(self):
        """Test episode with no preconditions (universal relevance)."""
        matcher = PreconditionMatcher()

        # Create episode without reflection
        episode_data = EpisodeCreate(
            customer_id="test-customer",  # Multi-tenancy required
            goal="Test goal for episode without reflection",
            tool_chain=["test"],
            actions_taken=["test"],
            error_trace="Error trace details for testing",
            error_class="unknown",
        )
        episode = Episode(
            create_data=episode_data,
            episode_id=123,
            reflection=None,  # No reflection
            created_at=datetime.now(UTC),
            retrieval_count=0,
        )

        result = matcher.check_preconditions(
            episode=episode,
            current_state={},
            threshold=0.5,
        )

        assert result.matched is True
        assert result.match_score == 1.0
        assert "universal relevance" in result.explanation.lower()

    def test_threshold_filtering(self, sample_episode: Episode):
        """Test precondition threshold filtering."""
        matcher = PreconditionMatcher()

        # State with no matching preconditions
        current_state = {
            "tool": "terraform",  # Different tool
            "error_class": "SyntaxError",  # Different error
        }

        result = matcher.check_preconditions(
            episode=sample_episode,
            current_state=current_state,
            threshold=0.7,  # High threshold
        )

        assert result.matched is False
        assert result.match_score < 0.7


class TestEpisodeRanker:
    """Test suite for weighted ranking."""

    def test_similarity_ranking(self, sample_episode: Episode):
        """Test ranking by similarity score."""
        ranker = EpisodeRanker()

        episodes = [sample_episode]
        similarity_scores = [0.95]
        precondition_scores = [0.5]
        matched_lists = [[]]

        weights = RankingWeights(
            similarity_weight=1.0,  # Only similarity
            precondition_weight=0.0,
            recency_weight=0.0,
            usage_weight=0.0,
        )

        results = ranker.rank_episodes(
            episodes=episodes,
            similarity_scores=similarity_scores,
            precondition_scores=precondition_scores,
            matched_preconditions_list=matched_lists,
            weights=weights,
        )

        assert len(results) == 1
        assert results[0].rank == 1
        assert results[0].scores["combined"] == 0.95
        assert results[0].scores["similarity"] == 0.95

    def test_recency_decay(self, sample_episode: Episode):
        """Test recency score decay over time."""
        ranker = EpisodeRanker()

        # Recent episode
        recent_time = datetime.now(UTC)

        # Old episode (90 days ago)
        old_time = datetime.now(UTC) - timedelta(days=90)

        recent_score = ranker._compute_recency_score(recent_time, recent_time)
        old_score = ranker._compute_recency_score(old_time, recent_time)

        assert recent_score > old_score
        assert recent_score >= 0.9  # Very recent
        assert old_score < 0.5  # Old episode

    def test_usage_score_logarithmic(self):
        """Test usage score with logarithmic scaling."""
        ranker = EpisodeRanker()

        score_0 = ranker._compute_usage_score(0)
        score_10 = ranker._compute_usage_score(10)
        score_100 = ranker._compute_usage_score(100)
        score_1000 = ranker._compute_usage_score(1000)

        assert score_0 == 0.0
        assert score_10 > score_0
        assert score_100 > score_10
        assert score_1000 > score_100

        # Verify scores are bounded
        assert 0.0 <= score_10 <= 1.0
        assert 0.0 <= score_100 <= 1.0
        assert score_1000 == 1.0  # At max count, should be 1.0

    def test_multi_episode_ranking(self):
        """Test ranking multiple episodes."""
        ranker = EpisodeRanker()

        # Create 3 episodes with different scores
        episode1_data = EpisodeCreate(
            customer_id="test-customer",  # Multi-tenancy required
            goal="Episode 1 goal description for ranking test",
            tool_chain=["tool1"],
            actions_taken=["action"],
            error_trace="Error trace for episode 1",
            error_class="unknown",
        )
        episode1 = Episode(
            create_data=episode1_data,
            episode_id=1,
            reflection=None,
            created_at=datetime.now(UTC),
            retrieval_count=100,  # High usage
        )

        episode2_data = EpisodeCreate(
            customer_id="test-customer",  # Multi-tenancy required
            goal="Episode 2 goal description for ranking test",
            tool_chain=["tool2"],
            actions_taken=["action"],
            error_trace="Error trace for episode 2",
            error_class="unknown",
        )
        episode2 = Episode(
            create_data=episode2_data,
            episode_id=2,
            reflection=None,
            created_at=datetime.now(UTC),
            retrieval_count=5,  # Low usage
        )

        episodes = [episode1, episode2]
        similarity_scores = [0.8, 0.9]  # Episode 2 has higher similarity
        precondition_scores = [0.7, 0.6]
        matched_lists = [[], []]

        weights = RankingWeights(
            similarity_weight=0.5,
            precondition_weight=0.2,
            recency_weight=0.1,
            usage_weight=0.2,
        )

        results = ranker.rank_episodes(
            episodes=episodes,
            similarity_scores=similarity_scores,
            precondition_scores=precondition_scores,
            matched_preconditions_list=matched_lists,
            weights=weights,
        )

        # Verify ranking
        assert len(results) == 2
        assert results[0].rank == 1
        assert results[1].rank == 2

        # Episode 1 should rank higher due to much higher usage (100 vs 5)
        # even though episode 2 has slightly higher similarity (0.9 vs 0.8)
        assert results[0].episode.episode_id == 1
        assert results[1].episode.episode_id == 2


class TestSearchPipeline:
    """Test suite for complete search pipeline."""

    @pytest.mark.asyncio
    async def test_basic_search(
        self,
        mock_kyrodb_router: KyroDBRouter,
        mock_embedding_service: EmbeddingService,
    ):
        """Test basic search flow."""
        pipeline = SearchPipeline(
            kyrodb_router=mock_kyrodb_router,
            embedding_service=mock_embedding_service,
        )

        request = SearchRequest(
            customer_id="test-customer",  # Multi-tenancy required
            goal="Deploy application with kubectl",
            current_state={
                "tool": "kubectl",
                "error_class": "ImagePullBackOff",
            },
            collection="failures",
            k=5,
        )

        response = await pipeline.search(request)

        # Assertions
        assert response.total_candidates >= 0
        assert response.total_returned >= 0
        assert response.search_latency_ms >= 0.0
        assert "embedding_ms" in response.breakdown
        assert "search_ms" in response.breakdown
        assert response.collection == "failures"

    @pytest.mark.asyncio
    async def test_search_latency_target(
        self,
        mock_kyrodb_router: KyroDBRouter,
        mock_embedding_service: EmbeddingService,
    ):
        """Test that search meets a reasonable latency target in tests."""
        pipeline = SearchPipeline(
            kyrodb_router=mock_kyrodb_router,
            embedding_service=mock_embedding_service,
        )

        request = SearchRequest(
            customer_id="test-customer",  # Multi-tenancy required
            goal="Test query for latency",
            collection="failures",
            k=5,
        )

        # Run multiple searches to check latency
        latencies = []
        for _ in range(10):
            response = await pipeline.search(request)
            latencies.append(response.search_latency_ms)

        # With mocks, latency should be low but can vary across environments
        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 300.0  # Sanity target for test environments

    @pytest.mark.asyncio
    async def test_metadata_filtering(
        self,
        mock_kyrodb_router: KyroDBRouter,
        mock_embedding_service: EmbeddingService,
    ):
        """Test metadata-based filtering."""
        pipeline = SearchPipeline(
            kyrodb_router=mock_kyrodb_router,
            embedding_service=mock_embedding_service,
        )

        # Search with tool filter
        request = SearchRequest(
            customer_id="test-customer",  # Multi-tenancy required
            goal="Test query for metadata filtering",
            collection="failures",
            tool_filter="kubectl",
            k=5,
        )

        response = await pipeline.search(request)

        # Verify filtering was applied (in real test, would check results)
        assert response.total_candidates >= 0

    @pytest.mark.asyncio
    async def test_search_stats(
        self,
        mock_kyrodb_router: KyroDBRouter,
        mock_embedding_service: EmbeddingService,
    ):
        """Test search pipeline statistics tracking."""
        pipeline = SearchPipeline(
            kyrodb_router=mock_kyrodb_router,
            embedding_service=mock_embedding_service,
        )

        # Initial stats
        stats = pipeline.get_stats()
        assert stats["total_searches"] == 0

        # Perform search
        request = SearchRequest(
            customer_id="test-customer",  # Multi-tenancy required
            goal="Test query for statistics tracking",
            collection="failures",
            k=5,
        )
        await pipeline.search(request)

        # Verify stats updated
        stats = pipeline.get_stats()
        assert stats["total_searches"] == 1
        assert stats["avg_latency_ms"] > 0.0
