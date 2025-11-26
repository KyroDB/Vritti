"""
Integration tests for Phase 4: Advanced Retrieval & Preconditions.

Tests end-to-end search pipeline integration with LLM semantic validation.
Uses OpenRouter API for LLM access.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.retrieval.search import SearchPipeline
from src.retrieval.preconditions import PreconditionMatcher, AdvancedPreconditionMatcher
from src.models.episode import Episode, EpisodeCreate, ErrorClass, Reflection
from src.models.search import SearchRequest, SearchResponse


class TestSearchPipelineBasic:
    """Test search pipeline without LLM validation."""
    
    @pytest.fixture
    def mock_kyrodb_router(self):
        """Create mock KyroDB router."""
        router = MagicMock()
        router.search_text = AsyncMock(return_value=MagicMock(results=[]))
        return router
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service."""
        service = MagicMock()
        service.embed_text = MagicMock(return_value=[0.1] * 384)
        return service
    
    @pytest.fixture
    def search_pipeline(self, mock_kyrodb_router, mock_embedding_service):
        """Create search pipeline with mocks."""
        return SearchPipeline(
            kyrodb_router=mock_kyrodb_router,
            embedding_service=mock_embedding_service
        )
    
    @pytest.mark.asyncio
    async def test_basic_search_flow(self, search_pipeline):
        """Test basic search flow without candidates."""
        request = SearchRequest(
            goal="Test query",
            customer_id="test_customer",
            collection="failures",
            k=5
        )
        
        response = await search_pipeline.search(request)
        
        assert isinstance(response, SearchResponse)
        assert response.total_returned == 0
        assert "embedding_ms" in response.breakdown
        assert "search_ms" in response.breakdown
    
    def test_stats_initialization(self, search_pipeline):
        """Test stats tracking initialization."""
        stats = search_pipeline.get_stats()
        
        assert stats["total_searches"] == 0
        assert stats["avg_latency_ms"] == 0.0
        assert stats["llm_validation_calls"] == 0
        assert stats["llm_rejections"] == 0


class TestSearchPipelineLLMIntegration:
    """Test search pipeline with LLM validation enabled via OpenRouter."""
    
    @pytest.fixture
    def sample_episode(self):
        """Create a sample episode for testing."""
        return Episode(
            episode_id=1,
            create_data=EpisodeCreate(
                goal="Delete files older than 7 days",
                error_class=ErrorClass.CONFIGURATION_ERROR,
                tool_chain=["find"],
                actions_taken=["find . -mtime +7 -delete"],
                error_trace="Error: Deleted wrong files"
            ),
            reflection=Reflection(
                root_cause="Deleted wrong files",
                resolution_strategy="Use exclusion patterns",
                preconditions=[],
                environment_factors=[],
                affected_components=[],
                generalization_score=0.8,
                confidence_score=0.9
            )
        )
    
    @pytest.fixture
    def mock_kyrodb_with_results(self, sample_episode):
        """Create mock KyroDB router with results."""
        router = MagicMock()
        
        # Mock search result
        mock_result = MagicMock()
        mock_result.doc_id = "test_doc_1"
        mock_result.score = 0.95  # High similarity
        mock_result.metadata = sample_episode.to_metadata_dict()
        
        search_response = MagicMock()
        search_response.results = [mock_result]
        
        router.search_text = AsyncMock(return_value=search_response)
        return router
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service."""
        service = MagicMock()
        service.embed_text = MagicMock(return_value=[0.1] * 384)
        return service
    
    def _mock_openrouter_response(self, content: str):
        """Create a mock httpx response for OpenRouter."""
        response = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        response.raise_for_status = MagicMock()
        return response
    
    @pytest.mark.asyncio
    async def test_llm_rejects_semantic_negation(self, mock_kyrodb_with_results, mock_embedding_service):
        """Test that LLM validation rejects semantically opposite queries."""
        # Create advanced matcher with OpenRouter
        advanced_matcher = AdvancedPreconditionMatcher(
            openrouter_api_key="test_key",
            enable_llm=True
        )
        
        # Mock OpenRouter to reject semantic negation
        mock_response = self._mock_openrouter_response(
            '{"compatible": false, "confidence": 0.95, "reason": "Opposite meaning due to EXCEPT"}'
        )
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value.post = AsyncMock(return_value=mock_response)
            
            # Create pipeline with LLM validation
            pipeline = SearchPipeline(
                kyrodb_router=mock_kyrodb_with_results,
                embedding_service=mock_embedding_service,
                advanced_precondition_matcher=advanced_matcher
            )
            
            # Search with semantically opposite query
            request = SearchRequest(
                goal="Delete files EXCEPT those older than 7 days",
                customer_id="test_customer",
                collection="failures",
                k=5
            )
            
            response = await pipeline.search(request)
            
            # Should have no results (LLM rejected)
            assert response.total_returned == 0
            
            # Check stats
            stats = pipeline.get_stats()
            assert stats["llm_validation_calls"] >= 1
            assert stats["llm_rejections"] >= 1
    
    @pytest.mark.asyncio
    async def test_llm_accepts_compatible_query(self, mock_kyrodb_with_results, mock_embedding_service):
        """Test that LLM validation accepts compatible queries."""
        # Create advanced matcher with OpenRouter
        advanced_matcher = AdvancedPreconditionMatcher(
            openrouter_api_key="test_key",
            enable_llm=True
        )
        
        # Mock OpenRouter to accept compatible query
        mock_response = self._mock_openrouter_response(
            '{"compatible": true, "confidence": 0.9, "reason": "Same goal"}'
        )
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value.post = AsyncMock(return_value=mock_response)
            
            # Create pipeline with LLM validation
            pipeline = SearchPipeline(
                kyrodb_router=mock_kyrodb_with_results,
                embedding_service=mock_embedding_service,
                advanced_precondition_matcher=advanced_matcher
            )
            
            # Search with compatible query
            request = SearchRequest(
                goal="Remove files older than 7 days",
                customer_id="test_customer",
                collection="failures",
                k=5
            )
            
            response = await pipeline.search(request)
            
            # Should have results (LLM accepted)
            assert response.total_returned >= 1
            
            # Check stats
            stats = pipeline.get_stats()
            assert stats["llm_validation_calls"] >= 1
            assert stats["llm_rejections"] == 0


class TestTwoStageValidation:
    """Test two-stage validation logic (heuristic + LLM)."""
    
    @pytest.fixture
    def mock_kyrodb_router(self):
        """Create mock KyroDB router."""
        router = MagicMock()
        router.search_text = AsyncMock(return_value=MagicMock(results=[]))
        return router
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service."""
        service = MagicMock()
        service.embed_text = MagicMock(return_value=[0.1] * 384)
        return service
    
    @pytest.mark.asyncio
    async def test_low_similarity_skips_llm(self, mock_kyrodb_router, mock_embedding_service):
        """Test that low similarity candidates skip LLM validation."""
        # Create mock advanced matcher
        advanced_matcher = MagicMock(spec=AdvancedPreconditionMatcher)
        advanced_matcher.check_preconditions_with_llm = AsyncMock()
        
        pipeline = SearchPipeline(
            kyrodb_router=mock_kyrodb_router,
            embedding_service=mock_embedding_service,
            advanced_precondition_matcher=advanced_matcher
        )
        
        # Search
        request = SearchRequest(
            goal="Test query",
            customer_id="test_customer",
            collection="failures",
            k=5
        )
        
        await pipeline.search(request)
        
        # LLM should not be called (no candidates with high similarity)
        stats = pipeline.get_stats()
        assert stats["llm_validation_calls"] == 0
    
    def test_graceful_fallback_on_llm_error(self):
        """Test graceful fallback when LLM validation fails."""
        pipeline = SearchPipeline(
            kyrodb_router=MagicMock(),
            embedding_service=MagicMock(),
            advanced_precondition_matcher=None  # LLM disabled
        )
        
        # Should work fine without LLM
        stats = pipeline.get_stats()
        assert stats["llm_validation_calls"] == 0


class TestPerformanceMetrics:
    """Test performance metrics tracking."""
    
    @pytest.mark.asyncio
    async def test_stats_tracking_without_llm(self):
        """Test stats tracking when LLM is disabled."""
        router = MagicMock()
        router.search_text = AsyncMock(return_value=MagicMock(results=[]))
        
        embedding = MagicMock()
        embedding.embed_text = MagicMock(return_value=[0.1] * 384)
        
        pipeline = SearchPipeline(
            kyrodb_router=router,
            embedding_service=embedding
        )
        
        # Perform search
        request = SearchRequest(
            goal="Test query for stats",
            customer_id="test",
            collection="failures",
            k=5
        )
        await pipeline.search(request)
        
        stats = pipeline.get_stats()
        assert stats["total_searches"] == 1
        assert stats["avg_latency_ms"] > 0
        assert stats["llm_validation_calls"] == 0
    
    @pytest.mark.asyncio
    async def test_stats_tracking_with_llm(self):
        """Test stats tracking with LLM enabled via OpenRouter."""
        router = MagicMock()
        router.search_text = AsyncMock(return_value=MagicMock(results=[]))
        
        embedding = MagicMock()
        embedding.embed_text = MagicMock(return_value=[0.1] * 384)
        
        advanced_matcher = AdvancedPreconditionMatcher(
            openrouter_api_key="test",
            enable_llm=True
        )
        
        pipeline = SearchPipeline(
            kyrodb_router=router,
            embedding_service=embedding,
            advanced_precondition_matcher=advanced_matcher
        )
        
        stats = pipeline.get_stats()
        
        # Should include LLM metrics
        assert "llm_cache_hits" in stats
        assert "llm_cache_hit_rate" in stats
        assert "llm_total_cost_usd" in stats


class TestConfigurationIntegration:
    """Test configuration-based LLM validation enabling/disabling."""
    
    @pytest.mark.asyncio
    async def test_llm_disabled_by_default(self):
        """Test that LLM validation is disabled by default."""
        with patch('src.retrieval.search.get_settings') as mock_settings:
            settings = MagicMock()
            settings.search.enable_llm_validation = False
            mock_settings.return_value = settings
            
            router = MagicMock()
            embedding = MagicMock()
            
            pipeline = SearchPipeline(
                kyrodb_router=router,
                embedding_service=embedding
            )
            
            assert pipeline.advanced_precondition_matcher is None
    
    @pytest.mark.asyncio
    async def test_llm_enabled_via_config(self):
        """Test that LLM validation can be enabled via configuration."""
        with patch('src.retrieval.search.get_settings') as mock_settings:
            settings = MagicMock()
            settings.search.enable_llm_validation = True
            settings.llm.openrouter_api_key = "test_key"
            mock_settings.return_value = settings
            
            router = MagicMock()
            embedding = MagicMock()
            
            # Should auto-initialize advanced matcher with OpenRouter
            pipeline = SearchPipeline(
                kyrodb_router=router,
                embedding_service=embedding
            )
            
            # Verify LLM validation was enabled via configuration
            assert settings.search.enable_llm_validation is True
            assert settings.llm.openrouter_api_key == "test_key"
            assert pipeline.advanced_precondition_matcher is not None
