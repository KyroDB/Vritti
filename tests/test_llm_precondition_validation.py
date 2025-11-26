"""
Unit tests for LLM-based precondition validation.

Tests semantic compatibility checking with mocked OpenRouter responses.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from src.retrieval.preconditions import AdvancedPreconditionMatcher
from src.models.episode import Episode, EpisodeCreate, ErrorClass, Reflection
from src.models.search import PreconditionCheckResult


class TestAdvancedPreconditionMatcher:
    """Test LLM-based precondition validation."""
    
    @pytest.fixture
    def matcher_without_llm(self):
        """Create matcher with LLM disabled."""
        return AdvancedPreconditionMatcher(
            openrouter_api_key=None,
            enable_llm=False
        )
    
    @pytest.fixture
    def sample_episode(self):
        """Create a sample episode."""
        return Episode(
            episode_id=1,
            create_data=EpisodeCreate(
                goal="Delete files older than 7 days",
                error_class=ErrorClass.CONFIGURATION_ERROR,
                tool_chain=["bash"],
                actions_taken=["rm -rf /tmp/*"],
                error_trace="Error: Deleted wrong files"
            ),
            reflection=Reflection(
                root_cause="Deleted wrong files",
                resolution_strategy="Use find with -mtime instead",
                preconditions=["Has files to delete"],
                environment_factors=[],
                affected_components=[],
                generalization_score=0.8,
                confidence_score=0.9
            )
        )
    
    @pytest.mark.asyncio
    async def test_llm_disabled_fallback(self, matcher_without_llm, sample_episode):
        """Test that matcher works without LLM."""
        result = await matcher_without_llm.check_preconditions_with_llm(
            candidate_episode=sample_episode,
            current_query="Delete old files",
            current_state={},
            threshold=0.3,  # Lower threshold since we have preconditions
            similarity_score=0.9
        )
        
        assert matcher_without_llm.stats["llm_calls"] == 0
    
    @pytest.mark.asyncio
    async def test_low_similarity_skips_llm(self, matcher_without_llm, sample_episode):
        """Test that LLM validation skipped for low similarity."""
        result = await matcher_without_llm.check_preconditions_with_llm(
            candidate_episode=sample_episode,
            current_query="Delete old files",
            current_state={},
            similarity_score=0.7
        )
        
        assert matcher_without_llm.stats["llm_calls"] == 0
    
    def test_input_sanitization(self, matcher_without_llm):
        """Test that inputs are sanitized."""
        long_text = "x" * 1000
        sanitized = matcher_without_llm._sanitize_input(long_text)
        
        assert len(sanitized) <= matcher_without_llm.MAX_QUERY_LENGTH + 3
        assert "..." in sanitized
        
        malicious = "```python\nimport os\nos.system('rm -rf /')\n```"
        sanitized = matcher_without_llm._sanitize_input(malicious)
        assert "```" not in sanitized
    
    def test_stats_tracking(self, matcher_without_llm):
        """Test that statistics are tracked correctly."""
        stats = matcher_without_llm.get_stats()
        
        assert "llm_calls" in stats
        assert "llm_rejections" in stats
        assert "cache_hits" in stats
        assert "timeouts" in stats
        assert "errors" in stats
        assert "total_cost_usd" in stats
        assert "cache_size" in stats
        assert "cache_hit_rate" in stats
    
    def test_cache_key_generation(self, matcher_without_llm):
        """Test cache key generation."""
        key1 = matcher_without_llm._get_cache_key("goal1", "query1")
        key2 = matcher_without_llm._get_cache_key("goal1", "query1")
        key3 = matcher_without_llm._get_cache_key("goal2", "query1")
        
        assert key1 == key2
        assert key1 != key3
    
    def test_cache_operations(self, matcher_without_llm):
        """Test cache add/get operations."""
        key = "test_key"
        
        assert matcher_without_llm._get_from_cache(key) is None
        
        matcher_without_llm._add_to_cache(key, True)
        
        assert matcher_without_llm._get_from_cache(key) is True
    
    @pytest.mark.asyncio
    async def test_basic_precondition_checking(self, matcher_without_llm):
        """Test basic precondition checking works."""
        episode = Episode(
            episode_id=1,
            create_data=EpisodeCreate(
                goal="Deploy to production",
                error_class=ErrorClass.CONFIGURATION_ERROR,
                tool_chain=["kubectl"],
                actions_taken=["kubectl apply"],
                error_trace="Error: Wrong configuration"
            ),
            reflection=Reflection(
                root_cause="Wrong config",
                resolution_strategy="Check environment",
                preconditions=["Using tool: kubectl"],
                environment_factors=[],
                affected_components=[],
                generalization_score=0.8,
                confidence_score=0.9
            )
        )
        
        result = await matcher_without_llm.check_preconditions_with_llm(
            candidate_episode=episode,
            current_query="Deploy app",
            current_state={"tool": "kubectl"},
            similarity_score=0.5
        )
        
        assert result.matched is True


class TestLLMValidationWithMocks:
    """Test LLM validation with mocked OpenRouter responses."""
    
    @pytest.fixture
    def sample_episode(self):
        """Create a sample episode."""
        return Episode(
            episode_id=1,
            create_data=EpisodeCreate(
                goal="Delete files older than 7 days",
                error_class=ErrorClass.CONFIGURATION_ERROR,
                tool_chain=["find"],
                actions_taken=["find . -mtime +7 -delete"],
                error_trace="Error: Deleted system files accidentally"
            ),
            reflection=Reflection(
                root_cause="Deleted system files accidentally",
                resolution_strategy="Add exclusions for system directories",
                preconditions=[],
                environment_factors=[],
                affected_components=[],
                generalization_score=0.8,
                confidence_score=0.9
            )
        )
    
    def _mock_openrouter_response(self, content: str):
        """Create a mock httpx response for OpenRouter."""
        response = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        response.raise_for_status = MagicMock()
        return response
    
    @pytest.mark.asyncio
    async def test_llm_rejects_semantic_negation(self, sample_episode):
        """Test that LLM correctly rejects semantically opposite queries."""
        matcher = AdvancedPreconditionMatcher(openrouter_api_key="test_key", enable_llm=True)
        
        mock_response = self._mock_openrouter_response(
            '{"compatible": false, "confidence": 0.95, "reason": "Opposite meaning due to EXCEPT keyword"}'
        )
        
        with patch('src.retrieval.preconditions.httpx.AsyncClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            
            result = await matcher.check_preconditions_with_llm(
                candidate_episode=sample_episode,
                current_query="Delete files EXCEPT those older than 7 days",
                current_state={},
                similarity_score=0.95
            )
            
            assert result.matched is False
            assert "Semantically incompatible" in result.explanation
            assert matcher.stats["llm_rejections"] == 1
    
    @pytest.mark.asyncio
    async def test_llm_accepts_compatible_query(self, sample_episode):
        """Test that LLM accepts semantically compatible queries."""
        matcher = AdvancedPreconditionMatcher(openrouter_api_key="test_key", enable_llm=True)
        # Clear cache using public method to ensure test isolation
        AdvancedPreconditionMatcher.clear_cache()
        
        mock_response = self._mock_openrouter_response(
            '{"compatible": true, "confidence": 0.9, "reason": "Same goal, different wording"}'
        )
        
        with patch('src.retrieval.preconditions.httpx.AsyncClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            
            result = await matcher.check_preconditions_with_llm(
                candidate_episode=sample_episode,
                current_query="Remove files older than 7 days - accept test",  # Unique
                current_state={},
                similarity_score=0.92
            )
            
            assert result.matched is True
            assert matcher.stats["llm_calls"] >= 1
    
    @pytest.mark.asyncio
    async def test_llm_rejects_environment_mismatch(self):
        """Test that LLM catches environment differences."""
        episode = Episode(
            episode_id=1,
            create_data=EpisodeCreate(
                goal="Deploy to production",
                error_class=ErrorClass.CONFIGURATION_ERROR,
                tool_chain=["kubectl"],
                actions_taken=["kubectl apply -f prod.yaml"],
                error_trace="Error: Deployment failed"
            ),
            reflection=Reflection(
                root_cause="Wrong config",
                resolution_strategy="Check environment",
                preconditions=[],
                environment_factors=[],
                affected_components=[],
                generalization_score=0.8,
                confidence_score=0.9
            )
        )
        
        matcher = AdvancedPreconditionMatcher(openrouter_api_key="test_key", enable_llm=True)
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"compatible": false, "confidence": 0.9, "reason": "Different environments: production vs staging"}'}}]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch('src.retrieval.preconditions.httpx.AsyncClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            
            result = await matcher.check_preconditions_with_llm(
                candidate_episode=episode,
                current_query="Deploy to staging",
                current_state={},
                similarity_score=0.95
            )
            
            assert result.matched is False
            assert matcher.stats["llm_rejections"] >= 1
    
    @pytest.mark.asyncio
    async def test_llm_validation_caching(self, sample_episode):
        """Test that LLM results are properly cached."""
        matcher = AdvancedPreconditionMatcher(openrouter_api_key="test_key", enable_llm=True)
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"compatible": true, "confidence": 0.9, "reason": "Compatible"}'}}]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch('src.retrieval.preconditions.httpx.AsyncClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            
            await matcher.check_preconditions_with_llm(
                candidate_episode=sample_episode,
                current_query="Delete old files",
                current_state={},
                similarity_score=0.9
            )
            
            await matcher.check_preconditions_with_llm(
                candidate_episode=sample_episode,
                current_query="Delete old files",
                current_state={},
                similarity_score=0.9
            )
            
            assert matcher.stats["cache_hits"] >= 1
    
    @pytest.mark.asyncio
    async def test_llm_low_confidence_rejection(self, sample_episode):
        """Test that low LLM confidence results are rejected."""
        matcher = AdvancedPreconditionMatcher(openrouter_api_key="test_key", enable_llm=True)
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"compatible": true, "confidence": 0.5, "reason": "Uncertain"}'}}]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch('src.retrieval.preconditions.httpx.AsyncClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            
            result = await matcher.check_preconditions_with_llm(
                candidate_episode=sample_episode,
                current_query="Delete files",
                current_state={},
                similarity_score=0.9
            )
            
            assert result.matched is False
    
    @pytest.mark.asyncio
    async def test_llm_malformed_response_fallback(self, sample_episode):
        """Test graceful handling of malformed LLM responses."""
        matcher = AdvancedPreconditionMatcher(openrouter_api_key="test_key", enable_llm=True)
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Invalid JSON response"}}]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch('src.retrieval.preconditions.httpx.AsyncClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            
            result = await matcher.check_preconditions_with_llm(
                candidate_episode=sample_episode,
                current_query="Delete old files",
                current_state={},
                similarity_score=0.9
            )
            
            # Falls back to accept on parse error
            assert result.matched is True
    
    @pytest.mark.asyncio
    async def test_llm_api_error_fallback(self, sample_episode):
        """Test graceful handling of LLM API errors."""
        matcher = AdvancedPreconditionMatcher(openrouter_api_key="test_key", enable_llm=True)
        # Clear cache using public method to ensure test isolation
        AdvancedPreconditionMatcher.clear_cache()
        
        with patch('src.retrieval.preconditions.httpx.AsyncClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.post = AsyncMock(side_effect=Exception("API Error"))
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            
            result = await matcher.check_preconditions_with_llm(
                candidate_episode=sample_episode,
                current_query="Delete old files - unique test",  # Unique query to avoid cached results
                current_state={},
                similarity_score=0.9
            )
            
            assert result.matched is True
            assert matcher.stats["errors"] >= 1


class TestTimeNegationScenarios:
    """Test real-world time negation scenarios."""
    
    @pytest.mark.asyncio
    async def test_time_negation_without_llm(self):
        """Demonstrate that basic matcher cannot catch time negation."""
        episode = Episode(
            episode_id=1,
            create_data=EpisodeCreate(
                goal="Delete files older than 7 days",
                error_class=ErrorClass.CONFIGURATION_ERROR,
                tool_chain=["find"],
                actions_taken=["find . -mtime +7 -delete"],
                error_trace="Error: Deleted system files"
            ),
            reflection=Reflection(
                root_cause="Deleted system files accidentally",
                resolution_strategy="Add exclusions for system directories",
                preconditions=[],
                environment_factors=[],
                affected_components=[],
                generalization_score=0.8,
                confidence_score=0.9
            )
        )
        
        matcher = AdvancedPreconditionMatcher(enable_llm=False)
        
        result = await matcher.check_preconditions_with_llm(
            candidate_episode=episode,
            current_query="Delete files EXCEPT those older than 7 days",
            current_state={},
            similarity_score=0.95
        )
        
        # Without LLM, basic matcher cannot detect the semantic negation
        assert result.matched is True
    
    @pytest.mark.asyncio
    async def test_time_negation_with_llm_validation(self):
        """Test that LLM correctly detects and rejects time negation."""
        episode = Episode(
            episode_id=1,
            create_data=EpisodeCreate(
                goal="Delete files older than 7 days",
                error_class=ErrorClass.CONFIGURATION_ERROR,
                tool_chain=["find"],
                actions_taken=["find . -mtime +7 -delete"],
                error_trace="Error: Deleted system files"
            ),
            reflection=Reflection(
                root_cause="Deleted system files accidentally",
                resolution_strategy="Add exclusions for system directories",
                preconditions=[],
                environment_factors=[],
                affected_components=[],
                generalization_score=0.8,
                confidence_score=0.9
            )
        )
        
        matcher = AdvancedPreconditionMatcher(openrouter_api_key="test_key", enable_llm=True)
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"compatible": false, "confidence": 0.98, "reason": "Opposite time conditions: older than vs except older than"}'}}]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch('src.retrieval.preconditions.httpx.AsyncClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
            
            result = await matcher.check_preconditions_with_llm(
                candidate_episode=episode,
                current_query="Delete files EXCEPT those older than 7 days",
                current_state={},
                similarity_score=0.95
            )
            
            assert result.matched is False
            assert "Semantically incompatible" in result.explanation
            assert matcher.stats["llm_rejections"] >= 1
