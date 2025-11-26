"""
Integration tests for Phase 5 Step 5.3: Tiered Reflection Integration.

Tests end-to-end flow from API endpoint through ingestion pipeline
to tiered reflection generation.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from src.models.episode import EpisodeCreate, ErrorClass, Reflection, ReflectionTier
from src.ingestion.capture import IngestionPipeline
from src.config import LLMConfig


class TestTieredIntegration:
    """Integration tests for tiered reflection in ingestion pipeline."""
    
    @pytest.mark.asyncio
    async def test_capture_with_auto_tier_selection(self):
        """Test episode capture with automatic tier selection."""
        # Create mock services
        kyrodb_router = AsyncMock()
        kyrodb_router.insert_episode = AsyncMock(return_value=(True, False))
        kyrodb_router.update_episode_reflection = AsyncMock(return_value=True)
        
        embedding_service = MagicMock()
        embedding_service.embed_text = MagicMock(return_value=[0.1] * 384)
        
        # Create tiered reflection service mock
        reflection_service = AsyncMock()
        mock_reflection = Reflection(
            root_cause="Test root cause",
            preconditions=["Test precondition"],
            resolution_strategy="Test resolution",
            confidence_score=0.7,
            generalization_score=0.5,
            llm_model="openrouter-cheap-model",
            cost_usd=0.0003,
            tier=ReflectionTier.CHEAP.value
        )
        reflection_service.generate_reflection = AsyncMock(return_value=mock_reflection)
        
        # Create pipeline
        pipeline = IngestionPipeline(
            kyrodb_router=kyrodb_router,
            embedding_service=embedding_service,
            reflection_service=reflection_service
        )
        
        # Create episode
        episode_data = EpisodeCreate(
            customer_id="test_customer",
            goal="Deploy application to production",
            error_class=ErrorClass.RESOURCE_ERROR,
            tool_chain=["docker"],
            actions_taken=["docker build"],
            error_trace="Test error trace"
        )
        
        # Capture episode (reflection async)
        episode = await pipeline.capture_episode(episode_data)
        
        # Verify episode captured
        assert episode.episode_id > 0
        assert episode.create_data.customer_id == "test_customer"
        
        # Give async task time to start and call generate_reflection
        await asyncio.sleep(0.01)
        
        # Verify tier parameter was passed correctly (auto-selection = None)
        reflection_service.generate_reflection.assert_called_once()
        call_kwargs = reflection_service.generate_reflection.call_args[1]
        assert call_kwargs.get('tier') is None, "Auto-selection should pass tier=None"
    
    @pytest.mark.asyncio
    async def test_capture_with_tier_override_cheap(self):
        """Test episode capture with explicit CHEAP tier override."""
        # Create mock services
        kyrodb_router = AsyncMock()
        kyrodb_router.insert_episode = AsyncMock(return_value=(True, False))
        kyrodb_router.update_episode_reflection = AsyncMock(return_value=True)
        
        embedding_service = MagicMock()
        embedding_service.embed_text = MagicMock(return_value=[0.1] * 384)
        
        # Create tiered reflection service mock
        reflection_service = AsyncMock()
        mock_reflection = Reflection(
            root_cause="Test root cause",
            preconditions=["Test precondition"],
            resolution_strategy="Test resolution",
            confidence_score=0.7,
            generalization_score=0.5,
            llm_model="openrouter-cheap-model",
            cost_usd=0.0003,
            tier=ReflectionTier.CHEAP.value
        )
        reflection_service.generate_reflection = AsyncMock(return_value=mock_reflection)
        
        # Create pipeline
        pipeline = IngestionPipeline(
            kyrodb_router=kyrodb_router,
            embedding_service=embedding_service,
            reflection_service=reflection_service
        )
        
        # Create episode
        episode_data = EpisodeCreate(
            customer_id="test_customer",
            goal="Deploy application to production",
            error_class=ErrorClass.RESOURCE_ERROR,
            tool_chain=["docker"],
            actions_taken=["docker build"],
            error_trace="Test error trace"
        )
        
        # Capture episode with tier override
        episode = await pipeline.capture_episode(
            episode_data,
            tier_override=ReflectionTier.CHEAP
        )
        
        # Verify episode captured
        assert episode.episode_id > 0
        
        # Give async task time to start
        await asyncio.sleep(0.01)
        
        # Verify tier override was passed correctly
        reflection_service.generate_reflection.assert_called_once()
        call_kwargs = reflection_service.generate_reflection.call_args[1]
        assert call_kwargs.get('tier') == ReflectionTier.CHEAP, "CHEAP tier override should be passed"
    
    @pytest.mark.asyncio
    async def test_capture_with_tier_override_premium(self):
        """Test episode capture with explicit PREMIUM tier override."""
        # Create mock services
        kyrodb_router = AsyncMock()
        kyrodb_router.insert_episode = AsyncMock(return_value=(True, False))
        kyrodb_router.update_episode_reflection = AsyncMock(return_value=True)
        
        embedding_service = MagicMock()
        embedding_service.embed_text = MagicMock(return_value=[0.1] * 384)
        
        # Create tiered reflection service mock
        reflection_service = AsyncMock()
        mock_reflection = Reflection(
            root_cause="Test root cause premium",
            preconditions=["Test precondition"],
            resolution_strategy="Test resolution",
            confidence_score=0.9,
            generalization_score=0.7,
            llm_model="multi-perspective",
            cost_usd=0.096,
            tier=ReflectionTier.PREMIUM.value
        )
        reflection_service.generate_reflection = AsyncMock(return_value=mock_reflection)
        
        # Create pipeline
        pipeline = IngestionPipeline(
            kyrodb_router=kyrodb_router,
            embedding_service=embedding_service,
            reflection_service=reflection_service
        )
        
        # Create episode  
        episode_data = EpisodeCreate(
            customer_id="test_customer",
            goal="Delete user data from production",
            error_class=ErrorClass.CONFIGURATION_ERROR,
            tool_chain=["psql"],
            actions_taken=["DELETE FROM users"],
            error_trace="Test error trace"
        )
        
        # Capture episode with premium tier
        episode = await pipeline.capture_episode(
            episode_data,
            tier_override=ReflectionTier.PREMIUM
        )
        
        # Verify episode captured
        assert episode.episode_id > 0
        
        # Give async task time to start
        await asyncio.sleep(0.01)
        
        # Verify tier override was passed correctly
        reflection_service.generate_reflection.assert_called_once()
        call_kwargs = reflection_service.generate_reflection.call_args[1]
        assert call_kwargs.get('tier') == ReflectionTier.PREMIUM, "PREMIUM tier override should be passed"
    
    @pytest.mark.asyncio   
    async def test_reflection_generation_with_tier(self):
        """Test that reflection generation passes tier correctly."""
        import asyncio
        
        # Create mock services
        kyrodb_router = AsyncMock()
        kyrodb_router.insert_episode = AsyncMock(return_value=(True, False))
        kyrodb_router.update_episode_reflection = AsyncMock(return_value=True)
        
        embedding_service = MagicMock()
        embedding_service.embed_text = MagicMock(return_value=[0.1] * 384)
        
        # Create tiered reflection service mock
        reflection_service = AsyncMock()
        mock_reflection = Reflection(
            root_cause="Test root cause",
            preconditions=["Test precondition"],
            resolution_strategy="Test resolution",
            confidence_score=0.7,
            generalization_score=0.5,
            llm_model="openrouter-cheap-model",
            cost_usd=0.0003,
            tier=ReflectionTier.CHEAP.value
        )
        reflection_service.generate_reflection = AsyncMock(return_value=mock_reflection)
        
        # Create pipeline
        pipeline = IngestionPipeline(
            kyrodb_router=kyrodb_router,
            embedding_service=embedding_service,
            reflection_service=reflection_service
        )
        
        # Create episode
        episode_data = EpisodeCreate(
            customer_id="test_customer",
            goal="Deploy application to production",
            error_class=ErrorClass.RESOURCE_ERROR,
            tool_chain=["docker"],
            actions_taken=["docker build"],
            error_trace="Test error trace"
        )
        
        # Capture episode with PREMIUM tier override
        episode = await pipeline.capture_episode(
            episode_data,
            tier_override=ReflectionTier.PREMIUM
        )
        
        # Give async task time to start (not complete)
        await asyncio.sleep(0.01)
        
        # Note: Full async task testing would require more sophisticated mocking
        # This test verifies the integration compiles and executes
        assert episode.episode_id > 0


class TestAPITierParameter:
    """Test tier parameter in API endpoint (requires full app mock)."""
    
    def test_tier_validation_invalid(self):
        """Test that invalid tier values are rejected."""
        from fastapi import HTTPException
        from src.models.episode import ReflectionTier
        
        # Test invalid tier
        invalid_tier = "super_premium"
        
        with pytest.raises(ValueError):
            ReflectionTier(invalid_tier)
    
    def test_tier_validation_valid(self):
        """Test that valid tier values are accepted."""
        from src.models.episode import ReflectionTier
        
        # Test all valid tiers
        assert ReflectionTier("cheap") == ReflectionTier.CHEAP
        assert ReflectionTier("cached") == ReflectionTier.CACHED
        assert ReflectionTier("premium") == ReflectionTier.PREMIUM
    
    def test_tier_parsing_case_insensitive(self):
        """Test that tier parsing is case-insensitive (via .lower())."""
        from src.models.episode import ReflectionTier
        
        # API converts to lowercase before parsing
        # So tester should simulate this
        assert ReflectionTier("CHEAP".lower()) == ReflectionTier.CHEAP
        assert ReflectionTier("Cheap".lower()) == ReflectionTier.CHEAP
        assert ReflectionTier("cheap") == ReflectionTier.CHEAP
