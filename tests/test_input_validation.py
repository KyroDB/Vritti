"""
Tests for input validation and error handling improvements.

Tests cover:
1. Config module logging import usage
2. Search fetch_k validation and bounds checking
3. Dead letter queue configuration and file rotation
"""

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import ServiceConfig, Settings, get_settings
from src.gating.service import GatingService
from src.ingestion.capture import IngestionPipeline
from src.models.episode import Episode, EpisodeCreate, ErrorClass, Reflection, ReflectionTier
from src.models.search import SearchRequest, SearchResponse
from src.retrieval.search import SearchPipeline


class TestConfigLoggingImport:
    """Test that config.py properly uses logging module."""

    def test_validate_configuration_uses_logging(self):
        """Test that validate_configuration can use logging without local import."""
        settings = Settings()
        
        # Should not raise any import errors
        settings.validate_configuration()
        
        # Test specific validation warnings
        settings_no_llm = Settings(llm={"openrouter_api_key": ""})
        settings_no_llm.validate_configuration()  # Should warn about no LLM key


class TestSearchFetchKValidation:
    """Test fetch_k validation in search pipeline."""

    @pytest.fixture
    def mock_search_pipeline(self):
        """Create a mock search pipeline for testing."""
        mock_router = AsyncMock()
        mock_router.search_text.return_value = AsyncMock(results=[])
        
        mock_embedding = MagicMock()
        mock_embedding.embed_text.return_value = [0.1] * 384
        
        return SearchPipeline(
            kyrodb_router=mock_router,
            embedding_service=mock_embedding
        )

    @pytest.mark.asyncio
    async def test_fetch_k_within_bounds(self, mock_search_pipeline):
        """Test that normal k values work correctly."""
        request = SearchRequest(
            customer_id="test_customer",
            goal="Test query",
            k=10  # Normal value
        )
        
        response = await mock_search_pipeline.search(request)
        
        # Should complete without errors
        assert response is not None
        assert response.total_returned >= 0

    @pytest.mark.asyncio
    async def test_fetch_k_capped_at_max(self, mock_search_pipeline):
        """Test that very large k values are capped to prevent memory issues."""
        settings = get_settings()
        max_k = settings.search.max_k
        
        # Request k that would result in fetch_k > max_k * 2
        request = SearchRequest(
            customer_id="test_customer",
            goal="Test query",
            k=max_k + 50  # This should be capped
        )
        
        # Mock the _fetch_candidates call to check the actual k used
        original_fetch = mock_search_pipeline._fetch_candidates
        called_with_k = []
        
        async def track_fetch_k(*args, **kwargs):
            called_with_k.append(kwargs.get('k'))
            return await original_fetch(*args, **kwargs)
        
        mock_search_pipeline._fetch_candidates = track_fetch_k
        
        response = await mock_search_pipeline.search(request)
        
        # Verify k was capped
        assert len(called_with_k) > 0
        assert called_with_k[0] <= max_k * 2

    @pytest.mark.asyncio
    async def test_fetch_k_warning_logged(self, mock_search_pipeline, caplog):
        """Test that a warning is logged when k is capped."""
        settings = get_settings()
        max_k = settings.search.max_k
        
        request = SearchRequest(
            customer_id="test_customer",
            goal="Test query",
            k=max_k + 100  # Definitely over the limit
        )
        
        with caplog.at_level("WARNING"):
            await mock_search_pipeline.search(request)
        
        # Check that a warning was logged about capping
        warning_messages = [rec.message for rec in caplog.records if rec.levelname == "WARNING"]
        assert any("capping at max_fetch_k" in msg for msg in warning_messages)


class TestDeadLetterQueueConfiguration:
    """Test dead letter queue configuration and file rotation."""

    @pytest.fixture
    def temp_dlq_dir(self):
        """Create a temporary directory for dead letter queue testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_ingestion_pipeline(self, temp_dlq_dir):
        """Create a mock ingestion pipeline with custom DLQ path."""
        mock_router = AsyncMock()
        mock_embedding = MagicMock()
        
        # Create custom settings with temp DLQ path
        with patch('src.ingestion.capture.get_settings') as mock_settings:
            settings = Settings()
            settings.service.dead_letter_queue_path = str(temp_dlq_dir / "test_dlq.log")
            settings.service.dead_letter_queue_max_size_mb = 1  # Small size for testing
            mock_settings.return_value = settings
            
            pipeline = IngestionPipeline(
                kyrodb_router=mock_router,
                embedding_service=mock_embedding
            )
            
            yield pipeline

    @pytest.mark.asyncio
    async def test_dlq_uses_configured_path(self, mock_ingestion_pipeline, temp_dlq_dir):
        """Test that DLQ uses the configured path instead of hardcoded one."""
        dlq_path = temp_dlq_dir / "test_dlq.log"
        
        # Create a mock reflection
        reflection = Reflection(
            root_cause="Test failure",
            preconditions=["test precondition"],
            resolution_strategy="Test resolution",
            environment_factors=["test env"],
            affected_components=["test component"],
            generalization_score=0.8,
            confidence_score=0.9,
            llm_model="test-model",
            generated_at=datetime.now(UTC),
            cost_usd=0.001,
            generation_latency_ms=100.0,
            tier=ReflectionTier.CHEAP
        )
        
        # Log to DLQ
        await mock_ingestion_pipeline._log_to_dead_letter_queue(
            episode_id=12345,
            customer_id="test_customer",
            reflection=reflection,
            failure_reason="test_failure"
        )
        
        # Verify file was created at configured path
        assert dlq_path.exists()
        
        # Verify content is valid JSON
        with open(dlq_path) as f:
            line = f.readline()
            entry = json.loads(line)
            assert entry["episode_id"] == 12345
            assert entry["customer_id"] == "test_customer"
            assert entry["failure_reason"] == "test_failure"

    @pytest.mark.asyncio
    async def test_dlq_file_rotation(self, mock_ingestion_pipeline, temp_dlq_dir):
        """Test that DLQ file is rotated when size exceeds limit."""
        dlq_path = temp_dlq_dir / "test_dlq.log"
        
        # Create a large initial file (over 1MB limit)
        dlq_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dlq_path, 'w') as f:
            # Write > 1MB of data
            f.write("x" * (1024 * 1024 + 100))
        
        original_size = dlq_path.stat().st_size
        assert original_size > 1024 * 1024
        
        # Create a mock reflection
        reflection = Reflection(
            root_cause="Test failure",
            preconditions=["test"],
            resolution_strategy="Test",
            environment_factors=["test"],
            affected_components=["test"],
            generalization_score=0.8,
            confidence_score=0.9,
            llm_model="test-model",
            generated_at=datetime.now(UTC),
            cost_usd=0.001,
            generation_latency_ms=100.0,
            tier=ReflectionTier.CHEAP
        )
        
        # Log to DLQ - should trigger rotation
        await mock_ingestion_pipeline._log_to_dead_letter_queue(
            episode_id=99999,
            customer_id="test_customer",
            reflection=reflection,
            failure_reason="rotation_test"
        )
        
        # Check that original file was rotated (should have timestamp suffix)
        rotated_files = list(temp_dlq_dir.glob("test_dlq.*.log"))
        assert len(rotated_files) > 0, "File should have been rotated"
        
        # Check that new file was created and is small
        assert dlq_path.exists()
        new_size = dlq_path.stat().st_size
        assert new_size < original_size, "New file should be smaller after rotation"


class TestGatingServiceDocumentation:
    """Test that gating service properly documents unused parameters."""

    def test_unused_parameters_documented(self):
        """Test that _determine_gating_recommendation has proper documentation."""
        mock_search = MagicMock()
        mock_router = AsyncMock()
        
        service = GatingService(mock_search, mock_router)
        
        # Get the method
        method = service._determine_gating_recommendation
        
        # Check docstring mentions the unused parameters
        docstring = method.__doc__
        assert docstring is not None
        assert "_proposed_action" in docstring
        assert "_current_state" in docstring
        assert "TODO" in docstring or "unused" in docstring.lower()
        
        # Verify the method signature still has the parameters (for future use)
        import inspect
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        assert "_proposed_action" in params
        assert "_current_state" in params
