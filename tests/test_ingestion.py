"""
Integration tests for episode ingestion pipeline.

Tests end-to-end ingestion flow:
- PII redaction
- ID generation
- Embedding generation
- KyroDB storage
- Reflection generation
"""

import pytest

from src.ingestion.capture import IngestionPipeline
from src.ingestion.embedding import EmbeddingService
from src.kyrodb.router import KyroDBRouter
from src.models.episode import Episode, EpisodeCreate


class TestIngestionPipeline:
    """Test suite for episode ingestion pipeline."""

    @pytest.mark.asyncio
    async def test_capture_episode_basic(
        self,
        ingestion_pipeline: IngestionPipeline,
        mock_kyrodb_router: KyroDBRouter,
        mock_embedding_service: EmbeddingService,
        sample_episode_create: EpisodeCreate,
    ):
        """Test basic episode capture without reflection."""
        # Capture episode
        episode = await ingestion_pipeline.capture_episode(
            episode_data=sample_episode_create,
            generate_reflection=False,
        )

        # Assertions
        assert episode.episode_id > 0
        assert episode.create_data == sample_episode_create
        assert episode.reflection is None
        assert episode.retrieval_count == 0
        assert episode.created_at is not None

        # Verify KyroDB insert was called
        mock_kyrodb_router.insert_episode.assert_called_once()

        # Verify embedding generation was called
        mock_embedding_service.embed_text_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_pii_redaction(
        self,
        ingestion_pipeline: IngestionPipeline,
        mock_kyrodb_router: KyroDBRouter,
        mock_embedding_service: EmbeddingService,
    ):
        """Test PII redaction during ingestion."""
        # Create episode with PII
        episode_data = EpisodeCreate(
            customer_id="test-customer",  # Multi-tenancy required
            goal="Deploy application to production environment",
            tool_chain=["kubectl"],
            actions_taken=["attempted deployment action"],
            error_trace=(
                "Error connecting to database at user:password@10.0.1.100\n"
                "Contact admin@example.com for help\n"
                "API key: sk-1234567890abcdef1234567890abcdef1234567890abcdef"
            ),
            error_class="network_error",
        )

        episode = await ingestion_pipeline.capture_episode(
            episode_data=episode_data,
            generate_reflection=False,
        )

        # Verify PII was redacted using standard placeholder format
        # Presidio NER uses [EMAIL_ADDRESS], regex fallback uses [EMAIL]
        # See: src/utils/pii_redaction.py:redact_all() for the two-stage redaction
        assert (
            "[EMAIL_ADDRESS]" in episode.create_data.error_trace
            or "[EMAIL]" in episode.create_data.error_trace
        ), f"Expected email PII placeholder in redacted trace, got: {episode.create_data.error_trace}"
        assert "[API_KEY]" in episode.create_data.error_trace
        assert "sk-1234567890" not in episode.create_data.error_trace
        assert "admin@example.com" not in episode.create_data.error_trace

    @pytest.mark.asyncio
    async def test_id_generation_uniqueness(
        self,
        ingestion_pipeline: IngestionPipeline,
        mock_kyrodb_router: KyroDBRouter,
        mock_embedding_service: EmbeddingService,
        sample_episode_create: EpisodeCreate,
    ):
        """Test that episode IDs are unique."""
        # Capture multiple episodes
        episode1 = await ingestion_pipeline.capture_episode(
            episode_data=sample_episode_create,
            generate_reflection=False,
        )

        episode2 = await ingestion_pipeline.capture_episode(
            episode_data=sample_episode_create,
            generate_reflection=False,
        )

        # Verify IDs are unique
        assert episode1.episode_id != episode2.episode_id
        assert episode1.episode_id > 0
        assert episode2.episode_id > 0

    @pytest.mark.asyncio
    async def test_multi_modal_embedding(
        self,
        ingestion_pipeline: IngestionPipeline,
        mock_kyrodb_router: KyroDBRouter,
        mock_embedding_service: EmbeddingService,
    ):
        """Test multi-modal embedding generation with screenshot."""
        episode_data = EpisodeCreate(
            customer_id="test-customer",  # Multi-tenancy required
            goal="Deploy with screenshot attachment",
            tool_chain=["kubectl"],
            actions_taken=["action1"],
            error_trace="Error occurred during deployment",
            error_class="unknown",
            screenshot_base64="ZmFrZS1pbWFnZS1ieXRlcw==",
        )

        _episode = await ingestion_pipeline.capture_episode(
            episode_data=episode_data,
            generate_reflection=False,
        )

        # Verify both text and image embeddings were generated
        mock_embedding_service.embed_text_async.assert_called_once()
        mock_embedding_service.embed_image_bytes_async.assert_called_once()

        # Verify KyroDB insert was called with both embeddings
        call_args = mock_kyrodb_router.insert_episode.call_args
        assert call_args.kwargs["text_embedding"] is not None
        assert call_args.kwargs["image_embedding"] is not None

    @pytest.mark.asyncio
    async def test_ingestion_stats(
        self,
        ingestion_pipeline: IngestionPipeline,
        mock_kyrodb_router: KyroDBRouter,
        mock_embedding_service: EmbeddingService,
        sample_episode_create: EpisodeCreate,
    ):
        """Test ingestion pipeline statistics tracking."""
        # Initial stats
        stats = ingestion_pipeline.get_stats()
        assert stats["total_ingested"] == 0
        assert stats["total_failures"] == 0

        # Capture episode
        await ingestion_pipeline.capture_episode(
            episode_data=sample_episode_create,
            generate_reflection=False,
        )

        # Verify stats updated
        stats = ingestion_pipeline.get_stats()
        assert stats["total_ingested"] == 1
        assert stats["total_failures"] == 0
        assert stats["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_metadata_serialization(
        self,
        ingestion_pipeline: IngestionPipeline,
        mock_kyrodb_router: KyroDBRouter,
        mock_embedding_service: EmbeddingService,
        sample_episode_create: EpisodeCreate,
    ):
        """Test episode metadata serialization for KyroDB."""
        episode = await ingestion_pipeline.capture_episode(
            episode_data=sample_episode_create,
            generate_reflection=False,
        )

        # Verify metadata format
        metadata = episode.to_metadata_dict()
        assert "episode_type" in metadata
        assert "error_class" in metadata
        assert "tool" in metadata
        assert "severity" in metadata
        assert "timestamp" in metadata
        assert "episode_json" in metadata
        assert "tags" in metadata

        # Verify all values are strings
        assert all(isinstance(v, str) for v in metadata.values())

        # Verify tags serialization
        assert metadata["tags"] == "production,deployment,critical"


class TestBulkIngestion:
    """Test suite for bulk episode ingestion."""

    @pytest.mark.asyncio
    async def test_bulk_capture(
        self,
        ingestion_pipeline: IngestionPipeline,
        mock_kyrodb_router: KyroDBRouter,
        mock_embedding_service: EmbeddingService,
        sample_episode_create: EpisodeCreate,
    ):
        """Test bulk episode capture."""
        # Create multiple episodes
        episodes_data = [sample_episode_create for _ in range(5)]

        # Bulk capture
        episodes = await ingestion_pipeline.bulk_capture(
            episodes=episodes_data,
            generate_reflections=False,
        )

        # Assertions
        assert len(episodes) == 5
        assert all(isinstance(ep, Episode) for ep in episodes)
        assert len({ep.episode_id for ep in episodes}) == 5  # All unique IDs

        # Verify stats
        stats = ingestion_pipeline.get_stats()
        assert stats["total_ingested"] == 5
