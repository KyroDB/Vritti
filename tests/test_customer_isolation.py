"""
Integration tests for multi-tenant customer isolation.

CRITICAL SECURITY TESTS:
Verifies zero data leakage between customers by testing:
- Customer A cannot access Customer B's episodes
- Search is scoped to customer namespace
- Insert uses customer namespace
- Validation fails if customer_id is missing

These tests validate the "no leakage" security mandate.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from src.ingestion.capture import IngestionPipeline
from src.ingestion.embedding import EmbeddingService
from src.kyrodb.router import KyroDBRouter, get_namespaced_collection
from src.models.episode import EpisodeCreate
from src.models.search import SearchRequest
from src.retrieval.search import SearchPipeline


class TestNamespaceIsolation:
    """Test suite for customer namespace isolation."""

    def test_get_namespaced_collection(self):
        """Test namespace generation for customer isolation."""
        # Valid namespace
        namespace = get_namespaced_collection("acme-corp", "failures")
        assert namespace == "acme-corp:failures"

        # Another customer
        namespace = get_namespaced_collection("techco", "failures")
        assert namespace == "techco:failures"

        # Different collection
        namespace = get_namespaced_collection("acme-corp", "skills")
        assert namespace == "acme-corp:skills"

    def test_get_namespaced_collection_validation(self):
        """Test namespace generation fails with empty customer_id."""
        # Empty customer_id should raise ValueError
        with pytest.raises(ValueError, match="customer_id is required"):
            get_namespaced_collection("", "failures")

        with pytest.raises(ValueError, match="customer_id is required"):
            get_namespaced_collection(None, "failures")


class TestIngestionIsolation:
    """Test suite for ingestion pipeline customer isolation."""

    @pytest.mark.asyncio
    async def test_insert_with_customer_namespace(
        self,
        ingestion_pipeline: IngestionPipeline,
        mock_kyrodb_router: KyroDBRouter,
        mock_embedding_service: EmbeddingService,
    ):
        """Test that episodes are inserted with customer-namespaced collection."""
        # Create episode for customer-a
        episode_data = EpisodeCreate(
            customer_id="customer-a",
            goal="Deploy application to production environment",
            tool_chain=["kubectl"],
            actions_taken=["action1"],
            error_trace="Error occurred during deployment process",
            error_class="unknown",
        )

        episode = await ingestion_pipeline.capture_episode(
            episode_data=episode_data,
            generate_reflection=False,
        )

        # Verify insert_episode was called with customer_id
        call_args = mock_kyrodb_router.insert_episode.call_args
        assert call_args.kwargs["customer_id"] == "customer-a"
        assert call_args.kwargs["collection"] == "failures"

        # Verify episode has correct customer_id
        assert episode.create_data.customer_id == "customer-a"

    @pytest.mark.asyncio
    async def test_insert_fails_without_customer_id(
        self,
        ingestion_pipeline: IngestionPipeline,
        mock_kyrodb_router: KyroDBRouter,
        mock_embedding_service: EmbeddingService,
    ):
        """Test that ingestion fails if customer_id is missing (security enforcement)."""
        # Create episode WITHOUT customer_id
        episode_data = EpisodeCreate(
            customer_id=None,  # Missing customer_id
            goal="Deploy application to production environment",
            tool_chain=["kubectl"],
            actions_taken=["action1"],
            error_trace="Error occurred during deployment process",
            error_class="unknown",
        )

        # Should raise ValueError due to multi-tenancy violation
        with pytest.raises(ValueError, match="customer_id is required"):
            await ingestion_pipeline.capture_episode(
                episode_data=episode_data,
                generate_reflection=False,
            )

    @pytest.mark.asyncio
    async def test_multiple_customers_different_namespaces(
        self,
        ingestion_pipeline: IngestionPipeline,
        mock_kyrodb_router: KyroDBRouter,
        mock_embedding_service: EmbeddingService,
    ):
        """Test that different customers use different namespaces."""
        # Customer A episode
        episode_a = EpisodeCreate(
            customer_id="customer-a",
            goal="Deploy customer A application to production",
            tool_chain=["kubectl"],
            actions_taken=["action1"],
            error_trace="Error occurred in customer A deployment",
            error_class="unknown",
        )

        # Customer B episode
        episode_b = EpisodeCreate(
            customer_id="customer-b",
            goal="Deploy customer B application to production",
            tool_chain=["docker"],
            actions_taken=["action1"],
            error_trace="Error occurred in customer B deployment",
            error_class="unknown",
        )

        # Ingest both
        await ingestion_pipeline.capture_episode(episode_data=episode_a, generate_reflection=False)
        await ingestion_pipeline.capture_episode(episode_data=episode_b, generate_reflection=False)

        # Verify both called insert_episode but with different customer_ids
        assert mock_kyrodb_router.insert_episode.call_count == 2

        # Get call arguments
        calls = mock_kyrodb_router.insert_episode.call_args_list

        # First call should be customer-a
        assert calls[0].kwargs["customer_id"] == "customer-a"

        # Second call should be customer-b
        assert calls[1].kwargs["customer_id"] == "customer-b"


class TestSearchIsolation:
    """Test suite for search pipeline customer isolation."""

    @pytest.mark.asyncio
    async def test_search_with_customer_namespace(
        self,
        mock_kyrodb_router: KyroDBRouter,
        mock_embedding_service: EmbeddingService,
    ):
        """Test that search is scoped to customer namespace."""
        pipeline = SearchPipeline(
            kyrodb_router=mock_kyrodb_router,
            embedding_service=mock_embedding_service,
        )

        # Search request for customer-a
        request = SearchRequest(
            customer_id="customer-a",
            goal="test query",
            collection="failures",
            k=5,
        )

        response = await pipeline.search(request)

        # Verify search completed successfully
        assert response is not None

        # Verify search_text was called with customer_id
        call_args = mock_kyrodb_router.search_text.call_args
        assert call_args.kwargs["customer_id"] == "customer-a"
        assert call_args.kwargs["collection"] == "failures"

    @pytest.mark.asyncio
    async def test_search_fails_without_customer_id(
        self,
        mock_kyrodb_router: KyroDBRouter,
        mock_embedding_service: EmbeddingService,
    ):
        """Test that search fails if customer_id is missing (security enforcement)."""
        pipeline = SearchPipeline(
            kyrodb_router=mock_kyrodb_router,
            embedding_service=mock_embedding_service,
        )

        # Search request WITHOUT customer_id
        request = SearchRequest(
            customer_id=None,  # Missing customer_id
            goal="test query",
            collection="failures",
            k=5,
        )

        # Should raise ValueError due to multi-tenancy violation
        with pytest.raises(ValueError, match="customer_id is required"):
            await pipeline.search(request)

    @pytest.mark.asyncio
    async def test_different_customers_separate_results(
        self,
        mock_kyrodb_router: KyroDBRouter,
        mock_embedding_service: EmbeddingService,
    ):
        """Test that different customers get separate search results."""
        pipeline = SearchPipeline(
            kyrodb_router=mock_kyrodb_router,
            embedding_service=mock_embedding_service,
        )

        # Customer A search
        request_a = SearchRequest(
            customer_id="customer-a",
            goal="kubernetes deployment error",
            collection="failures",
            k=5,
        )

        # Customer B search
        request_b = SearchRequest(
            customer_id="customer-b",
            goal="kubernetes deployment error",  # Same query!
            collection="failures",
            k=5,
        )

        # Execute both searches
        await pipeline.search(request_a)
        await pipeline.search(request_b)

        # Verify search_text was called twice with different customer_ids
        assert mock_kyrodb_router.search_text.call_count == 2

        calls = mock_kyrodb_router.search_text.call_args_list

        # First call: customer-a
        assert calls[0].kwargs["customer_id"] == "customer-a"

        # Second call: customer-b
        assert calls[1].kwargs["customer_id"] == "customer-b"

        # Even though queries are identical, they search different namespaces


class TestKyroDBRouterIsolation:
    """Test suite for KyroDB router namespace isolation."""

    @pytest.mark.asyncio
    async def test_insert_episode_namespace_format(self):
        """Test that insert_episode uses correct namespace format."""
        # Create a real KyroDBRouter for testing (with mocked clients)
        from src.config import KyroDBConfig

        config = KyroDBConfig(
            text_host="localhost",
            text_port=50051,
            image_host="localhost",
            image_port=50052,
        )

        router = KyroDBRouter(config)

        # Mock the clients
        router.text_client = AsyncMock()
        router.image_client = AsyncMock()

        # Mock insert responses
        insert_response = Mock()
        insert_response.success = True
        router.text_client.insert.return_value = insert_response
        router.image_client.insert.return_value = insert_response

        # Insert episode for customer-a
        text_embedding = [0.1] * 384
        image_embedding = [0.2] * 512

        await router.insert_episode(
            episode_id=123,
            customer_id="acme-corp",
            collection="failures",
            text_embedding=text_embedding,
            image_embedding=image_embedding,
            metadata={"test": "data"},
        )

        # Verify text client was called with namespaced collection
        text_call = router.text_client.insert.call_args
        assert text_call.kwargs["namespace"] == "acme-corp:failures"
        assert text_call.kwargs["doc_id"] == 123

        # Verify image client was called with namespaced collection + _images suffix
        image_call = router.image_client.insert.call_args
        assert image_call.kwargs["namespace"] == "acme-corp:failures_images"
        assert image_call.kwargs["doc_id"] == 123

    @pytest.mark.asyncio
    async def test_search_text_namespace_format(self):
        """Test that search_text uses correct namespace format."""
        from src.config import KyroDBConfig

        config = KyroDBConfig(
            text_host="localhost",
            text_port=50051,
            image_host="localhost",
            image_port=50052,
        )

        router = KyroDBRouter(config)

        # Mock the client
        router.text_client = AsyncMock()

        # Mock search response
        search_response = Mock()
        search_response.results = []
        router.text_client.search.return_value = search_response

        # Search for customer-a
        query_embedding = [0.1] * 384

        await router.search_text(
            query_embedding=query_embedding,
            k=10,
            customer_id="acme-corp",
            collection="failures",
            min_score=0.6,
        )

        # Verify search was called with namespaced collection
        call_args = router.text_client.search.call_args
        assert call_args.kwargs["namespace"] == "acme-corp:failures"
        assert call_args.kwargs["k"] == 10

    @pytest.mark.asyncio
    async def test_insert_fails_without_customer_id(self):
        """Test that insert_episode fails without customer_id."""
        from src.config import KyroDBConfig

        config = KyroDBConfig(
            text_host="localhost",
            text_port=50051,
            image_host="localhost",
            image_port=50052,
        )

        router = KyroDBRouter(config)

        # Mock the clients
        router.text_client = AsyncMock()
        router.image_client = AsyncMock()

        # Attempt insert without customer_id
        with pytest.raises(ValueError, match="customer_id is required"):
            await router.insert_episode(
                episode_id=123,
                customer_id=None,  # Missing customer_id
                collection="failures",
                text_embedding=[0.1] * 384,
            )

        # Verify no insert calls were made
        router.text_client.insert.assert_not_called()
        router.image_client.insert.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_fails_without_customer_id(self):
        """Test that search_text fails without customer_id."""
        from src.config import KyroDBConfig

        config = KyroDBConfig(
            text_host="localhost",
            text_port=50051,
            image_host="localhost",
            image_port=50052,
        )

        router = KyroDBRouter(config)

        # Mock the client
        router.text_client = AsyncMock()

        # Attempt search without customer_id
        with pytest.raises(ValueError, match="customer_id is required"):
            await router.search_text(
                query_embedding=[0.1] * 384,
                k=10,
                customer_id=None,  # Missing customer_id
                collection="failures",
            )

        # Verify no search calls were made
        router.text_client.search.assert_not_called()


class TestMetadataIsolation:
    """Test suite for metadata customer_id presence."""

    @pytest.mark.asyncio
    async def test_episode_metadata_includes_customer_id(
        self,
        ingestion_pipeline: IngestionPipeline,
        mock_kyrodb_router: KyroDBRouter,
        mock_embedding_service: EmbeddingService,
    ):
        """Test that episode metadata includes customer_id."""
        episode_data = EpisodeCreate(
            customer_id="acme-corp",
            goal="Deploy application to production environment",
            tool_chain=["kubectl"],
            actions_taken=["action1"],
            error_trace="Error occurred during deployment process",
            error_class="unknown",
        )

        episode = await ingestion_pipeline.capture_episode(
            episode_data=episode_data,
            generate_reflection=False,
        )

        # Verify metadata includes customer_id
        metadata = episode.to_metadata_dict()
        assert "customer_id" in metadata
        assert metadata["customer_id"] == "acme-corp"

        # Verify metadata was passed to KyroDB insert
        call_args = mock_kyrodb_router.insert_episode.call_args
        assert "customer_id" in call_args.kwargs["metadata"]
        assert call_args.kwargs["metadata"]["customer_id"] == "acme-corp"
