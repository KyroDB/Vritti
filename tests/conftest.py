"""
Pytest configuration and fixtures for integration tests.

Provides shared fixtures for:
- Mock KyroDB instances
- Embedding service mocks
- Test episodes
- FastAPI test client
"""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from fastapi.testclient import TestClient

from src.config import (
    EmbeddingConfig,
    HygieneConfig,
    KyroDBConfig,
    LLMConfig,
    Settings,
)
from src.ingestion.embedding import EmbeddingService
from src.kyrodb.client import KyroDBClient
from src.kyrodb.router import KyroDBRouter
from src.models.episode import (
    Episode,
    EpisodeCreate,
    EpisodeType,
    ErrorClass,
    Reflection,
)


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with mock configuration."""
    return Settings(
        kyrodb=KyroDBConfig(
            text_host="localhost",
            text_port=50051,
            image_host="localhost",
            image_port=50052,
            request_timeout_seconds=5,
        ),
        embedding=EmbeddingConfig(
            text_model_name="all-MiniLM-L6-v2",
            text_dimension=384,
            image_model_name="openai/clip-vit-base-patch32",
            image_dimension=512,
            device="cpu",  # Force CPU for tests
        ),
        llm=LLMConfig(
            api_key="sk-test-key-12345",
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=1000,
        ),
        hygiene=HygieneConfig(),
    )


@pytest.fixture
def sample_episode_create() -> EpisodeCreate:
    """Create sample episode creation data with customer_id."""
    return EpisodeCreate(
        customer_id="test-customer",  # Multi-tenancy: default test customer
        episode_type=EpisodeType.FAILURE,
        goal="Deploy web application to Kubernetes production cluster",
        tool_chain=["kubectl", "docker", "helm"],
        actions_taken=[
            "Built Docker image with tag v1.2.3",
            "Pushed image to container registry",
            "Applied Kubernetes deployment manifest",
            "Observed pod status showing ImagePullBackOff",
        ],
        error_trace=(
            "Error: ImagePullBackOff\n"
            "Failed to pull image 'myapp:latest': not found in registry\n"
            "Pod stuck in ImagePullBackOff state"
        ),
        error_class=ErrorClass.RESOURCE_ERROR,
        code_state_diff=(
            "--- a/deployment.yaml\n"
            "+++ b/deployment.yaml\n"
            "@@ -10,7 +10,7 @@\n"
            "     spec:\n"
            "       containers:\n"
            "       - name: myapp\n"
            "-        image: myapp:v1.2.2\n"
            "+        image: myapp:latest\n"
        ),
        environment_info={
            "os": "Darwin",
            "os_version": "14.1",
            "kubectl_version": "1.28.0",
            "docker_version": "24.0.6",
            "cluster": "production",
            "namespace": "default",
        },
        resolution="Changed image tag from 'latest' to 'v1.2.3' to match pushed image",
        tags=["production", "deployment", "critical"],
        severity=1,
    )


@pytest.fixture
def sample_episode(sample_episode_create: EpisodeCreate) -> Episode:
    """Create sample episode with reflection."""
    reflection = Reflection(
        root_cause="Image tag mismatch between manifest and pushed image",
        preconditions=[
            "Using tool: kubectl",
            "Error class: ImagePullBackOff",
            "OS: Darwin",
            "Component: kubernetes",
        ],
        resolution_strategy=(
            "1. Check image tag in deployment manifest\n"
            "2. Verify image exists in registry with that tag\n"
            "3. Update manifest to use correct tag or push image with expected tag"
        ),
        environment_factors=["kubectl_version", "docker_version", "cluster"],
        affected_components=["kubernetes", "docker", "container-registry"],
        generalization_score=0.8,
        confidence_score=0.9,
        llm_model="gpt-4",
    )

    return Episode(
        create_data=sample_episode_create,
        episode_id=1234567890,
        reflection=reflection,
        created_at=datetime.now(UTC),
        retrieval_count=0,
    )


@pytest.fixture
def mock_kyrodb_client() -> AsyncMock:
    """Create mock KyroDB client."""
    client = AsyncMock(spec=KyroDBClient)

    # Mock insert response
    insert_response = Mock()
    insert_response.success = True
    insert_response.error = ""
    client.insert.return_value = insert_response

    # Mock search response
    search_result1 = Mock()
    search_result1.doc_id = 123
    search_result1.score = 0.92
    search_result1.metadata = {
        "episode_type": "failure",
        "error_class": "resource_error",
        "tool": "kubectl",
        "episode_json": """{"create_data": {"episode_type": "failure", "goal": "test goal", "tool_chain": ["kubectl"], "actions_taken": ["action1"], "error_trace": "error", "error_class": "resource_error", "tags": [], "severity": 3}, "episode_id": 123, "created_at": "2024-01-01T00:00:00Z", "retrieval_count": 0}""",
    }

    search_response = Mock()
    search_response.results = [search_result1]
    search_response.num_results = 1
    client.search.return_value = search_response

    # Mock query response
    query_response = Mock()
    query_response.found = True
    query_response.doc_id = 123
    query_response.metadata = search_result1.metadata
    client.query.return_value = query_response

    # Mock health check
    client.health_check.return_value = None  # Success = no exception

    # Mock connect/close
    client.connect.return_value = None
    client.close.return_value = None

    return client


@pytest.fixture
async def mock_kyrodb_router(mock_kyrodb_client: AsyncMock) -> KyroDBRouter:
    """Create mock KyroDB router with mock clients."""
    router = MagicMock(spec=KyroDBRouter)
    router.text_client = mock_kyrodb_client
    router.image_client = mock_kyrodb_client

    # Mock insert_episode
    async def mock_insert_episode(*args, **kwargs):
        return (True, True)  # text_success, image_success

    router.insert_episode = AsyncMock(side_effect=mock_insert_episode)

    # Mock search_text
    search_result = Mock()
    search_result.doc_id = 123
    search_result.score = 0.92
    search_result.metadata = mock_kyrodb_client.search.return_value.results[0].metadata

    search_response = Mock()
    search_response.results = [search_result]
    search_response.num_results = 1

    async def mock_search_text(*args, **kwargs):
        return search_response

    router.search_text = AsyncMock(side_effect=mock_search_text)

    # Mock health_check
    async def mock_health_check():
        return {"text": True, "image": True}

    router.health_check = AsyncMock(side_effect=mock_health_check)

    # Mock connect/close
    router.connect = AsyncMock()
    router.close = AsyncMock()

    return router


@pytest.fixture
def mock_embedding_service() -> EmbeddingService:
    """Create mock embedding service."""
    service = MagicMock(spec=EmbeddingService)

    # Mock text embedding (384-dim)
    def mock_embed_text(text: str) -> list[float]:
        # Return consistent embedding based on text hash
        import hashlib

        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        import random

        random.seed(hash_val)
        return [random.random() for _ in range(384)]

    service.embed_text = Mock(side_effect=mock_embed_text)

    # Mock image embedding (512-dim)
    def mock_embed_image(image_path: Path) -> list[float]:
        import random

        random.seed(42)
        return [random.random() for _ in range(512)]

    service.embed_image = Mock(side_effect=mock_embed_image)

    return service


@pytest.fixture
def app_client(
    test_settings: Settings,
    mock_kyrodb_router: KyroDBRouter,
    mock_embedding_service: EmbeddingService,
):
    """Create FastAPI test client with mocked dependencies."""
    # Patch global instances
    import src.main as main_module
    from src.main import app

    main_module.kyrodb_router = mock_kyrodb_router
    main_module.embedding_service = mock_embedding_service
    main_module.reflection_service = None  # Skip LLM for tests

    # Create test client (skip lifespan to avoid actual connections)
    client = TestClient(app)

    yield client
