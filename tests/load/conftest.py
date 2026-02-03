"""
Load test fixtures.

These fixtures intentionally use real components (KyroDBRouter, EmbeddingService,
SearchPipeline, GatingService) so load tests validate the actual system behavior.

Load tests are opt-in via markers / env vars; they are not meant for CI.
"""

from __future__ import annotations

import os
import socket

import pytest

from src.config import KyroDBConfig, get_settings
from src.gating.service import GatingService
from src.ingestion.capture import IngestionPipeline
from src.ingestion.embedding import EmbeddingService
from src.kyrodb.router import KyroDBRouter
from src.models.episode import EpisodeCreate, ErrorClass
from src.retrieval.search import SearchPipeline

LOAD_CUSTOMER_ID = "load-test-customer"


def _is_port_open(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            return sock.connect_ex((host, port)) == 0
    except Exception:
        return False


@pytest.fixture
def skip_if_no_kyrodb():
    """Skip load tests if KyroDB is not available locally."""
    if not _is_port_open("localhost", 50051):
        pytest.skip("KyroDB text instance not running on localhost:50051")
    if not _is_port_open("localhost", 50052):
        pytest.skip("KyroDB image instance not running on localhost:50052")


@pytest.fixture
def load_customer_id() -> str:
    """Single customer namespace for load tests."""
    return LOAD_CUSTOMER_ID


@pytest.fixture(scope="session", autouse=True)
def _configure_load_customer_db(tmp_path_factory):
    """
    Isolate load tests from any developer/customer DB.

    Load tests allocate many doc_ids and update the local episode/skill indices; they must not
    touch a developer's real DB file.
    """
    db_path = tmp_path_factory.mktemp("load_customer_db") / "customers.db"
    os.environ["STORAGE_CUSTOMER_DB_PATH"] = str(db_path)

    import src.config as config_module

    config_module._settings = None

    import src.storage.database as customer_db_module

    customer_db_module._db = None


@pytest.fixture(scope="session", autouse=True)
async def _ensure_load_customer(_configure_load_customer_db):
    """Ensure the load-test customer exists so local indexing FK constraints succeed."""
    from src.models.customer import CustomerCreate, SubscriptionTier
    from src.storage.database import get_customer_db

    db = await get_customer_db()
    existing = await db.get_customer(LOAD_CUSTOMER_ID)
    if existing:
        return
    await db.create_customer(
        CustomerCreate(
            customer_id=LOAD_CUSTOMER_ID,
            organization_name="Load Test",
            email="load-test@vritti.local",
            subscription_tier=SubscriptionTier.PRO,
        )
    )


@pytest.fixture
def sample_episode_create(load_customer_id: str) -> EpisodeCreate:
    """Sample episode for load testing."""
    return EpisodeCreate(
        customer_id=load_customer_id,
        goal="Deploy application to production",
        tool_chain=["kubectl", "helm"],
        actions_taken=["kubectl apply -f deployment.yaml"],
        error_trace="ImagePullBackOff: failed to pull image from registry",
        error_class=ErrorClass.RESOURCE_ERROR,
        tags=["kubernetes", "deployment"],
        environment_info={"env": "load-test"},
    )


@pytest.fixture
async def kyrodb_router(skip_if_no_kyrodb):
    settings = get_settings()
    config = KyroDBConfig(
        text_host=settings.kyrodb.text_host,
        text_port=settings.kyrodb.text_port,
        image_host=settings.kyrodb.image_host,
        image_port=settings.kyrodb.image_port,
        enable_tls=False,
        request_timeout_seconds=settings.kyrodb.request_timeout_seconds,
    )
    router = KyroDBRouter(config=config)
    await router.connect()
    try:
        yield router
    finally:
        await router.close()


@pytest.fixture(scope="session")
def embedding_service():
    settings = get_settings()
    service = EmbeddingService(config=settings.embedding)
    service.warmup()
    return service


@pytest.fixture
def ingestion_pipeline(kyrodb_router: KyroDBRouter, embedding_service: EmbeddingService):
    return IngestionPipeline(
        kyrodb_router=kyrodb_router,
        embedding_service=embedding_service,
        reflection_service=None,  # Keep load tests offline by default
    )


@pytest.fixture
def search_pipeline(kyrodb_router: KyroDBRouter, embedding_service: EmbeddingService):
    return SearchPipeline(
        kyrodb_router=kyrodb_router,
        embedding_service=embedding_service,
    )


@pytest.fixture
def gating_service(search_pipeline: SearchPipeline, kyrodb_router: KyroDBRouter):
    return GatingService(search_pipeline=search_pipeline, kyrodb_router=kyrodb_router)
