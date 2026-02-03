"""
Integration test fixtures for real KyroDB connection.

These fixtures provide access to running KyroDB instances for integration testing.
Requires:
- Text KyroDB server on localhost:50051 (384-dim)
- Image KyroDB server on localhost:50052 (512-dim)

Prerequisites:
    1. Build KyroDB: cargo build --release -p kyrodb-engine --bin kyrodb_server
    2. Start server: ./target/release/kyrodb_server --config /tmp/kyrodb_test_config.toml
    3. Run tests: pytest tests/integration/ -v
"""

import os
import socket

import pytest

from src.config import KyroDBConfig
from src.kyrodb.client import KyroDBClient
from src.kyrodb.router import KyroDBRouter


def is_kyrodb_running(host: str = "localhost", port: int = 50051) -> bool:
    """
    Check if KyroDB is running on the specified host:port.

    Uses context manager for socket to ensure proper resource cleanup.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            return result == 0
    except Exception:
        return False


@pytest.fixture
def skip_if_no_kyrodb():
    """Skip test if KyroDB is not running on localhost:50051/50052."""
    if not is_kyrodb_running(host="localhost", port=50051):
        pytest.skip("KyroDB text instance not running on localhost:50051")
    if not is_kyrodb_running(host="localhost", port=50052):
        pytest.skip("KyroDB image instance not running on localhost:50052")


@pytest.fixture(scope="session", autouse=True)
def _configure_test_customer_db(tmp_path_factory):
    """
    Isolate integration tests from any developer/customer DB.

    Integration tests hit real KyroDB instances; doc_id allocation must be global and stable
    to avoid KyroDB doc_id collisions. We therefore use the real CustomerDatabase-backed
    allocator, but point it at a temp SQLite file.
    """
    db_path = tmp_path_factory.mktemp("customer_db") / "customers.db"
    os.environ["STORAGE_CUSTOMER_DB_PATH"] = str(db_path)

    import src.config as config_module

    config_module._settings = None

    import src.storage.database as customer_db_module

    customer_db_module._db = None


@pytest.fixture
async def customer_db():
    from src.storage.database import get_customer_db

    db = await get_customer_db()
    yield db


@pytest.fixture
async def kyrodb_client():
    """
    Create a real KyroDB client connected to localhost:50051.
    
    Uses try/finally to ensure proper resource cleanup even on test failure.
    
    Usage:
        @pytest.mark.asyncio
        async def test_something(skip_if_no_kyrodb, kyrodb_client):
            response = await kyrodb_client.health_check()
            assert response.status in [1, 2]
    """
    client = KyroDBClient(
        host="localhost",
        port=50051,
        timeout_seconds=30,
        max_retries=3,
        enable_tls=False,
    )
    await client.connect()
    try:
        yield client
    finally:
        await client.close()


@pytest.fixture
async def kyrodb_router():
    """
    Create a KyroDBRouter connected to localhost:50051.
    
    Uses same port for both text and image (single-instance testing).
    Uses try/finally to ensure proper resource cleanup even on test failure.
    Skips test if KyroDB is not available.
    
    Usage:
        @pytest.mark.asyncio
        async def test_episode_insert(kyrodb_router):
            success, _ = await kyrodb_router.insert_episode(...)
            assert success
    """
    config = KyroDBConfig(
        text_host="localhost",
        text_port=50051,
        image_host="localhost",
        image_port=50052,
        enable_tls=False,
        request_timeout_seconds=30,
    )
    router = KyroDBRouter(config=config)
    try:
        await router.connect()
    except Exception as e:
        pytest.skip(f"KyroDB not available at localhost:50051: {e}")
    try:
        yield router
    finally:
        await router.close()


@pytest.fixture
def test_customer_id() -> str:
    """
    Generate a test customer ID for namespace isolation.
    
    Uses millisecond precision and includes random suffix to prevent collisions
    when tests run in rapid succession.
    """
    import random
    import time
    timestamp_ms = int(time.time() * 1000)
    random_suffix = random.randint(1000, 9999)
    return f"test-customer-{timestamp_ms}-{random_suffix}"


@pytest.fixture
def test_collection() -> str:
    """Get the test collection name."""
    return "test_failures"


@pytest.fixture
def embedding_384() -> list[float]:
    """Generate a 384-dimensional test embedding (all-MiniLM-L6-v2 dimensions)."""
    return [0.1] * 384


@pytest.fixture
def embedding_factory():
    """
    Factory for generating test embeddings with slight variations.
    
    Usage:
        def test_search(embedding_factory):
            embeddings = [embedding_factory(i) for i in range(5)]
            # embeddings[0] is most similar to embedding_factory(0)
    """
    def _make_embedding(index: int = 0, dim: int = 384) -> list[float]:
        return [0.1 + (index * 0.01)] * dim
    return _make_embedding


@pytest.fixture
async def unique_doc_id(customer_db) -> int:
    """Allocate a unique KyroDB-safe doc_id via the real allocator."""
    return await customer_db.allocate_doc_id()


@pytest.fixture
def doc_id_factory(customer_db):
    """
    Factory for allocating unique document IDs via CustomerDatabase.

    Usage:
        episode_id = await doc_id_factory()
    """
    async def _alloc() -> int:
        return await customer_db.allocate_doc_id()

    return _alloc
