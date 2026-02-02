"""
Real KyroDB integration test.

Tests actual connection and operations against running KyroDB instances.
Requires:
- Text instance on localhost:50051 (384-dim)
- Image instance on localhost:50052 (512-dim)

Run with:
    pytest tests/integration/test_kyrodb_real.py -v -s

Prerequisites:
    cd <path_to_kyrodb_repo>
    ./target/release/kyrodb_server --port 50051 --data-dir ./data
"""

import time
from datetime import UTC

import pytest

from src.kyrodb.client import KyroDBClient
from src.kyrodb.router import KyroDBRouter


def is_kyrodb_running(host: str = "localhost", port: int = 50051) -> bool:
    """Check if KyroDB is running on host:port."""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


@pytest.fixture
def skip_if_no_kyrodb():
    """Skip test if KyroDB is not running."""
    if not is_kyrodb_running(host="localhost", port=50051):
        pytest.skip("KyroDB text instance not running on localhost:50051")
    if not is_kyrodb_running(host="localhost", port=50052):
        pytest.skip("KyroDB image instance not running on localhost:50052")


@pytest.fixture
async def kyrodb_client():
    """Create a real KyroDB client."""
    client = KyroDBClient(
        host="localhost",
        port=50051,
        timeout_seconds=30,
        max_retries=3,
        enable_tls=False,
    )
    await client.connect()
    yield client
    await client.close()


@pytest.fixture
async def kyrodb_router():
    """Create a real KyroDB router (text+image instances)."""
    from src.config import KyroDBConfig
    
    config = KyroDBConfig(
        text_host="localhost",
        text_port=50051,
        image_host="localhost",
        image_port=50052,
        enable_tls=False,
    )
    router = KyroDBRouter(config=config)
    await router.connect()
    yield router
    await router.close()


class TestKyroDBConnection:
    """Test basic KyroDB connectivity."""

    @pytest.mark.asyncio
    async def test_health_check(self, skip_if_no_kyrodb, kyrodb_client: KyroDBClient):
        """Test health check against real KyroDB."""
        response = await kyrodb_client.health_check()
        
        print("\n--- KyroDB Health Check ---")
        print(f"Status: {response.status}")
        print(f"Version: {response.version}")
        print(f"Uptime: {response.uptime_seconds}s")
        for key, value in response.components.items():
            print(f"  {key}: {value}")
        
        # Status 1 = HEALTHY, Status 2 = DEGRADED (acceptable for empty database)
        assert response.status in [1, 2], f"Expected HEALTHY (1) or DEGRADED (2), got {response.status}"
        assert response.version, "Version should be set"
        assert response.uptime_seconds >= 0, "Uptime should be non-negative"

    @pytest.mark.asyncio
    async def test_insert_and_query(self, skip_if_no_kyrodb, kyrodb_client: KyroDBClient):
        """Test insert and query operations."""
        # Generate a unique doc_id using timestamp
        doc_id = int(time.time() * 1000) % (2**63 - 1)
        
        # Create a test embedding (384 dimensions for all-MiniLM-L6-v2)
        embedding = [0.1] * 384
        namespace = "test:integration"
        metadata = {
            "episode_type": "failure",
            "customer_id": "test-customer",
            "test_run": "real_integration",
        }
        
        print("\n--- Insert Test ---")
        print(f"Doc ID: {doc_id}")
        print(f"Namespace: {namespace}")
        print(f"Embedding dims: {len(embedding)}")
        
        try:
            # Insert
            insert_response = await kyrodb_client.insert(
                doc_id=doc_id,
                embedding=embedding,
                namespace=namespace,
                metadata=metadata,
            )
            
            print(f"Insert success: {insert_response.success}")
            assert insert_response.success, f"Insert failed: {insert_response.error}"
            
            # Query back
            query_response = await kyrodb_client.query(
                doc_id=doc_id,
                namespace=namespace,
                include_embedding=True,
            )
            
            print("\n--- Query Test ---")
            print(f"Found: {query_response.found}")
            print(f"Metadata keys: {list(query_response.metadata.keys())}")
            
            assert query_response.found, "Document not found after insert"
            assert query_response.doc_id == doc_id
            assert len(query_response.embedding) == 384
            assert query_response.metadata.get("episode_type") == "failure"
        finally:
            # Cleanup - always runs even if assertions fail
            delete_response = await kyrodb_client.delete(doc_id=doc_id, namespace=namespace)
            print("\n--- Cleanup ---")
            print(f"Delete success: {delete_response.success}")

    @pytest.mark.asyncio
    async def test_search(self, skip_if_no_kyrodb, kyrodb_client: KyroDBClient):
        """Test k-NN search operation."""
        # Insert some test vectors first
        namespace = "test:search"
        base_time = int(time.time() * 1000) % (2**60)
        doc_ids = []
        
        print("\n--- Inserting test vectors ---")
        for i in range(5):
            doc_id = base_time + i
            doc_ids.append(doc_id)
            
            # Create slightly different embeddings
            embedding = [0.1 + (i * 0.01)] * 384
            
            await kyrodb_client.insert(
                doc_id=doc_id,
                embedding=embedding,
                namespace=namespace,
                metadata={"index": str(i)},
            )
        
        print(f"Inserted {len(doc_ids)} vectors")
        
        try:
            # Search for similar vectors
            query_embedding = [0.1] * 384  # Should be most similar to first vector
            
            search_response = await kyrodb_client.search(
                query_embedding=query_embedding,
                k=5,
                namespace=namespace,
                min_score=-1.0,
            )
            
            print("\n--- Search Results ---")
            print(f"Results found: {len(search_response.results)}")
            print(f"Search latency: {search_response.search_latency_ms:.2f}ms")
            
            for result in search_response.results:
                print(f"  Doc {result.doc_id}: score={result.score:.4f}")
            
            assert len(search_response.results) > 0, "No search results"
        finally:
            # Cleanup - always runs even if assertions fail
            print("\n--- Cleanup ---")
            for doc_id in doc_ids:
                await kyrodb_client.delete(doc_id=doc_id, namespace=namespace)
            print(f"Deleted {len(doc_ids)} vectors")


class TestKyroDBRouterReal:
    """Test KyroDBRouter against real KyroDB."""

    @pytest.mark.asyncio
    async def test_router_health(self, skip_if_no_kyrodb, kyrodb_router: KyroDBRouter):
        """Test router health check."""
        health = await kyrodb_router.health_check()
        
        print("\n--- Router Health ---")
        print(f"Text instance: {health.get('text', 'N/A')}")
        print(f"Image instance: {health.get('image', 'N/A')}")
        
        assert health.get("text") is True, "Text instance not healthy"

    @pytest.mark.asyncio
    async def test_insert_episode(self, skip_if_no_kyrodb, kyrodb_router: KyroDBRouter):
        """Test episode insertion via router."""
        episode_id = int(time.time() * 1000) % (2**63 - 1)
        customer_id = "test-customer"
        collection = "test_failures"  # Use test-specific namespace
        
        # Create embedding
        embedding = [0.1] * 384
        
        metadata = {
            "episode_type": "failure",
            "goal": "Test integration",
            "error_class": "test_error",
            "test_run": "real_integration",
            "customer_id": customer_id,  # Include customer_id in metadata
        }
        
        print("\n--- Insert Episode via Router ---")
        print(f"Episode ID: {episode_id}")
        print(f"Customer: {customer_id}")
        print(f"Collection: {collection}")
        
        try:
            text_success, image_success = await kyrodb_router.insert_episode(
                episode_id=episode_id,
                customer_id=customer_id,
                collection=collection,
                text_embedding=embedding,
                image_embedding=None,
                metadata=metadata,
            )
            
            print(f"Text insert: {text_success}")
            print(f"Image insert: {image_success}")
            
            assert text_success, "Text embedding insert failed"
            
            # Query back with customer_id for namespace isolation
            result = await kyrodb_router.get_episode(
                episode_id=episode_id,
                customer_id=customer_id,
                collection=collection,
            )
            
            print("\n--- Query Episode ---")
            print(f"Found: {result is not None}")
            
            assert result is not None, "Episode not found after insert"
            assert result.get("doc_id") == episode_id
        finally:
            # Cleanup - always runs even if assertions fail
            await kyrodb_router.delete_episode(
                episode_id=episode_id,
                customer_id=customer_id,
                collection=collection,
            )
            print("Episode cleaned up")

    @pytest.mark.asyncio
    async def test_search_episodes(self, skip_if_no_kyrodb, kyrodb_router: KyroDBRouter):
        """Test episode search via router."""
        customer_id = "test-customer"
        collection = "test_failures"  # Use test-specific namespace
        base_time = int(time.time() * 1000) % (2**60)
        episode_ids = []
        
        print("\n--- Inserting test episodes ---")
        
        # Insert test episodes
        for i in range(3):
            episode_id = base_time + i
            episode_ids.append(episode_id)
            
            embedding = [0.1 + (i * 0.02)] * 384
            metadata = {
                "episode_type": "failure",
                "goal": f"Test goal {i}",
                "index": str(i),
                "customer_id": customer_id,
            }
            
            await kyrodb_router.insert_episode(
                episode_id=episode_id,
                customer_id=customer_id,
                collection=collection,
                text_embedding=embedding,
                metadata=metadata,
            )
        
        print(f"Inserted {len(episode_ids)} episodes")
        
        try:
            # Search - now uses customer_id for namespace isolation
            query_embedding = [0.1] * 384
            
            response = await kyrodb_router.search_episodes(
                query_embedding=query_embedding,
                customer_id=customer_id,
                collection=collection,
                k=10,
                min_score=-1.0,  # Allow all scores for testing
            )
            
            print("\n--- Search Results ---")
            print(f"Found: {len(response.results)} episodes")
            
            for result in response.results:
                print(f"  Episode {result.doc_id}: score={result.score:.4f}")
            
            assert len(response.results) >= 3, f"Expected at least 3 results, got {len(response.results)}"
        finally:
            # Cleanup - always runs even if assertions fail
            print("\n--- Cleanup ---")
            for episode_id in episode_ids:
                await kyrodb_router.delete_episode(
                    episode_id=episode_id,
                    customer_id=customer_id,
                    collection=collection,
                )
            print(f"Cleaned up {len(episode_ids)} episodes")


class TestReflectionPersistence:
    """Test reflection update and retrieval with real KyroDB."""

    @pytest.mark.asyncio
    async def test_update_episode_reflection(self, skip_if_no_kyrodb, kyrodb_router: KyroDBRouter):
        """Test updating episode with reflection metadata."""
        from datetime import datetime

        from src.models.episode import Reflection, ReflectionTier
        
        episode_id = int(time.time() * 1000) % (2**63 - 1)
        customer_id = "test-reflection"
        collection = "test_failures"
        
        try:
            # Insert episode first - include customer_id in metadata for ownership check
            embedding = [0.1] * 384
            initial_metadata = {
                "episode_type": "failure",
                "goal": "Test reflection persistence",
                "customer_id": customer_id,  # Required for update_episode_reflection ownership check
            }
            
            print("\n--- Insert Episode ---")
            await kyrodb_router.insert_episode(
                episode_id=episode_id,
                customer_id=customer_id,
                collection=collection,
                text_embedding=embedding,
                metadata=initial_metadata,
            )
            print(f"Episode {episode_id} inserted")
            
            # Create reflection
            reflection = Reflection(
                consensus=None,
                root_cause="Missing dependency in requirements.txt",
                preconditions=["pip install used", "requirements.txt exists"],
                resolution_strategy="Add missing package to requirements.txt",
                environment_factors=["python_version", "pip_version"],
                affected_components=["dependency_management"],
                generalization_score=0.7,
                confidence_score=0.85,
                llm_model="openrouter-consensus",
                generated_at=datetime.now(UTC),
                cost_usd=0.0,
                generation_latency_ms=1500.0,
                tier=ReflectionTier.PREMIUM.value,
            )
            
            # Update with reflection
            print("\n--- Update with Reflection ---")
            success = await kyrodb_router.update_episode_reflection(
                episode_id=episode_id,
                customer_id=customer_id,
                collection=collection,
                reflection=reflection,
            )
            
            print(f"Update success: {success}")
            assert success, "Reflection update failed"
            
            # Query and verify reflection metadata with customer_id
            result = await kyrodb_router.get_episode(
                episode_id=episode_id,
                customer_id=customer_id,
                collection=collection,
            )
            
            print("\n--- Verify Reflection ---")
            print(f"Metadata keys: {list(result.get('metadata', {}).keys())}")
            
            metadata = result.get("metadata", {})
            assert "reflection_root_cause" in metadata, "Missing reflection_root_cause"
            assert metadata.get("reflection_root_cause") == reflection.root_cause
            assert "reflection_confidence" in metadata
            
            print(f"Root cause: {metadata.get('reflection_root_cause')}")
            print(f"Confidence: {metadata.get('reflection_confidence')}")
            print(f"Model: {metadata.get('reflection_model')}")
        finally:
            # Cleanup
            await kyrodb_router.delete_episode(
                episode_id=episode_id,
                customer_id=customer_id,
                collection=collection,
            )
            print("\nEpisode cleaned up")


class TestSkillsIntegration:
    """Test skills storage with real KyroDB."""

    @pytest.mark.asyncio
    async def test_insert_and_search_skill(self, skip_if_no_kyrodb, kyrodb_router: KyroDBRouter):
        """Test skill insertion and search."""
        from datetime import datetime

        from src.models.skill import Skill
        
        skill_id = int(time.time() * 1000) % (2**63 - 1)
        customer_id = "test-skills"
        
        try:
            # Create skill matching the actual Skill model
            skill = Skill(
                skill_id=skill_id,
                customer_id=customer_id,
                name="Fix missing Python dependency",
                docstring="This skill fixes the common issue of missing Python packages by adding them to requirements.txt and running pip install.",
                code="pip install missing_package && echo 'missing_package' >> requirements.txt",
                language="bash",
                error_class="ModuleNotFoundError",  # Required field
                source_episodes=[skill_id - 100, skill_id - 200],
                usage_count=5,
                success_count=4,
                failure_count=1,
                created_at=datetime.now(UTC),
            )
            
            # Create embedding
            embedding = [0.15] * 384
            
            print("\n--- Insert Skill ---")
            print(f"Skill ID: {skill_id}")
            print(f"Name: {skill.name}")
            
            success = await kyrodb_router.insert_skill(
                skill=skill,
                embedding=embedding,
            )
            
            print(f"Insert success: {success}")
            assert success, "Skill insert failed"
            
            # Search for skill
            query_embedding = [0.15] * 384
            
            results = await kyrodb_router.search_skills(
                query_embedding=query_embedding,
                customer_id=customer_id,
                k=5,
            )
            
            print("\n--- Search Skills ---")
            print(f"Found: {len(results)} skills")
            
            for skill_result, score in results:
                print(f"  Skill: {skill_result.name[:40]}... (score={score:.4f})")
            
            # Should find our skill
            found = any(s.skill_id == skill_id for s, _ in results)
            assert found, f"Skill {skill_id} not found in search results"
        finally:
            # Cleanup - delete via the underlying namespace
            namespace = f"{customer_id}:skills"
            await kyrodb_router.text_client.delete(doc_id=skill_id, namespace=namespace)
            print("\nSkill cleaned up")
