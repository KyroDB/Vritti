"""
Full Pipeline Integration Test with Real KyroDB.

Tests the complete episode lifecycle:
1. Capture episode → 2. Generate reflection → 3. Persist to KyroDB
4. Search → 5. Pre-action gating

Requires KyroDB servers:
- Text: localhost:50051 (384-dim)
- Image: localhost:50052 (512-dim)

Run with:
    pytest tests/integration/test_full_pipeline.py -v -s
"""

import time
from datetime import UTC, datetime

import pytest

from src.config import KyroDBConfig
from src.kyrodb.router import KyroDBRouter
from src.models.episode import Reflection, ReflectionTier


def is_kyrodb_running(host: str, port: int) -> bool:
    """Check if KyroDB is running on the given host:port."""
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
    if not is_kyrodb_running("localhost", 50051):
        pytest.skip("KyroDB text instance not running on localhost:50051")
    if not is_kyrodb_running("localhost", 50052):
        pytest.skip("KyroDB image instance not running on localhost:50052")


@pytest.fixture
async def kyrodb_router():
    """Create a real KyroDB router."""
    config = KyroDBConfig(
        text_host="localhost",
        text_port=50051,
        image_host="localhost",
        image_port=50052,
        enable_tls=False,
        request_timeout_seconds=30,
    )
    router = KyroDBRouter(config=config)
    await router.connect()
    yield router
    await router.close()


def generate_embedding(seed: float = 0.1, dim: int = 384) -> list[float]:
    """Generate a 384-dimensional test embedding."""
    # Create varied embedding by using seed to offset values
    return [seed + (i * 0.0001) for i in range(dim)]


class TestFullPipelineWithRealKyroDB:
    """End-to-end tests using real KyroDB instance."""

    @pytest.mark.asyncio
    async def test_capture_reflect_search_pipeline(self, skip_if_no_kyrodb, kyrodb_router: KyroDBRouter):
        """
        Full pipeline: Capture → Reflect → Persist → Search → Retrieve
        
        This tests the complete episode lifecycle with real KyroDB.
        """
        customer_id = f"test-pipeline-{int(time.time())}"
        collection = "failures"
        
        # Generate unique episode ID
        episode_id = int(time.time() * 1000) % (2**63 - 1)
        
        print(f"\n{'='*60}")
        print(f"Full Pipeline Test - Episode ID: {episode_id}")
        print(f"Customer: {customer_id}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Create episode with initial metadata (simulating capture)
            print("\n--- Step 1: Capture Episode ---")
            
            episode_embedding = generate_embedding(0.1)
            episode_metadata = {
                "customer_id": customer_id,
                "episode_type": "failure",
                "goal": "Deploy application to production",
                "tool_chain": "kubectl,docker",
                "error_class": "ImagePullBackOff",
                "error_trace": "Error: ImagePullBackOff - manifest not found",
                "created_at": datetime.now(UTC).isoformat(),
            }
            
            text_success, _ = await kyrodb_router.insert_episode(
                episode_id=episode_id,
                customer_id=customer_id,
                collection=collection,
                text_embedding=episode_embedding,
                metadata=episode_metadata,
            )
            
            assert text_success, "Episode capture failed"
            print(f"  Episode {episode_id} captured successfully")
            
            # Step 2: Generate and persist reflection
            print("\n--- Step 2: Generate Reflection ---")
            
            reflection = Reflection(
                consensus=None,
                root_cause="Docker image tag 'latest' not found in registry",
                preconditions=["Using 'latest' tag", "Registry configured"],
                resolution_strategy="Use specific image version tag instead of 'latest'",
                environment_factors=["docker_registry", "image_tag"],
                affected_components=["deployment", "container"],
                generalization_score=0.75,
                confidence_score=0.85,
                llm_model="openrouter-consensus",
                generated_at=datetime.now(UTC),
                cost_usd=0.0025,
                generation_latency_ms=1200.0,
                tier=ReflectionTier.PREMIUM.value,
            )
            
            # Update episode with reflection
            update_success = await kyrodb_router.update_episode_reflection(
                episode_id=episode_id,
                customer_id=customer_id,
                collection=collection,
                reflection=reflection,
            )
            
            assert update_success, "Reflection persistence failed"
            print(f"  Reflection persisted for episode {episode_id}")
            print(f"    Root cause: {reflection.root_cause[:50]}...")
            print(f"    Confidence: {reflection.confidence_score:.2f}")
            
            # Step 3: Verify episode with reflection is searchable
            print("\n--- Step 3: Search for Similar Episodes ---")
            
            # Create a slightly different query embedding (should still match)
            query_embedding = generate_embedding(0.11)  # Similar to original
            
            search_response = await kyrodb_router.search_episodes(
                query_embedding=query_embedding,
                customer_id=customer_id,
                collection=collection,
                k=10,
                min_score=0.5,
            )
            
            print(f"  Search returned {len(search_response.results)} results")
            
            # Should find our episode
            found_episode = None
            for result in search_response.results:
                print(f"    Doc {result.doc_id}: score={result.score:.4f}")
                if result.doc_id == episode_id:
                    found_episode = result
            
            assert found_episode is not None, f"Episode {episode_id} not found in search results"
            print(f"  Found target episode with score: {found_episode.score:.4f}")
            
            # Step 4: Retrieve full episode and verify reflection
            print("\n--- Step 4: Retrieve Full Episode ---")
            
            # Get episode with customer_id for namespace isolation
            retrieved_dict = await kyrodb_router.get_episode(
                episode_id=episode_id,
                customer_id=customer_id,
                collection=collection,
            )
            
            assert retrieved_dict is not None and retrieved_dict.get("found"), "Episode not found after search"
            
            metadata = retrieved_dict.get("metadata", {})
            print(f"  Retrieved episode with {len(metadata)} metadata fields")
            
            # Verify reflection fields
            assert "reflection_root_cause" in metadata, "Missing reflection_root_cause"
            assert "reflection_confidence" in metadata, "Missing reflection_confidence"
            assert "reflection_model" in metadata, "Missing reflection_model"
            
            print(f"    Root cause: {metadata.get('reflection_root_cause', 'N/A')[:50]}...")
            print(f"    Confidence: {metadata.get('reflection_confidence', 'N/A')}")
            print(f"    Model: {metadata.get('reflection_model', 'N/A')}")
            
            # Step 5: Simulate precondition matching
            print("\n--- Step 5: Precondition Matching ---")
            
            # Search with precondition-relevant query
            precondition_embedding = generate_embedding(0.105)  # Very similar
            
            precondition_results = await kyrodb_router.search_episodes(
                query_embedding=precondition_embedding,
                customer_id=customer_id,
                collection=collection,
                k=5,
                min_score=0.7,
            )
            
            print(f"  Precondition search returned {len(precondition_results.results)} results")
            
            if precondition_results.results:
                top_result = precondition_results.results[0]
                print(f"    Top match: Doc {top_result.doc_id} (score={top_result.score:.4f})")
                
                # Would trigger BLOCK or REWRITE recommendation
                if top_result.score > 0.9:
                    print("    Recommendation: BLOCK (high similarity)")
                elif top_result.score > 0.7:
                    print("    Recommendation: REWRITE or HINT")
            
            print(f"\n{'='*60}")
            print("Full Pipeline Test PASSED")
            print(f"{'='*60}")
            
        finally:
            # Cleanup
            print("\n--- Cleanup ---")
            await kyrodb_router.delete_episode(
                episode_id=episode_id,
                customer_id=customer_id,
                collection=collection,
            )
            print(f"  Episode {episode_id} deleted")

    @pytest.mark.asyncio
    async def test_search_latency_benchmark(self, skip_if_no_kyrodb, kyrodb_router: KyroDBRouter):
        """
        Benchmark search latency with 100 episodes.
        Target: <50ms P99
        """
        customer_id = f"test-bench-{int(time.time())}"
        collection = "failures"
        base_id = int(time.time() * 1000) % (2**60)
        episode_ids = []
        
        print(f"\n{'='*60}")
        print("Search Latency Benchmark")
        print(f"{'='*60}")
        
        try:
            # Insert 100 test episodes
            print("\n--- Inserting 100 test episodes ---")
            
            for i in range(100):
                episode_id = base_id + i
                episode_ids.append(episode_id)
                
                embedding = generate_embedding(0.1 + (i * 0.001))
                metadata = {
                    "customer_id": customer_id,
                    "episode_type": "failure",
                    "goal": f"Test goal {i}",
                    "index": str(i),
                }
                
                await kyrodb_router.insert_episode(
                    episode_id=episode_id,
                    customer_id=customer_id,
                    collection=collection,
                    text_embedding=embedding,
                    metadata=metadata,
                )
            
            print(f"  Inserted {len(episode_ids)} episodes")
            
            # Run 50 searches and measure latency
            print("\n--- Running 50 search queries ---")
            latencies = []
            
            for i in range(50):
                query_embedding = generate_embedding(0.1 + (i * 0.002))
                
                start = time.perf_counter()
                await kyrodb_router.search_episodes(
                    query_embedding=query_embedding,
                    customer_id=customer_id,
                    collection=collection,
                    k=10,
                    min_score=0.5,
                )
                end = time.perf_counter()
                
                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)
            
            # Calculate statistics
            latencies.sort()
            avg_latency = sum(latencies) / len(latencies)
            p50_latency = latencies[len(latencies) // 2]
            p95_latency = latencies[int(len(latencies) * 0.95)]
            p99_latency = latencies[int(len(latencies) * 0.99)]
            min_latency = latencies[0]
            max_latency = latencies[-1]
            
            print("\n--- Search Latency Results ---")
            print("  Queries: 50")
            print("  Episodes: 100")
            print(f"  Min: {min_latency:.2f}ms")
            print(f"  Avg: {avg_latency:.2f}ms")
            print(f"  P50: {p50_latency:.2f}ms")
            print(f"  P95: {p95_latency:.2f}ms")
            print(f"  P99: {p99_latency:.2f}ms")
            print(f"  Max: {max_latency:.2f}ms")
            
            # Assert target
            assert p99_latency < 50.0, f"P99 latency {p99_latency:.2f}ms exceeds 50ms target"
            print(f"\n  Target P99 < 50ms: PASSED ({p99_latency:.2f}ms)")
            
        finally:
            # Cleanup
            print("\n--- Cleanup ---")
            deleted = 0
            for episode_id in episode_ids:
                try:
                    await kyrodb_router.delete_episode(
                        episode_id=episode_id,
                        customer_id=customer_id,
                        collection=collection,
                    )
                    deleted += 1
                except Exception:
                    pass
            print(f"  Deleted {deleted}/{len(episode_ids)} episodes")

    @pytest.mark.asyncio
    async def test_reflection_searchable_after_persistence(self, skip_if_no_kyrodb, kyrodb_router: KyroDBRouter):
        """
        Verify reflection metadata is searchable via vector similarity.
        """
        customer_id = f"test-reflection-search-{int(time.time())}"
        collection = "failures"
        episode_id = int(time.time() * 1000) % (2**63 - 1)
        
        print(f"\n{'='*60}")
        print("Reflection Searchability Test")
        print(f"{'='*60}")
        
        try:
            # Insert episode
            embedding = generate_embedding(0.2)
            metadata = {
                "customer_id": customer_id,
                "episode_type": "failure",
                "goal": "Fix authentication error",
                "error_class": "AuthenticationError",
            }
            
            await kyrodb_router.insert_episode(
                episode_id=episode_id,
                customer_id=customer_id,
                collection=collection,
                text_embedding=embedding,
                metadata=metadata,
            )
            print(f"\n  Inserted episode {episode_id}")
            
            # Add reflection
            reflection = Reflection(
                consensus=None,
                root_cause="OAuth token expired",
                preconditions=["OAuth configured", "Token has expiry"],
                resolution_strategy="Implement token refresh mechanism",
                environment_factors=["oauth_provider"],
                affected_components=["auth_service"],
                generalization_score=0.8,
                confidence_score=0.9,
                llm_model="test-model",
                generated_at=datetime.now(UTC),
                cost_usd=0.001,
                generation_latency_ms=500.0,
                tier=ReflectionTier.CHEAP.value,
            )
            
            await kyrodb_router.update_episode_reflection(
                episode_id=episode_id,
                customer_id=customer_id,
                collection=collection,
                reflection=reflection,
            )
            print("  Added reflection to episode")
            
            # Search and verify reflection fields in results
            search_response = await kyrodb_router.search_episodes(
                query_embedding=embedding,
                customer_id=customer_id,
                collection=collection,
                k=5,
            )
            
            assert len(search_response.results) > 0, "No search results"
            
            # Get full metadata of first result
            top_result = search_response.results[0]
            result_metadata = dict(top_result.metadata)
            
            print("\n  Search result metadata fields:")
            for key in sorted(result_metadata.keys()):
                if key.startswith("reflection_"):
                    value = result_metadata[key]
                    if len(str(value)) > 50:
                        value = str(value)[:50] + "..."
                    print(f"    {key}: {value}")
            
            # Verify reflection fields are present
            assert "reflection_root_cause" in result_metadata
            assert "reflection_resolution" in result_metadata
            assert "reflection_confidence" in result_metadata
            
            print("\n  Reflection searchability: VERIFIED")
            
        finally:
            await kyrodb_router.delete_episode(
                episode_id=episode_id,
                customer_id=customer_id,
                collection=collection,
            )
            print(f"\n  Cleaned up episode {episode_id}")


class TestPreconditionMatching:
    """Test precondition matching for pre-action gating."""

    @pytest.mark.asyncio
    async def test_precondition_similarity_scoring(self, skip_if_no_kyrodb, kyrodb_router: KyroDBRouter):
        """
        Test that similar preconditions return high similarity scores.
        """
        customer_id = f"test-precondition-{int(time.time())}"
        collection = "failures"
        base_id = int(time.time() * 1000) % (2**60)
        episode_ids = []
        
        print(f"\n{'='*60}")
        print("Precondition Matching Test")
        print(f"{'='*60}")
        
        try:
            # Insert episodes with varying preconditions
            scenarios = [
                {
                    "id": base_id + 1,
                    "embedding": generate_embedding(0.1),
                    "goal": "Deploy to production",
                    "preconditions": "kubernetes_cluster,docker_registry",
                },
                {
                    "id": base_id + 2,
                    "embedding": generate_embedding(0.15),
                    "goal": "Deploy to staging",
                    "preconditions": "kubernetes_cluster,staging_env",
                },
                {
                    "id": base_id + 3,
                    "embedding": generate_embedding(0.5),
                    "goal": "Run database migration",
                    "preconditions": "postgres_running,backup_complete",
                },
            ]
            
            print("\n--- Inserting test scenarios ---")
            for scenario in scenarios:
                episode_ids.append(scenario["id"])
                await kyrodb_router.insert_episode(
                    episode_id=scenario["id"],
                    customer_id=customer_id,
                    collection=collection,
                    text_embedding=scenario["embedding"],
                    metadata={
                        "customer_id": customer_id,
                        "goal": scenario["goal"],
                        "preconditions": scenario["preconditions"],
                    },
                )
                print(f"  Inserted: {scenario['goal']}")
            
            # Query similar to first scenario (deploy to production)
            print("\n--- Searching for 'Deploy to production' similarity ---")
            query_embedding = generate_embedding(0.1)  # Exact match
            
            results = await kyrodb_router.search_episodes(
                query_embedding=query_embedding,
                customer_id=customer_id,
                collection=collection,
                k=3,
            )
            
            print("\n  Results (ordered by similarity):")
            for i, result in enumerate(results.results):
                goal = result.metadata.get("goal", "N/A")
                preconditions = result.metadata.get("preconditions", "N/A")
                print(f"    {i+1}. Score={result.score:.4f}: {goal}")
                print(f"       Preconditions: {preconditions}")
            
            # Explicit assertion that results were returned - do not silently skip
            assert len(results.results) > 0, (
                "Search should return at least one result. "
                "Verify episodes were inserted and collection namespace is correct."
            )
            
            # First result should be very similar
            assert results.results[0].score > 0.9, (
                f"Top result should have high similarity (>0.9), got {results.results[0].score:.4f}"
            )
            print(f"\n  Top match similarity: {results.results[0].score:.4f} (>0.9 required)")
            
            # Query for database migration (different domain)
            print("\n--- Searching for 'Database migration' similarity ---")
            db_embedding = generate_embedding(0.5)  # Match migration scenario
            
            db_results = await kyrodb_router.search_episodes(
                query_embedding=db_embedding,
                customer_id=customer_id,
                collection=collection,
                k=3,
            )
            
            # Explicit assertion for second search as well
            assert len(db_results.results) > 0, (
                "Database migration search should return results"
            )
            top_goal = db_results.results[0].metadata.get("goal", "N/A")
            print(f"  Top match: {top_goal} (score={db_results.results[0].score:.4f})")
            
        finally:
            print("\n--- Cleanup ---")
            for episode_id in episode_ids:
                await kyrodb_router.delete_episode(
                    episode_id=episode_id,
                    customer_id=customer_id,
                    collection=collection,
                )
            print(f"  Deleted {len(episode_ids)} episodes")
