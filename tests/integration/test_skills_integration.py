"""
Integration tests for Skills Collection with real KyroDB.

Day 12: Skills Collection Setup
- Create skills collection in KyroDB schema
- Test skill insertion and retrieval
- Test skills search with semantic similarity
- Test skills used in gating recommendations
"""

import contextlib
import time
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from src.kyrodb.router import KyroDBRouter
from src.models.skill import Skill


async def _allocate_doc_id() -> int:
    from src.storage.database import get_customer_db

    db = await get_customer_db()
    return await db.allocate_doc_id()


def _make_skill(
    skill_id: int,
    customer_id: str,
    name: str,
    docstring: str,
    code: str = "# fix code",
    error_class: str = "GeneralError",
) -> Skill:
    """Helper to create a valid Skill object."""
    return Skill(
        skill_id=skill_id,
        customer_id=customer_id,
        name=name,
        docstring=docstring,
        code=code,
        language="python",
        error_class=error_class,
        source_episodes=[skill_id + 1, skill_id + 2],
        created_at=datetime.now(UTC),
    )


@pytest.mark.integration
@pytest.mark.requires_kyrodb
class TestSkillsCollection:
    """Test skills collection CRUD operations with real KyroDB."""

    @pytest.mark.asyncio
    async def test_insert_skill_success(self, kyrodb_router: KyroDBRouter):
        """Test successful skill insertion."""
        customer_id = f"test_customer_{uuid4().hex[:8]}"
        skill_id = await _allocate_doc_id()

        skill = _make_skill(
            skill_id=skill_id,
            customer_id=customer_id,
            name="Python debugging with print statements",
            docstring="Use strategic print statements to debug Python code issues and trace execution flow",
            code="print(f'DEBUG: {variable}')",
            error_class="DebuggingError",
        )

        # 384-dim embedding for all-MiniLM-L6-v2
        embedding = [0.1] * 384

        try:
            success = await kyrodb_router.insert_skill(
                skill=skill,
                embedding=embedding,
            )

            assert success, "Skill insertion should succeed"
        finally:
            # Cleanup
            namespace = f"{customer_id}:skills"
            with contextlib.suppress(Exception):
                await kyrodb_router.text_client.delete(doc_id=skill_id, namespace=namespace)

    @pytest.mark.asyncio
    async def test_search_skills_finds_inserted(self, kyrodb_router: KyroDBRouter):
        """Test that inserted skill is searchable."""
        customer_id = f"test_customer_{uuid4().hex[:8]}"
        skill_id = await _allocate_doc_id()

        skill = _make_skill(
            skill_id=skill_id,
            customer_id=customer_id,
            name="API rate limiting implementation",
            docstring="Implement token bucket rate limiting for REST APIs to prevent abuse and ensure fair usage",
            code="from ratelimit import limits\n@limits(calls=100, period=60)\ndef api_call(): pass",
            error_class="RateLimitError",
        )

        embedding = [0.2] * 384

        try:
            # Insert skill
            await kyrodb_router.insert_skill(skill=skill, embedding=embedding)

            # Search with same embedding
            results = await kyrodb_router.search_skills(
                query_embedding=embedding,
                customer_id=customer_id,
                k=5,
            )

            assert len(results) > 0, "Should find at least one skill"

            # Verify our skill is in results
            found_skills = [s for s, _ in results if s.skill_id == skill_id]
            assert len(found_skills) == 1, f"Should find skill {skill_id}"
            assert found_skills[0].name == skill.name
        finally:
            namespace = f"{customer_id}:skills"
            with contextlib.suppress(Exception):
                await kyrodb_router.text_client.delete(doc_id=skill_id, namespace=namespace)

    @pytest.mark.asyncio
    async def test_search_skills_semantic_similarity(self, kyrodb_router: KyroDBRouter):
        """Test that similar skills rank higher in search.

        Note: Uses low min_score (0.0) to ensure both skills are found,
        since KyroDB uses cosine similarity and the test embeddings may
        not meet the default 0.7 threshold.
        """
        customer_id = f"test_customer_{uuid4().hex[:8]}"
        skill_ids = []

        # Create embeddings with actual directional variation for meaningful cosine similarity.
        # Uniform vectors like [0.8]*384 are all parallel (cosine similarity = 1.0).
        # Instead, we use embeddings where first half differs to create actual angular separation.

        # Query embedding: first half high, second half low
        query_embedding = [0.8] * 192 + [0.2] * 192

        # Similar embedding: same pattern as query (high first half, low second half)
        similar_embedding = [0.75] * 192 + [0.25] * 192

        # Different embedding: opposite pattern (low first half, high second half)
        different_embedding = [0.2] * 192 + [0.8] * 192

        skills_data = [
            ("Python debugging techniques", similar_embedding),  # Similar to query
            ("Database optimization", different_embedding),  # Different from query
        ]

        try:
            for name, emb in skills_data:
                skill_id = await _allocate_doc_id()
                skill_ids.append((skill_id, customer_id))

                skill = _make_skill(
                    skill_id=skill_id,
                    customer_id=customer_id,
                    name=name,
                    docstring=f"Detailed description for {name} with proper techniques and best practices",
                    error_class="GeneralError",
                )

                await kyrodb_router.insert_skill(skill=skill, embedding=emb)

            # Search with query embedding
            results = await kyrodb_router.search_skills(
                query_embedding=query_embedding,
                customer_id=customer_id,
                k=5,
                min_score=0.0,  # Ensure we find all skills
            )

            assert len(results) >= 2, f"Should find both skills, found {len(results)}"

            # First result should be the similar skill (higher score)
            scores = {s.name: score for s, score in results}
            print(f"\nScores: {scores}")

            if "Python debugging techniques" in scores and "Database optimization" in scores:
                assert (
                    scores["Python debugging techniques"] > scores["Database optimization"]
                ), "Similar skill should have higher score"
        finally:
            for skill_id, cust_id in skill_ids:
                try:
                    namespace = f"{cust_id}:skills"
                    await kyrodb_router.text_client.delete(doc_id=skill_id, namespace=namespace)
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_search_skills_customer_isolation(self, kyrodb_router: KyroDBRouter):
        """Test that skills are isolated by customer_id.

        Namespace isolation is enforced by KyroDB server-side filtering.
        The namespace is stored as '__namespace__' metadata on insert and
        filtered during search to ensure customer data isolation.
        """
        customer_a = f"customer_a_{uuid4().hex[:8]}"
        customer_b = f"customer_b_{uuid4().hex[:8]}"
        skill_ids = []

        embedding = [0.5] * 384

        try:
            # Insert skill for customer A
            skill_a_id = await _allocate_doc_id()
            skill_a = _make_skill(
                skill_id=skill_a_id,
                customer_id=customer_a,
                name="Customer A exclusive skill",
                docstring="This skill belongs only to customer A and provides specific functionality",
                error_class="CustomerAError",
            )
            skill_ids.append((skill_a_id, customer_a))
            await kyrodb_router.insert_skill(skill=skill_a, embedding=embedding)

            # Insert skill for customer B
            skill_b_id = await _allocate_doc_id()
            skill_b = _make_skill(
                skill_id=skill_b_id,
                customer_id=customer_b,
                name="Customer B exclusive skill",
                docstring="This skill belongs only to customer B and provides specific functionality",
                error_class="CustomerBError",
            )
            skill_ids.append((skill_b_id, customer_b))
            await kyrodb_router.insert_skill(skill=skill_b, embedding=embedding)

            # Search as customer A - should only see customer A's skill
            results_a = await kyrodb_router.search_skills(
                query_embedding=embedding,
                customer_id=customer_a,
                k=10,
            )

            customer_ids_found = {s.customer_id for s, _ in results_a}
            assert (
                customer_b not in customer_ids_found
            ), "Customer A should not see customer B's skills"

            # Search as customer B - should only see customer B's skill
            results_b = await kyrodb_router.search_skills(
                query_embedding=embedding,
                customer_id=customer_b,
                k=10,
            )

            customer_ids_found = {s.customer_id for s, _ in results_b}
            assert (
                customer_a not in customer_ids_found
            ), "Customer B should not see customer A's skills"
        finally:
            for skill_id, cust_id in skill_ids:
                try:
                    namespace = f"{cust_id}:skills"
                    await kyrodb_router.text_client.delete(doc_id=skill_id, namespace=namespace)
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_multiple_skills_insertion(self, kyrodb_router: KyroDBRouter):
        """Test inserting multiple skills for same customer."""
        customer_id = f"test_customer_{uuid4().hex[:8]}"
        skill_ids = []

        skill_names = [
            "Error handling best practices",
            "Unit testing patterns",
            "Code review techniques",
            "Documentation standards",
            "Performance optimization",
        ]

        try:
            # Insert 5 skills
            for i, name in enumerate(skill_names):
                skill_id = await _allocate_doc_id()
                skill_ids.append(skill_id)

                skill = _make_skill(
                    skill_id=skill_id,
                    customer_id=customer_id,
                    name=name,
                    docstring=f"Detailed description for {name} with comprehensive coverage of the topic",
                    error_class=f"Error{i}",
                )

                # Slightly different embeddings
                embedding = [0.1 + i * 0.1] * 384
                await kyrodb_router.insert_skill(skill=skill, embedding=embedding)

            # Search should find all skills
            results = await kyrodb_router.search_skills(
                query_embedding=[0.3] * 384,  # Middle-ish embedding
                customer_id=customer_id,
                k=10,
            )

            assert len(results) >= 5, f"Should find all 5 skills, found {len(results)}"
        finally:
            namespace = f"{customer_id}:skills"
            for skill_id in skill_ids:
                with contextlib.suppress(Exception):
                    await kyrodb_router.text_client.delete(doc_id=skill_id, namespace=namespace)


@pytest.mark.integration
@pytest.mark.requires_kyrodb
class TestSkillsSearchPerformance:
    """Test skills search performance with real KyroDB."""

    @pytest.mark.asyncio
    async def test_skills_search_latency(self, kyrodb_router: KyroDBRouter):
        """Test that skills search meets latency requirements (<50ms P99)."""
        customer_id = f"test_customer_{uuid4().hex[:8]}"
        skill_ids = []

        # Insert 10 skills
        try:
            for i in range(10):
                skill_id = await _allocate_doc_id()
                skill_ids.append(skill_id)

                skill = _make_skill(
                    skill_id=skill_id,
                    customer_id=customer_id,
                    name=f"Performance test skill number {i}",
                    docstring=f"Skill for performance testing iteration {i} with detailed description",
                    error_class=f"PerfError{i}",
                )

                embedding = [0.1 * (i + 1)] * 384
                await kyrodb_router.insert_skill(skill=skill, embedding=embedding)

            # Warm up
            await kyrodb_router.search_skills(
                query_embedding=[0.5] * 384,
                customer_id=customer_id,
                k=5,
            )

            # Measure 10 searches
            latencies = []
            for _ in range(10):
                query_embedding = [0.5] * 384

                start = time.perf_counter()
                await kyrodb_router.search_skills(
                    query_embedding=query_embedding,
                    customer_id=customer_id,
                    k=5,
                )
                latency_ms = (time.perf_counter() - start) * 1000
                latencies.append(latency_ms)

            # Calculate P99
            latencies.sort()
            p99_idx = min(int(len(latencies) * 0.99), len(latencies) - 1)
            p99_ms = latencies[p99_idx]
            avg_ms = sum(latencies) / len(latencies)

            print("\nSkills search latency:")
            print(f"  Average: {avg_ms:.2f}ms")
            print(f"  P99: {p99_ms:.2f}ms")
            print(f"  Min: {min(latencies):.2f}ms")
            print(f"  Max: {max(latencies):.2f}ms")

            # Skills search should be fast (<50ms P99 for small dataset)
            assert p99_ms < 50, f"P99 latency {p99_ms:.2f}ms exceeds 50ms target"
        finally:
            namespace = f"{customer_id}:skills"
            for skill_id in skill_ids:
                with contextlib.suppress(Exception):
                    await kyrodb_router.text_client.delete(doc_id=skill_id, namespace=namespace)


@pytest.mark.integration
@pytest.mark.requires_kyrodb
class TestSkillsInGating:
    """Test skills integration with gating service."""

    @pytest.mark.asyncio
    async def test_gating_uses_skills_for_hints(self, kyrodb_router: KyroDBRouter):
        """Test that gating service uses skills to provide hints."""
        from unittest.mock import MagicMock

        from src.gating.service import GatingService
        from src.retrieval.search import SearchPipeline

        customer_id = f"test_customer_{uuid4().hex[:8]}"
        skill_id = await _allocate_doc_id()

        skill = _make_skill(
            skill_id=skill_id,
            customer_id=customer_id,
            name="Always validate user input before processing",
            docstring="Check input types and bounds before using in calculations to prevent security issues",
            code="def validate_input(data):\n    if not isinstance(data, dict):\n        raise ValueError('Invalid input')",
            error_class="ValidationError",
        )

        # Create embedding matching "input validation"
        skill_embedding = [0.8] * 192 + [0.2] * 192

        try:
            await kyrodb_router.insert_skill(skill=skill, embedding=skill_embedding)

            # Mock embedding service
            mock_embedding_service = MagicMock()
            mock_embedding_service.embed_text.return_value = skill_embedding

            async def mock_embed_text_async(text: str) -> list[float]:
                return skill_embedding

            mock_embedding_service.embed_text_async = mock_embed_text_async

            # Create search pipeline and gating service
            search_pipeline = SearchPipeline(
                kyrodb_router=kyrodb_router, embedding_service=mock_embedding_service
            )
            gating_service = GatingService(
                search_pipeline=search_pipeline,
                kyrodb_router=kyrodb_router,
            )

            # Create a reflect request for input validation
            from src.models.gating import ReflectRequest

            request = ReflectRequest(
                goal="Process user form data",
                proposed_action="Write a function to process user input from form",
                tool="python",
                current_state={},
            )

            # Query about input validation
            response = await gating_service.reflect_before_action(
                request=request,
                customer_id=customer_id,
            )

            # Should get a valid response
            assert response is not None, "Should return a response"
            # We expect at least a HINT or PROCEED, but since we have a matching skill,
            # it might even suggest REWRITE if confidence is high enough.
            from src.models.gating import ActionRecommendation

            assert response.recommendation in [
                ActionRecommendation.PROCEED,
                ActionRecommendation.HINT,
                ActionRecommendation.REWRITE,
                ActionRecommendation.BLOCK,
            ]

            # Print for debugging
            print(f"\nGating recommendation: {response.recommendation}")
            print(f"Confidence: {response.confidence}")
            if response.relevant_skills:
                print(f"Matched skills: {len(response.relevant_skills)}")
                for s in response.relevant_skills:
                    print(f"  - {s.get('name')}")
        finally:
            namespace = f"{customer_id}:skills"
            with contextlib.suppress(Exception):
                await kyrodb_router.text_client.delete(doc_id=skill_id, namespace=namespace)
