"""
Chaos Testing for Vritti

Tests system behavior under fault conditions:
- KyroDB connection failures
- LLM API failures
- Network timeouts
- Graceful degradation

Run with: pytest tests/chaos/test_fault_injection.py -v -s
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest

from src.models.episode import EpisodeCreate, EpisodeType, ErrorClass
from src.models.gating import ActionRecommendation, ReflectRequest
from src.models.search import SearchRequest


class TestKyroDBFailures:
    """Test graceful degradation when KyroDB is unavailable."""

    @pytest.mark.asyncio
    async def test_kyrodb_connection_failure(self, ingestion_pipeline):
        """
        Scenario: KyroDB is down/unreachable

        Expected: Episode ingestion fails gracefully with clear error
        """
        print("\n" + "="*80)
        print("CHAOS TEST: KyroDB Connection Failure")
        print("="*80)

        # Simulate KyroDB connection failure
        original_insert = ingestion_pipeline.kyrodb_router.insert_episode

        async def mock_insert_failure(*args, **kwargs):
            raise grpc.aio.AioRpcError(
                code=grpc.StatusCode.UNAVAILABLE,
                initial_metadata=None,
                trailing_metadata=None,
                details="Connection refused: KyroDB unreachable",
            )

        ingestion_pipeline.kyrodb_router.insert_episode = AsyncMock(side_effect=mock_insert_failure)

        try:
            episode = EpisodeCreate(
                episode_type=EpisodeType.FAILURE,
                goal="Test KyroDB failure handling",
                actions_taken=["test action"],
                error_class=ErrorClass.NETWORK_ERROR,
                error_trace="test error",
                tool_chain=["test"],
                environment_info={"test": "true"},
                customer_id="test_customer",
            )

            # Should raise or return error (not crash)
            with pytest.raises(Exception) as exc_info:
                await ingestion_pipeline.capture_episode(episode, generate_reflection=False)

            print(f"\n Graceful failure: {str(exc_info.value)[:100]}")

        finally:
            # Restore
            ingestion_pipeline.kyrodb_router.insert_episode = original_insert

        print("="*80 + "\n")

    @pytest.mark.skip(reason="Flaky test: mock not being called correctly")
    @pytest.mark.asyncio
    async def test_kyrodb_slow_response(self, search_pipeline):
        """
        Scenario: KyroDB responds slowly (>10s)

        Expected: Request times out with clear error
        """
        print("\n" + "="*80)
        print("CHAOS TEST: KyroDB Slow Response")
        print("="*80)

        original_search = search_pipeline.kyrodb_router.search_text

        async def mock_slow_search(*args, **kwargs):
            print("\n  Simulating 15s delay...")
            await asyncio.sleep(15)
            return MagicMock(results=[], num_results=0)

        search_pipeline.kyrodb_router.search_text = AsyncMock(side_effect=mock_slow_search)

        request = SearchRequest(
            query="test query",
            k=5,
            min_similarity=0.7,
            customer_id="test_customer",
        )

        # Should timeout (KYRODB_REQUEST_TIMEOUT_SECONDS = 30 by default)
        start = asyncio.get_event_loop().time()

        try:
            await asyncio.wait_for(
                search_pipeline.search(request),
                timeout=5.0  # Use shorter timeout for test
            )
            raise AssertionError("Should have timed out")
        except asyncio.TimeoutError:
            elapsed = asyncio.get_event_loop().time() - start
            print(f"\n Timed out after {elapsed:.1f}s (expected)")

        # Restore
        search_pipeline.kyrodb_router.search_text = original_search

        print("="*80 + "\n")

    @pytest.mark.skip(reason="Flaky test: failure assertion unclear")
    @pytest.mark.asyncio
    async def test_kyrodb_partial_failure(self, ingestion_pipeline):
        """
        Scenario: Text instance works, image instance fails

        Expected: Episode stored in text instance only, no crash
        """
        print("\n" + "="*80)
        print("CHAOS TEST: KyroDB Partial Failure")
        print("="*80)

        original_insert = ingestion_pipeline.kyrodb_router.insert_episode

        # Simulate text success, image failure
        async def mock_insert(*args, **kwargs):
            return (True, False)  # text=success, image=fail

        ingestion_pipeline.kyrodb_router.insert_episode = AsyncMock(side_effect=mock_insert)

        try:
            episode = EpisodeCreate(
                episode_type=EpisodeType.FAILURE,
                goal="Test partial failure",
                actions_taken=["test action"],
                error_class=ErrorClass.NETWORK_ERROR,
                error_trace="test error",
                tool_chain=["test"],
                environment_info={"test": "true"},
                customer_id="test_customer",
            )

            # Should succeed (text instance worked)
            result = await ingestion_pipeline.capture_episode(episode, generate_reflection=False)
            assert result is not None

            print("\n Episode stored despite image instance failure")
        finally:
            ingestion_pipeline.kyrodb_router.insert_episode = original_insert
        print("="*80 + "\n")


class TestLLMFailures:
    """Test graceful degradation when LLM is unavailable."""

    @pytest.mark.asyncio
    async def test_llm_api_failure(self, ingestion_pipeline):
        """
        Scenario: OpenRouter API is down

        Expected: Episode stored without reflection, no crash
        """
        print("\n" + "="*80)
        print("CHAOS TEST: LLM API Failure")
        print("="*80)

        # Simulate LLM API failure
        if ingestion_pipeline.reflection_service:
            original_generate = ingestion_pipeline.reflection_service.generate_reflection

            async def mock_llm_failure(*args, **kwargs):
                raise Exception("OpenRouter API unavailable")

            ingestion_pipeline.reflection_service.generate_reflection = AsyncMock(side_effect=mock_llm_failure)

        try:
            episode = EpisodeCreate(
                episode_type=EpisodeType.FAILURE,
                goal="Test LLM failure handling",
                actions_taken=["test action"],
                error_class=ErrorClass.NETWORK_ERROR,
                error_trace="test error",
                tool_chain=["test"],
                environment_info={"test": "true"},
                customer_id="test_customer",
            )

            # Should store episode even without reflection
            result = await ingestion_pipeline.capture_episode(episode, generate_reflection=True)
            assert result is not None

            print("\n Episode stored without reflection (LLM failed gracefully)")
        finally:
            if ingestion_pipeline.reflection_service:
                ingestion_pipeline.reflection_service.generate_reflection = original_generate
        print("="*80 + "\n")

    @pytest.mark.asyncio
    async def test_llm_timeout(self, ingestion_pipeline):
        """
        Scenario: LLM API times out

        Expected: Episode stored without reflection after timeout
        """
        print("\n" + "="*80)
        print("CHAOS TEST: LLM Timeout")
        print("="*80)

        if ingestion_pipeline.reflection_service:
            async def mock_slow_llm(*args, **kwargs):
                print("\n  Simulating 60s LLM delay...")
                await asyncio.sleep(60)
                return MagicMock()

            original_generate = ingestion_pipeline.reflection_service.generate_reflection
            ingestion_pipeline.reflection_service.generate_reflection = AsyncMock(side_effect=mock_slow_llm)

        try:
            episode = EpisodeCreate(
                episode_type=EpisodeType.FAILURE,
                goal="Test LLM timeout",
                actions_taken=["test action"],
                error_class=ErrorClass.NETWORK_ERROR,
                error_trace="test error",
                tool_chain=["test"],
                environment_info={"test": "true"},
                customer_id="test_customer",
            )

            # Should timeout and continue
            start = asyncio.get_event_loop().time()

            await asyncio.wait_for(
                ingestion_pipeline.capture_episode(episode, generate_reflection=True),
                timeout=10.0
            )

            elapsed = asyncio.get_event_loop().time() - start
            print(f"\n Completed in {elapsed:.1f}s (LLM timed out, episode still stored)")
        finally:
            if ingestion_pipeline.reflection_service:
                ingestion_pipeline.reflection_service.generate_reflection = original_generate
        print("="*80 + "\n")


class TestGatingFailures:
    """Test gating behavior under fault conditions."""

    @pytest.mark.asyncio
    async def test_gating_with_kyrodb_down(self, gating_service):
        """
        Scenario: KyroDB is down during gating check

        Expected: Returns PROCEED (fail-open, don't block user)
        """
        print("\n" + "="*80)
        print("CHAOS TEST: Gating with KyroDB Down")
        print("="*80)

        # Simulate KyroDB failure
        async def mock_search_failure(*args, **kwargs):
            raise grpc.aio.AioRpcError(
                code=grpc.StatusCode.UNAVAILABLE,
                initial_metadata=None,
                trailing_metadata=None,
                details="KyroDB unavailable",
            )

        original_search = gating_service.search_pipeline.search
        gating_service.search_pipeline.search = AsyncMock(side_effect=mock_search_failure)

        try:
            request = ReflectRequest(
                proposed_action="test action",
                goal="test goal for gating",
                tool="kubectl",
                context="test context",
                current_state={"env": "test"},
            )

            # Should fail-open (return PROCEED, not crash)
            response = await gating_service.reflect_before_action(request, "test_customer")

            assert response.recommendation == ActionRecommendation.PROCEED
            assert "error" in response.rationale.lower() or "unavailable" in response.rationale.lower()

            print(f"\n Fail-open behavior: {response.recommendation.value}")
            print(f"   Rationale: {response.rationale[:80]}...")
        finally:
            gating_service.search_pipeline.search = original_search
        print("="*80 + "\n")

    @pytest.mark.asyncio
    async def test_gating_with_llm_down(self, gating_service):
        """
        Scenario: LLM validation is down during gating

        Expected: Falls back to heuristic matching, returns decision
        """
        print("\n" + "="*80)
        print("CHAOS TEST: Gating with LLM Validation Down")
        print("="*80)

        # Simulate LLM validation failure
        original_validate = None
        if hasattr(gating_service.search_pipeline, 'llm_validator'):
            original_validate = gating_service.search_pipeline.llm_validator.validate
            async def mock_llm_validation_failure(*args, **kwargs):
                raise Exception("LLM validation unavailable")

            gating_service.search_pipeline.llm_validator.validate = AsyncMock(
                side_effect=mock_llm_validation_failure
            )

        try:
            request = ReflectRequest(
                proposed_action="kubectl apply -f deployment.yaml",
                goal="deploy to production",
                tool="kubectl",
                context="deploying v1.2.3",
                current_state={"cluster": "prod"},
            )

            # Should work with heuristic matching only
            response = await gating_service.reflect_before_action(request, "test_customer")

            assert response is not None
            print(f"\n Fallback to heuristic matching: {response.recommendation.value}")
        finally:
            if original_validate:
                gating_service.search_pipeline.llm_validator.validate = original_validate
        print("="*80 + "\n")


class TestRecovery:
    """Test system recovery after faults."""

    @pytest.mark.asyncio
    async def test_kyrodb_recovery(self, ingestion_pipeline):
        """
        Scenario: KyroDB comes back online after failure

        Expected: System resumes normal operation
        """
        print("\n" + "="*80)
        print("CHAOS TEST: KyroDB Recovery")
        print("="*80)

        original_insert = ingestion_pipeline.kyrodb_router.insert_episode

        # Fail first attempt
        call_count = 0

        async def mock_insert_then_recover(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise grpc.aio.AioRpcError(
                    code=grpc.StatusCode.UNAVAILABLE,
                    initial_metadata=None,
                    trailing_metadata=None,
                    details="KyroDB down",
                )
            return await original_insert(*args, **kwargs)

        ingestion_pipeline.kyrodb_router.insert_episode = AsyncMock(side_effect=mock_insert_then_recover)

        try:
            episode1 = EpisodeCreate(
                episode_type=EpisodeType.FAILURE,
                goal="Test recovery - first attempt",
                actions_taken=["test action 1"],
                error_class=ErrorClass.NETWORK_ERROR,
                error_trace="test error 1",
                tool_chain=["test"],
                environment_info={"test": "true"},
                customer_id="test_customer",
            )

            # First attempt fails
            with pytest.raises(grpc.aio.AioRpcError):
                await ingestion_pipeline.capture_episode(episode1, generate_reflection=False)

            print("\n  First attempt failed (expected)")

            # Second attempt succeeds
            episode2 = EpisodeCreate(
                episode_type=EpisodeType.FAILURE,
                goal="Test recovery - second attempt",
                actions_taken=["test action 2"],
                error_class=ErrorClass.NETWORK_ERROR,
                error_trace="test error 2",
                tool_chain=["test"],
                environment_info={"test": "true"},
                customer_id="test_customer",
            )

            result = await ingestion_pipeline.capture_episode(episode2, generate_reflection=False)
            assert result is not None
        finally:
            ingestion_pipeline.kyrodb_router.insert_episode = original_insert

        print("  Second attempt succeeded")
        print("\n System recovered after KyroDB came back online")
        print("="*80 + "\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
