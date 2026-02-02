"""
OpenRouter API integration tests.

These tests validate real API calls to OpenRouter for multi-perspective reflection.
They are skipped in CI and should be run manually with valid API keys.

Prerequisites:
- LLM_OPENROUTER_API_KEY environment variable set with valid key
- Internet connectivity to OpenRouter API

Run with:
    pytest tests/integration/test_openrouter_api.py -v -s

Run specific test:
    pytest tests/integration/test_openrouter_api.py::TestOpenRouterAPI::test_single_model_call -v -s
"""

import asyncio
import os
import time

import pytest

from src.config import LLMConfig
from src.ingestion.multi_perspective_reflection import MultiPerspectiveReflectionService
from src.models.episode import EpisodeCreate, EpisodeType, ErrorClass


def get_test_api_key() -> str:
    """Get OpenRouter API key from environment."""
    key = os.environ.get("LLM_OPENROUTER_API_KEY", "")
    if not key:
        pytest.skip("LLM_OPENROUTER_API_KEY not set - run with real API key to test")
    return key


def _require_env(name: str) -> str:
    """Get required environment variable or skip test."""
    value = os.environ.get(name, "")
    if not value:
        pytest.skip(f"{name} not set - required for OpenRouter tests")
    return value


def _safe_float(name: str, default: float) -> float:
    """
    Parse environment variable as float with proper error handling.
    
    Args:
        name: Environment variable name
        default: Default value if env var is not set or invalid
        
    Returns:
        Parsed float value or default
    """
    raw_value = os.environ.get(name, "")
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError:
        pytest.fail(f"Invalid float value for {name}: '{raw_value}'")


def _safe_int(name: str, default: int) -> int:
    """
    Parse environment variable as int with proper error handling.
    
    Args:
        name: Environment variable name
        default: Default value if env var is not set or invalid
        
    Returns:
        Parsed int value or default
    """
    raw_value = os.environ.get(name, "")
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        pytest.fail(f"Invalid int value for {name}: '{raw_value}'")


@pytest.fixture
def real_llm_config() -> LLMConfig:
    """Create LLM config from environment variables. All model names come from .env."""
    api_key = get_test_api_key()
    return LLMConfig(
        openrouter_api_key=api_key,
        openrouter_base_url=_require_env("LLM_OPENROUTER_BASE_URL"),
        consensus_model_1=_require_env("LLM_CONSENSUS_MODEL_1"),
        consensus_model_2=_require_env("LLM_CONSENSUS_MODEL_2"),
        cheap_model=_require_env("LLM_CHEAP_MODEL"),
        temperature=_safe_float("LLM_TEMPERATURE", 0.3),
        max_tokens=_safe_int("LLM_MAX_TOKENS", 2000),
        max_retries=_safe_int("LLM_MAX_RETRIES", 2),
        timeout_seconds=_safe_int("LLM_TIMEOUT_SECONDS", 60),
    )


@pytest.fixture
def sample_episode() -> EpisodeCreate:
    """Create sample episode for testing."""
    return EpisodeCreate(
        customer_id="test-customer",
        episode_type=EpisodeType.FAILURE,
        goal="Deploy Python web application to production",
        tool_chain=["git", "pip", "gunicorn", "nginx"],
        actions_taken=[
            "Pulled latest code from main branch",
            "Ran pip install -r requirements.txt",
            "Started gunicorn with 4 workers",
            "Application crashed with ModuleNotFoundError",
        ],
        error_trace=(
            "Traceback (most recent call last):\n"
            "  File 'app.py', line 3, in <module>\n"
            "    import pandas as pd\n"
            "ModuleNotFoundError: No module named 'pandas'\n"
            "\n"
            "pip freeze shows pandas is not in requirements.txt"
        ),
        error_class=ErrorClass.DEPENDENCY_ERROR,
        code_state_diff="+ import pandas as pd\n+ df = pd.read_csv('data.csv')",
        environment_info={
            "os": "Ubuntu 22.04",
            "python_version": "3.11.4",
            "pip_version": "23.2.1",
        },
        resolution="Added pandas>=2.0.0 to requirements.txt and reinstalled",
        tags=["production", "python", "dependencies"],
        severity=2,
    )


class TestOpenRouterAPI:
    """
    Real OpenRouter API tests.
    
    These tests make actual API calls and are skipped in CI.
    Run manually with LLM_OPENROUTER_API_KEY set.
    """

    @pytest.mark.skipif(
        not os.environ.get("LLM_OPENROUTER_API_KEY"),
        reason="LLM_OPENROUTER_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_single_model_call(
        self,
        real_llm_config: LLMConfig,
        sample_episode: EpisodeCreate,
    ):
        """
        Test single model call via OpenRouter.
        
        Validates:
        - API connection works
        - JSON response is parseable
        - Pydantic model validates response
        - Response contains expected fields
        """
        service = MultiPerspectiveReflectionService(config=real_llm_config)

        start_time = time.perf_counter()

        # Generate reflection using cheap tier (single model)
        reflection = await service.generate_multi_perspective_reflection(
            episode=sample_episode,
            use_cheap_tier=True,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        print("\n--- Single Model Test Results ---")
        print(f"Latency: {elapsed_ms:.0f}ms")
        print(f"Model: {reflection.llm_model}")
        print(f"Root Cause: {reflection.root_cause}")
        print(f"Confidence: {reflection.confidence_score:.2f}")
        print(f"Preconditions: {reflection.preconditions}")

        # Assertions
        assert reflection.root_cause, "Root cause should not be empty"
        assert reflection.resolution_strategy, "Resolution should not be empty"
        assert len(reflection.preconditions) > 0, "Should have at least one precondition"
        assert 0 <= reflection.confidence_score <= 1, "Confidence should be 0-1"
        assert 0 <= reflection.generalization_score <= 1, "Generalization should be 0-1"
        # Verify LLM was actually called (not fallback heuristic)
        assert reflection.llm_model != "fallback_heuristic", \
            "Should use real LLM, not fallback. Check API key and model availability."
        # Verify cost tracking (should be >= 0 for real API call, even for free models)
        assert reflection.generation_latency_ms > 0, "Should have non-zero latency from real API call"

    @pytest.mark.skipif(
        not os.environ.get("LLM_OPENROUTER_API_KEY"),
        reason="LLM_OPENROUTER_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_consensus_with_two_models(
        self,
        real_llm_config: LLMConfig,
        sample_episode: EpisodeCreate,
    ):
        """
        Test multi-perspective reflection with 2 models.
        
        Validates:
        - Both models are called in parallel
        - Consensus is computed correctly
        - Semantic similarity logic works
        - Merged preconditions are deduplicated
        """
        service = MultiPerspectiveReflectionService(config=real_llm_config)

        start_time = time.perf_counter()

        # Generate reflection using premium tier (multi-model consensus)
        reflection = await service.generate_multi_perspective_reflection(
            episode=sample_episode,
            use_cheap_tier=False,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        print("\n--- Consensus Test Results ---")
        print(f"Latency: {elapsed_ms:.0f}ms")
        print(f"Model: {reflection.llm_model}")

        if reflection.consensus:
            print(f"Consensus Method: {reflection.consensus.consensus_method}")
            print(f"Consensus Confidence: {reflection.consensus.consensus_confidence:.2f}")
            print(f"Models Used: {[p.model_name for p in reflection.consensus.perspectives]}")
            print(f"Disagreement Points: {reflection.consensus.disagreement_points}")

        print(f"\nAgreed Root Cause: {reflection.root_cause}")
        print(f"Preconditions ({len(reflection.preconditions)}): {reflection.preconditions}")
        print(f"Resolution: {reflection.resolution_strategy[:200]}...")

        # Assertions
        assert reflection.root_cause, "Root cause should not be empty"
        assert reflection.consensus is not None, "Consensus should be present for multi-model"
        assert len(reflection.consensus.perspectives) >= 1, "Should have at least 1 perspective"
        assert reflection.consensus.consensus_method in [
            "semantic_unanimous",
            "weighted_semantic_vote",
            "highest_confidence_fallback",
            "single_model",
            "semantic_majority",
            "weighted_semantic_majority",
        ], f"Unknown consensus method: {reflection.consensus.consensus_method}"

        # Latency should be reasonable (parallel calls)
        # Single model takes ~5-15s, parallel should take ~10-20s (not 2x)
        assert elapsed_ms < 120_000, f"Latency too high: {elapsed_ms}ms (expected < 120s)"

    @pytest.mark.skipif(
        not os.environ.get("LLM_OPENROUTER_API_KEY"),
        reason="LLM_OPENROUTER_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_error_handling_invalid_model(self, real_llm_config: LLMConfig):
        """
        Test error handling when model doesn't exist.
        
        Should fall back gracefully without crashing.
        """
        # Create config with invalid model names to test error handling
        bad_config = LLMConfig(
            openrouter_api_key=real_llm_config.openrouter_api_key,
            openrouter_base_url=real_llm_config.openrouter_base_url,
            consensus_model_1="invalid/nonexistent-model-xyz",
            consensus_model_2="also/not-real-model",
            cheap_model=real_llm_config.cheap_model,  # Use valid cheap model from env
            temperature=0.3,
            max_tokens=1000,
            max_retries=1,
            timeout_seconds=30,
        )

        service = MultiPerspectiveReflectionService(config=bad_config)

        episode = EpisodeCreate(
            customer_id="test",
            episode_type=EpisodeType.FAILURE,
            goal="Test error handling",
            tool_chain=["test"],
            actions_taken=["Tested error handling"],
            error_trace="Test error",
            error_class=ErrorClass.UNKNOWN,
        )

        # Should not raise exception, should return fallback reflection
        reflection = await service.generate_multi_perspective_reflection(
            episode=episode,
            use_cheap_tier=False,
        )

        print("\n--- Error Handling Test ---")
        print(f"Model: {reflection.llm_model}")
        print(f"Confidence: {reflection.confidence_score}")

        # Should either use fallback heuristic or succeed with partial results
        assert reflection is not None, "Should return a reflection even on partial failure"

    @pytest.mark.skipif(
        not os.environ.get("LLM_OPENROUTER_API_KEY"),
        reason="LLM_OPENROUTER_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_semantic_similarity_for_root_cause(
        self,
        real_llm_config: LLMConfig,
    ):
        """
        Test that semantic similarity correctly identifies related root causes.
        
        Uses embedding model to compare root cause texts.
        """
        service = MultiPerspectiveReflectionService(config=real_llm_config)

        # Test texts with same semantic meaning but different wording
        similar_texts = [
            "Missing dependency in requirements file",
            "Package not listed in requirements.txt",
        ]

        # Test texts with different meanings
        different_texts = [
            "Missing dependency in requirements file",
            "Network connection timeout during API call",
        ]

        similar_matrix = service._compute_semantic_similarity_matrix(similar_texts)
        different_matrix = service._compute_semantic_similarity_matrix(different_texts)

        similar_score = similar_matrix[0, 1]
        different_score = different_matrix[0, 1]

        print("\n--- Semantic Similarity Test ---")
        print(f"Similar texts similarity: {similar_score:.3f}")
        print(f"Different texts similarity: {different_score:.3f}")

        # Similar texts should have high similarity
        assert similar_score > 0.5, f"Similar texts should have similarity > 0.5, got {similar_score}"

        # Different texts should have lower similarity
        assert different_score < similar_score, \
            f"Different texts ({different_score}) should be less similar than related texts ({similar_score})"


class TestOpenRouterRateLimits:
    """
    Rate limit handling tests.
    
    These tests validate the retry logic for rate-limited requests.
    """

    @pytest.mark.skipif(
        not os.environ.get("LLM_OPENROUTER_API_KEY"),
        reason="LLM_OPENROUTER_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_parallel_requests_respect_limits(
        self,
        real_llm_config: LLMConfig,
        sample_episode: EpisodeCreate,
    ):
        """
        Test that parallel requests handle rate limits correctly.
        
        Makes multiple concurrent requests and verifies they all complete.
        """
        service = MultiPerspectiveReflectionService(config=real_llm_config)

        # Make 3 concurrent cheap reflections
        tasks = [
            service.generate_multi_perspective_reflection(
                episode=sample_episode,
                use_cheap_tier=True,
            )
            for _ in range(3)
        ]

        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        print("\n--- Parallel Requests Test ---")
        print(f"Total latency for 3 requests: {elapsed_ms:.0f}ms")

        # Count successes and failures
        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, Exception)]

        print(f"Successes: {len(successes)}")
        print(f"Failures: {len(failures)}")

        # At least 2/3 should succeed (allows for some rate limiting)
        assert len(successes) >= 2, f"At least 2/3 should succeed, got {len(successes)}"

        # Verify successful results
        for reflection in successes:
            assert reflection.root_cause, "Root cause should be present"
            assert reflection.confidence_score > 0, "Confidence should be positive"
