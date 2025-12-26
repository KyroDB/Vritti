"""
Tests for verifying that proposed_action and current_state parameters 
are properly used in gating recommendations.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.gating.service import GatingService
from src.models.episode import Episode, EpisodeCreate, Reflection
from src.models.gating import ActionRecommendation, ReflectRequest
from src.models.search import SearchResponse, SearchResult
from src.models.skill import Skill


@pytest.fixture
def mock_search_pipeline():
    pipeline = MagicMock()
    pipeline.embedding_service = MagicMock()
    pipeline.embedding_service.embed_text.return_value = [0.1] * 384
    return pipeline


@pytest.fixture
def mock_kyrodb_router():
    router = AsyncMock()
    router.search_skills.return_value = []
    return router


@pytest.fixture
def gating_service(mock_search_pipeline, mock_kyrodb_router):
    return GatingService(mock_search_pipeline, mock_kyrodb_router)


@pytest.mark.asyncio
async def test_proposed_action_included_in_skill_hints(gating_service, mock_search_pipeline, mock_kyrodb_router):
    """Test that proposed_action is included in hints when a high-confidence skill is found."""
    
    # Setup skill with high confidence
    skill = Skill(
        skill_id=1,
        customer_id="test_customer",
        name="Deploy with safe config",
        docstring="Safe deployment procedure",
        code="kubectl apply -f safe-config.yaml",
        usage_count=10,
        success_count=9,
        failure_count=1,
        error_class="deployment_error",
        source_episodes=[1, 2, 3]
    )
    
    mock_kyrodb_router.search_skills.return_value = [(skill, 0.9)]
    mock_search_pipeline.search = AsyncMock(return_value=SearchResponse(
        results=[],
        total_candidates=0,
        total_filtered=0,
        total_returned=0,
        search_latency_ms=10,
        collection="failures",
        query_embedding_dimension=384
    ))
    
    request = ReflectRequest(
        goal="Deploy to prod",
        proposed_action="kubectl apply -f my-config.yaml",
        tool="kubectl",
        current_state={"os": "linux", "version": "1.28"}
    )
    
    response = await gating_service.reflect_before_action(request, "test_customer")
    
    # Verify skill is recommended
    assert response.recommendation == ActionRecommendation.REWRITE
    assert response.confidence == 0.9
    
    # Verify proposed action is in hints
    assert any("Original action" in hint for hint in response.hints)
    assert any("kubectl apply -f my-config.yaml" in hint for hint in response.hints)


@pytest.mark.asyncio
async def test_environment_mismatch_warning_in_block(gating_service, mock_search_pipeline):
    """Test that environment mismatch warning is shown when blocking."""
    
    episode = Episode(
        create_data=EpisodeCreate(
            goal="Deploy to prod",
            tool_chain=["kubectl"],
            actions_taken=["kubectl apply"],
            error_trace="Error: Resource quota exceeded",
            error_class="resource_error"
        ),
        episode_id=1,
        reflection=Reflection(
            root_cause="Resource limit exceeded",
            resolution_strategy="Increase quota",
            preconditions=["kubernetes cluster"],
            environment_factors=["ubuntu-20.04", "kubernetes-1.24"],
            affected_components=["deployment"],
            generalization_score=0.8,
            confidence_score=0.9
        )
    )
    
    search_result = SearchResult(
        episode=episode,
        scores={"similarity": 0.95, "precondition": 0.8},
        rank=1
    )
    
    mock_search_pipeline.search = AsyncMock(return_value=SearchResponse(
        results=[search_result],
        total_candidates=1,
        total_filtered=1,
        total_returned=1,
        search_latency_ms=10,
        collection="failures",
        query_embedding_dimension=384
    ))
    
    request = ReflectRequest(
        goal="Deploy to prod",
        proposed_action="kubectl apply -f deployment.yaml",
        tool="kubectl",
        current_state={"os": "windows", "version": "1.30"}  # Different environment
    )
    
    response = await gating_service.reflect_before_action(request, "test_customer")
    
    # Verify blocking recommendation
    assert response.recommendation == ActionRecommendation.BLOCK
    
    # Verify environment warning is present
    assert any("Environment differs" in hint for hint in response.hints)
    assert any("Blocked action" in hint for hint in response.hints)


@pytest.mark.asyncio
async def test_environment_match_in_rewrite(gating_service, mock_search_pipeline):
    """Test that environment match is mentioned in rewrite recommendations."""
    
    episode = Episode(
        create_data=EpisodeCreate(
            goal="Deploy to prod",
            tool_chain=["kubectl"],
            actions_taken=["kubectl apply"],
            error_trace="Error: Resource quota exceeded",
            error_class="resource_error"
        ),
        episode_id=1,
        reflection=Reflection(
            root_cause="Bad config",
            resolution_strategy="Use safe config",
            preconditions=["kubernetes cluster"],
            environment_factors=["linux", "kubernetes"],
            affected_components=["deployment"],
            generalization_score=0.8,
            confidence_score=0.9
        )
    )
    
    search_result = SearchResult(
        episode=episode,
        scores={"similarity": 0.85, "precondition": 0.6},
        rank=1
    )
    
    mock_search_pipeline.search = AsyncMock(return_value=SearchResponse(
        results=[search_result],
        total_candidates=1,
        total_filtered=1,
        total_returned=1,
        search_latency_ms=10,
        collection="failures",
        query_embedding_dimension=384
    ))
    
    request = ReflectRequest(
        goal="Deploy to prod",
        proposed_action="kubectl apply",
        tool="kubectl",
        current_state={"os": "linux", "k8s": "kubernetes-1.28"}  # Matching environment
    )
    
    response = await gating_service.reflect_before_action(request, "test_customer")
    
    # Verify rewrite recommendation
    assert response.recommendation == ActionRecommendation.REWRITE
    
    # Verify environment match is mentioned in hints
    assert any("Environment matches" in hint for hint in response.hints)


def test_check_environment_match_empty_inputs(gating_service):
    """Test environment matching with empty inputs."""
    
    # Empty current_state should return True (conservative)
    assert gating_service._check_environment_match({}, ["linux"]) is True
    
    # Empty environment_factors should return True (conservative)
    assert gating_service._check_environment_match({"os": "linux"}, []) is True
    
    # Both empty should return True
    assert gating_service._check_environment_match({}, []) is True


def test_check_environment_match_with_matches(gating_service):
    """Test environment matching with actual matches."""
    
    current_state = {
        "os": "linux",
        "version": "kubernetes-1.28",
        "cloud": "aws"
    }
    
    # Should match when factor is present
    assert gating_service._check_environment_match(
        current_state, ["linux", "ubuntu"]
    ) is True
    
    # Should match when at least one factor is present
    assert gating_service._check_environment_match(
        current_state, ["kubernetes"]
    ) is True
    
    # Should not match when no factors are present
    assert gating_service._check_environment_match(
        current_state, ["windows", "azure"]
    ) is False


def test_check_environment_match_case_insensitive(gating_service):
    """Test that environment matching is case-insensitive."""
    
    current_state = {"os": "Linux", "version": "KUBERNETES-1.28"}
    
    # Should match regardless of case
    assert gating_service._check_environment_match(
        current_state, ["linux"]
    ) is True
    
    assert gating_service._check_environment_match(
        current_state, ["kubernetes"]
    ) is True
