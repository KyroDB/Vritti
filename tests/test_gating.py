from unittest.mock import AsyncMock, MagicMock

import pytest

from src.gating.service import GatingService
from src.models.episode import Episode, EpisodeCreate, Reflection
from src.models.gating import ActionRecommendation, ReflectRequest
from src.models.search import SearchResponse, SearchResult


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
async def test_reflect_block_high_confidence(gating_service, mock_search_pipeline):
    # Setup mock search result
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
            resolution_strategy="Fix config",
            preconditions=[],
            environment_factors=[],
            affected_components=[],
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
        proposed_action="kubectl apply",
        tool="kubectl"
    )

    response = await gating_service.reflect_before_action(request, "cust_1")

    assert response.recommendation == ActionRecommendation.BLOCK
    assert response.confidence >= 0.9
    assert response.suggested_action == "Fix config"

@pytest.mark.asyncio
async def test_reflect_hint_medium_confidence(gating_service, mock_search_pipeline):
    # Setup mock search result
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
            resolution_strategy="Fix config",
            preconditions=[],
            environment_factors=[],
            affected_components=[],
            generalization_score=0.8,
            confidence_score=0.9
        )
    )
    
    search_result = SearchResult(
        episode=episode,
        scores={"similarity": 0.75, "precondition": 0.5},
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
        tool="kubectl"
    )

    response = await gating_service.reflect_before_action(request, "cust_1")

    assert response.recommendation == ActionRecommendation.HINT
    assert response.confidence == 0.7
    assert len(response.hints) > 0

@pytest.mark.asyncio
async def test_reflect_proceed_low_confidence(gating_service, mock_search_pipeline):
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
        proposed_action="kubectl apply",
        tool="kubectl"
    )

    response = await gating_service.reflect_before_action(request, "cust_1")

    assert response.recommendation == ActionRecommendation.PROCEED
    assert response.confidence == 1.0  # Proceed with confidence if no failures found
