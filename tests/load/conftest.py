# Test fixtures for load testing

import uuid
from unittest.mock import AsyncMock

import pytest

from src.models.episode import EpisodeCreate


@pytest.fixture
def sample_episode_create():
    """Sample episode for load testing."""
    return EpisodeCreate(
        customer_id=f"load-test-{uuid.uuid4().hex[:8]}",
        goal="Deploy application to production",
        tool_chain=["kubectl", "helm"],
        actions_taken=["kubectl apply -f deployment.yaml"],
        error_trace="ImagePullBackOff: failed to pull image",
        error_class="resource_error",
        tags=["kubernetes", "deployment"]
    )


@pytest.fixture
def ingestion_pipeline(mocker):
    """Mock ingestion pipeline for load tests."""
    pipeline = mocker.MagicMock()
    pipeline.capture_episode = AsyncMock(return_value=mocker.MagicMock(episode_id=1))
    return pipeline


@pytest.fixture
def search_pipeline(mocker):
    """Mock search pipeline for load tests."""
    pipeline = mocker.MagicMock()
    search_response = mocker.MagicMock()
    search_response.results = []
    pipeline.search = AsyncMock(return_value=search_response)
    return pipeline


@pytest.fixture
def gating_service(mocker):
    """Mock gating service for load tests."""
    service = mocker.MagicMock()
    response = mocker.MagicMock()
    response.recommendation = "proceed"
    service.reflect_before_action = AsyncMock(return_value=response)
    return service
