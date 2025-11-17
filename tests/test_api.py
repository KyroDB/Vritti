"""
Integration tests for FastAPI endpoints.

Tests:
- Health check endpoint
- Statistics endpoint
- Episode capture endpoint
- Episode search endpoint
- Error handling
"""

import pytest
from fastapi.testclient import TestClient

from src.models.episode import EpisodeCreate, EpisodeType, ErrorClass


class TestHealthEndpoint:
    """Test suite for health check endpoint."""

    def test_health_check_success(self, app_client: TestClient):
        """Test successful health check."""
        response = app_client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "kyrodb_connected" in data
        assert "embedding_service_ready" in data
        assert "reflection_service_ready" in data

    def test_health_check_structure(self, app_client: TestClient):
        """Test health check response structure."""
        response = app_client.get("/health")
        data = response.json()

        assert isinstance(data["kyrodb_connected"], bool)
        assert isinstance(data["embedding_service_ready"], bool)
        assert isinstance(data["reflection_service_ready"], bool)


class TestStatsEndpoint:
    """Test suite for statistics endpoint."""

    def test_stats_endpoint(self, app_client: TestClient):
        """Test statistics endpoint."""
        response = app_client.get("/stats")

        assert response.status_code == 200
        data = response.json()

        assert "ingestion_stats" in data
        assert "search_stats" in data


class TestCaptureEndpoint:
    """Test suite for episode capture endpoint."""

    def test_capture_episode_success(
        self, app_client: TestClient, sample_episode_create: EpisodeCreate
    ):
        """Test successful episode capture."""
        # Skip if services not initialized
        if not hasattr(app_client.app, "state"):
            pytest.skip("App state not initialized")

        response = app_client.post(
            "/api/v1/capture",
            json=sample_episode_create.model_dump(mode="json"),
        )

        # May return 503 if services not fully initialized
        if response.status_code == 503:
            pytest.skip("Services not initialized in test environment")

        assert response.status_code == 201
        data = response.json()

        assert "episode_id" in data
        assert "collection" in data
        assert "ingestion_latency_ms" in data
        assert data["episode_id"] > 0
        assert data["collection"] in ["failures", "skills", "rules"]

    def test_capture_episode_minimal(self, app_client: TestClient):
        """Test episode capture with minimal data."""
        minimal_episode = {
            "goal": "Minimal test episode for validation purposes",
            "tool_chain": ["test_tool"],
            "actions_taken": ["test action"],
            "error_trace": "test error trace",
        }

        response = app_client.post("/api/v1/capture", json=minimal_episode)

        # May return 503 if services not initialized, or 422 for validation
        if response.status_code == 503:
            pytest.skip("Services not initialized")

        assert response.status_code in [201, 422]

    def test_capture_episode_validation_error(self, app_client: TestClient):
        """Test episode capture with invalid data."""
        invalid_episode = {
            "goal": "Test",  # Too short
            "tool_chain": [],  # Empty
            "actions_taken": ["action"],
            "error_trace": "error",
        }

        response = app_client.post("/api/v1/capture", json=invalid_episode)

        assert response.status_code in [422, 503]  # Validation error or service unavailable


class TestSearchEndpoint:
    """Test suite for episode search endpoint."""

    def test_search_basic(self, app_client: TestClient):
        """Test basic episode search."""
        search_request = {
            "goal": "Test search query for deployment issues",
            "collection": "failures",
            "k": 5,
        }

        response = app_client.post("/api/v1/search", json=search_request)

        # May return 503 if services not initialized
        if response.status_code == 503:
            pytest.skip("Services not initialized")

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert "total_candidates" in data
        assert "total_returned" in data
        assert "search_latency_ms" in data
        assert "breakdown" in data
        assert isinstance(data["results"], list)

    def test_search_with_filters(self, app_client: TestClient):
        """Test search with metadata filters."""
        search_request = {
            "goal": "Search with filters",
            "collection": "failures",
            "tool_filter": "kubectl",
            "k": 5,
            "min_similarity": 0.7,
        }

        response = app_client.post("/api/v1/search", json=search_request)

        if response.status_code == 503:
            pytest.skip("Services not initialized")

        assert response.status_code == 200

    def test_search_with_ranking_weights(self, app_client: TestClient):
        """Test search with custom ranking weights."""
        search_request = {
            "goal": "Search with custom weights",
            "collection": "failures",
            "k": 5,
            "ranking_weights": {
                "similarity_weight": 0.6,
                "precondition_weight": 0.2,
                "recency_weight": 0.1,
                "usage_weight": 0.1,
            },
        }

        response = app_client.post("/api/v1/search", json=search_request)

        if response.status_code == 503:
            pytest.skip("Services not initialized")

        assert response.status_code == 200

    def test_search_validation_error(self, app_client: TestClient):
        """Test search with invalid parameters."""
        invalid_search = {
            "goal": "Test",  # Too short
            "collection": "failures",
            "k": 0,  # Invalid k
        }

        response = app_client.post("/api/v1/search", json=invalid_search)

        assert response.status_code in [422, 503]


class TestRootEndpoint:
    """Test suite for root endpoint."""

    def test_root_endpoint(self, app_client: TestClient):
        """Test root endpoint returns API information."""
        response = app_client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "service" in data
        assert "version" in data
        assert "docs" in data
        assert data["service"] == "Episodic Memory API"


class TestErrorHandling:
    """Test suite for error handling."""

    def test_not_found_endpoint(self, app_client: TestClient):
        """Test 404 for non-existent endpoint."""
        response = app_client.get("/nonexistent")

        assert response.status_code == 404

    def test_method_not_allowed(self, app_client: TestClient):
        """Test 405 for wrong HTTP method."""
        response = app_client.get("/api/v1/capture")  # Should be POST

        assert response.status_code == 405
