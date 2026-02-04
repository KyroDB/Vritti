"""
Tests for admin endpoints.

Tests the /admin/budget and /admin/reflection/stats endpoints.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Test admin key (must be at least 32 chars to pass validation)
TEST_ADMIN_KEY = "test-admin-key-for-testing-purposes-only-at-least-32-chars"


@pytest.fixture
def mock_reflection_service():
    """Create mock reflection service with stats."""
    service = MagicMock()
    service.get_stats.return_value = {
        "total_cost_usd": 5.50,
        "total_reflections": 100,
        "average_cost_per_reflection": 0.055,
        "cost_savings_usd": 4.10,
        "cost_savings_percentage": 42.7,
        "daily_cost": {
            "date": "2025-11-27",
            "daily_cost_usd": 2.35,
            "warning_threshold_usd": 10.0,
            "limit_threshold_usd": 50.0,
            "warning_triggered": False,
            "limit_exceeded": False,
            "budget_remaining_usd": 47.65,
        },
        "cost_by_tier": {
            "cheap": 0.0,
            "cached": 0.0,
            "premium": 5.50,
        },
        "count_by_tier": {
            "cheap": 85,
            "cached": 5,
            "premium": 10,
        },
        "percentage_by_tier": {
            "cheap": 85.0,
            "cached": 5.0,
            "premium": 10.0,
        },
    }
    return service


@pytest.fixture
def admin_headers():
    """Get headers with admin API key."""
    return {"X-Admin-API-Key": TEST_ADMIN_KEY}


@pytest.fixture
def mock_admin_key():
    """
    Fixture to patch admin key for testing.

    Uses patch on the config module before importing the app to ensure
    the admin_api_key is available.
    """
    from src.config import get_settings

    # Capture original admin_api_key
    settings = get_settings()
    original_key = settings.admin_api_key

    # Temporarily set admin key
    settings.admin_api_key = TEST_ADMIN_KEY

    yield

    # Restore original
    settings.admin_api_key = original_key


@pytest.fixture
def mock_no_admin_key():
    """Fixture to ensure no admin key is set."""
    from src.config import get_settings

    # Capture original admin_api_key
    settings = get_settings()
    original_key = settings.admin_api_key

    # Temporarily unset admin key
    settings.admin_api_key = None

    yield

    # Restore original
    settings.admin_api_key = original_key


class TestBudgetEndpoint:
    """Tests for /admin/budget endpoint."""

    def test_budget_endpoint_returns_stats(
        self, mock_reflection_service, admin_headers, mock_admin_key
    ):
        """Test that budget endpoint returns current cost stats."""
        with patch("src.main.reflection_service", mock_reflection_service):
            from src.main import app

            client = TestClient(app)

            response = client.get("/admin/budget", headers=admin_headers)

            assert response.status_code == 200
            data = response.json()

            assert data["date"] == "2025-11-27"
            assert data["daily_cost_usd"] == 2.35
            assert data["warning_threshold_usd"] == 10.0
            assert data["limit_threshold_usd"] == 50.0
            assert data["warning_triggered"] is False
            assert data["limit_exceeded"] is False
            assert data["budget_remaining_usd"] == 47.65
            assert data["premium_tier_blocked"] is False

    def test_budget_endpoint_when_warning_triggered(
        self, mock_reflection_service, admin_headers, mock_admin_key
    ):
        """Test budget endpoint when warning threshold exceeded."""
        mock_reflection_service.get_stats.return_value["daily_cost"]["daily_cost_usd"] = 12.50
        mock_reflection_service.get_stats.return_value["daily_cost"]["warning_triggered"] = True
        mock_reflection_service.get_stats.return_value["daily_cost"]["budget_remaining_usd"] = 37.50

        with patch("src.main.reflection_service", mock_reflection_service):
            from src.main import app

            client = TestClient(app)

            response = client.get("/admin/budget", headers=admin_headers)

            assert response.status_code == 200
            data = response.json()

            assert data["warning_triggered"] is True
            assert data["limit_exceeded"] is False

    def test_budget_endpoint_when_limit_exceeded(
        self, mock_reflection_service, admin_headers, mock_admin_key
    ):
        """Test budget endpoint when hard limit exceeded."""
        mock_reflection_service.get_stats.return_value["daily_cost"]["daily_cost_usd"] = 55.0
        mock_reflection_service.get_stats.return_value["daily_cost"]["warning_triggered"] = True
        mock_reflection_service.get_stats.return_value["daily_cost"]["limit_exceeded"] = True
        mock_reflection_service.get_stats.return_value["daily_cost"]["budget_remaining_usd"] = 0.0

        with patch("src.main.reflection_service", mock_reflection_service):
            from src.main import app

            client = TestClient(app)

            response = client.get("/admin/budget", headers=admin_headers)

            assert response.status_code == 200
            data = response.json()

            assert data["limit_exceeded"] is True
            assert data["premium_tier_blocked"] is True
            assert data["budget_remaining_usd"] == 0.0

    def test_budget_endpoint_when_service_unavailable(self, admin_headers, mock_admin_key):
        """Test budget endpoint when reflection service is not initialized."""
        with patch("src.main.reflection_service", None):
            from src.main import app

            client = TestClient(app)

            response = client.get("/admin/budget", headers=admin_headers)

            assert response.status_code == 200
            data = response.json()

            assert data["daily_cost_usd"] == 0.0
            assert data["budget_remaining_usd"] == 50.0
            assert data["premium_tier_blocked"] is False


class TestReflectionStatsEndpoint:
    """Tests for /admin/reflection/stats endpoint."""

    def test_stats_endpoint_returns_stats(
        self, mock_reflection_service, admin_headers, mock_admin_key
    ):
        """Test that stats endpoint returns reflection statistics."""
        with patch("src.main.reflection_service", mock_reflection_service):
            from src.main import app

            client = TestClient(app)

            response = client.get("/admin/reflection/stats", headers=admin_headers)

            assert response.status_code == 200
            data = response.json()

            assert data["total_cost_usd"] == 5.50
            assert data["total_reflections"] == 100
            assert data["average_cost_per_reflection"] == 0.055
            assert data["cost_savings_usd"] == 4.10
            assert data["cost_savings_percentage"] == 42.7
            assert "daily_cost" in data
            assert "cost_by_tier" in data
            assert "count_by_tier" in data
            assert "percentage_by_tier" in data

    def test_stats_endpoint_when_service_unavailable(self, admin_headers, mock_admin_key):
        """Test stats endpoint when reflection service is not initialized."""
        with patch("src.main.reflection_service", None):
            from src.main import app

            client = TestClient(app)

            response = client.get("/admin/reflection/stats", headers=admin_headers)

            assert response.status_code == 200
            data = response.json()

            assert data["total_cost_usd"] == 0.0
            assert data["total_reflections"] == 0
            assert data["cost_savings_usd"] == 0.0


class TestAdminAccessControl:
    """Tests for admin access control."""

    def test_admin_endpoint_without_key_returns_401(self, mock_no_admin_key):
        """Test that admin endpoints require API key."""
        from src.main import app

        client = TestClient(app)

        response = client.get("/admin/budget")

        # Should fail because ADMIN_API_KEY not set and no header
        assert response.status_code == 401

    def test_admin_endpoint_with_wrong_key_returns_401(self, mock_admin_key):
        """Test that wrong API key returns 401."""
        from src.main import app

        client = TestClient(app)

        response = client.get("/admin/budget", headers={"X-Admin-API-Key": "wrong-key-not-valid"})

        assert response.status_code == 401
