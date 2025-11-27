"""
Tests for admin endpoints.

Tests the /admin/budget and /admin/reflection/stats endpoints.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


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


class TestBudgetEndpoint:
    """Tests for /admin/budget endpoint."""

    def test_budget_endpoint_returns_stats(self, mock_reflection_service):
        """Test that budget endpoint returns current cost stats."""
        with patch("src.main.reflection_service", mock_reflection_service):
            from src.main import app
            client = TestClient(app)
            
            response = client.get("/admin/budget")
            
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

    def test_budget_endpoint_when_warning_triggered(self, mock_reflection_service):
        """Test budget endpoint when warning threshold exceeded."""
        mock_reflection_service.get_stats.return_value["daily_cost"]["daily_cost_usd"] = 12.50
        mock_reflection_service.get_stats.return_value["daily_cost"]["warning_triggered"] = True
        mock_reflection_service.get_stats.return_value["daily_cost"]["budget_remaining_usd"] = 37.50
        
        with patch("src.main.reflection_service", mock_reflection_service):
            from src.main import app
            client = TestClient(app)
            
            response = client.get("/admin/budget")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["warning_triggered"] is True
            assert data["limit_exceeded"] is False

    def test_budget_endpoint_when_limit_exceeded(self, mock_reflection_service):
        """Test budget endpoint when hard limit exceeded."""
        mock_reflection_service.get_stats.return_value["daily_cost"]["daily_cost_usd"] = 55.0
        mock_reflection_service.get_stats.return_value["daily_cost"]["warning_triggered"] = True
        mock_reflection_service.get_stats.return_value["daily_cost"]["limit_exceeded"] = True
        mock_reflection_service.get_stats.return_value["daily_cost"]["budget_remaining_usd"] = 0.0
        
        with patch("src.main.reflection_service", mock_reflection_service):
            from src.main import app
            client = TestClient(app)
            
            response = client.get("/admin/budget")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["limit_exceeded"] is True
            assert data["premium_tier_blocked"] is True
            assert data["budget_remaining_usd"] == 0.0

    def test_budget_endpoint_when_service_unavailable(self):
        """Test budget endpoint when reflection service is not initialized."""
        with patch("src.main.reflection_service", None):
            from src.main import app
            client = TestClient(app)
            
            response = client.get("/admin/budget")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["daily_cost_usd"] == 0.0
            assert data["budget_remaining_usd"] == 50.0
            assert data["premium_tier_blocked"] is False


class TestReflectionStatsEndpoint:
    """Tests for /admin/reflection/stats endpoint."""

    def test_stats_endpoint_returns_stats(self, mock_reflection_service):
        """Test that stats endpoint returns reflection statistics."""
        with patch("src.main.reflection_service", mock_reflection_service):
            from src.main import app
            client = TestClient(app)
            
            response = client.get("/admin/reflection/stats")
            
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

    def test_stats_endpoint_when_service_unavailable(self):
        """Test stats endpoint when reflection service is not initialized."""
        with patch("src.main.reflection_service", None):
            from src.main import app
            client = TestClient(app)
            
            response = client.get("/admin/reflection/stats")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["total_cost_usd"] == 0.0
            assert data["total_reflections"] == 0
            assert data["cost_savings_usd"] == 0.0
