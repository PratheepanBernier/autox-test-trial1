import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

import backend.main

@pytest.fixture
def client():
    # Patch the router to avoid importing actual api.routes.router
    with patch("backend.main.router"):
        # Re-import app with router mocked
        from backend.main import app
        yield TestClient(app)

def test_health_endpoint_returns_running_status(client):
    with patch.object(backend.main.logger, "info") as mock_log:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "running"}
        mock_log.assert_called_once_with("Health check endpoint called")

def test_health_endpoint_method_not_allowed(client):
    # Only GET is allowed; POST should return 405
    response = client.post("/")
    assert response.status_code == 405

def test_health_endpoint_logger_error_handling(client):
    # Simulate logger raising an exception; endpoint should still return 200
    with patch.object(backend.main.logger, "info", side_effect=Exception("Logging failed")):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "running"}

def test_app_metadata(client):
    # Check app metadata is set as expected
    from backend.main import app
    assert app.title == "Logistics Document Intelligence Assistant"
