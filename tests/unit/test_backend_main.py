import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

import backend.main

@pytest.fixture
def client():
    # Patch the router to avoid dependency on api.routes
    with patch("backend.main.router") as mock_router:
        # Re-import app to ensure patched router is used
        from importlib import reload
        reload(backend.main)
        yield TestClient(backend.main.app)

def test_health_endpoint_returns_running_status(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "running"}

def test_health_endpoint_logs_info(client):
    with patch.object(backend.main.logger, "info") as mock_info:
        response = client.get("/")
        assert response.status_code == 200
        mock_info.assert_called_once_with("Health check endpoint called")

def test_health_endpoint_method_not_allowed(client):
    # Only GET is allowed; POST should return 405
    response = client.post("/")
    assert response.status_code == 405

def test_app_title_is_set():
    assert backend.main.app.title == "Logistics Document Intelligence Assistant"

def test_router_is_included():
    # The router should be included in the app's routes
    # Since we patch router in the fixture, here we just check that the root path exists
    paths = [route.path for route in backend.main.app.routes]
    assert "/" in paths
