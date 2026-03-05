# source_hash: 93b912fd61c124bd
# import_target: backend.main
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

import backend.main

@pytest.fixture
def client():
    return TestClient(backend.main.app)

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
    response = client.post("/")
    assert response.status_code == 405

def test_health_endpoint_with_query_params_ignores_params(client):
    response = client.get("/?foo=bar")
    assert response.status_code == 200
    assert response.json() == {"status": "running"}

def test_app_title_is_set_correctly():
    assert backend.main.app.title == "Logistics Document Intelligence Assistant"

def test_router_is_included():
    # The router should be included in app.routes
    route_paths = [route.path for route in backend.main.app.routes]
    # "/" from health, and at least one from router (assume /api or similar)
    assert "/" in route_paths
    # We can't know the exact router path, but we can check that more than one route exists
    assert len(route_paths) > 1

def test_logging_configuration(monkeypatch):
    # Patch logging.basicConfig and StreamHandler to ensure they're called as expected
    with patch("logging.basicConfig") as mock_basicConfig, \
         patch("logging.StreamHandler") as mock_StreamHandler:
        import importlib
        import backend.main as main_module
        importlib.reload(main_module)
        mock_basicConfig.assert_called()
        mock_StreamHandler.assert_called_with(sys.stdout)

def test_health_endpoint_handles_large_number_of_requests(client):
    for _ in range(10):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "running"}

def test_health_endpoint_handles_unicode_query_params(client):
    response = client.get("/?q=测试")
    assert response.status_code == 200
    assert response.json() == {"status": "running"}
