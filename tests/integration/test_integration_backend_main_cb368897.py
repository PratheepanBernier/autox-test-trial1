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

def test_health_endpoint_with_large_headers(client):
    headers = {"X-Custom-Header": "a" * 10000}
    response = client.get("/", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"status": "running"}

def test_health_endpoint_with_unusual_accept_header(client):
    response = client.get("/", headers={"Accept": "application/xml"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")

def test_router_inclusion_and_other_route_returns_404(client):
    # Assuming /nonexistent is not defined in router
    response = client.get("/nonexistent")
    assert response.status_code == 404

def test_logging_configuration_is_set():
    # Check that logger has StreamHandler to sys.stdout
    handlers = backend.main.logger.handlers
    assert any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in handlers)

def test_app_title_is_set_correctly():
    assert backend.main.app.title == "Logistics Document Intelligence Assistant"
