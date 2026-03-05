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
    with patch("backend.main.logger") as mock_logger:
        response = client.get("/")
        assert response.status_code == 200
        mock_logger.info.assert_called_with("Health check endpoint called")


def test_health_endpoint_method_not_allowed(client):
    response = client.post("/")
    assert response.status_code == 405
    assert "detail" in response.json()
    assert response.json()["detail"] == "Method Not Allowed"


def test_health_endpoint_boundary_conditions(client):
    # Test with query params (should be ignored)
    response = client.get("/?foo=bar")
    assert response.status_code == 200
    assert response.json() == {"status": "running"}

    # Test with trailing slash
    response = client.get("//")
    assert response.status_code == 200
    assert response.json() == {"status": "running"}


def test_app_title_and_router_integration(client):
    assert backend.main.app.title == "Logistics Document Intelligence Assistant"
    # The router is included; test a non-root path returns 404 (since api.routes.router is mocked)
    response = client.get("/nonexistent")
    assert response.status_code == 404


def test_logging_configuration(monkeypatch):
    # Patch logging.basicConfig and StreamHandler to ensure correct configuration
    with patch("logging.basicConfig") as mock_basicConfig, \
         patch("logging.StreamHandler") as mock_stream_handler:
        import importlib
        importlib.reload(backend.main)
        mock_basicConfig.assert_called()
        mock_stream_handler.assert_called_with(sys.stdout)
