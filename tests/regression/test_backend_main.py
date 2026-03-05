import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

import backend.main

@pytest.fixture
def client():
    # Patch the router to avoid dependency on api.routes
    with patch("backend.main.router") as mock_router:
        # Re-import app to ensure patched router is used
        from backend.main import app
        yield TestClient(app)

def test_health_endpoint_returns_running_status(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "running"}

def test_health_endpoint_logs_info(client):
    with patch("backend.main.logger") as mock_logger:
        response = client.get("/")
        assert response.status_code == 200
        mock_logger.info.assert_called_once_with("Health check endpoint called")

def test_health_endpoint_method_not_allowed(client):
    # Only GET is allowed; test POST, PUT, DELETE
    for method in ["post", "put", "delete", "patch"]:
        resp = getattr(client, method)("/")
        assert resp.status_code == 405
        assert resp.json()["detail"] == "Method Not Allowed"

def test_health_endpoint_content_type(client):
    response = client.get("/")
    assert response.headers["content-type"].startswith("application/json")

def test_app_title_is_set():
    from backend.main import app
    assert app.title == "Logistics Document Intelligence Assistant"

def test_router_is_included():
    # The router is included; test that the app has at least one route from the router
    # Since we mock the router, we check that include_router was called
    with patch("backend.main.FastAPI.include_router") as mock_include_router:
        from importlib import reload
        reload(backend.main)
        mock_include_router.assert_called_once()

def test_logging_configuration(monkeypatch):
    # Patch logging.basicConfig and StreamHandler to check configuration
    with patch("logging.basicConfig") as mock_basicConfig, \
         patch("logging.StreamHandler") as mock_stream_handler:
        import importlib
        importlib.reload(backend.main)
        mock_basicConfig.assert_called_once()
        args, kwargs = mock_basicConfig.call_args
        assert kwargs["level"] == backend.main.logging.INFO
        assert "format" in kwargs
        assert isinstance(kwargs["handlers"][0], MagicMock)
