# source_hash: 93b912fd61c124bd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Patch the router import to avoid dependency on actual api.routes.router
@pytest.fixture(autouse=True)
def patch_router(monkeypatch):
    from fastapi import APIRouter
    dummy_router = APIRouter()
    monkeypatch.setattr("api.routes.router", dummy_router)
    yield

# Import main after patching router to ensure isolation
@pytest.fixture
def app(patch_router):
    import importlib
    import sys
    # Remove backend.main from sys.modules to force reload
    sys.modules.pop("backend.main", None)
    main = importlib.import_module("backend.main")
    return main.app

@pytest.fixture
def client(app):
    return TestClient(app)

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
    # Only GET is allowed, test POST, PUT, DELETE
    for method in ["post", "put", "delete", "patch"]:
        resp = getattr(client, method)("/")
        assert resp.status_code == 405
        assert resp.json()["detail"] == "Method Not Allowed"

def test_health_endpoint_boundary_conditions(client):
    # No query params, headers, or body should affect the response
    response = client.get("/", params={"foo": "bar"}, headers={"X-Test": "1"})
    assert response.status_code == 200
    assert response.json() == {"status": "running"}

def test_health_endpoint_content_type(client):
    response = client.get("/")
    assert response.headers["content-type"].startswith("application/json")

def test_health_endpoint_reconciliation(client):
    # Reconciliation: direct call vs. HTTP call
    import backend.main
    direct_result = backend.main.health()
    http_result = client.get("/").json()
    assert direct_result == http_result

def test_app_title_and_router_inclusion(app):
    # Regression: app title and router inclusion
    assert app.title == "Logistics Document Intelligence Assistant"
    # The router should be included (even if dummy)
    assert len(app.routes) >= 1
    # Health endpoint should be present
    health_paths = [route.path for route in app.routes if getattr(route, "path", None) == "/"]
    assert "/" in health_paths
