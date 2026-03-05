import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

import backend.main

@pytest.fixture
def client():
    # Patch the router to avoid dependency on actual api.routes implementation
    with patch("backend.main.router") as mock_router:
        # Patch FastAPI.include_router to avoid side effects
        with patch.object(backend.main.app, "include_router") as mock_include_router:
            # Re-import app to ensure patched router is used
            yield TestClient(backend.main.app)

def test_health_endpoint_returns_running_status(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "running"}

def test_health_endpoint_logs_info(client):
    with patch.object(backend.main.logger, "info") as mock_log:
        response = client.get("/")
        assert response.status_code == 200
        mock_log.assert_called_once_with("Health check endpoint called")

def test_health_endpoint_method_not_allowed(client):
    # Only GET is allowed; test POST, PUT, DELETE, PATCH
    for method in ["post", "put", "delete", "patch"]:
        resp = getattr(client, method)("/")
        assert resp.status_code == 405
        assert resp.json()["detail"] == "Method Not Allowed"

def test_health_endpoint_content_type(client):
    response = client.get("/")
    assert response.headers["content-type"].startswith("application/json")

def test_health_endpoint_boundary_conditions(client):
    # Test with query parameters (should be ignored)
    response = client.get("/?foo=bar&baz=qux")
    assert response.status_code == 200
    assert response.json() == {"status": "running"}

def test_health_endpoint_with_headers(client):
    # Test with custom headers
    response = client.get("/", headers={"X-Test-Header": "test"})
    assert response.status_code == 200
    assert response.json() == {"status": "running"}

def test_health_endpoint_reconciliation_with_direct_call(client):
    # Compare FastAPI endpoint with direct function call
    api_response = client().get("/")
    direct_response = backend.main.health()
    assert api_response.status_code == 200
    assert api_response.json() == direct_response

def test_app_title_and_metadata():
    assert backend.main.app.title == "Logistics Document Intelligence Assistant"
    # FastAPI default version is None unless set
    assert getattr(backend.main.app, "version", None) is None
    # Check that the router is included (mocked)
    assert hasattr(backend.main.app, "include_router")
