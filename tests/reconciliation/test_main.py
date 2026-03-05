# source_hash: 93b912fd61c124bd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Patch the router import to avoid dependency on api.routes
@pytest.fixture(autouse=True)
def patch_router(monkeypatch):
    from fastapi import APIRouter
    dummy_router = APIRouter()
    monkeypatch.setattr("api.routes.router", dummy_router)
    yield

# Import main after patching
@pytest.fixture
def app():
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
        mock_logger.info.assert_called_with("Health check endpoint called")

def test_health_endpoint_method_not_allowed(client):
    # Only GET is allowed
    response = client.post("/")
    assert response.status_code == 405
    assert "detail" in response.json()

def test_health_endpoint_boundary_conditions(client):
    # Test with query parameters (should be ignored)
    response = client.get("/?foo=bar")
    assert response.status_code == 200
    assert response.json() == {"status": "running"}

def test_health_endpoint_equivalent_paths(client):
    # The "/" endpoint should be equivalent regardless of trailing slash
    response1 = client.get("/")
    response2 = client.get("")
    assert response1.status_code == response2.status_code
    assert response1.json() == response2.json()

def test_health_endpoint_with_unusual_headers(client):
    # Should not affect output
    response = client.get("/", headers={"X-Custom-Header": "value"})
    assert response.status_code == 200
    assert response.json() == {"status": "running"}

def test_health_endpoint_reconciliation_with_direct_call(app):
    # Compare FastAPI route call with direct function call
    from backend.main import health
    with patch("backend.main.logger") as mock_logger:
        direct_result = health()
        mock_logger.info.assert_called_with("Health check endpoint called")
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == direct_result

def test_health_endpoint_error_handling_on_logger_failure(client):
    # Simulate logger failure
    with patch("backend.main.logger.info", side_effect=Exception("Logger failed")):
        response = client.get("/")
        # Endpoint should still return 200 and correct JSON
        assert response.status_code == 200
        assert response.json() == {"status": "running"}
