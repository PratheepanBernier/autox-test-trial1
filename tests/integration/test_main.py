import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Patch the router import to avoid dependency on api.routes
import sys
import types

# Create a dummy router to mock api.routes.router
from fastapi import APIRouter

dummy_router = APIRouter()

# Insert dummy api.routes module into sys.modules
api_module = types.ModuleType("api")
routes_module = types.ModuleType("api.routes")
routes_module.router = dummy_router
api_module.routes = routes_module
sys.modules["api"] = api_module
sys.modules["api.routes"] = routes_module

# Now import the app from backend.main
from backend.main import app

client = TestClient(app)

def test_health_endpoint_returns_running_status(monkeypatch):
    # Patch logger to avoid actual logging during test
    with patch("backend.main.logger") as mock_logger:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "running"}
        mock_logger.info.assert_called_once_with("Health check endpoint called")

def test_health_endpoint_method_not_allowed():
    # Only GET is allowed, test POST, PUT, DELETE
    for method in ["post", "put", "delete", "patch"]:
        resp = getattr(client, method)("/")
        assert resp.status_code == 405
        assert resp.json()["detail"] == "Method Not Allowed"

def test_health_endpoint_boundary_conditions(monkeypatch):
    # No query params or headers should be required
    with patch("backend.main.logger"):
        response = client.get("/", params={"unexpected": "param"})
        assert response.status_code == 200
        assert response.json() == {"status": "running"}

def test_health_endpoint_with_large_headers(monkeypatch):
    # Send large headers to test edge case
    large_header = "x" * 10000
    with patch("backend.main.logger"):
        response = client.get("/", headers={"X-Large-Header": large_header})
        assert response.status_code == 200
        assert response.json() == {"status": "running"}

def test_health_endpoint_logger_error_handling(monkeypatch):
    # Simulate logger raising an exception, should not affect endpoint response
    def raise_exc(*args, **kwargs):
        raise Exception("Logger failed")
    monkeypatch.setattr("backend.main.logger.info", raise_exc)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "running"}

def test_router_inclusion_and_health_endpoint(monkeypatch):
    # Ensure router is included and health endpoint still works
    # Add a dummy route to the dummy_router and check both endpoints
    @dummy_router.get("/dummy")
    def dummy():
        return {"dummy": True}
    with patch("backend.main.logger"):
        resp_health = client.get("/")
        resp_dummy = client.get("/dummy")
        assert resp_health.status_code == 200
        assert resp_health.json() == {"status": "running"}
        assert resp_dummy.status_code == 200
        assert resp_dummy.json() == {"dummy": True}
