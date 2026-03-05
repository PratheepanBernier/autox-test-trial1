import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Patch the router import to avoid dependency on api.routes
import sys
import types

# Create a dummy router to mock api.routes.router
dummy_router = types.SimpleNamespace()
dummy_router.__class__ = type("DummyRouter", (), {})
setattr(dummy_router, "routes", [])

# Patch sys.modules so that backend.main imports a dummy router
sys.modules["api"] = types.ModuleType("api")
sys.modules["api.routes"] = types.ModuleType("api.routes")
setattr(sys.modules["api.routes"], "router", dummy_router)

import backend.main

@pytest.fixture
def client():
    return TestClient(backend.main.app)

def test_health_endpoint_happy_path_and_logging(client):
    with patch.object(backend.main.logger, "info") as mock_log:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "running"}
        mock_log.assert_called_once_with("Health check endpoint called")

def test_health_endpoint_multiple_calls_consistency(client):
    # Reconciliation: Ensure repeated calls yield same output
    responses = [client.get("/") for _ in range(3)]
    for resp in responses:
        assert resp.status_code == 200
        assert resp.json() == {"status": "running"}
    # All responses should be identical
    assert all(r.json() == responses[0].json() for r in responses)

def test_health_endpoint_boundary_methods(client):
    # Only GET should be allowed
    for method in ["post", "put", "delete", "patch"]:
        resp = getattr(client, method)("/")
        assert resp.status_code == 405  # Method Not Allowed

def test_health_endpoint_edge_case_headers(client):
    # Reconciliation: Different headers should not affect output
    headers_list = [
        {},
        {"Accept": "application/json"},
        {"User-Agent": "pytest"},
        {"X-Custom-Header": "test"},
    ]
    results = []
    for headers in headers_list:
        resp = client.get("/", headers=headers)
        results.append(resp.json())
        assert resp.status_code == 200
        assert resp.json() == {"status": "running"}
    # All outputs should be identical
    assert all(r == results[0] for r in results)

def test_health_endpoint_error_handling_invalid_path(client):
    # Reconciliation: Invalid path should not return health output
    resp = client.get("/invalid")
    assert resp.status_code == 404
    assert resp.json().get("detail") == "Not Found"

def test_health_endpoint_reconciliation_with_direct_function_call():
    # Reconciliation: Compare FastAPI route vs direct function call
    with patch.object(backend.main.logger, "info"):
        api_result = backend.main.health()
    # Simulate what FastAPI would serialize
    assert api_result == {"status": "running"}

def test_health_endpoint_reconciliation_with_and_without_logging(client):
    # Reconciliation: Output should be same even if logger fails
    with patch.object(backend.main.logger, "info", side_effect=Exception("log fail")):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json() == {"status": "running"}
