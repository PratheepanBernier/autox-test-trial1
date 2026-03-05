# source_hash: 93b912fd61c124bd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Patch the router import to avoid dependency on api.routes
import sys

# Create a dummy router to mock api.routes.router
from fastapi import APIRouter

dummy_router = APIRouter()

sys.modules["api.routes"] = MagicMock()
sys.modules["api.routes"].router = dummy_router

# Now import the app from main.py
from backend.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_endpoint_returns_running_status_and_logs_info(client):
    with patch("backend.main.logger") as mock_logger:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "running"}
        mock_logger.info.assert_called_once_with("Health check endpoint called")

def test_health_endpoint_method_not_allowed(client):
    # Only GET is allowed, test POST, PUT, DELETE
    for method in ["post", "put", "delete", "patch"]:
        resp = getattr(client, method)("/")
        assert resp.status_code == 405
        assert resp.json()["detail"] == "Method Not Allowed"

def test_health_endpoint_boundary_conditions(client):
    # No query params or headers should be required
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "running"}

def test_health_endpoint_with_extra_headers_and_query_params(client):
    # Extra headers and query params should be ignored
    response = client.get("/?foo=bar", headers={"X-Test": "value"})
    assert response.status_code == 200
    assert response.json() == {"status": "running"}

def test_health_endpoint_content_negotiation(client):
    # Accept header variations
    response = client.get("/", headers={"Accept": "application/json"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"status": "running"}

def test_health_endpoint_regression_equivalence(client):
    # Regression: repeated calls should yield same result
    result1 = client.get("/").json()
    result2 = client.get("/").json()
    assert result1 == result2 == {"status": "running"}

def test_health_endpoint_reconciliation_equivalence(client):
    # Reconciliation: GET / and GET /? should be equivalent
    resp1 = client.get("/")
    resp2 = client.get("/?")
    assert resp1.status_code == resp2.status_code == 200
    assert resp1.json() == resp2.json() == {"status": "running"}
