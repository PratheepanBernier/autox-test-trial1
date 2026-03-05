import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

import backend.main

@pytest.fixture
def client():
    # Patch the router to avoid side effects from api.routes
    with patch("backend.main.router") as mock_router:
        # Re-import app with router mocked
        from backend.main import app
        yield TestClient(app)

def test_health_endpoint_returns_running_status_and_logs(client):
    with patch("backend.main.logger") as mock_logger:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "running"}
        mock_logger.info.assert_called_once_with("Health check endpoint called")

def test_health_endpoint_multiple_calls_are_consistent(client):
    with patch("backend.main.logger") as mock_logger:
        for _ in range(3):
            response = client.get("/")
            assert response.status_code == 200
            assert response.json() == {"status": "running"}
        assert mock_logger.info.call_count == 3

def test_health_endpoint_method_not_allowed(client):
    # Only GET is allowed, so POST should fail
    response = client.post("/")
    assert response.status_code == 405

def test_health_endpoint_reconciliation_with_direct_function_call(client):
    # Compare FastAPI route vs direct function call
    from backend.main import health
    with patch("backend.main.logger") as mock_logger:
        # Call via FastAPI
        response = client.get("/")
        api_result = response.json()
        # Call direct function
        direct_result = health()
        assert api_result == direct_result

def test_health_endpoint_boundary_conditions(client):
    # No query params or headers required, but test with extra headers
    response = client.get("/", headers={"X-Test-Header": "test"})
    assert response.status_code == 200
    assert response.json() == {"status": "running"}

def test_health_endpoint_error_handling_does_not_crash_on_logger_failure(client):
    # Simulate logger raising an exception; endpoint should still return 200
    def raise_exc(*args, **kwargs):
        raise Exception("Logger failed")
    with patch("backend.main.logger") as mock_logger:
        mock_logger.info.side_effect = raise_exc
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "running"}
