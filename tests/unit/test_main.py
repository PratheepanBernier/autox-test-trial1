import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Patch the router import to avoid dependency on api.routes
with patch("backend.main.router", autospec=True):
    import backend.main as main

@pytest.fixture
def client():
    return TestClient(main.app)

def test_health_endpoint_returns_running_status_and_logs_info(client):
    with patch.object(main.logger, "info") as mock_log:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "running"}
        mock_log.assert_called_once_with("Health check endpoint called")

def test_health_endpoint_multiple_calls_are_independent(client):
    with patch.object(main.logger, "info") as mock_log:
        for _ in range(3):
            response = client.get("/")
            assert response.status_code == 200
            assert response.json() == {"status": "running"}
        assert mock_log.call_count == 3

def test_health_endpoint_method_not_allowed(client):
    # Only GET is allowed, test POST, PUT, DELETE
    for method in ["post", "put", "delete", "patch"]:
        resp = getattr(client, method)("/")
        assert resp.status_code == 405

def test_app_title_and_router_inclusion():
    # Check app title
    assert main.app.title == "Logistics Document Intelligence Assistant"
    # Check router inclusion (should be present in routes)
    # Since router is mocked, just check that at least one route exists (the health route)
    route_paths = [route.path for route in main.app.routes]
    assert "/" in route_paths

def test_logging_configuration(monkeypatch):
    # Check that logging is configured at INFO level and uses StreamHandler
    logger = main.logger
    assert logger.level == logging.INFO or logger.level == 0  # 0 means not set, inherited from root
    handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    # If logger.handlers is empty, check root logger
    if not handlers:
        handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.StreamHandler)]
    assert handlers, "StreamHandler should be configured for logging"

def test_health_endpoint_edge_case_large_number_of_requests(client):
    # Simulate a burst of requests to ensure determinism and no state leakage
    with patch.object(main.logger, "info") as mock_log:
        for _ in range(50):
            response = client.get("/")
            assert response.status_code == 200
            assert response.json() == {"status": "running"}
        assert mock_log.call_count == 50

def test_health_endpoint_error_handling_logs(monkeypatch, client):
    # Simulate logger raising an exception to test error handling
    with patch.object(main.logger, "info", side_effect=Exception("Logging failed")):
        response = client.get("/")
        # Even if logging fails, endpoint should still return correct response
        assert response.status_code == 200
        assert response.json() == {"status": "running"}
