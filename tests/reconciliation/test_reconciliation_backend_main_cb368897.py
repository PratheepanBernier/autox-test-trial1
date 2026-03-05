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


def test_health_endpoint_happy_path_and_logging(client):
    with patch.object(backend.main.logger, "info") as mock_logger_info:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "running"}
        mock_logger_info.assert_called_once_with("Health check endpoint called")


def test_health_endpoint_multiple_calls_consistent_output(client):
    with patch.object(backend.main.logger, "info") as mock_logger_info:
        for _ in range(3):
            response = client.get("/")
            assert response.status_code == 200
            assert response.json() == {"status": "running"}
        assert mock_logger_info.call_count == 3


def test_health_endpoint_method_not_allowed(client):
    # POST should not be allowed on health endpoint
    response = client.post("/")
    assert response.status_code == 405
    assert "detail" in response.json()


def test_health_endpoint_with_query_params_ignored(client):
    with patch.object(backend.main.logger, "info") as mock_logger_info:
        response = client.get("/?foo=bar&baz=qux")
        assert response.status_code == 200
        assert response.json() == {"status": "running"}
        mock_logger_info.assert_called_once_with("Health check endpoint called")


def test_health_endpoint_reconciliation_with_direct_function_call(client):
    # Compare FastAPI route vs direct function call
    with patch.object(backend.main.logger, "info") as mock_logger_info:
        api_response = client.get("/")
        # Direct function call
        direct_response = backend.main.health()
        assert api_response.status_code == 200
        assert api_response.json() == direct_response
        mock_logger_info.assert_called()


def test_health_endpoint_logger_error_handling(client):
    # Simulate logger raising an exception, health endpoint should still return correct response
    with patch.object(backend.main.logger, "info", side_effect=Exception("Logger failure")):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "running"}


def test_app_router_inclusion_reconciliation(client):
    # The app should include the router from api.routes
    # We check that a route from the router exists in app.routes
    # For reconciliation, compare router route paths with app route paths
    from api.routes import router as imported_router
    app_route_paths = {route.path for route in backend.main.app.routes}
    router_paths = {route.path for route in imported_router.routes}
    # All router paths should be present in app
    assert router_paths.issubset(app_route_paths)


def test_logging_configuration_reconciliation():
    # Reconcile logging config: logger should inherit level and handlers from root logger
    root_logger = logging.getLogger()
    main_logger = backend.main.logger
    assert main_logger.level == root_logger.level or main_logger.level == logging.NOTSET
    assert any(isinstance(h, logging.StreamHandler) for h in main_logger.handlers + root_logger.handlers)


def test_health_endpoint_boundary_conditions(client):
    # Test with long query string
    long_query = "a" * 1000
    response = client.get(f"/?q={long_query}")
    assert response.status_code == 200
    assert response.json() == {"status": "running"}
    # Test with unusual headers
    response = client.get("/", headers={"X-Test-Header": "test"})
    assert response.status_code == 200
    assert response.json() == {"status": "running"}
