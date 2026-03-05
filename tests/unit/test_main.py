# source_hash: 93b912fd61c124bd
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

def test_health_endpoint_method_not_allowed(client):
    # Only GET is allowed, so POST should fail
    response = client.post("/")
    assert response.status_code == 405

def test_app_title_is_set_correctly():
    assert main.app.title == "Logistics Document Intelligence Assistant"

def test_logging_configuration(monkeypatch):
    # Patch logging.basicConfig and StreamHandler to verify configuration
    mock_basicConfig = MagicMock()
    mock_StreamHandler = MagicMock(return_value="handler")
    monkeypatch.setattr("logging.basicConfig", mock_basicConfig)
    monkeypatch.setattr("logging.StreamHandler", mock_StreamHandler)
    # Re-import to trigger logging config
    import importlib
    importlib.reload(main)
    mock_basicConfig.assert_called_once()
    args, kwargs = mock_basicConfig.call_args
    assert kwargs["level"] == main.logging.INFO
    assert "format" in kwargs
    assert "handlers" in kwargs
    assert kwargs["handlers"][0] == "handler"

def test_router_included_once(monkeypatch):
    mock_router = MagicMock()
    mock_app = MagicMock()
    monkeypatch.setattr(main, "router", mock_router)
    monkeypatch.setattr(main, "app", mock_app)
    # Re-import to trigger include_router
    import importlib
    importlib.reload(main)
    mock_app.include_router.assert_called_once_with(mock_router)
