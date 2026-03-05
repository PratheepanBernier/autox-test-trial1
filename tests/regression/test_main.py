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

# Import main after patching router
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
    # Only GET is allowed
    response = client.post("/")
    assert response.status_code == 405
    assert "detail" in response.json()
    assert response.json()["detail"] == "Method Not Allowed"

def test_health_endpoint_boundary_conditions(client):
    # No query params, should still work
    response = client.get("/")
    assert response.status_code == 200
    # Extra query params should be ignored
    response = client.get("/?foo=bar")
    assert response.status_code == 200
    assert response.json() == {"status": "running"}

def test_health_endpoint_path_not_found(client):
    response = client.get("/nonexistent")
    assert response.status_code == 404
    assert "detail" in response.json()
    assert response.json()["detail"] == "Not Found"

def test_app_title_is_set(app):
    assert app.title == "Logistics Document Intelligence Assistant"

def test_logging_configuration(monkeypatch):
    import importlib
    import sys
    # Remove backend.main from sys.modules to force reload
    sys.modules.pop("backend.main", None)
    # Patch logging.basicConfig to check configuration
    mock_basicConfig = MagicMock()
    monkeypatch.setattr("logging.basicConfig", mock_basicConfig)
    # Patch router to avoid import error
    from fastapi import APIRouter
    monkeypatch.setattr("api.routes.router", APIRouter())
    import backend.main
    mock_basicConfig.assert_called_once()
    kwargs = mock_basicConfig.call_args.kwargs
    assert kwargs["level"] == 20  # logging.INFO
    assert "format" in kwargs
    assert "handlers" in kwargs
    assert any(isinstance(h, type(sys.stdout)) or hasattr(h, "stream") for h in kwargs["handlers"])
