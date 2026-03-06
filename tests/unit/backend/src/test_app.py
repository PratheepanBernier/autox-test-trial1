import pytest
from unittest.mock import patch, MagicMock

import sys

import backend.src.app as app_module


@pytest.fixture
def mock_settings(monkeypatch):
    class MockSettings:
        APP_NAME = "TestApp"
    monkeypatch.setattr("backend.src.core.config.settings", MockSettings)
    return MockSettings


@pytest.fixture
def mock_router(monkeypatch):
    mock_router = MagicMock()
    monkeypatch.setattr("backend.src.api.routes.router", mock_router)
    return mock_router


def test_create_app_happy_path(mock_settings, mock_router):
    # Patch FastAPI to observe instantiation and router inclusion
    with patch("backend.src.app.FastAPI") as mock_fastapi_cls:
        mock_app_instance = MagicMock()
        mock_fastapi_cls.return_value = mock_app_instance

        result = app_module.create_app()

        # FastAPI instantiated with correct title
        mock_fastapi_cls.assert_called_once_with(title=mock_settings.APP_NAME)
        # Router included
        mock_app_instance.include_router.assert_called_once_with(mock_router)
        # Health endpoint registered
        assert result == mock_app_instance


def test_health_endpoint_returns_running_status(monkeypatch, mock_settings, mock_router):
    # Use the real FastAPI class to test the health endpoint logic
    from fastapi.testclient import TestClient

    # Patch router to avoid side effects
    monkeypatch.setattr("backend.src.api.routes.router", mock_router)

    # Patch settings
    monkeypatch.setattr("backend.src.core.config.settings", mock_settings)

    # Re-import to ensure patched settings are used
    app = app_module.create_app()
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "running"}


def test_health_endpoint_logs_info(monkeypatch, mock_settings, mock_router):
    # Patch logger to capture log output
    with patch.object(app_module, "logger") as mock_logger:
        from fastapi.testclient import TestClient

        monkeypatch.setattr("backend.src.api.routes.router", mock_router)
        monkeypatch.setattr("backend.src.core.config.settings", mock_settings)

        app = app_module.create_app()
        client = TestClient(app)

        client.get("/")
        mock_logger.info.assert_any_call("Health check endpoint called")


def test_create_app_with_empty_app_name(monkeypatch, mock_router):
    class EmptySettings:
        APP_NAME = ""
    monkeypatch.setattr("backend.src.core.config.settings", EmptySettings)
    monkeypatch.setattr("backend.src.api.routes.router", mock_router)

    from fastapi import FastAPI

    app = app_module.create_app()
    assert isinstance(app, FastAPI)
    # The title should be empty string
    assert app.title == ""


def test_create_app_router_inclusion_failure(monkeypatch, mock_settings):
    # Simulate router inclusion raising an exception
    class FailingRouter:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("Router inclusion failed")
    failing_router = FailingRouter()
    monkeypatch.setattr("backend.src.api.routes.router", failing_router)
    monkeypatch.setattr("backend.src.core.config.settings", mock_settings)

    from fastapi import FastAPI

    # Patch FastAPI to return a real instance for include_router
    app_instance = FastAPI(title=mock_settings.APP_NAME)
    with patch("backend.src.app.FastAPI", return_value=app_instance):
        # Patch include_router to raise
        with patch.object(app_instance, "include_router", side_effect=RuntimeError("Router inclusion failed")):
            with pytest.raises(RuntimeError, match="Router inclusion failed"):
                app_module.create_app()


def test_logging_configuration(monkeypatch):
    # Remove all handlers and re-import to test logging.basicConfig
    for handler in list(app_module.logging.root.handlers):
        app_module.logging.root.removeHandler(handler)

    # Patch sys.stdout to avoid writing to real stdout
    with patch("sys.stdout"):
        # Re-import app.py to trigger logging.basicConfig
        import importlib
        importlib.reload(app_module)

        # Check that at least one handler is attached
        assert app_module.logging.root.handlers
        # Check that the log level is set to INFO
        assert app_module.logging.root.level == app_module.logging.INFO
