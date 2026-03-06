import pytest
from unittest.mock import patch, MagicMock

import logging

import backend.src.app as app_module


@pytest.fixture
def mock_settings(monkeypatch):
    class DummySettings:
        APP_NAME = "TestApp"
    monkeypatch.setattr("backend.src.core.config.settings", DummySettings())
    yield


@pytest.fixture
def mock_router(monkeypatch):
    dummy_router = MagicMock(name="router")
    monkeypatch.setattr("backend.src.api.routes.router", dummy_router)
    yield dummy_router


def test_create_app_happy_path(mock_settings, mock_router):
    # Arrange
    # (fixtures already patch settings and router)
    # Act
    fastapi_app = app_module.create_app()
    # Assert
    assert fastapi_app.title == "TestApp"
    # Router should be included
    mock_router.include_router.assert_not_called()  # FastAPI's include_router is called, not router's
    # The router should be passed to app.include_router
    # Check that the root path is registered
    routes = [route.path for route in fastapi_app.routes]
    assert "/" in routes


def test_health_endpoint_returns_running_status(monkeypatch, mock_settings, mock_router):
    # Arrange
    fastapi_app = app_module.create_app()
    # Find the health endpoint function
    health_route = next(
        (route for route in fastapi_app.routes if route.path == "/" and "GET" in route.methods), None
    )
    assert health_route is not None
    # Patch logger to check logging
    with patch.object(app_module, "logger") as mock_logger:
        # Act
        response = health_route.endpoint()
        # Assert
        assert response == {"status": "running"}
        mock_logger.info.assert_called_once_with("Health check endpoint called")


def test_create_app_with_empty_app_name(monkeypatch, mock_router):
    # Arrange
    class DummySettings:
        APP_NAME = ""
    monkeypatch.setattr("backend.src.core.config.settings", DummySettings())
    # Act
    fastapi_app = app_module.create_app()
    # Assert
    assert fastapi_app.title == ""


def test_create_app_router_included(monkeypatch):
    # Arrange
    dummy_router = MagicMock(name="router")
    monkeypatch.setattr("backend.src.api.routes.router", dummy_router)
    class DummySettings:
        APP_NAME = "RouterTest"
    monkeypatch.setattr("backend.src.core.config.settings", DummySettings())
    # Patch FastAPI to check include_router call
    with patch("backend.src.app.FastAPI") as mock_fastapi_cls:
        mock_app_instance = MagicMock()
        mock_fastapi_cls.return_value = mock_app_instance
        # Act
        app_module.create_app()
        # Assert
        mock_app_instance.include_router.assert_called_once_with(dummy_router)
        mock_fastapi_cls.assert_called_once_with(title="RouterTest")


def test_create_app_logger_config(monkeypatch):
    # Arrange
    # Patch logging.basicConfig and getLogger to check configuration
    with patch("backend.src.app.logging.basicConfig") as mock_basicConfig, \
         patch("backend.src.app.logging.getLogger") as mock_getLogger:
        # Re-import the module to trigger logging config
        import importlib
        importlib.reload(app_module)
        # Assert
        mock_basicConfig.assert_called_once()
        mock_getLogger.assert_called_once_with(app_module.__name__)


def test_create_app_multiple_calls_return_distinct_apps(mock_settings, mock_router):
    # Arrange/Act
    app1 = app_module.create_app()
    app2 = app_module.create_app()
    # Assert
    assert app1 is not app2
    assert app1.title == app2.title
    # Each app should have its own routes list object
    assert app1.routes is not app2.routes


def test_health_endpoint_route_methods(mock_settings, mock_router):
    # Arrange
    fastapi_app = app_module.create_app()
    # Act
    health_route = next(
        (route for route in fastapi_app.routes if route.path == "/"), None
    )
    # Assert
    assert health_route is not None
    assert "GET" in health_route.methods
    assert callable(health_route.endpoint)


def test_create_app_with_nonstring_app_name(monkeypatch, mock_router):
    # Arrange
    class DummySettings:
        APP_NAME = 12345  # Non-string app name
    monkeypatch.setattr("backend.src.core.config.settings", DummySettings())
    # Act
    fastapi_app = app_module.create_app()
    # Assert
    # FastAPI coerces title to string
    assert fastapi_app.title == "12345"
