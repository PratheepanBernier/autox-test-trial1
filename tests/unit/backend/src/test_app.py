import pytest
from unittest.mock import patch, MagicMock

import logging

import sys

import types

import importlib

from fastapi import FastAPI
from fastapi.testclient import TestClient

# Patch settings and router before importing app.py
@pytest.fixture(autouse=True)
def patch_settings_and_router(monkeypatch):
    # Patch settings
    class DummySettings:
        APP_NAME = "TestApp"
    monkeypatch.setattr("backend.src.core.config.settings", DummySettings())

    # Patch router
    dummy_router = MagicMock(name="router")
    monkeypatch.setattr("backend.src.api.routes.router", dummy_router)
    yield

@pytest.fixture
def app_module(patch_settings_and_router):
    # Import the app module fresh for each test
    import backend.src.app as app_module
    importlib.reload(app_module)
    return app_module

@pytest.fixture
def fastapi_app(app_module):
    return app_module.create_app()

@pytest.fixture
def client(fastapi_app):
    return TestClient(fastapi_app)

def test_create_app_returns_fastapi_instance(app_module):
    app = app_module.create_app()
    assert isinstance(app, FastAPI)
    # Title should be set from settings
    assert app.title == "TestApp"

def test_router_included_once(app_module):
    # Patch FastAPI.include_router to track calls
    with patch.object(FastAPI, "include_router") as mock_include_router:
        app = app_module.create_app()
        # Should be called exactly once with the router
        from backend.src.api.routes import router
        mock_include_router.assert_called_once_with(router)

def test_health_endpoint_returns_running_status(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "running"}

def test_health_endpoint_logs_info(app_module, caplog):
    app = app_module.create_app()
    client = TestClient(app)
    with caplog.at_level(logging.INFO):
        client.get("/")
        # Should log the health check message
        assert any("Health check endpoint called" in m for m in caplog.messages)

def test_health_endpoint_method_not_allowed(client):
    # Only GET is allowed
    response = client.post("/")
    assert response.status_code == 405

def test_app_object_is_fastapi_instance(app_module):
    # The module-level 'app' should be a FastAPI instance
    assert isinstance(app_module.app, FastAPI)

def test_logging_configuration(monkeypatch):
    # Patch logging.basicConfig and StreamHandler to check configuration
    mock_basicConfig = MagicMock()
    mock_StreamHandler = MagicMock(return_value="handler")
    monkeypatch.setattr(logging, "basicConfig", mock_basicConfig)
    monkeypatch.setattr(logging, "StreamHandler", mock_StreamHandler)
    # Re-import the module to trigger logging config
    import importlib
    import backend.src.app as app_module
    importlib.reload(app_module)
    mock_basicConfig.assert_called_once()
    args, kwargs = mock_basicConfig.call_args
    assert kwargs["level"] == logging.INFO
    assert "format" in kwargs
    assert "handlers" in kwargs
    assert kwargs["handlers"] == ["handler"]
    mock_StreamHandler.assert_called_once_with(sys.stdout)

def test_create_app_multiple_calls_are_independent(app_module):
    app1 = app_module.create_app()
    app2 = app_module.create_app()
    assert app1 is not app2
    assert isinstance(app1, FastAPI)
    assert isinstance(app2, FastAPI)
    # Both should have the health endpoint
    client1 = TestClient(app1)
    client2 = TestClient(app2)
    assert client1.get("/").status_code == 200
    assert client2.get("/").status_code == 200

def test_health_endpoint_response_type(client):
    response = client.get("/")
    assert isinstance(response.json(), dict)
    assert set(response.json().keys()) == {"status"}
    assert isinstance(response.json()["status"], str)

def test_health_endpoint_boundary_conditions(client):
    # There are no query/path params, so boundary is just the root path
    response = client.get("/")
    assert response.status_code == 200
    # Try with trailing slash (should still work)
    response2 = client.get("//")
    # FastAPI normalizes, so this should also work
    assert response2.status_code == 200

def test_health_endpoint_invalid_methods(client):
    # Try all invalid HTTP methods
    for method in ["post", "put", "delete", "patch", "options"]:
        resp = getattr(client, method)("/")
        assert resp.status_code == 405

def test_app_title_from_settings(monkeypatch):
    # Change settings.APP_NAME and verify app title
    class DummySettings:
        APP_NAME = "AnotherApp"
    monkeypatch.setattr("backend.src.core.config.settings", DummySettings())
    from backend.src import app as app_module
    importlib.reload(app_module)
    assert app_module.app.title == "AnotherApp"
