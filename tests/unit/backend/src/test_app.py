import pytest
from unittest.mock import patch, MagicMock

import logging

import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient

# Patch settings and router before importing app.py
@pytest.fixture(autouse=True)
def patch_settings_and_router(monkeypatch):
    # Patch settings.APP_NAME
    mock_settings = MagicMock()
    mock_settings.APP_NAME = "TestApp"
    monkeypatch.setattr("backend.src.core.config.settings", mock_settings)

    # Patch router
    mock_router = MagicMock()
    monkeypatch.setattr("backend.src.api.routes.router", mock_router)
    yield

# Import after patching
import backend.src.app as app_module

def test_create_app_returns_fastapi_instance():
    # Arrange & Act
    application = app_module.create_app()
    # Assert
    assert isinstance(application, FastAPI)
    assert application.title == "TestApp"

def test_create_app_includes_router(monkeypatch):
    # Arrange
    mock_router = MagicMock()
    monkeypatch.setattr("backend.src.api.routes.router", mock_router)
    # Act
    application = app_module.create_app()
    # Assert
    # include_router should have been called with mock_router
    # FastAPI stores routers in application.router.routes, but since we mock router, check call
    # Since we can't check internal FastAPI state with a MagicMock, check that include_router was called
    # Patch FastAPI to track include_router calls
    with patch.object(FastAPI, "include_router") as mock_include_router:
        app_module.create_app()
        mock_include_router.assert_called_with(mock_router)

def test_health_endpoint_returns_status_running(monkeypatch):
    # Arrange
    # Patch router to avoid side effects
    mock_router = MagicMock()
    monkeypatch.setattr("backend.src.api.routes.router", mock_router)
    application = app_module.create_app()
    client = TestClient(application)
    # Act
    response = client.get("/")
    # Assert
    assert response.status_code == 200
    assert response.json() == {"status": "running"}

def test_health_endpoint_logs_info(monkeypatch, caplog):
    # Arrange
    mock_router = MagicMock()
    monkeypatch.setattr("backend.src.api.routes.router", mock_router)
    application = app_module.create_app()
    client = TestClient(application)
    # Act
    with caplog.at_level(logging.INFO):
        client.get("/")
    # Assert
    assert any("Health check endpoint called" in message for message in caplog.messages)

def test_create_app_with_empty_app_name(monkeypatch):
    # Arrange
    mock_settings = MagicMock()
    mock_settings.APP_NAME = ""
    monkeypatch.setattr("backend.src.core.config.settings", mock_settings)
    mock_router = MagicMock()
    monkeypatch.setattr("backend.src.api.routes.router", mock_router)
    # Act
    application = app_module.create_app()
    # Assert
    assert application.title == ""

def test_create_app_with_long_app_name(monkeypatch):
    # Arrange
    long_name = "A" * 256
    mock_settings = MagicMock()
    mock_settings.APP_NAME = long_name
    monkeypatch.setattr("backend.src.core.config.settings", mock_settings)
    mock_router = MagicMock()
    monkeypatch.setattr("backend.src.api.routes.router", mock_router)
    # Act
    application = app_module.create_app()
    # Assert
    assert application.title == long_name

def test_create_app_multiple_calls_return_independent_instances(monkeypatch):
    # Arrange
    mock_settings = MagicMock()
    mock_settings.APP_NAME = "TestApp"
    monkeypatch.setattr("backend.src.core.config.settings", mock_settings)
    mock_router = MagicMock()
    monkeypatch.setattr("backend.src.api.routes.router", mock_router)
    # Act
    app1 = app_module.create_app()
    app2 = app_module.create_app()
    # Assert
    assert app1 is not app2
    assert isinstance(app1, FastAPI)
    assert isinstance(app2, FastAPI)

def test_app_global_instance_is_fastapi():
    # Assert
    assert isinstance(app_module.app, FastAPI)
    # The title should match the patched settings.APP_NAME from the fixture
    assert app_module.app.title == "TestApp"
