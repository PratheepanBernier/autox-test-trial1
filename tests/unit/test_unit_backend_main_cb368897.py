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
from unittest.mock import patch, MagicMock

import backend.main


def test_health_endpoint_returns_running_status(monkeypatch):
    mock_logger = MagicMock()
    monkeypatch.setattr(backend.main, "logger", mock_logger)
    response = backend.main.health()
    assert response == {"status": "running"}
    mock_logger.info.assert_called_once_with("Health check endpoint called")


def test_app_has_correct_title():
    assert backend.main.app.title == "Logistics Document Intelligence Assistant"


def test_app_includes_router(monkeypatch):
    # Patch FastAPI.include_router to check if called with correct router
    mock_app = MagicMock()
    monkeypatch.setattr(backend.main, "app", mock_app)
    mock_router = MagicMock()
    monkeypatch.setattr(backend.main, "router", mock_router)
    # Re-import to trigger include_router
    import importlib
    importlib.reload(backend.main)
    mock_app.include_router.assert_called_with(mock_router)


def test_logging_configuration(monkeypatch):
    # Patch logging.basicConfig and StreamHandler to verify configuration
    mock_basicConfig = MagicMock()
    mock_StreamHandler = MagicMock(return_value="handler")
    monkeypatch.setattr("logging.basicConfig", mock_basicConfig)
    monkeypatch.setattr("logging.StreamHandler", mock_StreamHandler)
    import importlib
    importlib.reload(backend.main)
    mock_basicConfig.assert_called()
    args, kwargs = mock_basicConfig.call_args
    assert kwargs["level"] == backend.main.logging.INFO
    assert "format" in kwargs
    assert "handlers" in kwargs
    assert kwargs["handlers"][0] == "handler"
    mock_StreamHandler.assert_called_with(sys.stdout)


def test_health_endpoint_logger_error(monkeypatch):
    # Simulate logger raising an exception, health should still return correct response
    def raise_error(*args, **kwargs):
        raise Exception("Logger failed")
    monkeypatch.setattr(backend.main.logger, "info", raise_error)
    response = backend.main.health()
    assert response == {"status": "running"}


def test_health_endpoint_return_type():
    result = backend.main.health()
    assert isinstance(result, dict)
    assert "status" in result
    assert result["status"] == "running"
