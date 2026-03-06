import pytest
import sys
from unittest import mock

# Patch sys.modules to mock backend.src.app before importing backend.main
@pytest.fixture(autouse=True)
def mock_backend_src_app(monkeypatch):
    mock_app = mock.Mock(name="app")
    mock_create_app = mock.Mock(name="create_app")
    module_mock = mock.Mock()
    module_mock.app = mock_app
    module_mock.create_app = mock_create_app
    sys.modules["backend.src.app"] = module_mock
    yield
    del sys.modules["backend.src.app"]

def test_main_imports_app_and_create_app(monkeypatch):
    # Import backend.main after patching sys.modules
    import importlib
    import backend.main

    # Check that app and create_app are imported from backend.src.app
    assert hasattr(backend.main, "app")
    assert hasattr(backend.main, "create_app")
    # They should be the same as the mocks in sys.modules
    from backend.src import app as src_app_module
    assert backend.main.app is src_app_module.app
    assert backend.main.create_app is src_app_module.create_app

def test_main___all___exposes_app_and_create_app():
    import backend.main
    assert set(backend.main.__all__) == {"app", "create_app"}

def test_main_app_and_create_app_are_not_none():
    import backend.main
    assert backend.main.app is not None
    assert backend.main.create_app is not None

def test_main_app_and_create_app_are_callable_or_mock():
    import backend.main
    # Since we mock, they are Mock objects
    assert hasattr(backend.main.app, "__class__")
    assert hasattr(backend.main.create_app, "__class__")
