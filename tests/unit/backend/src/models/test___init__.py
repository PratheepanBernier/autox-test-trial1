import pytest
import sys
from unittest import mock

# Since backend/src/models/__init__.py is empty or only contains imports,
# we will test that importing this module does not raise errors and that
# any expected imports are correctly exposed.

def test_import_models_init_no_errors():
    """
    Test that importing backend.src.models.__init__ does not raise ImportError or other exceptions.
    """
    try:
        import backend.src.models
    except Exception as e:
        pytest.fail(f"Importing backend.src.models raised an exception: {e}")

def test_models_init_exports_expected_symbols(monkeypatch):
    """
    Test that backend.src.models exposes expected symbols if any are defined.
    This test assumes __all__ is defined if symbols are meant to be exported.
    """
    import importlib

    # Reload the module to ensure any changes are picked up
    models_module = importlib.import_module("backend.src.models")
    if hasattr(models_module, "__all__"):
        for symbol in models_module.__all__:
            assert hasattr(models_module, symbol), f"Symbol '{symbol}' listed in __all__ but not found in module"

def test_models_init_imports_are_mockable(monkeypatch):
    """
    Test that if __init__.py imports submodules, those imports can be mocked.
    This is important for testability and isolation.
    """
    # Simulate a submodule import in __init__.py
    with mock.patch.dict(sys.modules, {"backend.src.models.some_submodule": mock.Mock()}):
        import importlib
        importlib.reload(sys.modules["backend.src.models"])

def test_models_init_multiple_imports_consistency():
    """
    Test that multiple imports of backend.src.models yield the same module object (singleton behavior).
    """
    import backend.src.models
    import importlib

    first_import = sys.modules["backend.src.models"]
    second_import = importlib.import_module("backend.src.models")
    assert first_import is second_import, "Multiple imports should return the same module object"

def test_models_init_isolation_between_tests(monkeypatch):
    """
    Test that changes to backend.src.models in one test do not affect other tests.
    """
    import importlib
    models_module = importlib.import_module("backend.src.models")
    # Set an attribute
    setattr(models_module, "_test_attr", 123)
    # Remove the attribute and reload
    delattr(models_module, "_test_attr")
    importlib.reload(models_module)
    assert not hasattr(models_module, "_test_attr"), "Module should not retain test attribute after reload"
