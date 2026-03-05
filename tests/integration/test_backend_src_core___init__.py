import pytest
import sys
import importlib
from unittest import mock

# Since backend/src/core/__init__.py is empty or not provided,
# we will test the importability and module-level behaviors.

def test_core_module_importable():
    """
    Test that the core module can be imported without error.
    """
    try:
        import backend.src.core
    except Exception as e:
        pytest.fail(f"Importing backend.src.core failed: {e}")

def test_core_module_is_package():
    """
    Test that backend.src.core is a package (has __path__ attribute).
    """
    import backend.src.core
    assert hasattr(backend.src.core, '__path__'), "backend.src.core should be a package"

def test_core_module_no_side_effects_on_import(monkeypatch):
    """
    Test that importing backend.src.core does not execute unexpected code or cause side effects.
    """
    # Patch sys.modules to detect any unexpected imports
    with mock.patch.dict(sys.modules, {}):
        try:
            importlib.reload(importlib.import_module("backend.src.core"))
        except Exception as e:
            pytest.fail(f"Importing backend.src.core caused an error: {e}")

def test_core_module_dir_is_empty_or_expected():
    """
    Test that dir(backend.src.core) only contains expected attributes for an empty __init__.py.
    """
    import backend.src.core
    attrs = dir(backend.src.core)
    # __file__, __path__, __package__, __name__, __loader__, __spec__ are standard
    expected = {
        '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__'
    }
    assert set(attrs).issuperset(expected)
    # Should not contain unexpected attributes
    unexpected = set(attrs) - expected
    assert not unexpected or all(a.startswith('__') and a.endswith('__') for a in unexpected)

def test_core_module_import_from_parent():
    """
    Test that importing core from backend.src works as expected.
    """
    import backend.src.core as core
    assert core is sys.modules['backend.src.core']

def test_core_module_multiple_imports_are_idempotent():
    """
    Test that multiple imports of backend.src.core return the same module object.
    """
    import backend.src.core
    mod1 = backend.src.core
    mod2 = importlib.import_module("backend.src.core")
    assert mod1 is mod2

def test_core_module_import_error_handling(monkeypatch):
    """
    Test that import error is raised if the module is missing.
    """
    # Remove backend.src.core from sys.modules if present
    sys.modules.pop('backend.src.core', None)
    # Simulate ImportError by removing from sys.path
    original_sys_path = sys.path.copy()
    monkeypatch.setattr(sys, "path", [])
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.src.core")
    sys.path = original_sys_path
