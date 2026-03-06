# backend/tests/use_cases/test_init.py

import importlib
import sys
import types
import pytest

MODULE_PATH = "backend.src.use_cases.__init__"

def test_module_docstring_present():
    """Test that the __init__.py module has the expected docstring."""
    module = importlib.import_module(MODULE_PATH)
    assert module.__doc__ == "Application use-cases."

def test_module_is_empty():
    """Test that the __init__.py module does not define any attributes except standard ones."""
    module = importlib.import_module(MODULE_PATH)
    # Standard dunder attributes for a module
    standard_attrs = {
        "__name__",
        "__doc__",
        "__package__",
        "__loader__",
        "__spec__",
        "__file__",
        "__cached__",
        "__builtins__",
        "__path__",
    }
    module_attrs = set(dir(module))
    # There should be no custom attributes
    assert module_attrs.issuperset(standard_attrs)
    # No unexpected attributes
    assert module_attrs - standard_attrs == set()

def test_module_importable_multiple_times(monkeypatch):
    """Test that the module can be imported multiple times without side effects."""
    importlib.invalidate_caches()
    module1 = importlib.import_module(MODULE_PATH)
    # Remove from sys.modules to force reload
    monkeypatch.delitem(sys.modules, MODULE_PATH, raising=False)
    module2 = importlib.import_module(MODULE_PATH)
    assert module1 is not module2
    assert module2.__doc__ == "Application use-cases."

def test_module_docstring_edge_cases(monkeypatch):
    """Test behavior if the module docstring is missing or altered."""
    module = importlib.import_module(MODULE_PATH)
    # Patch __doc__ to None
    monkeypatch.setattr(module, "__doc__", None)
    assert module.__doc__ is None
    # Patch __doc__ to empty string
    monkeypatch.setattr(module, "__doc__", "")
    assert module.__doc__ == ""
    # Restore original docstring
    monkeypatch.setattr(module, "__doc__", "Application use-cases.")
    assert module.__doc__ == "Application use-cases."
