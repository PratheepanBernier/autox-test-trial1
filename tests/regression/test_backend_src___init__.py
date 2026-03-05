import sys
import types
import importlib
import pytest

# Since backend/src/__init__.py is empty or not provided,
# we will test basic importability and module properties.

def test_import_backend_src_init_module():
    """
    Test that the backend.src.__init__ module can be imported without error.
    """
    try:
        mod = importlib.import_module("backend.src")
    except Exception as e:
        pytest.fail(f"Importing backend.src failed: {e}")
    assert isinstance(mod, types.ModuleType)
    assert mod.__name__ == "backend.src"

def test_backend_src_init_module_attributes():
    """
    Test that backend.src module has expected default attributes.
    """
    mod = importlib.import_module("backend.src")
    # Standard module attributes
    assert hasattr(mod, "__name__")
    assert hasattr(mod, "__doc__")
    assert hasattr(mod, "__package__")
    assert hasattr(mod, "__loader__")
    assert hasattr(mod, "__spec__")

def test_backend_src_init_module_is_empty():
    """
    Test that backend.src module does not define unexpected attributes.
    """
    mod = importlib.import_module("backend.src")
    # Only standard dunder attributes should be present
    public_attrs = [a for a in dir(mod) if not a.startswith("__")]
    assert public_attrs == []

def test_backend_src_init_module_reimport_idempotency():
    """
    Test that re-importing backend.src returns the same module object (idempotency).
    """
    mod1 = importlib.import_module("backend.src")
    mod2 = importlib.import_module("backend.src")
    assert mod1 is mod2

def test_backend_src_init_module_sys_modules_consistency():
    """
    Test that backend.src is present in sys.modules after import.
    """
    importlib.import_module("backend.src")
    assert "backend.src" in sys.modules
    assert isinstance(sys.modules["backend.src"], types.ModuleType)
