import pytest
import sys
import importlib

@pytest.fixture(autouse=True)
def cleanup_backend_module():
    """
    Ensure backend is not cached in sys.modules before each test.
    """
    sys.modules.pop("backend", None)
    yield
    sys.modules.pop("backend", None)

def test_backend_import_does_not_raise():
    """
    Test that importing backend.__init__ does not raise any exceptions (happy path).
    """
    try:
        import backend
    except Exception as e:
        pytest.fail(f"Importing backend.__init__ raised an exception: {e}")

def test_backend_module_is_singleton():
    """
    Test that importing backend multiple times returns the same module object (singleton property).
    """
    import backend
    module1 = sys.modules["backend"]
    import backend as backend2
    module2 = sys.modules["backend"]
    assert module1 is module2
    assert backend is backend2

def test_backend_module_has_no_attributes():
    """
    Test that backend module has no unexpected attributes (edge case for empty __init__.py).
    """
    import backend
    # __doc__, __file__, __loader__, __name__, __package__, __path__, __spec__ are standard
    standard_attrs = {
        "__doc__", "__file__", "__loader__", "__name__", "__package__", "__path__", "__spec__"
    }
    attrs = set(dir(backend))
    extra_attrs = attrs - standard_attrs
    # Allow dunder attributes that may be added by Python, but no custom ones
    custom_attrs = [a for a in extra_attrs if not (a.startswith("__") and a.endswith("__"))]
    assert not custom_attrs, f"backend module has unexpected attributes: {custom_attrs}"

def test_backend_import_with_missing_path(monkeypatch):
    """
    Test importing backend when its __path__ is missing (boundary condition).
    """
    import backend
    monkeypatch.delattr(backend, "__path__")
    # Re-import should still succeed
    importlib.reload(backend)
    assert hasattr(backend, "__name__")
    assert backend.__name__ == "backend"

def test_backend_import_error(monkeypatch):
    """
    Simulate an ImportError during backend import and ensure it propagates (error handling).
    """
    # Remove backend from sys.modules to force re-import
    sys.modules.pop("backend", None)
    # Patch builtins.__import__ to raise ImportError when importing backend
    import builtins
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "backend":
            raise ImportError("Simulated import error")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="Simulated import error"):
        importlib.import_module("backend")
