import pytest
import sys
from importlib import reload

@pytest.fixture(autouse=True)
def cleanup_core_module():
    """
    Ensure backend.src.core is not cached in sys.modules before each test.
    """
    sys.modules.pop("backend.src.core", None)
    sys.modules.pop("backend.src.core.__init__", None)
    yield
    sys.modules.pop("backend.src.core", None)
    sys.modules.pop("backend.src.core.__init__", None)

def test_core_init_imports_without_error():
    """
    Happy path: importing backend.src.core.__init__ should not raise errors.
    """
    try:
        import backend.src.core
    except Exception as e:
        pytest.fail(f"Importing backend.src.core failed: {e}")

def test_core_init_idempotent_import():
    """
    Edge case: importing backend.src.core multiple times should not raise errors.
    """
    import backend.src.core
    try:
        reload(sys.modules["backend.src.core"])
    except Exception as e:
        pytest.fail(f"Reloading backend.src.core failed: {e}")

def test_core_init_module_attributes_are_accessible():
    """
    Boundary: __init__.py may define __doc__, __file__, __package__, etc.
    """
    import backend.src.core
    assert hasattr(backend.src.core, "__doc__")
    assert hasattr(backend.src.core, "__file__")
    assert hasattr(backend.src.core, "__package__")

def test_core_init_import_error_handling(monkeypatch):
    """
    Error handling: simulate ImportError in __init__.py (if any imports).
    """
    # Patch builtins.__import__ to raise ImportError for a fake submodule
    import builtins

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("backend.src.core.fake_submodule"):
            raise ImportError("Simulated import error")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # If __init__.py does not import fake_submodule, this is a no-op.
    # If it does, this will test error handling.
    try:
        import backend.src.core
    except ImportError as e:
        assert "Simulated import error" in str(e)
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")

def test_core_init_module_repr_is_string():
    """
    Regression: __repr__ of the module should be a string.
    """
    import backend.src.core
    assert isinstance(repr(backend.src.core), str)
