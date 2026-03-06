import pytest
import sys
import importlib

@pytest.fixture(autouse=True)
def cleanup_core_module():
    """
    Ensure 'core' is not cached in sys.modules before each test for isolation.
    """
    sys.modules.pop("core", None)
    sys.modules.pop("core.__init__", None)
    yield
    sys.modules.pop("core", None)
    sys.modules.pop("core.__init__", None)

def test_core_init_imports_without_error():
    """
    Test that importing backend.src.core.__init__ does not raise ImportError or other exceptions.
    """
    try:
        import backend.src.core
    except Exception as e:
        pytest.fail(f"Importing backend.src.core failed with exception: {e}")

def test_core_init_module_attributes():
    """
    Test that backend.src.core.__init__ defines expected dunder attributes.
    """
    import backend.src.core
    core_module = backend.src.core
    # __file__ and __package__ should exist
    assert hasattr(core_module, "__file__")
    assert hasattr(core_module, "__package__")
    # __name__ should be 'backend.src.core'
    assert getattr(core_module, "__name__") == "backend.src.core"

def test_core_init_is_idempotent():
    """
    Test that re-importing backend.src.core does not cause errors (idempotency).
    """
    import backend.src.core
    importlib.reload(backend.src.core)
    # Should not raise

def test_core_init_no_side_effects_on_import(monkeypatch):
    """
    Test that importing backend.src.core does not modify unrelated global state.
    """
    # Example: sys.path should not be changed by importing core
    original_sys_path = list(sys.path)
    import backend.src.core
    assert sys.path == original_sys_path

def test_core_init_import_as_module():
    """
    Test importing backend.src.core.__init__ as a module.
    """
    module = importlib.import_module("backend.src.core")
    assert module is not None
    assert module.__name__ == "backend.src.core"
