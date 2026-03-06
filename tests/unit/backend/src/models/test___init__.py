import sys
import types
import pytest

# Since backend/src/models/__init__.py is empty or only contains imports,
# we will test that the imports are correct and that the module loads as expected.
# This is a regression and compatibility test for the __init__.py file.

def test_models_init_imports(monkeypatch):
    """
    Test that importing backend.src.models.__init__ does not raise ImportError,
    and that all expected symbols are present if __init__.py defines any.
    """
    # Simulate the presence of modules that __init__.py might import
    dummy_module = types.ModuleType("dummy")
    monkeypatch.setitem(sys.modules, "backend.src.models.some_model", dummy_module)
    monkeypatch.setitem(sys.modules, "backend.src.models.another_model", dummy_module)

    # Attempt to import the __init__.py module
    try:
        import backend.src.models
    except ImportError as e:
        pytest.fail(f"Importing backend.src.models failed: {e}")

def test_models_init_idempotent_import():
    """
    Test that importing backend.src.models multiple times does not cause errors.
    """
    import importlib
    import backend.src.models
    # Re-import to check idempotency
    importlib.reload(backend.src.models)
    importlib.reload(backend.src.models)

def test_models_init_module_attributes():
    """
    Test that backend.src.models module has expected attributes if any are defined.
    """
    import backend.src.models
    # If __init__.py defines __all__, check that it is a list or tuple
    if hasattr(backend.src.models, "__all__"):
        assert isinstance(backend.src.models.__all__, (list, tuple))
        # __all__ should only contain strings
        assert all(isinstance(item, str) for item in backend.src.models.__all__)

def test_models_init_no_side_effects(monkeypatch):
    """
    Test that importing backend.src.models does not modify global state.
    """
    # Save a snapshot of sys.modules before import
    before = set(sys.modules.keys())
    import backend.src.models
    after = set(sys.modules.keys())
    # The only new modules should be backend.src.models and its direct children
    new_modules = after - before
    allowed_prefix = "backend.src.models"
    for mod in new_modules:
        assert mod.startswith(allowed_prefix)
