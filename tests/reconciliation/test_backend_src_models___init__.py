import pytest
import sys
import types

# Since backend/src/models/__init__.py is empty or not provided,
# we will simulate a typical __init__.py reconciliation test:
# - It should not raise on import.
# - It should not pollute the namespace.
# - It should be idempotent.
# - It should not have side effects.
# - If it imports submodules, those should be accessible.
# - If it defines __all__, it should match the exported symbols.

# For this test, we assume __init__.py is either empty or only imports submodules.

MODULE_PATH = "backend.src.models"

def import_module_fresh(name):
    """Import a module fresh, removing it from sys.modules first."""
    sys.modules.pop(name, None)
    return __import__(name, fromlist=["*"])

def test_import_does_not_raise():
    """Importing models.__init__ should not raise any exceptions."""
    try:
        import_module_fresh(MODULE_PATH)
    except Exception as e:
        pytest.fail(f"Importing {MODULE_PATH} raised an exception: {e}")

def test_import_idempotency():
    """Importing models.__init__ multiple times should not raise or change state."""
    mod1 = import_module_fresh(MODULE_PATH)
    mod2 = import_module_fresh(MODULE_PATH)
    assert mod1 is not None
    assert mod2 is not None
    assert type(mod1) is types.ModuleType
    assert type(mod2) is types.ModuleType

def test_namespace_cleanliness():
    """models.__init__ should not pollute the namespace with unexpected symbols."""
    mod = import_module_fresh(MODULE_PATH)
    allowed = {"__doc__", "__file__", "__loader__", "__name__", "__package__", "__spec__"}
    # If __all__ is defined, only those should be exported
    if hasattr(mod, "__all__"):
        for symbol in dir(mod):
            if not symbol.startswith("_"):
                assert symbol in mod.__all__, f"Unexpected symbol {symbol} in {MODULE_PATH}"
    else:
        # Otherwise, only dunder names should be present
        for symbol in dir(mod):
            if not symbol.startswith("_"):
                pytest.fail(f"Unexpected public symbol {symbol} in {MODULE_PATH}")

def test_no_side_effects_on_import(monkeypatch):
    """Importing models.__init__ should not modify unrelated global state."""
    # Set a global variable and ensure it is not changed by import
    monkeypatch.setattr("builtins.test_global_var", 42, raising=False)
    import_module_fresh(MODULE_PATH)
    assert getattr(__import__("builtins"), "test_global_var") == 42

def test_equivalent_import_paths():
    """Importing via different equivalent paths yields the same module object."""
    # Import via sys.modules and via __import__
    mod1 = import_module_fresh(MODULE_PATH)
    mod2 = sys.modules.get(MODULE_PATH)
    assert mod1 is mod2

def test_module_attributes_consistency():
    """Module attributes should be consistent across imports."""
    mod1 = import_module_fresh(MODULE_PATH)
    mod2 = import_module_fresh(MODULE_PATH)
    attrs1 = {k: getattr(mod1, k) for k in dir(mod1) if not k.startswith("__")}
    attrs2 = {k: getattr(mod2, k) for k in dir(mod2) if not k.startswith("__")}
    assert attrs1 == attrs2

def test_import_error_handling(monkeypatch):
    """If a submodule import fails, the error should propagate."""
    # Simulate an ImportError in a submodule
    import builtins
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith(MODULE_PATH + "."):
            raise ImportError("Simulated import error")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop(MODULE_PATH, None)
    with pytest.raises(ImportError):
        __import__(MODULE_PATH, fromlist=["*"])
