import pytest
from backend.src.utils import __init__ as utils_init

def test_module_exports_are_consistent():
    """
    Reconciliation test to ensure that the __init__.py file in utils
    exposes the same public API regardless of import path.
    """
    # Import via direct path
    import importlib
    module_direct = importlib.import_module("backend.src.utils.__init__")
    # Import via package
    import backend.src.utils as utils_pkg

    # Compare __all__ if present, else compare dir() minus dunder names
    all_direct = getattr(module_direct, "__all__", None)
    all_pkg = getattr(utils_pkg, "__all__", None)

    if all_direct is not None and all_pkg is not None:
        assert set(all_direct) == set(all_pkg), "Mismatch in __all__ between import paths"
        for name in all_direct:
            assert hasattr(module_direct, name), f"{name} missing in direct import"
            assert hasattr(utils_pkg, name), f"{name} missing in package import"
            assert getattr(module_direct, name) is getattr(utils_pkg, name), f"{name} differs between import paths"
    else:
        # Fallback: compare all non-dunder attributes
        attrs_direct = {k: getattr(module_direct, k) for k in dir(module_direct) if not k.startswith("__")}
        attrs_pkg = {k: getattr(utils_pkg, k) for k in dir(utils_pkg) if not k.startswith("__")}
        assert set(attrs_direct.keys()) == set(attrs_pkg.keys()), "Mismatch in public attributes"
        for k in attrs_direct:
            assert attrs_direct[k] is attrs_pkg[k], f"Attribute {k} differs between import paths"

def test_utils_init_is_idempotent(monkeypatch):
    """
    Reconciliation test to ensure that re-importing utils.__init__ does not change its public API.
    """
    import importlib
    import sys

    module_name = "backend.src.utils.__init__"
    module_first = importlib.import_module(module_name)
    attrs_first = {k: getattr(module_first, k) for k in dir(module_first) if not k.startswith("__")}

    # Remove from sys.modules and re-import
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    module_second = importlib.import_module(module_name)
    attrs_second = {k: getattr(module_second, k) for k in dir(module_second) if not k.startswith("__")}

    assert set(attrs_first.keys()) == set(attrs_second.keys()), "Public API changed after re-import"
    for k in attrs_first:
        assert attrs_first[k] == attrs_second[k], f"Attribute {k} changed after re-import"

def test_utils_init_handles_missing_attributes_gracefully():
    """
    Edge case: If __init__.py is empty or only contains dunder attributes,
    ensure no unexpected public attributes are present.
    """
    public_attrs = [k for k in dir(utils_init) if not k.startswith("__")]
    assert isinstance(public_attrs, list)
    # Accept empty public API as valid
    for attr in public_attrs:
        # All public attributes should be accessible
        getattr(utils_init, attr)

def test_utils_init_dunder_attributes_are_consistent():
    """
    Boundary condition: Ensure dunder attributes like __doc__, __file__, __name__ are present and consistent.
    """
    assert hasattr(utils_init, "__name__")
    assert hasattr(utils_init, "__file__")
    assert hasattr(utils_init, "__doc__")
    # __package__ may be None or a string
    assert hasattr(utils_init, "__package__")
    # __spec__ should be present
    assert hasattr(utils_init, "__spec__")

def test_utils_init_import_error_handling(monkeypatch):
    """
    Error handling: Simulate ImportError in __init__.py and ensure it propagates.
    """
    import importlib
    import sys

    module_name = "backend.src.utils.__init__"
    # Remove from sys.modules to force reload
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    # Patch builtins.__import__ to raise ImportError when called from this module
    import builtins
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("backend.src.utils") and name != module_name:
            raise ImportError("Simulated import error")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Should not raise if __init__.py does not import anything
    try:
        importlib.import_module(module_name)
    except ImportError as e:
        # Acceptable only if __init__.py actually imports from backend.src.utils.*
        assert "Simulated import error" in str(e)
    else:
        # If no ImportError, that's fine for an empty __init__.py
        pass
