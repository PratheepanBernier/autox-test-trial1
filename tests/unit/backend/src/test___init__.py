import pytest
import sys
import importlib

@pytest.fixture(autouse=True)
def cleanup_sys_modules():
    # Ensure sys.modules is clean before and after each test
    before = set(sys.modules.keys())
    yield
    after = set(sys.modules.keys())
    for mod in after - before:
        sys.modules.pop(mod, None)

def test_import_backend_src_init_no_errors():
    """
    Test that importing backend.src.__init__ does not raise errors.
    This covers the happy path and ensures the module is importable.
    """
    try:
        import backend.src.__init__
    except Exception as e:
        pytest.fail(f"Importing backend.src.__init__ raised an exception: {e}")

def test_backend_src_init_idempotent_import():
    """
    Test that importing backend.src.__init__ multiple times is idempotent.
    """
    import backend.src.__init__
    importlib.reload(importlib.import_module("backend.src.__init__"))
    # No assertion needed; test passes if no exception is raised

def test_backend_src_init_module_attributes():
    """
    Test that backend.src.__init__ does not expose unexpected attributes.
    Edge case: module should not have arbitrary attributes.
    """
    import backend.src.__init__ as init_mod
    allowed_attrs = {"__doc__", "__loader__", "__name__", "__package__", "__spec__"}
    attrs = set(dir(init_mod))
    # Allow only standard module attributes
    assert attrs.issuperset(allowed_attrs)
    # Should not have unexpected attributes (boundary condition)
    unexpected_attrs = attrs - allowed_attrs
    assert not unexpected_attrs, f"Unexpected attributes found: {unexpected_attrs}"

def test_backend_src_init_import_error(monkeypatch):
    """
    Test error handling: simulate ImportError in backend.src.__init__.
    Edge case: forcibly remove __init__ from sys.modules and simulate import error.
    """
    module_name = "backend.src.__init__"
    sys.modules.pop(module_name, None)
    monkeypatch.setitem(sys.modules, module_name, None)
    with pytest.raises(ImportError):
        importlib.reload(importlib.import_module(module_name))
