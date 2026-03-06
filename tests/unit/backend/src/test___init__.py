import pytest
import sys
import importlib

@pytest.fixture(autouse=True)
def cleanup_sys_modules():
    # Ensure sys.modules is clean before/after each test
    original_modules = sys.modules.copy()
    yield
    sys.modules.clear()
    sys.modules.update(original_modules)

def test_import_backend_src_init_no_side_effects(monkeypatch):
    """
    Test importing backend.src.__init__ does not raise errors or produce side effects.
    """
    # Arrange: Remove backend.src.__init__ from sys.modules if present
    sys.modules.pop("backend.src.__init__", None)
    sys.modules.pop("backend.src", None)
    sys.modules.pop("backend", None)

    # Act & Assert: Import should succeed and not raise
    try:
        import backend.src.__init__
    except Exception as e:
        pytest.fail(f"Importing backend.src.__init__ raised an exception: {e}")

def test_backend_src_init_module_attributes():
    """
    Test that backend.src.__init__ defines no unexpected attributes.
    """
    import backend.src.__init__ as init_mod
    # By default, __init__.py should not define any attributes except __doc__, __file__, etc.
    allowed_attrs = {"__doc__", "__file__", "__loader__", "__name__", "__package__", "__spec__"}
    attrs = set(dir(init_mod))
    # All attributes should be in allowed set
    assert attrs.issuperset(allowed_attrs)
    # No unexpected attributes
    assert attrs - allowed_attrs == set()

def test_backend_src_init_import_idempotency():
    """
    Test that importing backend.src.__init__ multiple times is idempotent.
    """
    import backend.src.__init__ as init_mod1
    importlib.reload(init_mod1)
    import backend.src.__init__ as init_mod2
    assert init_mod1 is init_mod2

def test_backend_src_init_import_from_parent():
    """
    Test importing backend.src from backend and ensure __init__ is loaded.
    """
    import backend.src
    # __init__ should be loaded as backend.src
    assert hasattr(backend, "src")
    assert backend.src.__name__ == "backend.src"
