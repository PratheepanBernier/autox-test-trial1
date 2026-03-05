import pytest
import sys
import types

# Since backend/src/utils/__init__.py is empty or only contains package-level code,
# we will test that importing it works, and that it does not introduce side effects or errors.

def test_import_utils_init_success(monkeypatch):
    """
    Test that importing backend.src.utils.__init__ succeeds and does not raise errors.
    """
    # Remove from sys.modules to force re-import
    module_name = "backend.src.utils"
    if module_name in sys.modules:
        del sys.modules[module_name]
    try:
        import backend.src.utils
    except Exception as e:
        pytest.fail(f"Importing backend.src.utils failed with exception: {e}")

def test_utils_init_is_module():
    """
    Test that backend.src.utils is a module and not None.
    """
    import backend.src.utils
    assert isinstance(backend.src.utils, types.ModuleType)
    assert backend.src.utils is not None

def test_utils_init_has_no_unexpected_attributes():
    """
    Test that backend.src.utils does not expose unexpected attributes.
    """
    import backend.src.utils
    # By default, __init__.py should not define any attributes except __doc__, __file__, etc.
    allowed = {"__doc__", "__file__", "__loader__", "__name__", "__package__", "__path__", "__spec__"}
    attrs = set(dir(backend.src.utils))
    # If the __init__.py is empty, only default module attributes should be present
    assert attrs.issuperset(allowed)
    # No unexpected attributes
    assert not (attrs - allowed)

def test_utils_init_import_idempotency():
    """
    Test that importing backend.src.utils multiple times does not cause errors or side effects.
    """
    import importlib
    import backend.src.utils
    before_id = id(backend.src.utils)
    # Re-import
    module = importlib.reload(backend.src.utils)
    after_id = id(module)
    assert before_id == after_id
    assert module is backend.src.utils

def test_utils_init_import_from_parent():
    """
    Test that importing utils from backend.src works as expected.
    """
    from backend.src import utils
    assert isinstance(utils, types.ModuleType)
    assert utils.__name__ == "backend.src.utils"
