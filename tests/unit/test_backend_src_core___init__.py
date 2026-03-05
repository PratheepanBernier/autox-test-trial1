import pytest
import sys
from importlib import reload

# Since backend/src/core/__init__.py is empty or not provided,
# we will test that importing it does not raise errors and that
# it does not pollute the namespace unexpectedly.

def test_import_core_init_no_errors():
    """
    Test that importing backend.src.core.__init__ does not raise any exceptions.
    """
    try:
        import backend.src.core
    except Exception as e:
        pytest.fail(f"Importing backend.src.core raised an exception: {e}")

def test_core_init_namespace_is_empty():
    """
    Test that backend.src.core.__init__ does not define unexpected attributes.
    """
    import backend.src.core
    # __init__ should not define any attributes except for standard ones
    allowed_attrs = {"__doc__", "__file__", "__loader__", "__name__", "__package__", "__path__", "__spec__"}
    attrs = set(dir(backend.src.core))
    unexpected = attrs - allowed_attrs
    assert not unexpected, f"Unexpected attributes in backend.src.core: {unexpected}"

def test_core_init_reloadable():
    """
    Test that backend.src.core can be reloaded without error.
    """
    import backend.src.core
    try:
        reload(backend.src.core)
    except Exception as e:
        pytest.fail(f"Reloading backend.src.core raised an exception: {e}")

def test_core_init_module_path():
    """
    Test that backend.src.core.__init__ is loaded from the expected path.
    """
    import backend.src.core
    assert backend.src.core.__file__.endswith("__init__.py") or backend.src.core.__file__.endswith("__init__.pyc")

def test_core_init_import_as_module():
    """
    Test that backend.src.core can be imported as a module using __import__.
    """
    try:
        mod = __import__("backend.src.core", fromlist=["*"])
        assert mod is not None
    except Exception as e:
        pytest.fail(f"__import__ for backend.src.core failed: {e}")
