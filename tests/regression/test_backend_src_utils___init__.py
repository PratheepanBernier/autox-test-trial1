import pytest
import sys
import types

# Since backend/src/utils/__init__.py is empty or only contains imports,
# we will test that importing it does not raise errors and that any
# imported symbols are available as expected.

def test_import_utils_init_no_errors():
    """
    Test that importing backend.src.utils.__init__ does not raise ImportError or other exceptions.
    """
    try:
        import backend.src.utils
    except Exception as e:
        pytest.fail(f"Importing backend.src.utils failed with exception: {e}")

def test_utils_init_namespace_consistency():
    """
    Test that the utils package namespace is consistent and contains expected attributes.
    """
    import backend.src.utils as utils_pkg

    # The __init__.py may be empty, but it should at least have __file__ and __path__ attributes
    assert hasattr(utils_pkg, "__file__")
    assert hasattr(utils_pkg, "__path__")
    assert isinstance(utils_pkg.__file__, str)
    assert isinstance(utils_pkg.__path__, list)

def test_utils_init_is_package():
    """
    Test that backend.src.utils is recognized as a package.
    """
    import backend.src.utils as utils_pkg
    assert hasattr(utils_pkg, "__path__")
    assert isinstance(utils_pkg.__path__, list)

def test_utils_init_imports_are_available(monkeypatch):
    """
    If __init__.py imports symbols from submodules, ensure they are available.
    This test is robust to an empty __init__.py.
    """
    import importlib

    # Dynamically reload in case of test isolation
    utils_pkg = importlib.import_module("backend.src.utils")

    # If __init__.py imports symbols, check they are present
    # For regression, we check that no unexpected symbols are present
    allowed_builtin_attrs = {"__doc__", "__file__", "__loader__", "__name__", "__package__", "__path__", "__spec__"}
    all_attrs = set(dir(utils_pkg))
    extra_attrs = all_attrs - allowed_builtin_attrs

    # If there are extra attributes, ensure they are not callables (i.e., not accidentally exposing functions/classes)
    for attr in extra_attrs:
        value = getattr(utils_pkg, attr)
        assert not callable(value), f"Unexpected callable '{attr}' found in utils package"

def test_utils_init_multiple_imports_idempotent():
    """
    Test that importing backend.src.utils multiple times does not cause errors or side effects.
    """
    import importlib

    for _ in range(3):
        importlib.reload(importlib.import_module("backend.src.utils"))

def test_utils_init_module_type():
    """
    Test that backend.src.utils is a module of type 'module'.
    """
    import backend.src.utils as utils_pkg
    assert isinstance(utils_pkg, types.ModuleType)
    assert utils_pkg.__name__ == "backend.src.utils"
