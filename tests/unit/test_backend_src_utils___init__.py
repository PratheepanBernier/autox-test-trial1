import pytest
import sys
import types

# Since backend/src/utils/__init__.py is empty or contains only package-level code,
# we test that the module loads correctly and behaves as an importable package.

def test_utils_package_importable():
    """
    Test that the utils package can be imported without error.
    """
    import backend.src.utils
    assert isinstance(backend.src.utils, types.ModuleType)

def test_utils_package_has_no_unexpected_attributes():
    """
    Test that the utils package does not expose unexpected attributes.
    """
    import backend.src.utils
    # Only __name__, __doc__, __package__, __loader__, __spec__, __path__, __file__ are expected
    allowed = {
        "__name__", "__doc__", "__package__", "__loader__", "__spec__", "__path__", "__file__"
    }
    attrs = set(dir(backend.src.utils))
    # There may be __cached__ in some Python versions
    allowed.add("__cached__")
    # There may be __builtins__ in some environments
    allowed.add("__builtins__")
    assert attrs.issubset(allowed)

def test_utils_package_path_is_correct():
    """
    Test that the __path__ attribute of the utils package points to the correct directory.
    """
    import backend.src.utils
    import os
    # __path__ is a list with one entry: the directory of the package
    assert isinstance(backend.src.utils.__path__, list)
    assert len(backend.src.utils.__path__) == 1
    # The last part of the path should be 'utils'
    assert os.path.basename(backend.src.utils.__path__[0]) == "utils"

def test_utils_package_import_from_sys_modules():
    """
    Test that the utils package can be accessed from sys.modules after import.
    """
    import backend.src.utils
    assert "backend.src.utils" in sys.modules
    assert sys.modules["backend.src.utils"] is backend.src.utils

def test_utils_package_double_import_is_idempotent():
    """
    Test that importing the utils package twice yields the same module object.
    """
    import backend.src.utils
    first = backend.src.utils
    import importlib
    second = importlib.import_module("backend.src.utils")
    assert first is second
