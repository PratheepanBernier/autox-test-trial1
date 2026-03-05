import sys
import types
import importlib
import pytest

# Since backend/src/__init__.py is empty or not provided,
# we will test basic importability and Python package semantics.

def test_import_backend_src_init_module():
    """
    Test that backend.src.__init__ can be imported without error.
    """
    try:
        import backend.src
    except Exception as e:
        pytest.fail(f"Importing backend.src failed: {e}")

def test_backend_src_init_module_is_package():
    """
    Test that backend.src is recognized as a package.
    """
    import backend.src
    assert hasattr(backend.src, '__path__'), "backend.src should be a package (have __path__)"

def test_backend_src_init_module_attributes_are_empty():
    """
    Test that backend.src.__init__ does not define unexpected attributes.
    """
    import backend.src
    # Only standard dunder attributes should be present
    allowed = {'__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__'}
    attrs = set(dir(backend.src))
    extra = attrs - allowed
    # Allow for Python internals, but no custom attributes
    assert not any(not a.startswith('__') for a in extra), f"Unexpected attributes in backend.src: {extra}"

def test_backend_src_init_module_reloadable():
    """
    Test that backend.src can be reloaded without error.
    """
    import backend.src
    try:
        importlib.reload(backend.src)
    except Exception as e:
        pytest.fail(f"Reloading backend.src failed: {e}")

def test_backend_src_init_module_sys_modules_consistency():
    """
    Test that backend.src is present in sys.modules after import.
    """
    import backend.src
    assert 'backend.src' in sys.modules
    assert sys.modules['backend.src'] is backend.src

def test_backend_src_init_module_import_star_behavior():
    """
    Test that 'from backend.src import *' does not import any symbols except dunders.
    """
    # Simulate import *
    import backend.src
    public_attrs = [a for a in dir(backend.src) if not a.startswith('_')]
    assert public_attrs == [], f"backend.src should not export public symbols, found: {public_attrs}"
