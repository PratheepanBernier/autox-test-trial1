import pytest
import sys
import importlib

# Since backend/src/core/__init__.py is empty or not provided,
# we will test the importability and module attributes for regression.

def test_core_init_importable():
    """
    Test that the core package can be imported without error.
    """
    try:
        import backend.src.core
    except Exception as e:
        pytest.fail(f"Importing backend.src.core failed: {e}")

def test_core_init_module_attributes():
    """
    Test that the core module has expected default attributes.
    """
    import backend.src.core
    # __file__ and __package__ should exist
    assert hasattr(backend.src.core, '__file__')
    assert hasattr(backend.src.core, '__package__')
    # __name__ should be 'backend.src.core'
    assert backend.src.core.__name__ == 'backend.src.core'

def test_core_init_import_via_importlib():
    """
    Test importing the core module using importlib.
    """
    module = importlib.import_module('backend.src.core')
    assert module is sys.modules['backend.src.core']

def test_core_init_double_import_no_side_effects():
    """
    Test that importing core twice does not raise or change state.
    """
    import backend.src.core
    before = dict(sys.modules)
    import backend.src.core
    after = dict(sys.modules)
    assert before == after

def test_core_init_dir_is_empty_or_standard():
    """
    Test that dir() of core module contains only standard dunder attributes.
    """
    import backend.src.core
    attrs = dir(backend.src.core)
    # Accept only standard dunder attributes for an empty __init__.py
    allowed = {'__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__'}
    assert set(attrs).issuperset(allowed)
    # No unexpected attributes
    assert all(a.startswith('__') and a.endswith('__') for a in attrs)

def test_core_init_import_nonexistent_attribute_raises():
    """
    Test that accessing a nonexistent attribute raises AttributeError.
    """
    import backend.src.core
    with pytest.raises(AttributeError):
        _ = backend.src.core.nonexistent_attribute
