import sys
import types
import importlib
import pytest

# Since backend/src/__init__.py is empty or not provided,
# we will test that importing it does not raise errors,
# and that it does not pollute the namespace.

def test_import_init_module_does_not_raise():
    """
    Test that importing backend.src.__init__ does not raise any exceptions.
    """
    try:
        import backend.src
    except Exception as e:
        pytest.fail(f"Importing backend.src raised an exception: {e}")

def test_init_module_namespace_is_empty_or_expected():
    """
    Test that backend.src.__init__ does not define unexpected attributes.
    """
    import backend.src
    # Only allow dunder attributes
    public_attrs = [attr for attr in dir(backend.src) if not attr.startswith('__')]
    assert public_attrs == [], f"backend.src should not define public attributes, found: {public_attrs}"

def test_reimport_init_module_is_idempotent():
    """
    Test that re-importing backend.src does not change its state or raise errors.
    """
    import backend.src
    before_attrs = set(dir(backend.src))
    importlib.reload(backend.src)
    after_attrs = set(dir(backend.src))
    assert before_attrs == after_attrs, "Module attributes changed after reload"

def test_import_init_module_multiple_times():
    """
    Test that importing backend.src multiple times does not raise and is consistent.
    """
    for _ in range(3):
        import backend.src
        assert hasattr(backend.src, '__file__')
        assert hasattr(backend.src, '__package__')

def test_init_module_is_module_type():
    """
    Test that backend.src is a module type.
    """
    import backend.src
    assert isinstance(backend.src, types.ModuleType)
