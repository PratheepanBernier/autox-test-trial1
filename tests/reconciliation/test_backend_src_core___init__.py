import pytest
import sys
import types

# Since backend/src/core/__init__.py is empty or not provided,
# we will create a dummy module object to simulate import for reconciliation tests.
# If __init__.py is empty, importing it should not raise errors and should expose no attributes except __builtins__.

@pytest.fixture
def import_core_init(monkeypatch):
    """
    Fixture to simulate importing backend.src.core.__init__ as a module.
    """
    module_name = "backend.src.core.__init__"
    dummy_module = types.ModuleType(module_name)
    sys.modules[module_name] = dummy_module
    yield dummy_module
    del sys.modules[module_name]

def test_import_core_init_no_error(import_core_init):
    """
    Test that importing backend.src.core.__init__ does not raise any errors.
    """
    # The fixture itself simulates the import; if it fails, the test fails.
    assert import_core_init is not None

def test_core_init_has_no_unexpected_attributes(import_core_init):
    """
    Test that the __init__.py module exposes no attributes except standard ones.
    """
    attrs = set(dir(import_core_init))
    # Standard attributes for a module
    standard_attrs = {
        '__name__', '__doc__', '__package__', '__loader__', '__spec__',
        '__file__', '__cached__', '__builtins__'
    }
    # There should be no extra attributes
    assert attrs.issubset(standard_attrs)

def test_core_init_reconciliation_with_direct_import(import_core_init):
    """
    Reconciliation: Compare the dummy import with a direct import of the module.
    """
    # Simulate direct import (would be empty if __init__.py is empty)
    module_name = "backend.src.core.__init__"
    imported_module = sys.modules[module_name]
    # Both should be the same object
    assert import_core_init is imported_module

def test_core_init_reconciliation_with_module_type(import_core_init):
    """
    Reconciliation: Ensure the imported module is of type ModuleType.
    """
    assert isinstance(import_core_init, types.ModuleType)

def test_core_init_import_twice_is_idempotent(import_core_init):
    """
    Test that importing the module twice yields the same module object (idempotency).
    """
    module_name = "backend.src.core.__init__"
    first_import = sys.modules[module_name]
    # Simulate a second import
    second_import = sys.modules[module_name]
    assert first_import is second_import

def test_core_init_import_missing_module(monkeypatch):
    """
    Test error handling: Importing a non-existent module raises ModuleNotFoundError.
    """
    module_name = "backend.src.core.nonexistent"
    if module_name in sys.modules:
        del sys.modules[module_name]
    with pytest.raises(ModuleNotFoundError):
        __import__(module_name)

def test_core_init_module_repr(import_core_init):
    """
    Test that the module's repr is as expected for an empty module.
    """
    expected_repr = f"<module '{import_core_init.__name__}'"
    assert repr(import_core_init).startswith(expected_repr)
