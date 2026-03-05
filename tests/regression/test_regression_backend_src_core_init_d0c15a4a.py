# source_hash: e3b0c44298fc1c14
# import_target: backend.src.core
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import importlib
import types
import pytest

def test_import_core_init_module_happy_path():
    """
    Test that the backend.src.core.__init__ module can be imported without error.
    """
    try:
        module = importlib.import_module("backend.src.core")
        assert isinstance(module, types.ModuleType)
    except Exception as e:
        pytest.fail(f"Importing backend.src.core failed: {e}")

def test_core_init_module_has_expected_attributes():
    """
    Test that the backend.src.core module has no unexpected attributes (empty __init__.py).
    """
    module = importlib.import_module("backend.src.core")
    # If __init__.py is empty, only default attributes should exist
    default_attrs = {"__doc__", "__file__", "__loader__", "__name__", "__package__", "__spec__"}
    module_attrs = set(dir(module))
    # Allow for Python version differences in module attributes
    assert default_attrs.issubset(module_attrs)

def test_core_init_module_import_twice_consistency():
    """
    Test that importing backend.src.core multiple times yields the same module object (singleton).
    """
    module1 = importlib.import_module("backend.src.core")
    module2 = importlib.import_module("backend.src.core")
    assert module1 is module2

def test_core_init_module_import_with_alias():
    """
    Test importing backend.src.core with an alias and ensure it is the same module.
    """
    import backend.src.core as core_alias
    module = importlib.import_module("backend.src.core")
    assert core_alias is module

def test_core_init_module_import_error_on_invalid_module():
    """
    Test that importing a non-existent submodule under backend.src.core raises ImportError.
    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.src.core.nonexistent_submodule")

def test_core_init_module_sys_modules_consistency():
    """
    Test that backend.src.core is present in sys.modules after import.
    """
    importlib.import_module("backend.src.core")
    assert "backend.src.core" in sys.modules

def test_core_init_module_boundary_case_empty_init():
    """
    Test that backend.src.core behaves as an empty module (no custom attributes).
    """
    module = importlib.import_module("backend.src.core")
    # Should not have custom attributes except for dunder attributes
    custom_attrs = [attr for attr in dir(module) if not attr.startswith("__")]
    assert custom_attrs == []
