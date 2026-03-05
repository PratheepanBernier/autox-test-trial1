# source_hash: e3b0c44298fc1c14
# import_target: backend.src.models
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

MODULE_PATH = "backend.src.models.__init__"

def test_module_imports_successfully():
    """Test that the backend.src.models.__init__ module imports without error."""
    try:
        importlib.import_module(MODULE_PATH)
    except Exception as e:
        pytest.fail(f"Importing {MODULE_PATH} failed with exception: {e}")

def test_module_attributes_are_accessible():
    """Test that all expected attributes in __init__.py are accessible, if any."""
    module = importlib.import_module(MODULE_PATH)
    # List all public attributes (not starting with _)
    public_attrs = [attr for attr in dir(module) if not attr.startswith("_")]
    # If there are no public attributes, this is a boundary condition
    if not public_attrs:
        assert public_attrs == []
    else:
        for attr in public_attrs:
            assert hasattr(module, attr)

def test_module_is_a_module_type():
    """Test that the imported object is a module."""
    module = importlib.import_module(MODULE_PATH)
    assert isinstance(module, types.ModuleType)

def test_reimport_module_does_not_raise():
    """Test that re-importing the module does not raise errors (idempotency)."""
    module1 = importlib.import_module(MODULE_PATH)
    importlib.reload(module1)
    module2 = importlib.import_module(MODULE_PATH)
    assert isinstance(module2, types.ModuleType)

def test_import_nonexistent_attribute_raises_attribute_error():
    """Test that accessing a non-existent attribute raises AttributeError."""
    module = importlib.import_module(MODULE_PATH)
    with pytest.raises(AttributeError):
        getattr(module, "nonexistent_attribute_12345")

def test_module_file_path_is_correct():
    """Test that the __file__ attribute points to an __init__.py file."""
    module = importlib.import_module(MODULE_PATH)
    assert module.__file__.endswith("__init__.py")
    assert Path(module.__file__).exists()
