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

# Since backend/src/models/__init__.py is likely an __init__ file, we will test
# that importing via different paths yields the same module object and contents.

def test_import_module_equivalence():
    """
    Reconciliation: Importing backend.src.models via different mechanisms yields the same module.
    """
    mod1 = importlib.import_module("backend.src.models")
    mod2 = importlib.import_module("backend.src.models.__init__")
    # Both should be the same module object
    assert mod1 is mod2 or mod1.__file__ == mod2.__file__

def test_module_attributes_consistency():
    """
    Reconciliation: All public attributes are consistent across import paths.
    """
    mod1 = importlib.import_module("backend.src.models")
    mod2 = importlib.import_module("backend.src.models.__init__")
    attrs1 = {k: v for k, v in vars(mod1).items() if not k.startswith("_")}
    attrs2 = {k: v for k, v in vars(mod2).items() if not k.startswith("_")}
    assert attrs1 == attrs2

def test_module_type_and_file():
    """
    Happy path: Module is a Python module and has a __file__ attribute.
    """
    mod = importlib.import_module("backend.src.models")
    assert isinstance(mod, types.ModuleType)
    assert hasattr(mod, "__file__")
    assert mod.__file__.endswith("__init__.py") or mod.__file__.endswith("__init__.pyc")

def test_module_dir_consistency():
    """
    Reconciliation: dir() output is consistent across import paths.
    """
    mod1 = importlib.import_module("backend.src.models")
    mod2 = importlib.import_module("backend.src.models.__init__")
    assert set(dir(mod1)) == set(dir(mod2))

def test_import_nonexistent_attribute_raises():
    """
    Error handling: Accessing a non-existent attribute raises AttributeError.
    """
    mod = importlib.import_module("backend.src.models")
    with pytest.raises(AttributeError):
        getattr(mod, "nonexistent_attribute_12345")

def test_module_isolation_from_other_modules(monkeypatch):
    """
    Edge case: Importing backend.src.models does not pollute sys.modules with unrelated modules.
    """
    before = set(sys.modules.keys())
    importlib.import_module("backend.src.models")
    after = set(sys.modules.keys())
    # Only modules under backend.src.models or its submodules may be added
    new_modules = after - before
    for modname in new_modules:
        assert modname.startswith("backend.src.models")

def test_module_reload_consistency():
    """
    Boundary: Reloading the module does not change its public attributes.
    """
    mod = importlib.import_module("backend.src.models")
    attrs_before = {k: v for k, v in vars(mod).items() if not k.startswith("_")}
    importlib.reload(mod)
    attrs_after = {k: v for k, v in vars(mod).items() if not k.startswith("_")}
    assert attrs_before == attrs_after
