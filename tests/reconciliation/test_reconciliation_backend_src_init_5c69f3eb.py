# source_hash: e3b0c44298fc1c14
# import_target: backend.src
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

MODULE_PATH = "backend.src.__init__"

def import_module_via_importlib():
    return importlib.import_module(MODULE_PATH)

def import_module_via___import__():
    return __import__(MODULE_PATH, fromlist=["*"])

def test_reconciliation_importlib_and___import___equivalence():
    mod1 = import_module_via_importlib()
    mod2 = import_module_via___import__()
    # Compare module names
    assert mod1.__name__ == mod2.__name__
    # Compare module dict keys (public API)
    keys1 = set(k for k in mod1.__dict__ if not k.startswith("__"))
    keys2 = set(k for k in mod2.__dict__ if not k.startswith("__"))
    assert keys1 == keys2
    # Compare values for each key
    for key in keys1:
        v1 = getattr(mod1, key)
        v2 = getattr(mod2, key)
        # For functions/classes, compare names and types
        if isinstance(v1, (types.FunctionType, type)):
            assert type(v1) == type(v2)
            assert getattr(v1, "__name__", None) == getattr(v2, "__name__", None)
        else:
            assert v1 == v2

def test_reconciliation_module_is_singleton():
    mod1 = import_module_via_importlib()
    mod2 = import_module_via___import__()
    assert mod1 is mod2

def test_reconciliation_import_multiple_times_consistent():
    mod1 = import_module_via_importlib()
    mod2 = import_module_via_importlib()
    assert mod1 is mod2

def test_reconciliation_import_with_reload_consistent(monkeypatch):
    mod1 = import_module_via_importlib()
    import importlib
    mod2 = importlib.reload(mod1)
    # After reload, should still be the same module name and API
    assert mod2.__name__ == mod1.__name__
    keys1 = set(k for k in mod1.__dict__ if not k.startswith("__"))
    keys2 = set(k for k in mod2.__dict__ if not k.startswith("__"))
    assert keys1 == keys2

def test_reconciliation_import_error_handling(monkeypatch):
    # Simulate import error by removing module from sys.modules and patching importlib
    sys_modules_backup = sys.modules.copy()
    if MODULE_PATH in sys.modules:
        del sys.modules[MODULE_PATH]
    original_import_module = importlib.import_module
    def raise_import_error(name, *args, **kwargs):
        if name == MODULE_PATH:
            raise ImportError("Simulated import error")
        return original_import_module(name, *args, **kwargs)
    monkeypatch.setattr(importlib, "import_module", raise_import_error)
    with pytest.raises(ImportError):
        importlib.import_module(MODULE_PATH)
    sys.modules.clear()
    sys.modules.update(sys_modules_backup)

def test_reconciliation_import_edge_case_empty_module(monkeypatch):
    # Simulate an empty module for edge case
    empty_mod = types.ModuleType(MODULE_PATH)
    sys_modules_backup = sys.modules.copy()
    sys.modules[MODULE_PATH] = empty_mod
    mod1 = import_module_via_importlib()
    mod2 = import_module_via___import__()
    assert mod1 is mod2
    assert mod1.__dict__ == mod2.__dict__
    sys.modules.clear()
    sys.modules.update(sys_modules_backup)
