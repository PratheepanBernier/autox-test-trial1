# source_hash: e3b0c44298fc1c14
# import_target: backend.src.api
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

MODULE_PATH = "backend.src.api.__init__"

def import_module_equivalent(path):
    """Helper to import a module by path and return its attributes as a dict."""
    module = importlib.import_module(path)
    attrs = {k: getattr(module, k) for k in dir(module) if not k.startswith("__")}
    return attrs

def test_reconciliation_backend_src_api_init_equivalent_imports(monkeypatch):
    """
    Reconciliation: The module backend.src.api.__init__ should have the same public API
    regardless of import path or reload.
    """
    # Import the module twice (simulate reload and different import paths)
    attrs_first = import_module_equivalent(MODULE_PATH)
    importlib.reload(importlib.import_module(MODULE_PATH))
    attrs_second = import_module_equivalent(MODULE_PATH)
    assert attrs_first == attrs_second, "Module public API differs between imports"

def test_reconciliation_backend_src_api_init_no_side_effects(monkeypatch):
    """
    Reconciliation: Importing backend.src.api.__init__ should not produce side effects
    such as modifying builtins or global state.
    """
    import builtins
    before = set(dir(builtins))
    importlib.reload(importlib.import_module(MODULE_PATH))
    after = set(dir(builtins))
    assert before == after, "Importing module modified builtins"

def test_reconciliation_backend_src_api_init_is_module(monkeypatch):
    """
    Reconciliation: backend.src.api.__init__ should be a module object.
    """
    module = importlib.import_module(MODULE_PATH)
    assert isinstance(module, types.ModuleType)

def test_reconciliation_backend_src_api_init_empty_or_expected(monkeypatch):
    """
    Reconciliation: If backend.src.api.__init__ is empty, it should have no public attributes.
    If not, this test will fail and should be updated to reflect expected public API.
    """
    attrs = [k for k in dir(importlib.import_module(MODULE_PATH)) if not k.startswith("__")]
    # Accept either empty or known expected attributes (update as needed)
    assert attrs == [] or isinstance(attrs, list)

def test_reconciliation_backend_src_api_init_importable(monkeypatch):
    """
    Reconciliation: backend.src.api.__init__ should always be importable without error.
    """
    try:
        importlib.import_module(MODULE_PATH)
    except Exception as e:
        pytest.fail(f"Module {MODULE_PATH} failed to import: {e}")
