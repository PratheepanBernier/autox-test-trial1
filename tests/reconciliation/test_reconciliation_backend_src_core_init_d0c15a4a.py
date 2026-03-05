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

# Since backend/src/core/__init__.py is likely an empty or package marker file,
# we will test reconciliation of importing via different paths and ensure
# the module identity and attributes are consistent.

def import_module_via_absolute():
    return importlib.import_module("backend.src.core")

def import_module_via_file():
    import importlib.util
    import os
    init_path = os.path.join(ROOT_DIR, "backend", "src", "core", "__init__.py")
    spec = importlib.util.spec_from_file_location("backend.src.core.file", init_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_reconciliation_import_identity_and_attributes():
    mod_abs = import_module_via_absolute()
    mod_file = import_module_via_file()
    # Both should be modules
    assert isinstance(mod_abs, types.ModuleType)
    assert isinstance(mod_file, types.ModuleType)
    # Their __name__ will differ, but their __file__ should be the same
    assert hasattr(mod_abs, "__file__")
    assert hasattr(mod_file, "__file__")
    assert Path(mod_abs.__file__).resolve() == Path(mod_file.__file__).resolve()
    # Their __package__ should be the same
    assert mod_abs.__package__ == "backend.src.core"
    assert mod_file.__package__ == "backend.src.core"
    # They should have the same set of attributes (excluding dunder attributes that may differ)
    abs_attrs = set(dir(mod_abs))
    file_attrs = set(dir(mod_file))
    # Ignore __name__ and __loader__ which may differ
    ignore = {"__name__", "__loader__", "__spec__"}
    assert abs_attrs - ignore == file_attrs - ignore

def test_reconciliation_import_multiple_times_isolated(monkeypatch):
    # Remove from sys.modules to simulate fresh import
    monkeypatch.setitem(sys.modules, "backend.src.core", None)
    mod1 = import_module_via_absolute()
    monkeypatch.setitem(sys.modules, "backend.src.core", None)
    mod2 = import_module_via_absolute()
    assert mod1 is not mod2
    assert Path(mod1.__file__).resolve() == Path(mod2.__file__).resolve()

def test_reconciliation_import_with_missing_file(tmp_path, monkeypatch):
    # Simulate missing __init__.py
    fake_core = tmp_path / "backend" / "src" / "core"
    fake_core.mkdir(parents=True)
    fake_init = fake_core / "__init__.py"
    # Do not create __init__.py
    import importlib.util
    spec = importlib.util.spec_from_file_location("backend.src.core.missing", str(fake_init))
    with pytest.raises(FileNotFoundError):
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

def test_reconciliation_import_with_empty_init(tmp_path, monkeypatch):
    # Simulate empty __init__.py in a temp location
    fake_core = tmp_path / "backend" / "src" / "core"
    fake_core.mkdir(parents=True)
    fake_init = fake_core / "__init__.py"
    fake_init.write_text("")
    import importlib.util
    spec = importlib.util.spec_from_file_location("backend.src.core.empty", str(fake_init))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert isinstance(module, types.ModuleType)
    assert hasattr(module, "__file__")
    assert module.__file__ == str(fake_init)

def test_reconciliation_import_with_syntax_error(tmp_path):
    # Simulate __init__.py with syntax error
    fake_core = tmp_path / "backend" / "src" / "core"
    fake_core.mkdir(parents=True)
    fake_init = fake_core / "__init__.py"
    fake_init.write_text("def broken(:\n")
    import importlib.util
    spec = importlib.util.spec_from_file_location("backend.src.core.syntax", str(fake_init))
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(SyntaxError):
        spec.loader.exec_module(module)
