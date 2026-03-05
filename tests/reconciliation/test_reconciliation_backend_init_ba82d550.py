# source_hash: e3b0c44298fc1c14
# import_target: backend
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import pytest
import importlib
from unittest import mock

MODULE_PATH = "backend.__init__"

def import_backend_init():
    return importlib.import_module(MODULE_PATH)

def test_backend_init_import_happy_path(monkeypatch):
    # Should import without error
    try:
        mod = import_backend_init()
    except Exception as e:
        pytest.fail(f"Importing backend.__init__ raised an exception: {e}")

def test_backend_init_import_multiple_times_is_idempotent():
    # Importing multiple times should not cause issues
    mod1 = import_backend_init()
    mod2 = import_backend_init()
    assert mod1 is mod2 or mod1.__name__ == mod2.__name__

def test_backend_init_import_with_mocked_sys_modules(monkeypatch):
    # Simulate sys.modules missing backend
    sys_modules_backup = sys.modules.copy()
    sys.modules.pop("backend", None)
    sys.modules.pop("backend.__init__", None)
    try:
        mod = import_backend_init()
        assert mod is not None
    finally:
        sys.modules.clear()
        sys.modules.update(sys_modules_backup)

def test_backend_init_import_with_empty_file(tmp_path, monkeypatch):
    # Simulate an empty __init__.py
    backend_dir = tmp_path / "backend"
    backend_dir.mkdir()
    init_file = backend_dir / "__init__.py"
    init_file.write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    mod = importlib.import_module("backend")
    assert hasattr(mod, "__file__")
    assert mod.__file__.endswith("__init__.py")

def test_backend_init_import_with_syntax_error(tmp_path, monkeypatch):
    # Simulate a syntax error in __init__.py
    backend_dir = tmp_path / "backend"
    backend_dir.mkdir()
    init_file = backend_dir / "__init__.py"
    init_file.write_text("def broken(:")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(SyntaxError):
        importlib.reload(importlib.import_module("backend"))

def test_backend_init_import_with_non_utf8_encoding(tmp_path, monkeypatch):
    # Simulate non-UTF8 encoded __init__.py
    backend_dir = tmp_path / "backend"
    backend_dir.mkdir()
    init_file = backend_dir / "__init__.py"
    # Write bytes that are invalid in utf-8
    init_file.write_bytes(b"\xff\xfe\xfd")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(SyntaxError):
        importlib.reload(importlib.import_module("backend"))

def test_backend_init_reconciliation_equivalent_imports(monkeypatch):
    # Reconciliation: importing via importlib and __import__ should yield equivalent modules
    mod1 = import_backend_init()
    mod2 = __import__("backend")
    assert mod1.__name__ == mod2.__name__
    assert hasattr(mod1, "__file__") == hasattr(mod2, "__file__")
    assert (getattr(mod1, "__file__", None) == getattr(mod2, "__file__", None))

def test_backend_init_import_with_missing_file(tmp_path, monkeypatch):
    # Simulate missing __init__.py
    backend_dir = tmp_path / "backend"
    backend_dir.mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend")

def test_backend_init_import_with_directory_named_backend(monkeypatch, tmp_path):
    # Simulate a directory named backend with no __init__.py
    backend_dir = tmp_path / "backend"
    backend_dir.mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend")

def test_backend_init_import_with_file_named_backend(monkeypatch, tmp_path):
    # Simulate a file named backend.py instead of a package
    backend_file = tmp_path / "backend.py"
    backend_file.write_text("# dummy backend file")
    monkeypatch.syspath_prepend(str(tmp_path))
    mod = importlib.import_module("backend")
    assert hasattr(mod, "__file__")
    assert mod.__file__.endswith("backend.py")
