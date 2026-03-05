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

def test_import_backend_src_init_happy_path():
    """
    Test that backend.src.__init__ can be imported without error (happy path).
    """
    try:
        mod = importlib.import_module("backend.src")
        assert isinstance(mod, types.ModuleType)
    except Exception as e:
        pytest.fail(f"Importing backend.src failed: {e}")

def test_import_backend_src_init_multiple_times_is_idempotent():
    """
    Test that importing backend.src.__init__ multiple times does not raise errors (idempotency).
    """
    try:
        mod1 = importlib.import_module("backend.src")
        mod2 = importlib.reload(mod1)
        assert mod1 is mod2 or isinstance(mod2, types.ModuleType)
    except Exception as e:
        pytest.fail(f"Multiple imports/reloads of backend.src failed: {e}")

def test_import_backend_src_init_edge_case_module_not_found(monkeypatch):
    """
    Test edge case where backend.src does not exist in sys.modules and import fails.
    """
    monkeypatch.setitem(sys.modules, "backend.src", None)
    # Remove the module from sys.modules to simulate missing module
    sys.modules.pop("backend.src", None)
    sys.modules.pop("backend", None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.src")

def test_import_backend_src_init_boundary_empty_init(tmp_path, monkeypatch):
    """
    Test importing a dynamically created empty __init__.py in a temporary backend/src package.
    """
    # Create backend/src/__init__.py in a temp directory
    backend_dir = tmp_path / "backend" / "src"
    backend_dir.mkdir(parents=True)
    init_file = backend_dir / "__init__.py"
    init_file.write_text("")  # empty __init__.py

    # Add tmp_path to sys.path
    monkeypatch.syspath_prepend(str(tmp_path))
    # Remove from sys.modules if present
    sys.modules.pop("backend.src", None)
    sys.modules.pop("backend", None)
    mod = importlib.import_module("backend.src")
    assert isinstance(mod, types.ModuleType)
    assert hasattr(mod, "__file__")
    assert str(mod.__file__).endswith("__init__.py")

def test_import_backend_src_init_error_handling(monkeypatch, tmp_path):
    """
    Test error handling when __init__.py contains invalid syntax.
    """
    backend_dir = tmp_path / "backend" / "src"
    backend_dir.mkdir(parents=True)
    init_file = backend_dir / "__init__.py"
    init_file.write_text("def broken(:")  # invalid syntax

    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("backend.src", None)
    sys.modules.pop("backend", None)
    with pytest.raises(SyntaxError):
        importlib.import_module("backend.src")
