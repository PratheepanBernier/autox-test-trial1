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
import pytest

def test_import_backend_src_api_init_happy_path():
    try:
        module = importlib.import_module("backend.src.api.__init__")
    except Exception as e:
        pytest.fail(f"Importing backend.src.api.__init__ raised an exception: {e}")

def test_import_backend_src_api_init_multiple_times_is_idempotent():
    module1 = importlib.import_module("backend.src.api.__init__")
    importlib.reload(module1)
    module2 = importlib.import_module("backend.src.api.__init__")
    assert module1 is module2 or module1.__name__ == module2.__name__

def test_import_backend_src_api_init_with_missing_dependency(monkeypatch):
    # Simulate missing dependency if any are imported in __init__.py
    # If no dependencies, this test will simply import as normal
    import builtins
    original_import = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name.startswith("backend.src.api") and name != "backend.src.api.__init__":
            raise ImportError("Simulated missing dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mocked_import)
    try:
        importlib.reload(importlib.import_module("backend.src.api.__init__"))
    except ImportError:
        pass  # Acceptable if __init__ imports submodules
    except Exception as e:
        pytest.fail(f"Unexpected exception during import: {e}")

def test_import_backend_src_api_init_edge_case_empty_module(tmp_path, monkeypatch):
    # Simulate an empty __init__.py
    api_dir = tmp_path / "backend" / "src" / "api"
    api_dir.mkdir(parents=True)
    init_file = api_dir / "__init__.py"
    init_file.write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        module = importlib.import_module("backend.src.api.__init__")
        assert hasattr(module, "__file__")
    finally:
        sys.modules.pop("backend.src.api.__init__", None)
        sys.modules.pop("backend.src.api", None)
        sys.modules.pop("backend.src", None)
        sys.modules.pop("backend", None)
