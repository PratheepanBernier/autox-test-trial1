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
import pytest

def test_import_backend_src_core_init_happy_path():
    """
    Test that backend.src.core.__init__ can be imported without error (happy path).
    """
    try:
        import backend.src.core
    except Exception as e:
        pytest.fail(f"Importing backend.src.core failed: {e}")

def test_import_backend_src_core_init_multiple_times_idempotent():
    """
    Test that importing backend.src.core.__init__ multiple times does not raise errors (idempotency).
    """
    try:
        import backend.src.core
        importlib.reload(importlib.import_module("backend.src.core"))
    except Exception as e:
        pytest.fail(f"Multiple imports of backend.src.core failed: {e}")

def test_import_backend_src_core_init_edge_case_module_not_found(monkeypatch):
    """
    Test edge case where the module path is incorrect, expecting ModuleNotFoundError.
    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.src.core.nonexistent")

def test_import_backend_src_core_init_boundary_empty_module(monkeypatch):
    """
    Test importing __init__ when the file is empty (boundary condition).
    """
    # This test assumes __init__.py is empty or contains only comments/whitespace.
    # If not, it still should import without error.
    try:
        import backend.src.core
    except Exception as e:
        pytest.fail(f"Importing empty backend.src.core.__init__ failed: {e}")

def test_import_backend_src_core_init_error_handling(monkeypatch):
    """
    Test error handling: simulate ImportError by removing backend.src.core from sys.modules.
    """
    module_name = "backend.src.core"
    if module_name in sys.modules:
        del sys.modules[module_name]
    try:
        import backend.src.core
    except ImportError as e:
        pytest.fail(f"ImportError raised unexpectedly: {e}")
