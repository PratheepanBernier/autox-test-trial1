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

import importlib
import types
import pytest

def test_backend_init_imports_without_error():
    try:
        import backend
    except Exception as e:
        pytest.fail(f"Importing backend.__init__ raised an exception: {e}")

def test_backend_init_idempotent_import():
    # Importing the module multiple times should not raise or change state
    import backend
    importlib.reload(backend)
    importlib.reload(backend)

def test_backend_init_module_attributes_are_consistent():
    import backend
    # Check that __name__ and __package__ are set as expected
    assert hasattr(backend, '__name__')
    assert backend.__name__ == 'backend'
    assert hasattr(backend, '__package__')
    assert backend.__package__ in ('backend', '')

def test_backend_init_module_type():
    import backend
    assert isinstance(backend, types.ModuleType)

def test_backend_init_no_unexpected_side_effects(monkeypatch):
    # Patch a global to detect side effects
    import builtins
    monkeypatch.setattr(builtins, 'backend_init_side_effect', False)
    import backend
    assert not getattr(builtins, 'backend_init_side_effect', False)

def test_backend_init_reload_does_not_raise():
    import backend
    try:
        importlib.reload(backend)
    except Exception as e:
        pytest.fail(f"Reloading backend.__init__ raised an exception: {e}")
