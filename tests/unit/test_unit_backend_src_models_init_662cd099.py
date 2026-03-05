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
import pytest

def test_models_init_imports_without_error():
    """
    Test that backend.src.models.__init__ imports without error (happy path).
    """
    try:
        import backend.src.models
    except Exception as e:
        pytest.fail(f"Importing backend.src.models failed: {e}")

def test_models_init_is_idempotent():
    """
    Test that importing backend.src.models multiple times does not raise errors (idempotency).
    """
    try:
        import backend.src.models
        importlib.reload(importlib.import_module("backend.src.models"))
    except Exception as e:
        pytest.fail(f"Re-importing backend.src.models failed: {e}")

def test_models_init_module_attributes_exist():
    """
    Test that backend.src.models module has expected attributes if any are defined.
    """
    import backend.src.models
    # Check for __file__ and __package__ as minimal attributes
    assert hasattr(backend.src.models, "__file__")
    assert hasattr(backend.src.models, "__package__")

def test_models_init_import_nonexistent_attribute_raises_attribute_error():
    """
    Test that accessing a non-existent attribute on backend.src.models raises AttributeError.
    """
    import backend.src.models
    with pytest.raises(AttributeError):
        _ = backend.src.models.NON_EXISTENT_ATTRIBUTE

def test_models_init_import_as_from_style():
    """
    Test that 'from backend.src import models' works as expected.
    """
    try:
        from backend.src import models
    except Exception as e:
        pytest.fail(f"from backend.src import models failed: {e}")
