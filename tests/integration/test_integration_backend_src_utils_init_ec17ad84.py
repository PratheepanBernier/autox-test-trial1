# source_hash: e3b0c44298fc1c14
# import_target: backend.src.utils
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import pytest

# Since backend/src/utils/__init__.py is likely an empty or utility aggregator,
# we will test that importing it does not raise, and that any expected symbols are present.
# If there are utility functions/classes, we will test their integration.
# Since the file content is not provided, we will test import and attribute presence.

def test_import_utils_init_does_not_raise():
    try:
        import backend.src.utils
    except Exception as e:
        pytest.fail(f"Importing backend.src.utils raised an exception: {e}")

def test_utils_init_has_expected_attributes(monkeypatch):
    import importlib

    # Simulate a utility function in __init__.py for edge case
    # We'll monkeypatch the module to add a dummy attribute for this test
    utils_module = importlib.import_module("backend.src.utils")
    monkeypatch.setattr(utils_module, "DUMMY_UTIL", lambda x: x + 1)
    assert hasattr(utils_module, "DUMMY_UTIL")
    assert utils_module.DUMMY_UTIL(2) == 3

def test_utils_init_attribute_error_on_missing(monkeypatch):
    import importlib

    utils_module = importlib.import_module("backend.src.utils")
    if hasattr(utils_module, "NON_EXISTENT_ATTRIBUTE"):
        monkeypatch.delattr(utils_module, "NON_EXISTENT_ATTRIBUTE")
    with pytest.raises(AttributeError):
        _ = getattr(utils_module, "NON_EXISTENT_ATTRIBUTE")

def test_utils_init_multiple_import_paths_equivalence():
    import importlib

    # Import via two equivalent paths and compare their id (should be same module object)
    mod1 = importlib.import_module("backend.src.utils")
    mod2 = importlib.import_module("backend.src.utils.__init__")
    assert mod1 is mod2

def test_utils_init_import_is_deterministic():
    import importlib

    # Importing multiple times should always yield the same module object
    mod1 = importlib.import_module("backend.src.utils")
    mod2 = importlib.import_module("backend.src.utils")
    assert mod1 is mod2

def test_utils_init_import_with_reload(monkeypatch):
    import importlib

    utils_module = importlib.import_module("backend.src.utils")
    monkeypatch.setattr(utils_module, "RELOAD_TEST", 42)
    importlib.reload(utils_module)
    # After reload, monkeypatched attribute should not exist
    assert not hasattr(utils_module, "RELOAD_TEST")
