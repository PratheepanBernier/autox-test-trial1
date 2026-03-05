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

# Attempt to import all public symbols from backend.src.utils.__init__
import importlib

utils_init = importlib.import_module("backend.src.utils.__init__")

def get_public_symbols(module):
    if hasattr(module, "__all__"):
        return module.__all__
    return [name for name in dir(module) if not name.startswith("_")]

@pytest.mark.parametrize("symbol", get_public_symbols(utils_init))
def test_symbol_consistency_with_direct_import(symbol):
    """
    Reconciliation test: Ensure that importing a symbol from backend.src.utils
    yields the same object as importing it from backend.src.utils.__init__.
    """
    utils = importlib.import_module("backend.src.utils")
    symbol_from_utils = getattr(utils, symbol, None)
    symbol_from_init = getattr(utils_init, symbol, None)
    assert symbol_from_utils is symbol_from_init, (
        f"Symbol '{symbol}' differs between backend.src.utils and backend.src.utils.__init__"
    )

def test_all_public_symbols_are_exposed():
    """
    Happy path: All public symbols in __init__ are exposed in backend.src.utils.
    """
    utils = importlib.import_module("backend.src.utils")
    for symbol in get_public_symbols(utils_init):
        assert hasattr(utils, symbol), f"backend.src.utils missing symbol '{symbol}'"

def test_no_extra_public_symbols_in_utils():
    """
    Edge case: backend.src.utils should not expose extra public symbols not in __init__.
    """
    utils = importlib.import_module("backend.src.utils")
    symbols_in_utils = set(get_public_symbols(utils))
    symbols_in_init = set(get_public_symbols(utils_init))
    extra = symbols_in_utils - symbols_in_init
    assert not extra, f"backend.src.utils exposes extra symbols: {extra}"

def test_import_error_on_missing_symbol(monkeypatch):
    """
    Error handling: Simulate missing symbol in __init__ and check AttributeError.
    """
    symbol = None
    for s in get_public_symbols(utils_init):
        symbol = s
        break
    if symbol is None:
        pytest.skip("No public symbols to test error handling.")
    monkeypatch.delattr(utils_init, symbol, raising=False)
    with pytest.raises(AttributeError):
        getattr(utils_init, symbol)

def test_boundary_empty_public_symbols(monkeypatch):
    """
    Boundary condition: If __all__ is empty, get_public_symbols returns no symbols.
    """
    monkeypatch.setattr(utils_init, "__all__", [], raising=False)
    assert get_public_symbols(utils_init) == []

def test_boundary_no_public_symbols(monkeypatch):
    """
    Boundary condition: If module has no public symbols, get_public_symbols returns empty.
    """
    class Dummy:
        pass
    dummy = Dummy()
    assert get_public_symbols(dummy) == []
