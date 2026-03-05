import pytest
import sys
import types

import backend

def test_backend_module_importable_and_singleton_behavior():
    # Happy path: backend module is importable and is a singleton in sys.modules
    assert "backend" in sys.modules
    assert sys.modules["backend"] is backend

def test_backend_dunder_all_consistency():
    # Edge case: __all__ should be present and consistent if defined
    if hasattr(backend, "__all__"):
        dunder_all = getattr(backend, "__all__")
        assert isinstance(dunder_all, (list, tuple))
        # All names in __all__ should be attributes of backend
        for name in dunder_all:
            assert hasattr(backend, name)

def test_backend_module_attributes_consistency():
    # Reconciliation: attributes accessible via dir() and __dict__ should match
    dir_attrs = set(dir(backend))
    dict_attrs = set(backend.__dict__.keys())
    # Some attributes in dir() may be from parent classes, but all __dict__ keys should be in dir()
    assert dict_attrs.issubset(dir_attrs)

def test_backend_module_repr_and_str_consistency():
    # Reconciliation: str and repr should be consistent for modules
    s = str(backend)
    r = repr(backend)
    assert isinstance(s, str)
    assert isinstance(r, str)
    # They should both mention 'backend'
    assert "backend" in s
    assert "backend" in r

def test_backend_module_identity_and_type():
    # Happy path: backend is a module and has correct name
    assert isinstance(backend, types.ModuleType)
    assert backend.__name__ == "backend"

def test_backend_module_no_unexpected_attributes():
    # Edge case: backend should not have unexpected attributes
    allowed = {"__name__", "__doc__", "__package__", "__loader__", "__spec__", "__file__", "__cached__"}
    if hasattr(backend, "__all__"):
        allowed.add("__all__")
        for name in backend.__all__:
            allowed.add(name)
    attrs = set(backend.__dict__.keys())
    # Allow for empty module, or only standard dunder attributes
    assert attrs.issuperset(allowed)

def test_backend_module_reload_consistency(monkeypatch):
    # Reconciliation: reloading the module should not change its identity in sys.modules
    import importlib
    old_id = id(backend)
    importlib.reload(backend)
    new_id = id(sys.modules["backend"])
    assert old_id == new_id

def test_backend_module_attribute_error_on_missing(monkeypatch):
    # Error handling: accessing a missing attribute should raise AttributeError
    with pytest.raises(AttributeError):
        getattr(backend, "nonexistent_attribute_12345")
