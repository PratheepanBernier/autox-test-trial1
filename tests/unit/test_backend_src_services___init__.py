import pytest
import sys
from unittest import mock

def test_import_services_module_success():
    """
    Test that the services module can be imported successfully.
    """
    try:
        import backend.src.services
    except Exception as e:
        pytest.fail(f"Importing backend.src.services failed: {e}")

def test_services_module_has_expected_attributes():
    """
    Test that the services module has no unexpected attributes (empty __init__).
    """
    import backend.src.services as services
    # Since __init__.py is empty, only dunder attributes should exist
    attrs = [attr for attr in dir(services) if not attr.startswith('__')]
    assert attrs == [], f"Unexpected attributes found: {attrs}"

def test_services_module_imports_are_idempotent():
    """
    Test that importing the services module multiple times does not raise errors.
    """
    import importlib
    import backend.src.services as services1
    services2 = importlib.reload(services1)
    assert services1 is services2 or services1.__name__ == services2.__name__

def test_services_module_sys_modules_entry():
    """
    Test that the services module is registered in sys.modules after import.
    """
    import backend.src.services
    assert "backend.src.services" in sys.modules

def test_services_module_import_with_mocked_sys_modules():
    """
    Test importing services module when sys.modules is mocked to simulate missing dependencies.
    """
    with mock.patch.dict('sys.modules', {}):
        # Should still be able to import, as __init__.py is empty
        import importlib
        module = importlib.import_module("backend.src.services")
        assert module is not None

def test_services_module_import_error_handling(monkeypatch):
    """
    Test that import errors in __init__.py would propagate (simulate by raising ImportError).
    """
    # Simulate ImportError by patching builtins.__import__ when importing this module
    import builtins

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "backend.src.services":
            raise ImportError("Simulated import error")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="Simulated import error"):
        __import__("backend.src.services")
