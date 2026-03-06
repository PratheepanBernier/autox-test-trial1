import pytest
import sys
from unittest import mock

# Since backend/src/utils/__init__.py is empty or only contains imports,
# we need to test that the imports work as expected and that the module
# can be imported without error. If __init__.py is empty, this is a valid test.

def test_utils_init_importable():
    """
    Test that the utils package __init__ can be imported without error.
    """
    try:
        import backend.src.utils
    except Exception as e:
        pytest.fail(f"Importing backend.src.utils failed: {e}")

def test_utils_init_module_attributes():
    """
    Test that the utils module has no unexpected attributes if __init__.py is empty.
    """
    import backend.src.utils as utils
    # If __init__.py is empty, only __doc__, __file__, __name__, __package__, __path__, __loader__, __spec__ should exist
    allowed_attrs = {
        "__doc__", "__file__", "__name__", "__package__", "__path__", "__loader__", "__spec__"
    }
    module_attrs = set(dir(utils))
    # All attributes should be in allowed_attrs or start with '__'
    for attr in module_attrs:
        if not attr.startswith("__"):
            assert attr in allowed_attrs, f"Unexpected attribute in utils module: {attr}"

def test_utils_init_imports_are_mockable(monkeypatch):
    """
    If __init__.py imports submodules, test that they can be mocked.
    """
    # Simulate a submodule import in __init__.py
    module_name = "backend.src.utils.some_submodule"
    fake_module = mock.Mock()
    sys.modules[module_name] = fake_module
    with mock.patch.dict(sys.modules, {module_name: fake_module}):
        # Re-import utils to trigger import of submodule if present
        import importlib
        import backend.src.utils
        importlib.reload(backend.src.utils)
        # If __init__.py imports some_submodule, it should now be the fake_module
        if hasattr(backend.src.utils, "some_submodule"):
            assert backend.src.utils.some_submodule is fake_module

def test_utils_init_multiple_imports(monkeypatch):
    """
    If __init__.py imports multiple submodules, test that all can be mocked and imported.
    """
    submodules = ["backend.src.utils.a", "backend.src.utils.b"]
    fake_modules = {name: mock.Mock() for name in submodules}
    with mock.patch.dict(sys.modules, fake_modules):
        import importlib
        import backend.src.utils
        importlib.reload(backend.src.utils)
        for name in submodules:
            attr = name.split(".")[-1]
            if hasattr(backend.src.utils, attr):
                assert getattr(backend.src.utils, attr) is fake_modules[name]

def test_utils_init_import_error(monkeypatch):
    """
    If __init__.py imports a missing submodule, test that ImportError is raised.
    """
    # Remove the submodule from sys.modules to simulate missing import
    missing_module = "backend.src.utils.missing_submodule"
    if missing_module in sys.modules:
        del sys.modules[missing_module]
    with mock.patch.dict(sys.modules, {}):
        import importlib
        # Patch __import__ to raise ImportError for the missing submodule
        original_import = __import__
        def fake_import(name, *args, **kwargs):
            if name == missing_module:
                raise ImportError("No module named 'missing_submodule'")
            return original_import(name, *args, **kwargs)
        with mock.patch("builtins.__import__", side_effect=fake_import):
            try:
                import backend.src.utils
                importlib.reload(backend.src.utils)
            except ImportError:
                pass  # Expected
            except Exception as e:
                pytest.fail(f"Unexpected exception: {e}")
            else:
                # If no ImportError, check that missing_submodule is not present
                assert not hasattr(backend.src.utils, "missing_submodule")
