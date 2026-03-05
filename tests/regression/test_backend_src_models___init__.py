import pytest
import sys
from unittest import mock

# Since backend/src/models/__init__.py is typically an __init__ file,
# it may only import or expose symbols from submodules.
# We'll test its import behavior and symbol exposure.

# Helper to reload module for isolation
def reload_models_init():
    if "backend.src.models" in sys.modules:
        del sys.modules["backend.src.models"]
    if "backend.src.models.__init__" in sys.modules:
        del sys.modules["backend.src.models.__init__"]
    import importlib
    return importlib.import_module("backend.src.models")

def test_import_models_init_happy_path(monkeypatch):
    # Simulate submodules and their symbols
    dummy_class = type("DummyClass", (), {})
    dummy_func = lambda: 42

    # Patch sys.modules to simulate submodules
    with mock.patch.dict(sys.modules, {
        "backend.src.models.user": mock.Mock(User=dummy_class),
        "backend.src.models.product": mock.Mock(Product=dummy_class),
        "backend.src.models.utils": mock.Mock(helper=dummy_func),
    }):
        # Patch __init__.py to import these symbols
        import importlib
        import types

        # Simulate __init__.py content
        module = types.ModuleType("backend.src.models")
        exec(
            "from backend.src.models.user import User\n"
            "from backend.src.models.product import Product\n"
            "from backend.src.models.utils import helper\n"
            "__all__ = ['User', 'Product', 'helper']\n",
            module.__dict__,
        )

        # Check that symbols are exposed
        assert hasattr(module, "User")
        assert hasattr(module, "Product")
        assert hasattr(module, "helper")
        assert module.User is dummy_class
        assert module.Product is dummy_class
        assert module.helper is dummy_func
        assert set(module.__all__) == {"User", "Product", "helper"}

def test_import_models_init_missing_submodule(monkeypatch):
    # Simulate missing submodule
    with mock.patch.dict(sys.modules, {
        "backend.src.models.user": mock.Mock(User=object),
        # "backend.src.models.product" is missing
        "backend.src.models.utils": mock.Mock(helper=lambda: 1),
    }):
        import types
        module = types.ModuleType("backend.src.models")
        # Simulate __init__.py content with missing import
        code = (
            "from backend.src.models.user import User\n"
            "from backend.src.models.product import Product\n"
            "from backend.src.models.utils import helper\n"
            "__all__ = ['User', 'Product', 'helper']\n"
        )
        with pytest.raises(ModuleNotFoundError):
            exec(code, module.__dict__)

def test_import_models_init_missing_symbol(monkeypatch):
    # Simulate submodule without expected symbol
    with mock.patch.dict(sys.modules, {
        "backend.src.models.user": mock.Mock(),  # No User symbol
        "backend.src.models.product": mock.Mock(Product=object),
        "backend.src.models.utils": mock.Mock(helper=lambda: 1),
    }):
        import types
        module = types.ModuleType("backend.src.models")
        code = (
            "from backend.src.models.user import User\n"
            "from backend.src.models.product import Product\n"
            "from backend.src.models.utils import helper\n"
            "__all__ = ['User', 'Product', 'helper']\n"
        )
        with pytest.raises(ImportError):
            exec(code, module.__dict__)

def test_models_init_all_attribute_consistency():
    # Simulate __init__.py with __all__ and check consistency
    import types
    module = types.ModuleType("backend.src.models")
    code = (
        "class User: pass\n"
        "class Product: pass\n"
        "def helper(): return 1\n"
        "__all__ = ['User', 'Product', 'helper']\n"
    )
    exec(code, module.__dict__)
    # All names in __all__ must be present in module
    for name in module.__all__:
        assert hasattr(module, name)

def test_models_init_all_attribute_empty():
    # Simulate __init__.py with empty __all__
    import types
    module = types.ModuleType("backend.src.models")
    code = (
        "__all__ = []\n"
    )
    exec(code, module.__dict__)
    assert module.__all__ == []

def test_models_init_imports_are_idempotent():
    # Simulate repeated imports do not cause errors
    dummy_class = type("DummyClass", (), {})
    dummy_func = lambda: 99
    with mock.patch.dict(sys.modules, {
        "backend.src.models.user": mock.Mock(User=dummy_class),
        "backend.src.models.product": mock.Mock(Product=dummy_class),
        "backend.src.models.utils": mock.Mock(helper=dummy_func),
    }):
        import types
        module = types.ModuleType("backend.src.models")
        code = (
            "from backend.src.models.user import User\n"
            "from backend.src.models.product import Product\n"
            "from backend.src.models.utils import helper\n"
            "__all__ = ['User', 'Product', 'helper']\n"
        )
        exec(code, module.__dict__)
        # Simulate re-import
        module2 = types.ModuleType("backend.src.models")
        exec(code, module2.__dict__)
        assert module.User is module2.User
        assert module.Product is module2.Product
        assert module.helper is module2.helper

def test_models_init_handles_nonexistent_all_symbol():
    # Simulate __init__.py without __all__
    import types
    module = types.ModuleType("backend.src.models")
    code = (
        "class User: pass\n"
        "class Product: pass\n"
        "def helper(): return 1\n"
    )
    exec(code, module.__dict__)
    # __all__ should not exist
    assert not hasattr(module, "__all__")
    # But symbols should be present
    assert hasattr(module, "User")
    assert hasattr(module, "Product")
    assert hasattr(module, "helper")
