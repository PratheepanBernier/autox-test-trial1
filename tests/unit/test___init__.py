import pytest
from unittest import mock

# Since backend/src/services/__init__.py is likely an __init__ file,
# it may import or expose symbols from submodules, or set up package-level logic.
# We'll assume it exposes some services for import, e.g.:
# from .user_service import UserService
# from .order_service import OrderService

# For this test, we'll simulate that __init__.py exposes UserService and OrderService.

# Mock the submodules and their classes
with mock.patch.dict('sys.modules', {
    'backend.src.services.user_service': mock.Mock(),
    'backend.src.services.order_service': mock.Mock(),
}):
    # Now import the __init__ file
    import importlib
    services_init = importlib.import_module('backend.src.services')

def test_imported_services_are_accessible():
    # Test that the expected services are accessible from the package
    assert hasattr(services_init, 'UserService'), "UserService should be exposed in __init__"
    assert hasattr(services_init, 'OrderService'), "OrderService should be exposed in __init__"

def test_imported_services_are_classes():
    # Test that the imported services are classes (or mocks in this case)
    assert isinstance(getattr(services_init, 'UserService'), mock.Mock), "UserService should be a class or mock"
    assert isinstance(getattr(services_init, 'OrderService'), mock.Mock), "OrderService should be a class or mock"

def test_imported_services_are_distinct():
    # Test that the services are not the same object
    assert getattr(services_init, 'UserService') is not getattr(services_init, 'OrderService'), \
        "UserService and OrderService should be distinct"

def test_missing_service_attribute_raises_attribute_error():
    # Test that accessing a non-existent service raises AttributeError
    with pytest.raises(AttributeError):
        getattr(services_init, 'NonExistentService')

def test_services_are_consistent_across_imports():
    # Test that importing the services via __init__ and directly from submodules yields the same object
    with mock.patch('backend.src.services.user_service.UserService', new=mock.Mock(name='UserServiceMock')) as user_service_mock:
        import importlib
        import backend.src.services.user_service as user_service_module
        import backend.src.services as services_init2
        assert services_init2.UserService is user_service_module.UserService

def test_services_do_not_leak_unexpected_attributes():
    # Test that __init__ does not expose unexpected attributes
    public_attrs = [attr for attr in dir(services_init) if not attr.startswith('_')]
    allowed = {'UserService', 'OrderService'}
    assert set(public_attrs).issubset(allowed), f"Unexpected public attributes: {set(public_attrs) - allowed}"
