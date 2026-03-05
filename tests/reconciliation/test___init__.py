import pytest
from unittest.mock import patch, MagicMock

# Assume backend/src/services/__init__.py exposes two ways to get a service:
# 1. via get_service_by_name(name)
# 2. via ServiceRegistry.get(name)
# We'll reconcile that both return the same instance/object for the same input.

# For demonstration, let's assume the following structure in __init__.py:
# def get_service_by_name(name): ...
# class ServiceRegistry:
#     @staticmethod
#     def get(name): ...

import backend.src.services.__init__ as services_init

@pytest.fixture(autouse=True)
def isolate_registry(monkeypatch):
    # Patch the registry to be a fresh dict for each test
    fake_registry = {}
    monkeypatch.setattr(services_init, "_SERVICE_REGISTRY", fake_registry)
    yield

def register_service(name, instance):
    # Helper to register a service in both code paths
    services_init._SERVICE_REGISTRY[name] = instance

def test_happy_path_reconciliation_between_get_service_by_name_and_registry_get():
    service_name = "email"
    fake_service = MagicMock(name="EmailService")
    register_service(service_name, fake_service)

    result_by_name = services_init.get_service_by_name(service_name)
    result_by_registry = services_init.ServiceRegistry.get(service_name)

    assert result_by_name is result_by_registry
    assert result_by_name is fake_service

def test_reconciliation_with_nonexistent_service_returns_none_or_raises():
    service_name = "nonexistent"
    # No registration

    try:
        result_by_name = services_init.get_service_by_name(service_name)
    except Exception as e1:
        result_by_name = e1

    try:
        result_by_registry = services_init.ServiceRegistry.get(service_name)
    except Exception as e2:
        result_by_registry = e2

    # Both should behave the same way: both None, or both raise and of same type
    if isinstance(result_by_name, Exception) or isinstance(result_by_registry, Exception):
        assert type(result_by_name) is type(result_by_registry)
    else:
        assert result_by_name is result_by_registry

def test_reconciliation_with_empty_string_service_name():
    service_name = ""
    fake_service = MagicMock(name="EmptyNameService")
    register_service(service_name, fake_service)

    result_by_name = services_init.get_service_by_name(service_name)
    result_by_registry = services_init.ServiceRegistry.get(service_name)

    assert result_by_name is result_by_registry
    assert result_by_name is fake_service

def test_reconciliation_with_none_service_name():
    service_name = None
    fake_service = MagicMock(name="NoneNameService")
    register_service(service_name, fake_service)

    result_by_name = services_init.get_service_by_name(service_name)
    result_by_registry = services_init.ServiceRegistry.get(service_name)

    assert result_by_name is result_by_registry
    assert result_by_name is fake_service

def test_reconciliation_with_boundary_service_names():
    # Test with very long name
    long_name = "x" * 1024
    fake_service = MagicMock(name="LongNameService")
    register_service(long_name, fake_service)

    result_by_name = services_init.get_service_by_name(long_name)
    result_by_registry = services_init.ServiceRegistry.get(long_name)
    assert result_by_name is result_by_registry
    assert result_by_name is fake_service

def test_reconciliation_when_service_registry_is_mutated():
    service_name = "cache"
    fake_service1 = MagicMock(name="CacheService1")
    fake_service2 = MagicMock(name="CacheService2")
    register_service(service_name, fake_service1)

    # Both should return the first registered service
    assert services_init.get_service_by_name(service_name) is fake_service1
    assert services_init.ServiceRegistry.get(service_name) is fake_service1

    # Mutate registry
    register_service(service_name, fake_service2)

    # Both should now return the new service
    assert services_init.get_service_by_name(service_name) is fake_service2
    assert services_init.ServiceRegistry.get(service_name) is fake_service2

def test_reconciliation_with_service_name_case_sensitivity():
    # Register lower case
    lower_name = "service"
    upper_name = "SERVICE"
    fake_service_lower = MagicMock(name="LowerService")
    fake_service_upper = MagicMock(name="UpperService")
    register_service(lower_name, fake_service_lower)
    register_service(upper_name, fake_service_upper)

    assert services_init.get_service_by_name(lower_name) is fake_service_lower
    assert services_init.ServiceRegistry.get(lower_name) is fake_service_lower
    assert services_init.get_service_by_name(upper_name) is fake_service_upper
    assert services_init.ServiceRegistry.get(upper_name) is fake_service_upper

def test_reconciliation_with_service_object_identity():
    # Register two different objects under different names
    name1 = "db"
    name2 = "db2"
    fake_service1 = MagicMock(name="DBService1")
    fake_service2 = MagicMock(name="DBService2")
    register_service(name1, fake_service1)
    register_service(name2, fake_service2)

    assert services_init.get_service_by_name(name1) is not services_init.get_service_by_name(name2)
    assert services_init.ServiceRegistry.get(name1) is not services_init.ServiceRegistry.get(name2)
    assert services_init.get_service_by_name(name1) is services_init.ServiceRegistry.get(name1)
    assert services_init.get_service_by_name(name2) is services_init.ServiceRegistry.get(name2)
