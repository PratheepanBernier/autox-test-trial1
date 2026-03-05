# source_hash: e3b0c44298fc1c14
import pytest
from unittest import mock

# Assuming backend/src/services/__init__.py exposes the following for reconciliation:
# - a function `get_service(name)` that returns a service instance by name
# - a function `list_services()` that returns all available service names
# - a function `service_factory(name)` that is an alternate path to get a service instance
# - services are registered in a registry dict: _SERVICE_REGISTRY

# Since the source code is not provided, we will mock the structure and focus on reconciliation tests.

import backend.src.services.__init__ as services_init

@pytest.fixture(autouse=True)
def isolate_registry(monkeypatch):
    # Patch the registry to a fresh dict for each test
    registry = {}
    monkeypatch.setattr(services_init, "_SERVICE_REGISTRY", registry)
    yield
    # No cleanup needed, as monkeypatch will revert

def dummy_service():
    return object()

def register_dummy_service(name="dummy", service=None):
    if service is None:
        service = dummy_service()
    services_init._SERVICE_REGISTRY[name] = service
    return service

def test_get_service_and_service_factory_return_same_instance_for_registered_service():
    service = register_dummy_service("alpha")
    result1 = services_init.get_service("alpha")
    result2 = services_init.service_factory("alpha")
    assert result1 is result2
    assert result1 is service

def test_get_service_and_service_factory_raise_for_unregistered_service():
    with pytest.raises(KeyError):
        services_init.get_service("ghost")
    with pytest.raises(KeyError):
        services_init.service_factory("ghost")

def test_list_services_matches_registry_keys():
    register_dummy_service("svc1")
    register_dummy_service("svc2")
    expected = set(services_init._SERVICE_REGISTRY.keys())
    actual = set(services_init.list_services())
    assert actual == expected

def test_list_services_empty_when_no_services_registered():
    assert services_init.list_services() == []

def test_registering_multiple_services_and_retrieving_them():
    s1 = register_dummy_service("svcA")
    s2 = register_dummy_service("svcB")
    assert services_init.get_service("svcA") is s1
    assert services_init.get_service("svcB") is s2
    assert set(services_init.list_services()) == {"svcA", "svcB"}

def test_service_factory_and_get_service_equivalence_for_multiple_services():
    s1 = register_dummy_service("svcX")
    s2 = register_dummy_service("svcY")
    assert services_init.service_factory("svcX") is services_init.get_service("svcX")
    assert services_init.service_factory("svcY") is services_init.get_service("svcY")

def test_get_service_and_service_factory_are_case_sensitive():
    register_dummy_service("CaseTest")
    with pytest.raises(KeyError):
        services_init.get_service("casetest")
    with pytest.raises(KeyError):
        services_init.service_factory("casetest")

def test_get_service_and_service_factory_with_boundary_service_names():
    # Empty string as name
    register_dummy_service("")
    assert services_init.get_service("") is services_init._SERVICE_REGISTRY[""]
    # Very long name
    long_name = "x" * 1024
    register_dummy_service(long_name)
    assert services_init.get_service(long_name) is services_init._SERVICE_REGISTRY[long_name]

def test_list_services_does_not_expose_internal_registry_reference():
    register_dummy_service("svc")
    result = services_init.list_services()
    assert isinstance(result, list)
    # Mutating result should not affect registry
    result.append("fake")
    assert "fake" not in services_init._SERVICE_REGISTRY

def test_get_service_and_service_factory_do_not_mutate_registry():
    register_dummy_service("immutable")
    before = dict(services_init._SERVICE_REGISTRY)
    services_init.get_service("immutable")
    services_init.service_factory("immutable")
    after = dict(services_init._SERVICE_REGISTRY)
    assert before == after

def test_list_services_consistency_after_service_registration():
    assert services_init.list_services() == []
    register_dummy_service("first")
    assert services_init.list_services() == ["first"]
    register_dummy_service("second")
    assert set(services_init.list_services()) == {"first", "second"}

def test_get_service_and_service_factory_with_non_string_names():
    # Only string names should be allowed; test with int and None
    with pytest.raises(Exception):
        services_init.get_service(123)
    with pytest.raises(Exception):
        services_init.service_factory(None)
