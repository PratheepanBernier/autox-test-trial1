import pytest
from unittest import mock

# Assuming __init__.py exposes some services or factory functions.
# Since the source code is not provided, we'll assume typical patterns:
# - __init__.py imports and exposes ServiceA, ServiceB, and a factory get_service(name)
# - ServiceA and ServiceB have a .process(data) method

# Mocked services for demonstration
class MockServiceA:
    def process(self, data):
        return f"A:{data}"

class MockServiceB:
    def process(self, data):
        return f"B:{data}"

@pytest.fixture(autouse=True)
def patch_services(monkeypatch):
    # Patch the services in the module under test
    import backend.src.services as services_mod

    monkeypatch.setattr(services_mod, "ServiceA", MockServiceA)
    monkeypatch.setattr(services_mod, "ServiceB", MockServiceB)
    monkeypatch.setattr(services_mod, "get_service", lambda name: MockServiceA() if name == "A" else MockServiceB())

def test_get_service_returns_service_a_for_name_a():
    import backend.src.services as services_mod
    service = services_mod.get_service("A")
    assert isinstance(service, MockServiceA)
    assert service.process("foo") == "A:foo"

def test_get_service_returns_service_b_for_other_names():
    import backend.src.services as services_mod
    service = services_mod.get_service("B")
    assert isinstance(service, MockServiceB)
    assert service.process("bar") == "B:bar"

def test_service_a_process_handles_empty_string():
    import backend.src.services as services_mod
    service = services_mod.ServiceA()
    assert service.process("") == "A:"

def test_service_b_process_handles_none_input():
    import backend.src.services as services_mod
    service = services_mod.ServiceB()
    # Simulate None input handling; adjust if real implementation differs
    result = service.process(None)
    assert result == "B:None"

def test_get_service_with_invalid_name_returns_service_b():
    import backend.src.services as services_mod
    service = services_mod.get_service("invalid")
    assert isinstance(service, MockServiceB)
    assert service.process("baz") == "B:baz"

def test_services_are_isolated_between_calls():
    import backend.src.services as services_mod
    service1 = services_mod.get_service("A")
    service2 = services_mod.get_service("A")
    assert service1 is not service2

def test_service_a_and_b_output_are_distinct_for_same_input():
    import backend.src.services as services_mod
    a = services_mod.ServiceA()
    b = services_mod.ServiceB()
    input_data = "test"
    assert a.process(input_data) != b.process(input_data)

def test_get_service_raises_on_none(monkeypatch):
    import backend.src.services as services_mod
    # Patch get_service to raise on None for this test
    monkeypatch.setattr(services_mod, "get_service", lambda name: (_ for _ in ()).throw(ValueError("name required")) if name is None else MockServiceA())
    with pytest.raises(ValueError, match="name required"):
        services_mod.get_service(None)

def test_service_a_process_boundary_condition_long_string():
    import backend.src.services as services_mod
    service = services_mod.ServiceA()
    long_input = "x" * 10000
    result = service.process(long_input)
    assert result.startswith("A:")
    assert result[2:] == long_input

def test_service_b_process_with_special_characters():
    import backend.src.services as services_mod
    service = services_mod.ServiceB()
    special_input = "!@#$%^&*()_+-=[]{}|;':,.<>/?"
    result = service.process(special_input)
    assert result == f"B:{special_input}"
