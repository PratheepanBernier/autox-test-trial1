# source_hash: c0f4f32a02e6e1f2
# import_target: backend.src.models.extraction_schema
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import pytest
from pydantic import ValidationError
from backend.src.models.extraction_schema import (
    Location,
    CommodityItem,
    CarrierInfo,
    DriverInfo,
    RateInfo,
    ShipmentData,
    ExtractionResponse,
)

def test_location_happy_path_and_equivalence():
    data = {
        "name": "Warehouse A",
        "address": "123 Main St",
        "city": "Metropolis",
        "state": "NY",
        "zip_code": "10001",
        "country": "USA",
        "appointment_time": "2024-06-01T10:00:00Z",
        "po_number": "PO12345"
    }
    loc1 = Location(**data)
    loc2 = Location.parse_obj(data)
    assert loc1 == loc2
    assert loc1.name == "Warehouse A"
    assert loc1.address == "123 Main St"
    assert loc1.city == "Metropolis"
    assert loc1.state == "NY"
    assert loc1.zip_code == "10001"
    assert loc1.country == "USA"
    assert loc1.appointment_time == "2024-06-01T10:00:00Z"
    assert loc1.po_number == "PO12345"

def test_location_all_fields_none_equivalence():
    loc1 = Location()
    loc2 = Location.parse_obj({})
    assert loc1 == loc2
    for field in loc1.__fields__:
        assert getattr(loc1, field) is None

def test_commodity_item_happy_path_and_equivalence():
    data = {
        "commodity_name": "Steel",
        "weight": "56000.00 lbs",
        "quantity": "10000 units",
        "description": "Rolled steel coils"
    }
    item1 = CommodityItem(**data)
    item2 = CommodityItem.parse_obj(data)
    assert item1 == item2
    assert item1.commodity_name == "Steel"
    assert item1.weight == "56000.00 lbs"
    assert item1.quantity == "10000 units"
    assert item1.description == "Rolled steel coils"

def test_commodity_item_empty_and_none_equivalence():
    item1 = CommodityItem()
    item2 = CommodityItem.parse_obj({})
    assert item1 == item2
    for field in item1.__fields__:
        assert getattr(item1, field) is None

def test_carrier_info_happy_path_and_equivalence():
    data = {
        "carrier_name": "CarrierX",
        "mc_number": "MC123456",
        "phone": "555-1234",
        "email": "contact@carrierx.com"
    }
    c1 = CarrierInfo(**data)
    c2 = CarrierInfo.parse_obj(data)
    assert c1 == c2
    assert c1.carrier_name == "CarrierX"
    assert c1.mc_number == "MC123456"
    assert c1.phone == "555-1234"
    assert c1.email == "contact@carrierx.com"

def test_driver_info_happy_path_and_equivalence():
    data = {
        "driver_name": "John Doe",
        "cell_number": "555-5678",
        "truck_number": "TRK123",
        "trailer_number": "TRL456"
    }
    d1 = DriverInfo(**data)
    d2 = DriverInfo.parse_obj(data)
    assert d1 == d2
    assert d1.driver_name == "John Doe"
    assert d1.cell_number == "555-5678"
    assert d1.truck_number == "TRK123"
    assert d1.trailer_number == "TRL456"

def test_rate_info_happy_path_and_equivalence():
    data = {
        "total_rate": 1500.50,
        "currency": "USD",
        "rate_breakdown": {"base": 1200, "fuel": 300.5}
    }
    r1 = RateInfo(**data)
    r2 = RateInfo.parse_obj(data)
    assert r1 == r2
    assert r1.total_rate == 1500.50
    assert r1.currency == "USD"
    assert r1.rate_breakdown == {"base": 1200, "fuel": 300.5}

def test_rate_info_none_and_empty_breakdown_equivalence():
    r1 = RateInfo()
    r2 = RateInfo.parse_obj({})
    assert r1 == r2
    assert r1.total_rate is None
    assert r1.currency is None
    assert r1.rate_breakdown is None

def test_shipment_data_full_happy_path_and_equivalence():
    data = {
        "reference_id": "REF123",
        "load_id": "LOAD456",
        "po_number": "PO789",
        "shipper": "Shipper Inc.",
        "consignee": "Consignee LLC",
        "carrier": {
            "carrier_name": "CarrierX",
            "mc_number": "MC123456",
            "phone": "555-1234",
            "email": "contact@carrierx.com"
        },
        "driver": {
            "driver_name": "John Doe",
            "cell_number": "555-5678",
            "truck_number": "TRK123",
            "trailer_number": "TRL456"
        },
        "pickup": {
            "name": "Warehouse A",
            "address": "123 Main St",
            "city": "Metropolis",
            "state": "NY",
            "zip_code": "10001",
            "country": "USA",
            "appointment_time": "2024-06-01T10:00:00Z",
            "po_number": "PO12345"
        },
        "drop": {
            "name": "Warehouse B",
            "address": "456 Side St",
            "city": "Gotham",
            "state": "NJ",
            "zip_code": "07001",
            "country": "USA",
            "appointment_time": "2024-06-02T12:00:00Z",
            "po_number": "PO54321"
        },
        "shipping_date": "2024-06-01",
        "delivery_date": "2024-06-02",
        "created_on": "2024-05-31",
        "booking_date": "2024-05-30",
        "equipment_type": "Flatbed",
        "equipment_size": "53",
        "load_type": "FTL",
        "commodities": [
            {
                "commodity_name": "Steel",
                "weight": "56000.00 lbs",
                "quantity": "10000 units",
                "description": "Rolled steel coils"
            }
        ],
        "rate_info": {
            "total_rate": 1500.50,
            "currency": "USD",
            "rate_breakdown": {"base": 1200, "fuel": 300.5}
        },
        "special_instructions": "Handle with care",
        "shipper_instructions": "Call before arrival",
        "carrier_instructions": "No overnight parking",
        "dispatcher_name": "Alice",
        "dispatcher_phone": "555-9999",
        "dispatcher_email": "alice@dispatch.com",
        "additional_data": {"custom_field": "custom_value"}
    }
    s1 = ShipmentData(**data)
    s2 = ShipmentData.parse_obj(data)
    assert s1 == s2
    assert s1.reference_id == "REF123"
    assert s1.carrier.carrier_name == "CarrierX"
    assert s1.driver.driver_name == "John Doe"
    assert s1.pickup.name == "Warehouse A"
    assert s1.drop.name == "Warehouse B"
    assert s1.commodities[0].commodity_name == "Steel"
    assert s1.rate_info.total_rate == 1500.50
    assert s1.additional_data["custom_field"] == "custom_value"

def test_shipment_data_minimal_and_equivalence():
    s1 = ShipmentData()
    s2 = ShipmentData.parse_obj({})
    assert s1 == s2
    for field in s1.__fields__:
        assert getattr(s1, field) is None

def test_shipment_data_commodities_empty_list_equivalence():
    s1 = ShipmentData(commodities=[])
    s2 = ShipmentData.parse_obj({"commodities": []})
    assert s1 == s2
    assert s1.commodities == []

def test_shipment_data_additional_data_empty_dict_equivalence():
    s1 = ShipmentData(additional_data={})
    s2 = ShipmentData.parse_obj({"additional_data": {}})
    assert s1 == s2
    assert s1.additional_data == {}

def test_extraction_response_happy_path_and_equivalence():
    shipment_data = ShipmentData(reference_id="REF123")
    resp1 = ExtractionResponse(data=shipment_data, document_id="DOC999")
    resp2 = ExtractionResponse.parse_obj({"data": {"reference_id": "REF123"}, "document_id": "DOC999"})
    assert resp1 == resp2
    assert resp1.data.reference_id == "REF123"
    assert resp1.document_id == "DOC999"

def test_extraction_response_document_id_none_equivalence():
    shipment_data = ShipmentData(reference_id="REF123")
    resp1 = ExtractionResponse(data=shipment_data)
    resp2 = ExtractionResponse.parse_obj({"data": {"reference_id": "REF123"}})
    assert resp1 == resp2
    assert resp1.document_id is None

def test_extraction_response_invalid_data_type_error():
    with pytest.raises(ValidationError):
        ExtractionResponse(data="not a shipment data object")

def test_shipment_data_invalid_carrier_type_error():
    with pytest.raises(ValidationError):
        ShipmentData(carrier="not a carrier info object")

def test_shipment_data_invalid_commodities_type_error():
    with pytest.raises(ValidationError):
        ShipmentData(commodities="not a list")

def test_rate_info_invalid_total_rate_type_error():
    with pytest.raises(ValidationError):
        RateInfo(total_rate="not a float")

def test_location_boundary_zip_code_length():
    loc = Location(zip_code="12345")
    assert loc.zip_code == "12345"
    loc2 = Location(zip_code="")
    assert loc2.zip_code == ""

def test_commodity_item_boundary_weight_and_quantity():
    item = CommodityItem(weight="0 lbs", quantity="0 units")
    assert item.weight == "0 lbs"
    assert item.quantity == "0 units"

def test_shipment_data_partial_nested_objects_equivalence():
    data = {
        "carrier": {"carrier_name": "CarrierX"},
        "driver": {"driver_name": "Jane"},
        "pickup": {"name": "Warehouse A"},
        "drop": {"name": "Warehouse B"},
        "commodities": [{"commodity_name": "Steel"}],
        "rate_info": {"currency": "USD"}
    }
    s1 = ShipmentData(**data)
    s2 = ShipmentData.parse_obj(data)
    assert s1 == s2
    assert s1.carrier.carrier_name == "CarrierX"
    assert s1.driver.driver_name == "Jane"
    assert s1.pickup.name == "Warehouse A"
    assert s1.drop.name == "Warehouse B"
    assert s1.commodities[0].commodity_name == "Steel"
    assert s1.rate_info.currency == "USD"
