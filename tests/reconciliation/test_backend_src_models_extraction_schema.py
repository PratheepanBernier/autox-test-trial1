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

def test_location_model_happy_path_and_equivalence():
    data = {
        "name": "Warehouse A",
        "address": "123 Main St",
        "city": "Springfield",
        "state": "IL",
        "zip_code": "62701",
        "country": "USA",
        "appointment_time": "2024-06-01T10:00:00Z",
        "po_number": "PO12345"
    }
    loc1 = Location(**data)
    loc2 = Location.parse_obj(data)
    assert loc1 == loc2
    assert loc1.name == "Warehouse A"
    assert loc1.address == "123 Main St"
    assert loc1.city == "Springfield"
    assert loc1.state == "IL"
    assert loc1.zip_code == "62701"
    assert loc1.country == "USA"
    assert loc1.appointment_time == "2024-06-01T10:00:00Z"
    assert loc1.po_number == "PO12345"

def test_location_model_edge_cases_and_defaults():
    loc = Location()
    assert loc.name is None
    assert loc.address is None
    assert loc.city is None
    assert loc.state is None
    assert loc.zip_code is None
    assert loc.country is None
    assert loc.appointment_time is None
    assert loc.po_number is None

def test_commodity_item_equivalent_paths():
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

def test_commodity_item_missing_fields():
    item = CommodityItem()
    assert item.commodity_name is None
    assert item.weight is None
    assert item.quantity is None
    assert item.description is None

def test_carrier_info_and_driver_info_equivalence():
    carrier_data = {
        "carrier_name": "Acme Trucking",
        "mc_number": "MC123456",
        "phone": "555-1234",
        "email": "dispatch@acmetrucking.com"
    }
    driver_data = {
        "driver_name": "John Doe",
        "cell_number": "555-5678",
        "truck_number": "TX1234",
        "trailer_number": "TR5678"
    }
    carrier1 = CarrierInfo(**carrier_data)
    carrier2 = CarrierInfo.parse_obj(carrier_data)
    driver1 = DriverInfo(**driver_data)
    driver2 = DriverInfo.parse_obj(driver_data)
    assert carrier1 == carrier2
    assert driver1 == driver2

def test_rate_info_equivalence_and_breakdown():
    breakdown = {"linehaul": 1200.0, "fuel": 200.0}
    data = {
        "total_rate": 1400.0,
        "currency": "USD",
        "rate_breakdown": breakdown
    }
    rate1 = RateInfo(**data)
    rate2 = RateInfo.parse_obj(data)
    assert rate1 == rate2
    assert rate1.total_rate == 1400.0
    assert rate1.currency == "USD"
    assert rate1.rate_breakdown == breakdown

def test_rate_info_missing_optional_fields():
    rate = RateInfo()
    assert rate.total_rate is None
    assert rate.currency is None
    assert rate.rate_breakdown is None

def test_shipment_data_full_and_equivalent_paths():
    shipment_dict = {
        "reference_id": "REF123",
        "load_id": "LOAD456",
        "po_number": "PO789",
        "shipper": "Shipper Inc.",
        "consignee": "Consignee LLC",
        "carrier": {
            "carrier_name": "Acme Trucking",
            "mc_number": "MC123456",
            "phone": "555-1234",
            "email": "dispatch@acmetrucking.com"
        },
        "driver": {
            "driver_name": "John Doe",
            "cell_number": "555-5678",
            "truck_number": "TX1234",
            "trailer_number": "TR5678"
        },
        "pickup": {
            "name": "Warehouse A",
            "address": "123 Main St",
            "city": "Springfield",
            "state": "IL",
            "zip_code": "62701",
            "country": "USA",
            "appointment_time": "2024-06-01T10:00:00Z",
            "po_number": "PO12345"
        },
        "drop": {
            "name": "Warehouse B",
            "address": "456 Market St",
            "city": "Chicago",
            "state": "IL",
            "zip_code": "60601",
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
            "total_rate": 1400.0,
            "currency": "USD",
            "rate_breakdown": {"linehaul": 1200.0, "fuel": 200.0}
        },
        "special_instructions": "Call before arrival",
        "shipper_instructions": "Load from dock 3",
        "carrier_instructions": "Seal trailer after loading",
        "dispatcher_name": "Jane Smith",
        "dispatcher_phone": "555-9999",
        "dispatcher_email": "jane@shipperinc.com",
        "additional_data": {"custom_field": "custom_value"}
    }
    shipment1 = ShipmentData(**shipment_dict)
    shipment2 = ShipmentData.parse_obj(shipment_dict)
    assert shipment1 == shipment2
    assert shipment1.reference_id == "REF123"
    assert shipment1.carrier.carrier_name == "Acme Trucking"
    assert shipment1.driver.driver_name == "John Doe"
    assert shipment1.pickup.name == "Warehouse A"
    assert shipment1.drop.name == "Warehouse B"
    assert shipment1.commodities[0].commodity_name == "Steel"
    assert shipment1.rate_info.total_rate == 1400.0
    assert shipment1.additional_data["custom_field"] == "custom_value"

def test_shipment_data_minimal_and_missing_fields():
    shipment = ShipmentData()
    assert shipment.reference_id is None
    assert shipment.carrier is None
    assert shipment.commodities is None
    assert shipment.additional_data is None

def test_shipment_data_commodities_empty_list():
    shipment = ShipmentData(commodities=[])
    assert shipment.commodities == []

def test_shipment_data_additional_data_empty_dict():
    shipment = ShipmentData(additional_data={})
    assert shipment.additional_data == {}

def test_extraction_response_equivalent_paths():
    shipment = ShipmentData(reference_id="REF123")
    resp1 = ExtractionResponse(data=shipment, document_id="DOC789")
    resp2 = ExtractionResponse.parse_obj({"data": shipment.dict(), "document_id": "DOC789"})
    assert resp1.data.reference_id == "REF123"
    assert resp2.data.reference_id == "REF123"
    assert resp1.document_id == "DOC789"
    assert resp2.document_id == "DOC789"
    # Reconciliation: resp1 and resp2 should be equivalent
    assert resp1 == resp2

def test_extraction_response_missing_document_id():
    shipment = ShipmentData(reference_id="REF123")
    resp = ExtractionResponse(data=shipment)
    assert resp.document_id is None

def test_invalid_rate_info_type_error():
    # total_rate expects float, not string
    with pytest.raises(ValidationError):
        RateInfo(total_rate="not_a_float")

def test_invalid_commodities_type_error():
    # commodities expects a list of CommodityItem, not a dict
    with pytest.raises(ValidationError):
        ShipmentData(commodities={"commodity_name": "Steel"})

def test_invalid_carrier_type_error():
    # carrier expects CarrierInfo, not a string
    with pytest.raises(ValidationError):
        ShipmentData(carrier="Acme Trucking")

def test_invalid_additional_data_type_error():
    # additional_data expects dict, not a list
    with pytest.raises(ValidationError):
        ShipmentData(additional_data=["not", "a", "dict"])

def test_boundary_condition_empty_strings_and_zero_values():
    shipment = ShipmentData(
        reference_id="",
        load_id="",
        po_number="",
        shipper="",
        consignee="",
        shipping_date="",
        delivery_date="",
        created_on="",
        booking_date="",
        equipment_type="",
        equipment_size="0",
        load_type="",
        dispatcher_name="",
        dispatcher_phone="",
        dispatcher_email="",
        special_instructions="",
        shipper_instructions="",
        carrier_instructions="",
        additional_data={}
    )
    assert shipment.reference_id == ""
    assert shipment.equipment_size == "0"
    assert shipment.additional_data == {}

def test_shipment_data_with_none_and_empty_lists():
    shipment = ShipmentData(commodities=None)
    assert shipment.commodities is None
    shipment2 = ShipmentData(commodities=[])
    assert shipment2.commodities == []

def test_location_and_commodityitem_equivalence_with_none():
    loc1 = Location()
    loc2 = Location.parse_obj({})
    assert loc1 == loc2
    item1 = CommodityItem()
    item2 = CommodityItem.parse_obj({})
    assert item1 == item2
