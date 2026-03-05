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

def test_location_happy_path():
    loc = Location(
        name="Warehouse A",
        address="123 Main St",
        city="Metropolis",
        state="NY",
        zip_code="10001",
        country="USA",
        appointment_time="2024-06-01T10:00:00Z",
        po_number="PO12345"
    )
    assert loc.name == "Warehouse A"
    assert loc.address == "123 Main St"
    assert loc.city == "Metropolis"
    assert loc.state == "NY"
    assert loc.zip_code == "10001"
    assert loc.country == "USA"
    assert loc.appointment_time == "2024-06-01T10:00:00Z"
    assert loc.po_number == "PO12345"

def test_location_all_fields_none():
    loc = Location()
    assert loc.name is None
    assert loc.address is None
    assert loc.city is None
    assert loc.state is None
    assert loc.zip_code is None
    assert loc.country is None
    assert loc.appointment_time is None
    assert loc.po_number is None

def test_commodity_item_happy_path():
    item = CommodityItem(
        commodity_name="Steel",
        weight="56000.00 lbs",
        quantity="10000 units",
        description="Rolled steel coils"
    )
    assert item.commodity_name == "Steel"
    assert item.weight == "56000.00 lbs"
    assert item.quantity == "10000 units"
    assert item.description == "Rolled steel coils"

def test_commodity_item_partial_fields():
    item = CommodityItem(commodity_name="Copper")
    assert item.commodity_name == "Copper"
    assert item.weight is None
    assert item.quantity is None
    assert item.description is None

def test_carrier_info_happy_path():
    carrier = CarrierInfo(
        carrier_name="FastTrans LLC",
        mc_number="MC123456",
        phone="555-1234",
        email="dispatch@fasttrans.com"
    )
    assert carrier.carrier_name == "FastTrans LLC"
    assert carrier.mc_number == "MC123456"
    assert carrier.phone == "555-1234"
    assert carrier.email == "dispatch@fasttrans.com"

def test_driver_info_happy_path():
    driver = DriverInfo(
        driver_name="John Doe",
        cell_number="555-5678",
        truck_number="TRK123",
        trailer_number="TRL456"
    )
    assert driver.driver_name == "John Doe"
    assert driver.cell_number == "555-5678"
    assert driver.truck_number == "TRK123"
    assert driver.trailer_number == "TRL456"

def test_rate_info_happy_path():
    rate = RateInfo(
        total_rate=2500.75,
        currency="USD",
        rate_breakdown={"base": 2000, "fuel": 500.75}
    )
    assert rate.total_rate == 2500.75
    assert rate.currency == "USD"
    assert rate.rate_breakdown == {"base": 2000, "fuel": 500.75}

def test_rate_info_none_fields():
    rate = RateInfo()
    assert rate.total_rate is None
    assert rate.currency is None
    assert rate.rate_breakdown is None

def test_shipment_data_minimal():
    shipment = ShipmentData()
    for field in shipment.__fields__:
        assert getattr(shipment, field) is None

def test_shipment_data_full():
    shipment = ShipmentData(
        reference_id="REF123",
        load_id="LOAD456",
        po_number="PO789",
        shipper="Shipper Inc.",
        consignee="Consignee LLC",
        carrier=CarrierInfo(carrier_name="CarrierX"),
        driver=DriverInfo(driver_name="Jane Smith"),
        pickup=Location(name="Origin"),
        drop=Location(name="Destination"),
        shipping_date="2024-06-01",
        delivery_date="2024-06-02",
        created_on="2024-05-31",
        booking_date="2024-05-30",
        equipment_type="Van",
        equipment_size="53",
        load_type="FTL",
        commodities=[CommodityItem(commodity_name="Widgets", weight="1000 lbs")],
        rate_info=RateInfo(total_rate=1000.0, currency="USD"),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No overnight parking",
        dispatcher_name="Dispatch Joe",
        dispatcher_phone="555-0000",
        dispatcher_email="joe@dispatch.com",
        additional_data={"custom_field": "custom_value"}
    )
    assert shipment.reference_id == "REF123"
    assert shipment.load_id == "LOAD456"
    assert shipment.po_number == "PO789"
    assert shipment.shipper == "Shipper Inc."
    assert shipment.consignee == "Consignee LLC"
    assert shipment.carrier.carrier_name == "CarrierX"
    assert shipment.driver.driver_name == "Jane Smith"
    assert shipment.pickup.name == "Origin"
    assert shipment.drop.name == "Destination"
    assert shipment.shipping_date == "2024-06-01"
    assert shipment.delivery_date == "2024-06-02"
    assert shipment.created_on == "2024-05-31"
    assert shipment.booking_date == "2024-05-30"
    assert shipment.equipment_type == "Van"
    assert shipment.equipment_size == "53"
    assert shipment.load_type == "FTL"
    assert shipment.commodities[0].commodity_name == "Widgets"
    assert shipment.commodities[0].weight == "1000 lbs"
    assert shipment.rate_info.total_rate == 1000.0
    assert shipment.rate_info.currency == "USD"
    assert shipment.special_instructions == "Handle with care"
    assert shipment.shipper_instructions == "Call before arrival"
    assert shipment.carrier_instructions == "No overnight parking"
    assert shipment.dispatcher_name == "Dispatch Joe"
    assert shipment.dispatcher_phone == "555-0000"
    assert shipment.dispatcher_email == "joe@dispatch.com"
    assert shipment.additional_data == {"custom_field": "custom_value"}

def test_shipment_data_commodities_empty_list():
    shipment = ShipmentData(commodities=[])
    assert shipment.commodities == []

def test_shipment_data_additional_data_empty_dict():
    shipment = ShipmentData(additional_data={})
    assert shipment.additional_data == {}

def test_shipment_data_invalid_rate_info_type():
    with pytest.raises(ValidationError):
        ShipmentData(rate_info="not_a_rate_info_object")

def test_shipment_data_invalid_commodities_type():
    with pytest.raises(ValidationError):
        ShipmentData(commodities="not_a_list")

def test_extraction_response_happy_path():
    shipment = ShipmentData(reference_id="REF999")
    resp = ExtractionResponse(data=shipment, document_id="DOC123")
    assert resp.data.reference_id == "REF999"
    assert resp.document_id == "DOC123"

def test_extraction_response_document_id_none():
    shipment = ShipmentData(reference_id="REF888")
    resp = ExtractionResponse(data=shipment)
    assert resp.data.reference_id == "REF888"
    assert resp.document_id is None

def test_extraction_response_missing_data_raises():
    with pytest.raises(ValidationError):
        ExtractionResponse()

def test_location_field_boundaries():
    # Test empty string and long string
    long_str = "a" * 1000
    loc = Location(name="", address=long_str)
    assert loc.name == ""
    assert loc.address == long_str

def test_rate_info_rate_breakdown_various_types():
    # Accepts any dict
    rate = RateInfo(rate_breakdown={"misc": [1, 2, 3], "notes": {"a": 1}})
    assert rate.rate_breakdown == {"misc": [1, 2, 3], "notes": {"a": 1}}

def test_shipment_data_equivalent_paths_for_carrier_info():
    # Reconciliation: direct dict vs. CarrierInfo object
    carrier_dict = {"carrier_name": "CarrierY", "mc_number": "MC999"}
    shipment1 = ShipmentData(carrier=CarrierInfo(**carrier_dict))
    shipment2 = ShipmentData(carrier=carrier_dict)
    assert shipment1.carrier == shipment2.carrier

def test_shipment_data_equivalent_paths_for_commodities():
    # Reconciliation: list of dicts vs. list of objects
    commodities_dicts = [{"commodity_name": "A"}, {"commodity_name": "B"}]
    shipment1 = ShipmentData(commodities=[CommodityItem(**d) for d in commodities_dicts])
    shipment2 = ShipmentData(commodities=commodities_dicts)
    assert shipment1.commodities == shipment2.commodities

def test_shipment_data_additional_data_arbitrary_types():
    shipment = ShipmentData(additional_data={"int": 1, "list": [1,2], "dict": {"a": "b"}})
    assert shipment.additional_data["int"] == 1
    assert shipment.additional_data["list"] == [1,2]
    assert shipment.additional_data["dict"] == {"a": "b"}

def test_shipment_data_invalid_field_types():
    with pytest.raises(ValidationError):
        ShipmentData(reference_id=123)  # Should be str or None

def test_rate_info_invalid_total_rate_type():
    with pytest.raises(ValidationError):
        RateInfo(total_rate="not_a_float")

def test_extraction_response_invalid_data_type():
    with pytest.raises(ValidationError):
        ExtractionResponse(data="not_a_shipment_data_object")
