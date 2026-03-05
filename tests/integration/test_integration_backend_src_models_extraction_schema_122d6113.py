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
        po_number="PO123456"
    )
    assert loc.name == "Warehouse A"
    assert loc.address == "123 Main St"
    assert loc.city == "Metropolis"
    assert loc.state == "NY"
    assert loc.zip_code == "10001"
    assert loc.country == "USA"
    assert loc.appointment_time == "2024-06-01T10:00:00Z"
    assert loc.po_number == "PO123456"

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

def test_commodity_item_edge_cases():
    item = CommodityItem(
        commodity_name="Widgets",
        weight="0 lbs",
        quantity="0 units",
        description=""
    )
    assert item.commodity_name == "Widgets"
    assert item.weight == "0 lbs"
    assert item.quantity == "0 units"
    assert item.description == ""

def test_carrier_info_partial_fields():
    carrier = CarrierInfo(
        carrier_name="CarrierX"
    )
    assert carrier.carrier_name == "CarrierX"
    assert carrier.mc_number is None
    assert carrier.phone is None
    assert carrier.email is None

def test_driver_info_all_fields():
    driver = DriverInfo(
        driver_name="John Doe",
        cell_number="555-1234",
        truck_number="TRK123",
        trailer_number="TRL456"
    )
    assert driver.driver_name == "John Doe"
    assert driver.cell_number == "555-1234"
    assert driver.truck_number == "TRK123"
    assert driver.trailer_number == "TRL456"

def test_rate_info_with_breakdown():
    breakdown = {"linehaul": 1000.0, "fuel": 150.0}
    rate = RateInfo(
        total_rate=1150.0,
        currency="USD",
        rate_breakdown=breakdown
    )
    assert rate.total_rate == 1150.0
    assert rate.currency == "USD"
    assert rate.rate_breakdown == breakdown

def test_rate_info_none_fields():
    rate = RateInfo()
    assert rate.total_rate is None
    assert rate.currency is None
    assert rate.rate_breakdown is None

def test_shipment_data_minimal():
    shipment = ShipmentData()
    assert shipment.reference_id is None
    assert shipment.load_id is None
    assert shipment.commodities is None
    assert shipment.carrier is None
    assert shipment.driver is None
    assert shipment.pickup is None
    assert shipment.drop is None
    assert shipment.rate_info is None
    assert shipment.additional_data is None

def test_shipment_data_full():
    shipment = ShipmentData(
        reference_id="REF123",
        load_id="LOAD456",
        po_number="PO789",
        shipper="Shipper Inc.",
        consignee="Consignee LLC",
        carrier=CarrierInfo(carrier_name="CarrierX", mc_number="MC123"),
        driver=DriverInfo(driver_name="Jane Smith", cell_number="555-6789"),
        pickup=Location(name="Origin", city="CityA"),
        drop=Location(name="Destination", city="CityB"),
        shipping_date="2024-06-01",
        delivery_date="2024-06-02",
        created_on="2024-05-31",
        booking_date="2024-05-30",
        equipment_type="Van",
        equipment_size="53",
        load_type="FTL",
        commodities=[
            CommodityItem(commodity_name="Widgets", weight="1000 lbs", quantity="100 units"),
            CommodityItem(commodity_name="Gadgets", weight="500 lbs", quantity="50 units"),
        ],
        rate_info=RateInfo(total_rate=2000.0, currency="USD"),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No weekend delivery",
        dispatcher_name="Dispatch Joe",
        dispatcher_phone="555-0000",
        dispatcher_email="dispatch@shipper.com",
        additional_data={"custom_field": "custom_value"}
    )
    assert shipment.reference_id == "REF123"
    assert shipment.carrier.carrier_name == "CarrierX"
    assert shipment.driver.driver_name == "Jane Smith"
    assert shipment.pickup.name == "Origin"
    assert shipment.drop.city == "CityB"
    assert shipment.commodities[0].commodity_name == "Widgets"
    assert shipment.rate_info.total_rate == 2000.0
    assert shipment.additional_data["custom_field"] == "custom_value"

def test_shipment_data_empty_commodities_and_additional_data():
    shipment = ShipmentData(
        commodities=[],
        additional_data={}
    )
    assert shipment.commodities == []
    assert shipment.additional_data == {}

def test_shipment_data_invalid_rate_info_type():
    with pytest.raises(ValidationError):
        ShipmentData(rate_info="not_a_rate_info_object")

def test_extraction_response_happy_path():
    shipment = ShipmentData(reference_id="REF999")
    response = ExtractionResponse(data=shipment, document_id="DOC123")
    assert response.data.reference_id == "REF999"
    assert response.document_id == "DOC123"

def test_extraction_response_missing_document_id():
    shipment = ShipmentData(reference_id="REF888")
    response = ExtractionResponse(data=shipment)
    assert response.data.reference_id == "REF888"
    assert response.document_id is None

def test_extraction_response_invalid_data_type():
    with pytest.raises(ValidationError):
        ExtractionResponse(data="not_a_shipment_data_object")

def test_location_invalid_field_type():
    with pytest.raises(ValidationError):
        Location(name=12345)

def test_commodity_item_missing_all_fields():
    item = CommodityItem()
    assert item.commodity_name is None
    assert item.weight is None
    assert item.quantity is None
    assert item.description is None

def test_rate_info_boundary_total_rate_zero():
    rate = RateInfo(total_rate=0.0, currency="USD")
    assert rate.total_rate == 0.0
    assert rate.currency == "USD"

def test_shipment_data_additional_data_various_types():
    shipment = ShipmentData(
        additional_data={
            "int_field": 1,
            "float_field": 2.5,
            "bool_field": True,
            "list_field": [1, 2, 3],
            "dict_field": {"nested": "value"}
        }
    )
    assert shipment.additional_data["int_field"] == 1
    assert shipment.additional_data["float_field"] == 2.5
    assert shipment.additional_data["bool_field"] is True
    assert shipment.additional_data["list_field"] == [1, 2, 3]
    assert shipment.additional_data["dict_field"] == {"nested": "value"}

def test_shipment_data_equivalent_paths_for_reference_id_and_load_id():
    shipment1 = ShipmentData(reference_id="REF1", load_id="LOAD1")
    shipment2 = ShipmentData(reference_id="REF1", load_id="LOAD1")
    assert shipment1.reference_id == shipment2.reference_id
    assert shipment1.load_id == shipment2.load_id
    assert shipment1.dict() == shipment2.dict()
