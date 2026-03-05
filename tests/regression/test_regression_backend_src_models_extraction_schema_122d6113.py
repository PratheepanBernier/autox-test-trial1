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

def test_commodity_item_minimal():
    item = CommodityItem()
    assert item.commodity_name is None
    assert item.weight is None
    assert item.quantity is None
    assert item.description is None

def test_carrier_info_happy_path():
    carrier = CarrierInfo(
        carrier_name="FastTrans",
        mc_number="MC123456",
        phone="555-1234",
        email="dispatch@fasttrans.com"
    )
    assert carrier.carrier_name == "FastTrans"
    assert carrier.mc_number == "MC123456"
    assert carrier.phone == "555-1234"
    assert carrier.email == "dispatch@fasttrans.com"

def test_carrier_info_empty():
    carrier = CarrierInfo()
    assert carrier.carrier_name is None
    assert carrier.mc_number is None
    assert carrier.phone is None
    assert carrier.email is None

def test_driver_info_happy_path():
    driver = DriverInfo(
        driver_name="John Doe",
        cell_number="555-6789",
        truck_number="TRK123",
        trailer_number="TRL456"
    )
    assert driver.driver_name == "John Doe"
    assert driver.cell_number == "555-6789"
    assert driver.truck_number == "TRK123"
    assert driver.trailer_number == "TRL456"

def test_driver_info_empty():
    driver = DriverInfo()
    assert driver.driver_name is None
    assert driver.cell_number is None
    assert driver.truck_number is None
    assert driver.trailer_number is None

def test_rate_info_happy_path():
    rate = RateInfo(
        total_rate=1500.75,
        currency="USD",
        rate_breakdown={"linehaul": 1200, "fuel": 300, "accessorial": 0.75}
    )
    assert rate.total_rate == 1500.75
    assert rate.currency == "USD"
    assert rate.rate_breakdown == {"linehaul": 1200, "fuel": 300, "accessorial": 0.75}

def test_rate_info_none_fields():
    rate = RateInfo()
    assert rate.total_rate is None
    assert rate.currency is None
    assert rate.rate_breakdown is None

def test_shipment_data_full_happy_path():
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
        equipment_type="Flatbed",
        equipment_size="53",
        load_type="FTL",
        commodities=[
            CommodityItem(commodity_name="Steel", weight="1000 lbs", quantity="10 units")
        ],
        rate_info=RateInfo(total_rate=2000.0, currency="USD"),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No weekend delivery",
        dispatcher_name="Dispatch John",
        dispatcher_phone="555-0000",
        dispatcher_email="dispatch@shipper.com",
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
    assert shipment.equipment_type == "Flatbed"
    assert shipment.equipment_size == "53"
    assert shipment.load_type == "FTL"
    assert len(shipment.commodities) == 1
    assert shipment.commodities[0].commodity_name == "Steel"
    assert shipment.rate_info.total_rate == 2000.0
    assert shipment.special_instructions == "Handle with care"
    assert shipment.shipper_instructions == "Call before arrival"
    assert shipment.carrier_instructions == "No weekend delivery"
    assert shipment.dispatcher_name == "Dispatch John"
    assert shipment.dispatcher_phone == "555-0000"
    assert shipment.dispatcher_email == "dispatch@shipper.com"
    assert shipment.additional_data == {"custom_field": "custom_value"}

def test_shipment_data_minimal():
    shipment = ShipmentData()
    for field in shipment.__fields__:
        assert getattr(shipment, field) is None

def test_shipment_data_commodities_empty_list():
    shipment = ShipmentData(commodities=[])
    assert shipment.commodities == []

def test_shipment_data_additional_data_empty_dict():
    shipment = ShipmentData(additional_data={})
    assert shipment.additional_data == {}

def test_shipment_data_invalid_commodities_type():
    with pytest.raises(ValidationError):
        ShipmentData(commodities="not_a_list")

def test_shipment_data_invalid_rate_info_type():
    with pytest.raises(ValidationError):
        ShipmentData(rate_info="not_a_dict")

def test_extraction_response_happy_path():
    shipment = ShipmentData(reference_id="REF999")
    response = ExtractionResponse(data=shipment, document_id="DOC123")
    assert response.data.reference_id == "REF999"
    assert response.document_id == "DOC123"

def test_extraction_response_document_id_none():
    shipment = ShipmentData(reference_id="REF888")
    response = ExtractionResponse(data=shipment)
    assert response.data.reference_id == "REF888"
    assert response.document_id is None

def test_extraction_response_invalid_data_type():
    with pytest.raises(ValidationError):
        ExtractionResponse(data="not_a_shipment_data")

def test_location_zip_code_boundary():
    loc = Location(zip_code="00000")
    assert loc.zip_code == "00000"

def test_rate_info_total_rate_boundary_zero():
    rate = RateInfo(total_rate=0.0)
    assert rate.total_rate == 0.0

def test_rate_info_total_rate_negative():
    rate = RateInfo(total_rate=-100.0)
    assert rate.total_rate == -100.0

def test_shipment_data_additional_data_various_types():
    shipment = ShipmentData(additional_data={"int": 1, "float": 2.5, "list": [1, 2], "dict": {"a": "b"}})
    assert shipment.additional_data["int"] == 1
    assert shipment.additional_data["float"] == 2.5
    assert shipment.additional_data["list"] == [1, 2]
    assert shipment.additional_data["dict"] == {"a": "b"}

def test_shipment_data_equivalent_paths_for_reconciliation():
    # Path 1: All fields explicitly set to None
    shipment1 = ShipmentData(
        reference_id=None,
        load_id=None,
        po_number=None,
        shipper=None,
        consignee=None,
        carrier=None,
        driver=None,
        pickup=None,
        drop=None,
        shipping_date=None,
        delivery_date=None,
        created_on=None,
        booking_date=None,
        equipment_type=None,
        equipment_size=None,
        load_type=None,
        commodities=None,
        rate_info=None,
        special_instructions=None,
        shipper_instructions=None,
        carrier_instructions=None,
        dispatcher_name=None,
        dispatcher_phone=None,
        dispatcher_email=None,
        additional_data=None
    )
    # Path 2: No fields set (defaults)
    shipment2 = ShipmentData()
    assert shipment1.dict() == shipment2.dict()
