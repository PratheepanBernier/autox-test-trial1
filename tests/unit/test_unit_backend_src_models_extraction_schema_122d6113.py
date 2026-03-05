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
from backend.src.models.extraction_schema import (
    Location,
    CommodityItem,
    CarrierInfo,
    DriverInfo,
    RateInfo,
    ShipmentData,
    ExtractionResponse,
)
from pydantic import ValidationError


def test_location_happy_path():
    loc = Location(
        name="Warehouse A",
        address="123 Main St",
        city="Metropolis",
        state="NY",
        zip_code="12345",
        country="USA",
        appointment_time="2024-06-01T10:00:00Z",
        po_number="PO123456"
    )
    assert loc.name == "Warehouse A"
    assert loc.address == "123 Main St"
    assert loc.city == "Metropolis"
    assert loc.state == "NY"
    assert loc.zip_code == "12345"
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


def test_commodity_item_partial_fields():
    item = CommodityItem(commodity_name="Steel", weight="1000 lbs")
    assert item.commodity_name == "Steel"
    assert item.weight == "1000 lbs"
    assert item.quantity is None
    assert item.description is None


def test_carrier_info_full_and_partial():
    carrier = CarrierInfo(
        carrier_name="CarrierX",
        mc_number="MC123456",
        phone="555-1234",
        email="contact@carrierx.com"
    )
    assert carrier.carrier_name == "CarrierX"
    assert carrier.mc_number == "MC123456"
    assert carrier.phone == "555-1234"
    assert carrier.email == "contact@carrierx.com"

    carrier_partial = CarrierInfo(carrier_name="CarrierY")
    assert carrier_partial.carrier_name == "CarrierY"
    assert carrier_partial.mc_number is None
    assert carrier_partial.phone is None
    assert carrier_partial.email is None


def test_driver_info_all_fields():
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


def test_rate_info_with_breakdown():
    breakdown = {"base": 1000.0, "fuel": 150.0}
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
    for field in shipment.__fields__:
        assert getattr(shipment, field) is None


def test_shipment_data_full():
    pickup = Location(name="Origin", city="StartCity")
    drop = Location(name="Destination", city="EndCity")
    carrier = CarrierInfo(carrier_name="CarrierZ")
    driver = DriverInfo(driver_name="Jane Smith")
    commodities = [
        CommodityItem(commodity_name="Widgets", weight="500 lbs", quantity="100 units"),
        CommodityItem(commodity_name="Gadgets", weight="200 lbs", quantity="50 units"),
    ]
    rate_info = RateInfo(total_rate=2000.0, currency="USD")
    additional_data = {"custom_field": "custom_value"}

    shipment = ShipmentData(
        reference_id="REF123",
        load_id="LOAD456",
        po_number="PO789",
        shipper="Shipper Inc.",
        consignee="Consignee LLC",
        carrier=carrier,
        driver=driver,
        pickup=pickup,
        drop=drop,
        shipping_date="2024-06-01",
        delivery_date="2024-06-02",
        created_on="2024-05-31",
        booking_date="2024-05-30",
        equipment_type="Van",
        equipment_size="53",
        load_type="FTL",
        commodities=commodities,
        rate_info=rate_info,
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No weekend delivery",
        dispatcher_name="Dispatch Joe",
        dispatcher_phone="555-0000",
        dispatcher_email="dispatch@shipper.com",
        additional_data=additional_data
    )
    assert shipment.reference_id == "REF123"
    assert shipment.load_id == "LOAD456"
    assert shipment.po_number == "PO789"
    assert shipment.shipper == "Shipper Inc."
    assert shipment.consignee == "Consignee LLC"
    assert shipment.carrier == carrier
    assert shipment.driver == driver
    assert shipment.pickup == pickup
    assert shipment.drop == drop
    assert shipment.shipping_date == "2024-06-01"
    assert shipment.delivery_date == "2024-06-02"
    assert shipment.created_on == "2024-05-31"
    assert shipment.booking_date == "2024-05-30"
    assert shipment.equipment_type == "Van"
    assert shipment.equipment_size == "53"
    assert shipment.load_type == "FTL"
    assert shipment.commodities == commodities
    assert shipment.rate_info == rate_info
    assert shipment.special_instructions == "Handle with care"
    assert shipment.shipper_instructions == "Call before arrival"
    assert shipment.carrier_instructions == "No weekend delivery"
    assert shipment.dispatcher_name == "Dispatch Joe"
    assert shipment.dispatcher_phone == "555-0000"
    assert shipment.dispatcher_email == "dispatch@shipper.com"
    assert shipment.additional_data == additional_data


def test_shipment_data_commodities_empty_list():
    shipment = ShipmentData(commodities=[])
    assert shipment.commodities == []


def test_shipment_data_additional_data_empty_dict():
    shipment = ShipmentData(additional_data={})
    assert shipment.additional_data == {}


def test_extraction_response_happy_path():
    shipment = ShipmentData(reference_id="REF999")
    response = ExtractionResponse(data=shipment, document_id="DOC123")
    assert response.data == shipment
    assert response.document_id == "DOC123"


def test_extraction_response_document_id_none():
    shipment = ShipmentData(reference_id="REF888")
    response = ExtractionResponse(data=shipment)
    assert response.data == shipment
    assert response.document_id is None


def test_location_invalid_field_type_raises():
    with pytest.raises(ValidationError):
        Location(name=123)  # name should be str or None


def test_commodity_item_weight_and_quantity_formats():
    item = CommodityItem(weight="56000.00 lbs", quantity="10000 units")
    assert item.weight == "56000.00 lbs"
    assert item.quantity == "10000 units"


def test_rate_info_rate_breakdown_various_types():
    breakdown = {"base": 1000, "extra": "50", "fuel": None}
    rate = RateInfo(rate_breakdown=breakdown)
    assert rate.rate_breakdown == breakdown


def test_shipment_data_unknown_fields_go_to_additional_data():
    # Pydantic will raise error for unknown fields unless extra="allow" is set, which is not the case here.
    with pytest.raises(TypeError):
        ShipmentData(unknown_field="value")


def test_extraction_response_missing_data_raises():
    with pytest.raises(ValidationError):
        ExtractionResponse()


def test_shipment_data_boundary_empty_strings():
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
        equipment_size="",
        load_type="",
        special_instructions="",
        shipper_instructions="",
        carrier_instructions="",
        dispatcher_name="",
        dispatcher_phone="",
        dispatcher_email=""
    )
    assert shipment.reference_id == ""
    assert shipment.load_id == ""
    assert shipment.po_number == ""
    assert shipment.shipper == ""
    assert shipment.consignee == ""
    assert shipment.shipping_date == ""
    assert shipment.delivery_date == ""
    assert shipment.created_on == ""
    assert shipment.booking_date == ""
    assert shipment.equipment_type == ""
    assert shipment.equipment_size == ""
    assert shipment.load_type == ""
    assert shipment.special_instructions == ""
    assert shipment.shipper_instructions == ""
    assert shipment.carrier_instructions == ""
    assert shipment.dispatcher_name == ""
    assert shipment.dispatcher_phone == ""
    assert shipment.dispatcher_email == ""


def test_shipment_data_none_and_empty_mix():
    shipment = ShipmentData(
        reference_id=None,
        load_id="",
        po_number=None,
        shipper="",
        consignee=None
    )
    assert shipment.reference_id is None
    assert shipment.load_id == ""
    assert shipment.po_number is None
    assert shipment.shipper == ""
    assert shipment.consignee is None
