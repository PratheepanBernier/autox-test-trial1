# source_hash: c0f4f32a02e6e1f2
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
        city="Springfield",
        state="IL",
        zip_code="62704",
        country="USA",
        appointment_time="2024-06-01T10:00:00Z",
        po_number="PO12345"
    )
    assert loc.name == "Warehouse A"
    assert loc.address == "123 Main St"
    assert loc.city == "Springfield"
    assert loc.state == "IL"
    assert loc.zip_code == "62704"
    assert loc.country == "USA"
    assert loc.appointment_time == "2024-06-01T10:00:00Z"
    assert loc.po_number == "PO12345"

def test_location_all_fields_none():
    loc = Location()
    for field in loc.__fields__:
        assert getattr(loc, field) is None

def test_commodity_item_partial_fields():
    item = CommodityItem(commodity_name="Steel", weight="1000 lbs")
    assert item.commodity_name == "Steel"
    assert item.weight == "1000 lbs"
    assert item.quantity is None
    assert item.description is None

def test_carrier_info_full_and_partial():
    carrier = CarrierInfo(
        carrier_name="Acme Logistics",
        mc_number="MC123456",
        phone="555-1234",
        email="contact@acme.com"
    )
    assert carrier.carrier_name == "Acme Logistics"
    assert carrier.mc_number == "MC123456"
    assert carrier.phone == "555-1234"
    assert carrier.email == "contact@acme.com"

    carrier_partial = CarrierInfo(carrier_name="Beta Transport")
    assert carrier_partial.carrier_name == "Beta Transport"
    assert carrier_partial.mc_number is None
    assert carrier_partial.phone is None
    assert carrier_partial.email is None

def test_driver_info_edge_cases():
    driver = DriverInfo(driver_name="", cell_number=None, truck_number="T123", trailer_number="")
    assert driver.driver_name == ""
    assert driver.cell_number is None
    assert driver.truck_number == "T123"
    assert driver.trailer_number == ""

def test_rate_info_happy_path_and_breakdown():
    breakdown = {"base": 1000.0, "fuel": 150.0}
    rate = RateInfo(total_rate=1150.0, currency="USD", rate_breakdown=breakdown)
    assert rate.total_rate == 1150.0
    assert rate.currency == "USD"
    assert rate.rate_breakdown == breakdown

def test_rate_info_none_and_boundary():
    rate = RateInfo()
    assert rate.total_rate is None
    assert rate.currency is None
    assert rate.rate_breakdown is None

    # Boundary: total_rate = 0
    rate_zero = RateInfo(total_rate=0.0)
    assert rate_zero.total_rate == 0.0

def test_shipment_data_minimal():
    shipment = ShipmentData()
    for field in shipment.__fields__:
        assert getattr(shipment, field) is None

def test_shipment_data_full_nested():
    shipment = ShipmentData(
        reference_id="REF123",
        load_id="LOAD456",
        po_number="PO789",
        shipper="Shipper Inc.",
        consignee="Consignee LLC",
        carrier=CarrierInfo(carrier_name="CarrierX", mc_number="MC999"),
        driver=DriverInfo(driver_name="John Doe", cell_number="555-6789"),
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
            CommodityItem(commodity_name="Widgets", weight="1000 lbs", quantity="100", description="Blue widgets"),
            CommodityItem(commodity_name="Gadgets", weight="500 lbs", quantity="50", description="Red gadgets")
        ],
        rate_info=RateInfo(total_rate=2000.0, currency="USD"),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No weekend delivery",
        dispatcher_name="Alice",
        dispatcher_phone="555-0000",
        dispatcher_email="alice@dispatch.com",
        additional_data={"custom_field": "custom_value"}
    )
    assert shipment.reference_id == "REF123"
    assert shipment.carrier.carrier_name == "CarrierX"
    assert shipment.commodities[1].commodity_name == "Gadgets"
    assert shipment.additional_data["custom_field"] == "custom_value"

def test_shipment_data_commodities_empty_list():
    shipment = ShipmentData(commodities=[])
    assert shipment.commodities == []

def test_shipment_data_additional_data_various_types():
    shipment = ShipmentData(additional_data={"int_field": 1, "list_field": [1, 2], "dict_field": {"a": "b"}})
    assert shipment.additional_data["int_field"] == 1
    assert shipment.additional_data["list_field"] == [1, 2]
    assert shipment.additional_data["dict_field"] == {"a": "b"}

def test_extraction_response_happy_path():
    shipment = ShipmentData(reference_id="R1")
    resp = ExtractionResponse(data=shipment, document_id="DOC123")
    assert resp.data.reference_id == "R1"
    assert resp.document_id == "DOC123"

def test_extraction_response_document_id_none():
    shipment = ShipmentData(reference_id="R2")
    resp = ExtractionResponse(data=shipment)
    assert resp.document_id is None

def test_invalid_rate_info_type_error():
    # total_rate expects float, not string
    with pytest.raises(ValidationError):
        RateInfo(total_rate="not_a_float")

def test_invalid_commodities_type_error():
    # commodities expects list of CommodityItem, not dict
    with pytest.raises(ValidationError):
        ShipmentData(commodities={"commodity_name": "Steel"})

def test_invalid_additional_data_type_error():
    # additional_data expects dict or None
    with pytest.raises(ValidationError):
        ShipmentData(additional_data="not_a_dict")

def test_shipment_data_equivalent_paths_for_po_number():
    # po_number at root and in pickup location
    shipment1 = ShipmentData(po_number="PO1", pickup=Location(po_number="PO1"))
    shipment2 = ShipmentData(po_number="PO1")
    assert shipment1.po_number == shipment2.po_number
    assert shipment1.pickup.po_number == "PO1"

def test_location_zip_code_boundary_conditions():
    # ZIP code as empty string, numeric string, and long string
    loc1 = Location(zip_code="")
    loc2 = Location(zip_code="12345")
    loc3 = Location(zip_code="12345678901234567890")
    assert loc1.zip_code == ""
    assert loc2.zip_code == "12345"
    assert loc3.zip_code == "12345678901234567890"

def test_rate_info_rate_breakdown_various_types():
    # Accepts any dict for rate_breakdown
    rate = RateInfo(rate_breakdown={"extra": "value", "amount": 123, "list": [1, 2, 3]})
    assert rate.rate_breakdown["extra"] == "value"
    assert rate.rate_breakdown["amount"] == 123
    assert rate.rate_breakdown["list"] == [1, 2, 3]
