import pytest
from pydantic import ValidationError
from backend.src.models import extraction_schema

def test_location_happy_path():
    loc = extraction_schema.Location(
        name="Warehouse A",
        address="123 Main St",
        city="Springfield",
        state="IL",
        zip_code="62701",
        country="USA",
        appointment_time="2024-06-01T10:00:00Z",
        po_number="PO123456"
    )
    assert loc.name == "Warehouse A"
    assert loc.address == "123 Main St"
    assert loc.city == "Springfield"
    assert loc.state == "IL"
    assert loc.zip_code == "62701"
    assert loc.country == "USA"
    assert loc.appointment_time == "2024-06-01T10:00:00Z"
    assert loc.po_number == "PO123456"

def test_location_all_fields_none():
    loc = extraction_schema.Location()
    for field in loc.__fields__:
        assert getattr(loc, field) is None

def test_commodity_item_partial_fields():
    item = extraction_schema.CommodityItem(
        commodity_name="Steel Beams",
        weight="56000.00 lbs"
    )
    assert item.commodity_name == "Steel Beams"
    assert item.weight == "56000.00 lbs"
    assert item.quantity is None
    assert item.description is None

def test_carrier_info_full_and_partial():
    carrier = extraction_schema.CarrierInfo(
        carrier_name="Acme Logistics",
        mc_number="MC123456",
        phone="555-1234",
        email="contact@acme.com"
    )
    assert carrier.carrier_name == "Acme Logistics"
    assert carrier.mc_number == "MC123456"
    assert carrier.phone == "555-1234"
    assert carrier.email == "contact@acme.com"

    carrier_partial = extraction_schema.CarrierInfo(carrier_name="Acme Logistics")
    assert carrier_partial.carrier_name == "Acme Logistics"
    assert carrier_partial.mc_number is None
    assert carrier_partial.phone is None
    assert carrier_partial.email is None

def test_driver_info_empty_and_full():
    driver = extraction_schema.DriverInfo()
    for field in driver.__fields__:
        assert getattr(driver, field) is None

    driver_full = extraction_schema.DriverInfo(
        driver_name="John Doe",
        cell_number="555-5678",
        truck_number="TRK123",
        trailer_number="TRL456"
    )
    assert driver_full.driver_name == "John Doe"
    assert driver_full.cell_number == "555-5678"
    assert driver_full.truck_number == "TRK123"
    assert driver_full.trailer_number == "TRL456"

def test_rate_info_with_breakdown():
    breakdown = {"base": 1000.0, "fuel": 150.0}
    rate = extraction_schema.RateInfo(
        total_rate=1150.0,
        currency="USD",
        rate_breakdown=breakdown
    )
    assert rate.total_rate == 1150.0
    assert rate.currency == "USD"
    assert rate.rate_breakdown == breakdown

def test_rate_info_none_fields():
    rate = extraction_schema.RateInfo()
    assert rate.total_rate is None
    assert rate.currency is None
    assert rate.rate_breakdown is None

def test_shipment_data_minimal():
    shipment = extraction_schema.ShipmentData()
    for field in shipment.__fields__:
        assert getattr(shipment, field) is None

def test_shipment_data_full():
    shipment = extraction_schema.ShipmentData(
        reference_id="REF123",
        load_id="LOAD456",
        po_number="PO789",
        shipper="Shipper Inc.",
        consignee="Consignee LLC",
        carrier=extraction_schema.CarrierInfo(carrier_name="CarrierX"),
        driver=extraction_schema.DriverInfo(driver_name="Jane Smith"),
        pickup=extraction_schema.Location(name="Origin"),
        drop=extraction_schema.Location(name="Destination"),
        shipping_date="2024-06-01",
        delivery_date="2024-06-02",
        created_on="2024-05-31",
        booking_date="2024-05-30",
        equipment_type="Van",
        equipment_size="53",
        load_type="FTL",
        commodities=[
            extraction_schema.CommodityItem(commodity_name="Widgets", weight="1000 lbs", quantity="100 units")
        ],
        rate_info=extraction_schema.RateInfo(total_rate=2000.0, currency="USD"),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No overnight parking",
        dispatcher_name="Dispatch Joe",
        dispatcher_phone="555-0000",
        dispatcher_email="dispatch@shipper.com",
        additional_data={"custom_field": "custom_value"}
    )
    assert shipment.reference_id == "REF123"
    assert shipment.carrier.carrier_name == "CarrierX"
    assert shipment.driver.driver_name == "Jane Smith"
    assert shipment.pickup.name == "Origin"
    assert shipment.drop.name == "Destination"
    assert shipment.commodities[0].commodity_name == "Widgets"
    assert shipment.rate_info.total_rate == 2000.0
    assert shipment.additional_data["custom_field"] == "custom_value"

def test_shipment_data_commodities_empty_list():
    shipment = extraction_schema.ShipmentData(commodities=[])
    assert shipment.commodities == []

def test_shipment_data_additional_data_edge_cases():
    shipment = extraction_schema.ShipmentData(additional_data={})
    assert shipment.additional_data == {}

    shipment2 = extraction_schema.ShipmentData(additional_data={"foo": None, "bar": 123})
    assert shipment2.additional_data["foo"] is None
    assert shipment2.additional_data["bar"] == 123

def test_extraction_response_happy_path():
    shipment = extraction_schema.ShipmentData(reference_id="REF999")
    resp = extraction_schema.ExtractionResponse(data=shipment, document_id="DOC123")
    assert resp.data.reference_id == "REF999"
    assert resp.document_id == "DOC123"

def test_extraction_response_document_id_none():
    shipment = extraction_schema.ShipmentData(reference_id="REF888")
    resp = extraction_schema.ExtractionResponse(data=shipment)
    assert resp.data.reference_id == "REF888"
    assert resp.document_id is None

def test_invalid_rate_info_total_rate_type():
    # total_rate expects float or None
    with pytest.raises(ValidationError):
        extraction_schema.RateInfo(total_rate="not_a_float")

def test_invalid_commodities_type():
    # commodities expects a list of CommodityItem or None
    with pytest.raises(ValidationError):
        extraction_schema.ShipmentData(commodities="not_a_list")

def test_invalid_additional_data_type():
    # additional_data expects dict or None
    with pytest.raises(ValidationError):
        extraction_schema.ShipmentData(additional_data="not_a_dict")

def test_location_zip_code_boundary():
    # Accepts any string, but test a short and long zip code
    loc_short = extraction_schema.Location(zip_code="1")
    assert loc_short.zip_code == "1"
    loc_long = extraction_schema.Location(zip_code="12345678901234567890")
    assert loc_long.zip_code == "12345678901234567890"

def test_commodity_item_empty_string_fields():
    item = extraction_schema.CommodityItem(
        commodity_name="",
        weight="",
        quantity="",
        description=""
    )
    assert item.commodity_name == ""
    assert item.weight == ""
    assert item.quantity == ""
    assert item.description == ""

def test_shipment_data_equivalent_paths_for_po_number():
    # po_number at root and in pickup location
    shipment1 = extraction_schema.ShipmentData(po_number="PO111")
    shipment2 = extraction_schema.ShipmentData(pickup=extraction_schema.Location(po_number="PO111"))
    # They are not the same field, but both should store the value correctly
    assert shipment1.po_number == "PO111"
    assert shipment2.pickup.po_number == "PO111"

def test_rate_info_rate_breakdown_various_types():
    # Accepts any dict for rate_breakdown
    breakdown = {"base": 1000, "extra": "fee", "nested": {"sub": 1}}
    rate = extraction_schema.RateInfo(rate_breakdown=breakdown)
    assert rate.rate_breakdown == breakdown

def test_shipment_data_missing_nested_models():
    # Should allow None for nested models
    shipment = extraction_schema.ShipmentData(carrier=None, driver=None, pickup=None, drop=None)
    assert shipment.carrier is None
    assert shipment.driver is None
    assert shipment.pickup is None
    assert shipment.drop is None
