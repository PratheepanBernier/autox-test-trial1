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
        zip_code="62701",
        country="USA",
        appointment_time="2024-06-01T10:00:00Z",
        po_number="PO12345"
    )
    assert loc.name == "Warehouse A"
    assert loc.address == "123 Main St"
    assert loc.city == "Springfield"
    assert loc.state == "IL"
    assert loc.zip_code == "62701"
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

def test_carrier_info_invalid_field_raises():
    with pytest.raises(TypeError):
        CarrierInfo(carrier_name="CarrierX", unknown_field="foo")

def test_driver_info_empty():
    driver = DriverInfo()
    assert driver.driver_name is None
    assert driver.cell_number is None
    assert driver.truck_number is None
    assert driver.trailer_number is None

def test_rate_info_with_breakdown():
    breakdown = {"base": 1000.0, "fuel": 150.0}
    rate = RateInfo(total_rate=1150.0, currency="USD", rate_breakdown=breakdown)
    assert rate.total_rate == 1150.0
    assert rate.currency == "USD"
    assert rate.rate_breakdown == breakdown

def test_rate_info_invalid_total_rate_type():
    with pytest.raises(ValidationError):
        RateInfo(total_rate="not_a_float")

def test_shipment_data_full_happy_path():
    shipment = ShipmentData(
        reference_id="REF123",
        load_id="LOAD456",
        po_number="PO789",
        shipper="Shipper Inc.",
        consignee="Consignee LLC",
        carrier=CarrierInfo(carrier_name="CarrierX", mc_number="MC123", phone="555-1234", email="carrier@example.com"),
        driver=DriverInfo(driver_name="John Doe", cell_number="555-5678", truck_number="TRK123", trailer_number="TRL456"),
        pickup=Location(name="Warehouse A", address="123 Main St"),
        drop=Location(name="Warehouse B", address="456 Elm St"),
        shipping_date="2024-06-01",
        delivery_date="2024-06-02",
        created_on="2024-05-31",
        booking_date="2024-05-30",
        equipment_type="Van",
        equipment_size="53",
        load_type="FTL",
        commodities=[
            CommodityItem(commodity_name="Steel", weight="1000 lbs", quantity="10 units", description="Steel rods"),
            CommodityItem(commodity_name="Aluminum", weight="500 lbs", quantity="5 units", description="Aluminum sheets"),
        ],
        rate_info=RateInfo(total_rate=2000.0, currency="USD", rate_breakdown={"base": 1800.0, "fuel": 200.0}),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No overnight parking",
        dispatcher_name="Jane Smith",
        dispatcher_phone="555-9999",
        dispatcher_email="dispatcher@example.com",
        additional_data={"custom_field": "custom_value"}
    )
    assert shipment.reference_id == "REF123"
    assert shipment.carrier.carrier_name == "CarrierX"
    assert shipment.commodities[0].commodity_name == "Steel"
    assert shipment.rate_info.total_rate == 2000.0
    assert shipment.additional_data["custom_field"] == "custom_value"

def test_shipment_data_minimal_fields():
    shipment = ShipmentData()
    for field in shipment.__fields__:
        assert getattr(shipment, field) is None

def test_shipment_data_commodities_empty_list():
    shipment = ShipmentData(commodities=[])
    assert shipment.commodities == []

def test_shipment_data_additional_data_arbitrary_types():
    shipment = ShipmentData(additional_data={"foo": 123, "bar": [1, 2, 3], "baz": {"nested": True}})
    assert shipment.additional_data["foo"] == 123
    assert shipment.additional_data["bar"] == [1, 2, 3]
    assert shipment.additional_data["baz"]["nested"] is True

def test_extraction_response_happy_path():
    shipment = ShipmentData(reference_id="REF1")
    resp = ExtractionResponse(data=shipment, document_id="DOC123")
    assert resp.data.reference_id == "REF1"
    assert resp.document_id == "DOC123"

def test_extraction_response_document_id_none():
    shipment = ShipmentData(reference_id="REF2")
    resp = ExtractionResponse(data=shipment)
    assert resp.document_id is None

def test_extraction_response_invalid_data_type():
    with pytest.raises(ValidationError):
        ExtractionResponse(data="not_a_shipment_data")

def test_shipment_data_invalid_carrier_type():
    with pytest.raises(ValidationError):
        ShipmentData(carrier="not_a_carrier_info")

def test_shipment_data_invalid_commodities_type():
    with pytest.raises(ValidationError):
        ShipmentData(commodities="not_a_list")

def test_shipment_data_commodities_with_invalid_item():
    with pytest.raises(ValidationError):
        ShipmentData(commodities=[{"commodity_name": "Steel", "weight": 1000}])  # weight should be str

def test_rate_info_rate_breakdown_arbitrary_types():
    rate = RateInfo(rate_breakdown={"misc": [1, 2, 3], "flag": True})
    assert rate.rate_breakdown["misc"] == [1, 2, 3]
    assert rate.rate_breakdown["flag"] is True

def test_location_zip_code_boundary_conditions():
    loc = Location(zip_code="00000")
    assert loc.zip_code == "00000"
    loc2 = Location(zip_code="99999")
    assert loc2.zip_code == "99999"

def test_location_invalid_field_raises():
    with pytest.raises(TypeError):
        Location(name="A", invalid_field="foo")

def test_commodity_item_empty_string_fields():
    item = CommodityItem(commodity_name="", weight="", quantity="", description="")
    assert item.commodity_name == ""
    assert item.weight == ""
    assert item.quantity == ""
    assert item.description == ""

def test_rate_info_none_fields():
    rate = RateInfo()
    assert rate.total_rate is None
    assert rate.currency is None
    assert rate.rate_breakdown is None

def test_shipment_data_reference_id_empty_string():
    shipment = ShipmentData(reference_id="")
    assert shipment.reference_id == ""

def test_shipment_data_additional_data_none():
    shipment = ShipmentData(additional_data=None)
    assert shipment.additional_data is None
