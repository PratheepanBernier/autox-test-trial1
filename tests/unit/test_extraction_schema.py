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

def test_driver_info_empty_and_full():
    driver = DriverInfo()
    assert driver.driver_name is None
    assert driver.cell_number is None
    assert driver.truck_number is None
    assert driver.trailer_number is None

    driver_full = DriverInfo(
        driver_name="John Doe",
        cell_number="555-6789",
        truck_number="TRK123",
        trailer_number="TRL456"
    )
    assert driver_full.driver_name == "John Doe"
    assert driver_full.cell_number == "555-6789"
    assert driver_full.truck_number == "TRK123"
    assert driver_full.trailer_number == "TRL456"

def test_rate_info_with_breakdown():
    breakdown = {"linehaul": 1000.0, "fuel": 200.0}
    rate = RateInfo(total_rate=1200.0, currency="USD", rate_breakdown=breakdown)
    assert rate.total_rate == 1200.0
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
        rate_info=RateInfo(total_rate=1500.0, currency="USD"),
        special_instructions="Handle with care",
        shipper_instructions="Call before delivery",
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
    assert shipment.drop.name == "Destination"
    assert shipment.commodities[0].commodity_name == "Widgets"
    assert shipment.rate_info.total_rate == 1500.0
    assert shipment.additional_data["custom_field"] == "custom_value"

def test_shipment_data_commodities_empty_list():
    shipment = ShipmentData(commodities=[])
    assert shipment.commodities == []

def test_shipment_data_additional_data_none_and_dict():
    shipment_none = ShipmentData()
    assert shipment_none.additional_data is None

    shipment_dict = ShipmentData(additional_data={"foo": "bar"})
    assert shipment_dict.additional_data == {"foo": "bar"}

def test_extraction_response_happy_path():
    shipment = ShipmentData(reference_id="REF1")
    resp = ExtractionResponse(data=shipment, document_id="DOC123")
    assert resp.data.reference_id == "REF1"
    assert resp.document_id == "DOC123"

def test_extraction_response_document_id_none():
    shipment = ShipmentData(reference_id="REF2")
    resp = ExtractionResponse(data=shipment)
    assert resp.data.reference_id == "REF2"
    assert resp.document_id is None

def test_invalid_rate_info_total_rate_type():
    with pytest.raises(ValidationError):
        RateInfo(total_rate="not_a_float")

def test_invalid_commodities_type_in_shipment_data():
    with pytest.raises(ValidationError):
        ShipmentData(commodities="not_a_list")

def test_invalid_rate_info_breakdown_type():
    with pytest.raises(ValidationError):
        RateInfo(rate_breakdown="not_a_dict")

def test_invalid_carrier_info_type_in_shipment_data():
    with pytest.raises(ValidationError):
        ShipmentData(carrier="not_a_carrierinfo")

def test_invalid_pickup_type_in_shipment_data():
    with pytest.raises(ValidationError):
        ShipmentData(pickup="not_a_location")

def test_invalid_additional_data_type():
    with pytest.raises(ValidationError):
        ShipmentData(additional_data="not_a_dict")

def test_boundary_empty_strings_and_zero_values():
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
        commodities=[CommodityItem(commodity_name="", weight="0")],
        rate_info=RateInfo(total_rate=0.0, currency="", rate_breakdown={}),
        special_instructions="",
        shipper_instructions="",
        carrier_instructions="",
        dispatcher_name="",
        dispatcher_phone="",
        dispatcher_email="",
        additional_data={}
    )
    assert shipment.reference_id == ""
    assert shipment.equipment_size == "0"
    assert shipment.commodities[0].weight == "0"
    assert shipment.rate_info.total_rate == 0.0
    assert shipment.additional_data == {}

def test_equivalent_paths_for_shipment_data_and_extraction_response():
    # Reconciliation: ShipmentData with same values, ExtractionResponse wraps it
    shipment1 = ShipmentData(reference_id="EQ1", po_number="PO1")
    shipment2 = ShipmentData(reference_id="EQ1", po_number="PO1")
    assert shipment1 == shipment2

    resp1 = ExtractionResponse(data=shipment1, document_id="D1")
    resp2 = ExtractionResponse(data=shipment2, document_id="D1")
    assert resp1 == resp2

def test_shipment_data_with_none_and_missing_fields_equivalence():
    # Regression: None fields and omitted fields should be equivalent
    shipment_none = ShipmentData(reference_id=None)
    shipment_missing = ShipmentData()
    assert shipment_none.reference_id == shipment_missing.reference_id
    assert shipment_none == shipment_missing or shipment_none.dict() == shipment_missing.dict()
