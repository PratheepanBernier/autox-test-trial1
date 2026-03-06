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
    for field in loc.__fields__:
        assert getattr(loc, field) is None

def test_commodity_item_partial_fields():
    item = CommodityItem(commodity_name="Steel", weight="1000 lbs")
    assert item.commodity_name == "Steel"
    assert item.weight == "1000 lbs"
    assert item.quantity is None
    assert item.description is None

def test_carrier_info_edge_cases():
    carrier = CarrierInfo(carrier_name="", mc_number=None, phone="000", email="test@example.com")
    assert carrier.carrier_name == ""
    assert carrier.mc_number is None
    assert carrier.phone == "000"
    assert carrier.email == "test@example.com"

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
    breakdown = {"base": 1000.0, "fuel": 150.0}
    rate = RateInfo(total_rate=1150.0, currency="USD", rate_breakdown=breakdown)
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
    shipment = ShipmentData(
        reference_id="REF123",
        load_id="LOAD456",
        po_number="PO789",
        shipper="Shipper Inc.",
        consignee="Consignee LLC",
        carrier=CarrierInfo(carrier_name="CarrierX", mc_number="MC123", phone="555-0000", email="carrier@example.com"),
        driver=DriverInfo(driver_name="Jane Roe", cell_number="555-5678", truck_number="TRK789", trailer_number="TRL012"),
        pickup=Location(name="Origin", address="1 Origin Rd", city="Start", state="CA", zip_code="90001", country="USA"),
        drop=Location(name="Destination", address="2 Dest Ave", city="End", state="TX", zip_code="73301", country="USA"),
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
        rate_info=RateInfo(total_rate=2000.0, currency="USD", rate_breakdown={"base": 1800.0, "fuel": 200.0}),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No overnight parking",
        dispatcher_name="Dispatch Dan",
        dispatcher_phone="555-9999",
        dispatcher_email="dispatch@example.com",
        additional_data={"custom_field": "custom_value"}
    )
    assert shipment.reference_id == "REF123"
    assert shipment.load_id == "LOAD456"
    assert shipment.po_number == "PO789"
    assert shipment.shipper == "Shipper Inc."
    assert shipment.consignee == "Consignee LLC"
    assert shipment.carrier.carrier_name == "CarrierX"
    assert shipment.driver.driver_name == "Jane Roe"
    assert shipment.pickup.name == "Origin"
    assert shipment.drop.name == "Destination"
    assert shipment.shipping_date == "2024-06-01"
    assert shipment.delivery_date == "2024-06-02"
    assert shipment.created_on == "2024-05-31"
    assert shipment.booking_date == "2024-05-30"
    assert shipment.equipment_type == "Van"
    assert shipment.equipment_size == "53"
    assert shipment.load_type == "FTL"
    assert len(shipment.commodities) == 2
    assert shipment.commodities[0].commodity_name == "Widgets"
    assert shipment.rate_info.total_rate == 2000.0
    assert shipment.special_instructions == "Handle with care"
    assert shipment.shipper_instructions == "Call before arrival"
    assert shipment.carrier_instructions == "No overnight parking"
    assert shipment.dispatcher_name == "Dispatch Dan"
    assert shipment.dispatcher_phone == "555-9999"
    assert shipment.dispatcher_email == "dispatch@example.com"
    assert shipment.additional_data == {"custom_field": "custom_value"}

def test_shipment_data_commodities_empty_list():
    shipment = ShipmentData(commodities=[])
    assert shipment.commodities == []

def test_shipment_data_additional_data_empty_dict():
    shipment = ShipmentData(additional_data={})
    assert shipment.additional_data == {}

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
    # total_rate expects float or None
    with pytest.raises(ValidationError):
        RateInfo(total_rate="not_a_float")

def test_invalid_commodities_type_in_shipment_data():
    # commodities expects a list of CommodityItem or None
    with pytest.raises(ValidationError):
        ShipmentData(commodities="not_a_list")

def test_invalid_additional_data_type():
    # additional_data expects dict or None
    with pytest.raises(ValidationError):
        ShipmentData(additional_data="not_a_dict")

def test_location_zip_code_boundary():
    # Accepts any string, but test a very long zip code
    loc = Location(zip_code="12345678901234567890")
    assert loc.zip_code == "12345678901234567890"

def test_rate_info_rate_breakdown_arbitrary_dict():
    breakdown = {"misc": [1, 2, 3], "notes": {"a": 1}}
    rate = RateInfo(rate_breakdown=breakdown)
    assert rate.rate_breakdown == breakdown

def test_shipment_data_equivalent_paths_for_po_number():
    # po_number at root and in pickup location
    shipment1 = ShipmentData(po_number="PO1")
    shipment2 = ShipmentData(pickup=Location(po_number="PO1"))
    # They are not strictly equivalent, but both store the value in a po_number field
    assert shipment1.po_number == shipment2.pickup.po_number

def test_shipment_data_none_fields_are_none():
    shipment = ShipmentData()
    for field in shipment.__fields__:
        assert getattr(shipment, field) is None

def test_extraction_response_requires_data():
    with pytest.raises(ValidationError):
        ExtractionResponse(document_id="DOC999")
