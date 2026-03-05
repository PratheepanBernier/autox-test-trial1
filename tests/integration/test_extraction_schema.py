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

def test_commodity_item_partial_fields():
    item = CommodityItem(commodity_name="Steel", weight="1000 lbs")
    assert item.commodity_name == "Steel"
    assert item.weight == "1000 lbs"
    assert item.quantity is None
    assert item.description is None

def test_carrier_info_and_driver_info_full_fields():
    carrier = CarrierInfo(
        carrier_name="FastTrans",
        mc_number="MC123456",
        phone="555-1234",
        email="contact@fasttrans.com"
    )
    driver = DriverInfo(
        driver_name="John Doe",
        cell_number="555-5678",
        truck_number="TRK123",
        trailer_number="TRL456"
    )
    assert carrier.carrier_name == "FastTrans"
    assert carrier.mc_number == "MC123456"
    assert carrier.phone == "555-1234"
    assert carrier.email == "contact@fasttrans.com"
    assert driver.driver_name == "John Doe"
    assert driver.cell_number == "555-5678"
    assert driver.truck_number == "TRK123"
    assert driver.trailer_number == "TRL456"

def test_rate_info_with_breakdown():
    breakdown = {"base": 1000.0, "fuel": 150.0}
    rate = RateInfo(total_rate=1150.0, currency="USD", rate_breakdown=breakdown)
    assert rate.total_rate == 1150.0
    assert rate.currency == "USD"
    assert rate.rate_breakdown == breakdown

def test_shipment_data_minimal_fields():
    shipment = ShipmentData()
    assert shipment.reference_id is None
    assert shipment.load_id is None
    assert shipment.commodities is None
    assert shipment.carrier is None
    assert shipment.pickup is None
    assert shipment.additional_data is None

def test_shipment_data_full_fields():
    shipment = ShipmentData(
        reference_id="REF123",
        load_id="LOAD456",
        po_number="PO789",
        shipper="Shipper Inc.",
        consignee="Consignee LLC",
        carrier=CarrierInfo(carrier_name="CarrierX"),
        driver=DriverInfo(driver_name="Jane Smith"),
        pickup=Location(name="Origin", city="Alpha"),
        drop=Location(name="Destination", city="Beta"),
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
        dispatcher_name="Dispatch Joe",
        dispatcher_phone="555-9999",
        dispatcher_email="dispatch@shipper.com",
        additional_data={"custom_field": "custom_value"}
    )
    assert shipment.reference_id == "REF123"
    assert shipment.carrier.carrier_name == "CarrierX"
    assert shipment.driver.driver_name == "Jane Smith"
    assert shipment.pickup.name == "Origin"
    assert shipment.drop.city == "Beta"
    assert shipment.commodities[0].commodity_name == "Widgets"
    assert shipment.rate_info.total_rate == 2000.0
    assert shipment.additional_data["custom_field"] == "custom_value"

def test_shipment_data_commodities_empty_list():
    shipment = ShipmentData(commodities=[])
    assert shipment.commodities == []

def test_shipment_data_additional_data_various_types():
    shipment = ShipmentData(additional_data={"int_field": 1, "list_field": [1,2,3], "dict_field": {"a": "b"}})
    assert shipment.additional_data["int_field"] == 1
    assert shipment.additional_data["list_field"] == [1,2,3]
    assert shipment.additional_data["dict_field"] == {"a": "b"}

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

def test_rate_info_invalid_total_rate_type():
    with pytest.raises(ValidationError):
        RateInfo(total_rate="not_a_float")

def test_shipment_data_invalid_commodities_type():
    with pytest.raises(ValidationError):
        ShipmentData(commodities="not_a_list")

def test_location_zip_code_boundary_length():
    # US ZIP codes: 5 or 9 digits
    loc = Location(zip_code="12345")
    assert loc.zip_code == "12345"
    loc9 = Location(zip_code="123456789")
    assert loc9.zip_code == "123456789"

def test_location_zip_code_invalid_type():
    with pytest.raises(ValidationError):
        Location(zip_code=12345)

def test_commodity_item_empty_string_fields():
    item = CommodityItem(commodity_name="", weight="", quantity="", description="")
    assert item.commodity_name == ""
    assert item.weight == ""
    assert item.quantity == ""
    assert item.description == ""

def test_shipment_data_equivalent_paths_for_po_number():
    # po_number at root and in pickup location
    shipment1 = ShipmentData(po_number="PO1", pickup=Location(po_number="PO1"))
    shipment2 = ShipmentData(po_number="PO1")
    assert shipment1.po_number == shipment2.po_number
    assert shipment1.pickup.po_number == "PO1"

def test_shipment_data_none_and_missing_fields_equivalence():
    # Omitted fields and explicit None should be equivalent
    shipment1 = ShipmentData()
    shipment2 = ShipmentData(reference_id=None, load_id=None)
    assert shipment1.reference_id == shipment2.reference_id
    assert shipment1.load_id == shipment2.load_id

def test_shipment_data_additional_data_none_and_empty_dict():
    shipment1 = ShipmentData(additional_data=None)
    shipment2 = ShipmentData(additional_data={})
    # None and empty dict are not the same, but both valid
    assert shipment1.additional_data is None
    assert shipment2.additional_data == {}

def test_shipment_data_rate_info_none_and_missing():
    shipment1 = ShipmentData()
    shipment2 = ShipmentData(rate_info=None)
    assert shipment1.rate_info == shipment2.rate_info

def test_extraction_response_invalid_data_type():
    with pytest.raises(ValidationError):
        ExtractionResponse(data="not_a_shipment_data")

def test_shipment_data_dispatcher_fields_partial():
    shipment = ShipmentData(dispatcher_name="Alice")
    assert shipment.dispatcher_name == "Alice"
    assert shipment.dispatcher_phone is None
    assert shipment.dispatcher_email is None

def test_rate_info_rate_breakdown_various_types():
    rate = RateInfo(rate_breakdown={"accessorial": "detention", "amount": 50})
    assert rate.rate_breakdown["accessorial"] == "detention"
    assert rate.rate_breakdown["amount"] == 50

def test_shipment_data_special_instructions_empty_string():
    shipment = ShipmentData(special_instructions="")
    assert shipment.special_instructions == ""

def test_shipment_data_commodities_none_and_empty_equivalence():
    shipment1 = ShipmentData()
    shipment2 = ShipmentData(commodities=None)
    shipment3 = ShipmentData(commodities=[])
    assert shipment1.commodities == shipment2.commodities
    assert shipment3.commodities == []

def test_location_field_descriptions():
    # Ensure field descriptions are preserved (regression)
    assert Location.__fields__["name"].field_info.description == "Location name"
    assert Location.__fields__["appointment_time"].field_info.description == "Appointment time"
    assert Location.__fields__["po_number"].field_info.description == "PO/Container number"
