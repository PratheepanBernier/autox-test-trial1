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
        carrier=CarrierInfo(carrier_name="CarrierX"),
        driver=DriverInfo(driver_name="Jane Smith"),
        pickup=Location(name="Origin", city="StartCity"),
        drop=Location(name="Destination", city="EndCity"),
        shipping_date="2024-06-01",
        delivery_date="2024-06-02",
        created_on="2024-05-31",
        booking_date="2024-05-30",
        equipment_type="Van",
        equipment_size="53",
        load_type="FTL",
        commodities=[
            CommodityItem(commodity_name="Widgets", weight="1000 lbs", quantity="100 units")
        ],
        rate_info=RateInfo(total_rate=2000.0, currency="USD"),
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
    assert shipment.drop.city == "EndCity"
    assert shipment.commodities[0].commodity_name == "Widgets"
    assert shipment.rate_info.total_rate == 2000.0
    assert shipment.additional_data["custom_field"] == "custom_value"

def test_shipment_data_commodities_empty_list():
    shipment = ShipmentData(commodities=[])
    assert shipment.commodities == []

def test_shipment_data_additional_data_edge_cases():
    shipment = ShipmentData(additional_data={})
    assert shipment.additional_data == {}

    shipment2 = ShipmentData(additional_data={"foo": None, "bar": 123})
    assert shipment2.additional_data["foo"] is None
    assert shipment2.additional_data["bar"] == 123

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

def test_invalid_rate_info_total_rate_type():
    with pytest.raises(ValidationError):
        RateInfo(total_rate="not_a_float")

def test_invalid_commodities_type_in_shipment_data():
    with pytest.raises(ValidationError):
        ShipmentData(commodities="not_a_list")

def test_invalid_additional_data_type():
    with pytest.raises(ValidationError):
        ShipmentData(additional_data="not_a_dict")

def test_location_zip_code_boundary():
    loc = Location(zip_code="00000")
    assert loc.zip_code == "00000"
    loc2 = Location(zip_code="99999")
    assert loc2.zip_code == "99999"

def test_rate_info_rate_breakdown_various_types():
    # Accepts any dict, including nested
    breakdown = {"base": 1000, "details": {"fuel": 200, "toll": 50}}
    rate = RateInfo(rate_breakdown=breakdown)
    assert rate.rate_breakdown["details"]["fuel"] == 200

def test_shipment_data_equivalent_paths_for_po_number():
    # po_number at root and in pickup
    shipment1 = ShipmentData(po_number="PO1", pickup=Location(po_number="PO1"))
    shipment2 = ShipmentData(po_number="PO1")
    assert shipment1.po_number == shipment2.po_number
    assert shipment1.pickup.po_number == "PO1"
    assert shipment2.pickup is None

def test_shipment_data_none_and_missing_fields_equivalence():
    # None and missing fields should be equivalent in output
    shipment1 = ShipmentData(reference_id=None)
    shipment2 = ShipmentData()
    assert shipment1.dict() == shipment2.dict()
