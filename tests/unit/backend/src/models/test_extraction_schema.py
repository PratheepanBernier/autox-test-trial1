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

def test_carrier_info_full_and_partial():
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

    carrier_partial = CarrierInfo(carrier_name="OnlyName")
    assert carrier_partial.carrier_name == "OnlyName"
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
    assert shipment.equipment_type == "Van"
    assert shipment.equipment_size == "53"
    assert shipment.load_type == "FTL"
    assert shipment.commodities[0].commodity_name == "Widgets"
    assert shipment.commodities[0].weight == "1000 lbs"
    assert shipment.rate_info.total_rate == 2000.0
    assert shipment.rate_info.currency == "USD"
    assert shipment.special_instructions == "Handle with care"
    assert shipment.shipper_instructions == "Call before arrival"
    assert shipment.carrier_instructions == "No overnight parking"
    assert shipment.dispatcher_name == "Dispatch Joe"
    assert shipment.dispatcher_phone == "555-0000"
    assert shipment.dispatcher_email == "dispatch@shipper.com"
    assert shipment.additional_data == {"custom_field": "custom_value"}

def test_shipment_data_commodities_empty_list():
    shipment = ShipmentData(commodities=[])
    assert shipment.commodities == []

def test_shipment_data_additional_data_edge_cases():
    shipment = ShipmentData(additional_data={})
    assert shipment.additional_data == {}

    shipment2 = ShipmentData(additional_data={"foo": None, "bar": 123})
    assert shipment2.additional_data == {"foo": None, "bar": 123}

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
    # total_rate expects a float, but string is provided
    with pytest.raises(ValidationError):
        RateInfo(total_rate="not_a_float")

def test_invalid_commodities_type_in_shipment_data():
    # commodities expects a list of CommodityItem, not a dict
    with pytest.raises(ValidationError):
        ShipmentData(commodities={"commodity_name": "Steel"})

def test_invalid_carrier_type_in_shipment_data():
    # carrier expects CarrierInfo, not a string
    with pytest.raises(ValidationError):
        ShipmentData(carrier="Carrier as string")

def test_invalid_pickup_type_in_shipment_data():
    # pickup expects Location, not an int
    with pytest.raises(ValidationError):
        ShipmentData(pickup=123)

def test_invalid_rate_info_type_in_shipment_data():
    # rate_info expects RateInfo, not a list
    with pytest.raises(ValidationError):
        ShipmentData(rate_info=[1, 2, 3])

def test_boundary_empty_strings_and_zeroes():
    loc = Location(
        name="",
        address="",
        city="",
        state="",
        zip_code="",
        country="",
        appointment_time="",
        po_number=""
    )
    assert loc.name == ""
    assert loc.address == ""
    assert loc.city == ""
    assert loc.state == ""
    assert loc.zip_code == ""
    assert loc.country == ""
    assert loc.appointment_time == ""
    assert loc.po_number == ""

    rate = RateInfo(total_rate=0.0, currency="", rate_breakdown={})
    assert rate.total_rate == 0.0
    assert rate.currency == ""
    assert rate.rate_breakdown == {}

def test_shipment_data_equivalent_paths_for_po_number():
    # po_number exists at both ShipmentData and Location
    shipment = ShipmentData(
        po_number="PO-SHIPMENT",
        pickup=Location(po_number="PO-PICKUP"),
        drop=Location(po_number="PO-DROP")
    )
    assert shipment.po_number == "PO-SHIPMENT"
    assert shipment.pickup.po_number == "PO-PICKUP"
    assert shipment.drop.po_number == "PO-DROP"
    # Reconciliation: values are independent
    assert shipment.po_number != shipment.pickup.po_number
    assert shipment.po_number != shipment.drop.po_number

def test_shipment_data_accepts_none_for_all_fields():
    # All fields are optional and should accept None
    shipment = ShipmentData(
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
    for field in shipment.__fields__:
        assert getattr(shipment, field) is None
