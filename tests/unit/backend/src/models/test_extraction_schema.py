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
        commodity_name="Steel Beams",
        weight="56000.00 lbs",
        quantity="10000 units",
        description="High quality steel beams"
    )
    assert item.commodity_name == "Steel Beams"
    assert item.weight == "56000.00 lbs"
    assert item.quantity == "10000 units"
    assert item.description == "High quality steel beams"

def test_commodity_item_missing_fields():
    item = CommodityItem()
    assert item.commodity_name is None
    assert item.weight is None
    assert item.quantity is None
    assert item.description is None

def test_carrier_info_happy_path():
    carrier = CarrierInfo(
        carrier_name="FastTrans LLC",
        mc_number="MC123456",
        phone="555-123-4567",
        email="dispatch@fasttrans.com"
    )
    assert carrier.carrier_name == "FastTrans LLC"
    assert carrier.mc_number == "MC123456"
    assert carrier.phone == "555-123-4567"
    assert carrier.email == "dispatch@fasttrans.com"

def test_driver_info_happy_path():
    driver = DriverInfo(
        driver_name="John Doe",
        cell_number="555-987-6543",
        truck_number="TX1234",
        trailer_number="TRL5678"
    )
    assert driver.driver_name == "John Doe"
    assert driver.cell_number == "555-987-6543"
    assert driver.truck_number == "TX1234"
    assert driver.trailer_number == "TRL5678"

def test_rate_info_happy_path():
    rate = RateInfo(
        total_rate=2500.75,
        currency="USD",
        rate_breakdown={"base": 2000, "fuel": 500.75}
    )
    assert rate.total_rate == 2500.75
    assert rate.currency == "USD"
    assert rate.rate_breakdown == {"base": 2000, "fuel": 500.75}

def test_rate_info_missing_fields():
    rate = RateInfo()
    assert rate.total_rate is None
    assert rate.currency is None
    assert rate.rate_breakdown is None

def test_shipment_data_happy_path():
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
        commodities=[CommodityItem(commodity_name="Widgets", weight="1000 lbs")],
        rate_info=RateInfo(total_rate=1000.0, currency="USD"),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No weekend delivery",
        dispatcher_name="Alice",
        dispatcher_phone="555-000-1111",
        dispatcher_email="alice@dispatch.com",
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
    assert shipment.commodities[0].commodity_name == "Widgets"
    assert shipment.commodities[0].weight == "1000 lbs"
    assert shipment.rate_info.total_rate == 1000.0
    assert shipment.rate_info.currency == "USD"
    assert shipment.special_instructions == "Handle with care"
    assert shipment.shipper_instructions == "Call before arrival"
    assert shipment.carrier_instructions == "No weekend delivery"
    assert shipment.dispatcher_name == "Alice"
    assert shipment.dispatcher_phone == "555-000-1111"
    assert shipment.dispatcher_email == "alice@dispatch.com"
    assert shipment.additional_data == {"custom_field": "custom_value"}

def test_shipment_data_minimal_fields():
    shipment = ShipmentData()
    assert shipment.reference_id is None
    assert shipment.carrier is None
    assert shipment.commodities is None
    assert shipment.additional_data is None

def test_shipment_data_empty_commodities():
    shipment = ShipmentData(commodities=[])
    assert shipment.commodities == []

def test_shipment_data_additional_data_types():
    shipment = ShipmentData(additional_data={"foo": 123, "bar": [1, 2, 3]})
    assert shipment.additional_data == {"foo": 123, "bar": [1, 2, 3]}

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
    # total_rate should be float or None; string should raise error
    with pytest.raises(ValidationError):
        RateInfo(total_rate="not_a_float")

def test_invalid_commodities_type_in_shipment_data():
    # commodities should be a list of CommodityItem or None
    with pytest.raises(ValidationError):
        ShipmentData(commodities="not_a_list")

def test_invalid_rate_breakdown_type():
    # rate_breakdown should be dict or None
    with pytest.raises(ValidationError):
        RateInfo(rate_breakdown="not_a_dict")

def test_invalid_additional_data_type():
    # additional_data should be dict or None
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
        equipment_size="",
        load_type="",
        special_instructions="",
        shipper_instructions="",
        carrier_instructions="",
        dispatcher_name="",
        dispatcher_phone="",
        dispatcher_email="",
        additional_data={}
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
    assert shipment.additional_data == {}

def test_rate_info_zero_and_negative_values():
    rate = RateInfo(total_rate=0.0, currency="USD", rate_breakdown={"base": 0, "discount": -100})
    assert rate.total_rate == 0.0
    assert rate.currency == "USD"
    assert rate.rate_breakdown == {"base": 0, "discount": -100}

def test_commodity_item_empty_and_zero_values():
    item = CommodityItem(commodity_name="", weight="0 lbs", quantity="0 units", description="")
    assert item.commodity_name == ""
    assert item.weight == "0 lbs"
    assert item.quantity == "0 units"
    assert item.description == ""
