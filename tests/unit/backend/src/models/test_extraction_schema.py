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

def test_commodity_item_happy_path():
    item = CommodityItem(
        commodity_name="Steel",
        weight="56000.00 lbs",
        quantity="10000 units",
        description="Rolled steel coils"
    )
    assert item.commodity_name == "Steel"
    assert item.weight == "56000.00 lbs"
    assert item.quantity == "10000 units"
    assert item.description == "Rolled steel coils"

def test_commodity_item_partial_fields():
    item = CommodityItem(commodity_name="Copper")
    assert item.commodity_name == "Copper"
    assert item.weight is None
    assert item.quantity is None
    assert item.description is None

def test_carrier_info_happy_path():
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

def test_driver_info_happy_path():
    driver = DriverInfo(
        driver_name="John Doe",
        cell_number="555-5678",
        truck_number="TX1234",
        trailer_number="TRL5678"
    )
    assert driver.driver_name == "John Doe"
    assert driver.cell_number == "555-5678"
    assert driver.truck_number == "TX1234"
    assert driver.trailer_number == "TRL5678"

def test_rate_info_happy_path():
    rate = RateInfo(
        total_rate=2500.50,
        currency="USD",
        rate_breakdown={"linehaul": 2000, "fuel": 500.5}
    )
    assert rate.total_rate == 2500.50
    assert rate.currency == "USD"
    assert rate.rate_breakdown == {"linehaul": 2000, "fuel": 500.5}

def test_rate_info_none_fields():
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
        equipment_type="Van",
        equipment_size="53",
        load_type="FTL",
        commodities=[CommodityItem(commodity_name="Widgets", weight="1000 lbs")],
        rate_info=RateInfo(total_rate=1000.0, currency="USD"),
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
    assert shipment.rate_info.total_rate == 1000.0
    assert shipment.rate_info.currency == "USD"
    assert shipment.special_instructions == "Handle with care"
    assert shipment.shipper_instructions == "Call before arrival"
    assert shipment.carrier_instructions == "No overnight parking"
    assert shipment.dispatcher_name == "Dispatch Joe"
    assert shipment.dispatcher_phone == "555-0000"
    assert shipment.dispatcher_email == "dispatch@shipper.com"
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
    shipment = ShipmentData(reference_id="REF1")
    resp = ExtractionResponse(data=shipment, document_id="DOC1")
    assert resp.data.reference_id == "REF1"
    assert resp.document_id == "DOC1"

def test_extraction_response_document_id_none():
    shipment = ShipmentData(reference_id="REF2")
    resp = ExtractionResponse(data=shipment)
    assert resp.data.reference_id == "REF2"
    assert resp.document_id is None

def test_invalid_rate_info_total_rate_type():
    # total_rate expects float or None
    with pytest.raises(ValidationError):
        RateInfo(total_rate="not_a_float")

def test_invalid_commodities_type():
    # commodities expects a list of CommodityItem or None
    with pytest.raises(ValidationError):
        ShipmentData(commodities="not_a_list")

def test_invalid_additional_data_type():
    # additional_data expects dict or None
    with pytest.raises(ValidationError):
        ShipmentData(additional_data="not_a_dict")

def test_invalid_carrier_type():
    # carrier expects CarrierInfo or None
    with pytest.raises(ValidationError):
        ShipmentData(carrier="not_a_carrierinfo")

def test_invalid_driver_type():
    # driver expects DriverInfo or None
    with pytest.raises(ValidationError):
        ShipmentData(driver=123)

def test_invalid_pickup_type():
    # pickup expects Location or None
    with pytest.raises(ValidationError):
        ShipmentData(pickup="not_a_location")

def test_invalid_drop_type():
    # drop expects Location or None
    with pytest.raises(ValidationError):
        ShipmentData(drop=123)

def test_invalid_rate_info_type():
    # rate_info expects RateInfo or None
    with pytest.raises(ValidationError):
        ShipmentData(rate_info="not_a_rateinfo")

def test_invalid_extraction_response_data_type():
    # data expects ShipmentData
    with pytest.raises(ValidationError):
        ExtractionResponse(data="not_a_shipmentdata")

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

def test_shipment_data_with_nulls_and_empty_lists():
    shipment = ShipmentData(
        commodities=[],
        additional_data={},
        carrier=None,
        driver=None,
        pickup=None,
        drop=None,
        rate_info=None
    )
    assert shipment.commodities == []
    assert shipment.additional_data == {}
    assert shipment.carrier is None
    assert shipment.driver is None
    assert shipment.pickup is None
    assert shipment.drop is None
    assert shipment.rate_info is None

def test_shipment_data_with_nested_none():
    shipment = ShipmentData(
        carrier=CarrierInfo(),
        driver=DriverInfo(),
        pickup=Location(),
        drop=Location(),
        rate_info=RateInfo(),
        commodities=[CommodityItem()]
    )
    assert shipment.carrier.carrier_name is None
    assert shipment.driver.driver_name is None
    assert shipment.pickup.name is None
    assert shipment.drop.name is None
    assert shipment.rate_info.total_rate is None
    assert shipment.commodities[0].commodity_name is None

def test_reconciliation_equivalent_paths_for_commodities():
    # commodities=None and commodities=[] should be distinguishable
    shipment_none = ShipmentData()
    shipment_empty = ShipmentData(commodities=[])
    assert shipment_none.commodities is None
    assert shipment_empty.commodities == []

def test_reconciliation_equivalent_paths_for_additional_data():
    # additional_data=None and additional_data={} should be distinguishable
    shipment_none = ShipmentData()
    shipment_empty = ShipmentData(additional_data={})
    assert shipment_none.additional_data is None
    assert shipment_empty.additional_data == {}
