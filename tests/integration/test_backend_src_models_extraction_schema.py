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

def test_carrier_info_full_and_partial():
    full = CarrierInfo(
        carrier_name="FastTrans",
        mc_number="MC123456",
        phone="555-1234",
        email="dispatch@fasttrans.com"
    )
    partial = CarrierInfo(carrier_name="QuickMove")
    assert full.carrier_name == "FastTrans"
    assert full.mc_number == "MC123456"
    assert full.phone == "555-1234"
    assert full.email == "dispatch@fasttrans.com"
    assert partial.carrier_name == "QuickMove"
    assert partial.mc_number is None
    assert partial.phone is None
    assert partial.email is None

def test_driver_info_edge_cases():
    driver = DriverInfo(driver_name="", cell_number=None, truck_number="TRK1", trailer_number="")
    assert driver.driver_name == ""
    assert driver.cell_number is None
    assert driver.truck_number == "TRK1"
    assert driver.trailer_number == ""

def test_rate_info_happy_path_and_breakdown():
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
    assert shipment.reference_id is None
    assert shipment.load_id is None
    assert shipment.commodities is None
    assert shipment.pickup is None
    assert shipment.rate_info is None

def test_shipment_data_full():
    shipment = ShipmentData(
        reference_id="REF123",
        load_id="LOAD456",
        po_number="PO789",
        shipper="Shipper Inc.",
        consignee="Consignee LLC",
        carrier=CarrierInfo(carrier_name="CarrierX"),
        driver=DriverInfo(driver_name="John Doe"),
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
            CommodityItem(commodity_name="Widgets", weight="1000 lbs", quantity="500 units")
        ],
        rate_info=RateInfo(total_rate=2000.0, currency="USD"),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No overnight parking",
        dispatcher_name="Alice",
        dispatcher_phone="555-6789",
        dispatcher_email="alice@dispatch.com",
        additional_data={"custom_field": "custom_value"}
    )
    assert shipment.reference_id == "REF123"
    assert shipment.carrier.carrier_name == "CarrierX"
    assert shipment.driver.driver_name == "John Doe"
    assert shipment.pickup.name == "Origin"
    assert shipment.drop.city == "EndCity"
    assert shipment.commodities[0].commodity_name == "Widgets"
    assert shipment.rate_info.total_rate == 2000.0
    assert shipment.additional_data["custom_field"] == "custom_value"

def test_shipment_data_commodities_empty_list():
    shipment = ShipmentData(commodities=[])
    assert shipment.commodities == []

def test_shipment_data_additional_data_various_types():
    shipment = ShipmentData(additional_data={"int": 1, "list": [1, 2], "dict": {"a": "b"}})
    assert shipment.additional_data["int"] == 1
    assert shipment.additional_data["list"] == [1, 2]
    assert shipment.additional_data["dict"] == {"a": "b"}

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

def test_invalid_carrier_info_type_in_shipment_data():
    with pytest.raises(ValidationError):
        ShipmentData(carrier="not_a_carrier_info")

def test_invalid_pickup_type_in_shipment_data():
    with pytest.raises(ValidationError):
        ShipmentData(pickup="not_a_location")

def test_invalid_rate_info_type_in_shipment_data():
    with pytest.raises(ValidationError):
        ShipmentData(rate_info="not_a_rate_info")

def test_invalid_additional_data_type_in_shipment_data():
    with pytest.raises(ValidationError):
        ShipmentData(additional_data="not_a_dict")

def test_shipment_data_boundary_empty_strings():
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
        dispatcher_email=""
    )
    assert shipment.reference_id == ""
    assert shipment.load_id == ""
    assert shipment.shipper == ""
    assert shipment.dispatcher_email == ""

def test_shipment_data_none_and_empty_mix():
    shipment = ShipmentData(
        reference_id=None,
        load_id="",
        po_number=None,
        shipper="",
        consignee=None
    )
    assert shipment.reference_id is None
    assert shipment.load_id == ""
    assert shipment.po_number is None
    assert shipment.shipper == ""
    assert shipment.consignee is None

def test_reconciliation_equivalent_paths_for_carrier_info():
    # Both should result in the same CarrierInfo object
    c1 = CarrierInfo(carrier_name="CarrierA", mc_number="123")
    c2 = CarrierInfo.parse_obj({"carrier_name": "CarrierA", "mc_number": "123"})
    assert c1 == c2

def test_reconciliation_equivalent_paths_for_shipment_data():
    # Both should result in the same ShipmentData object
    s1 = ShipmentData(reference_id="R1", carrier=CarrierInfo(carrier_name="C"))
    s2 = ShipmentData.parse_obj({"reference_id": "R1", "carrier": {"carrier_name": "C"}})
    assert s1 == s2

def test_reconciliation_equivalent_paths_for_extraction_response():
    s = ShipmentData(reference_id="R2")
    e1 = ExtractionResponse(data=s, document_id="D1")
    e2 = ExtractionResponse.parse_obj({"data": {"reference_id": "R2"}, "document_id": "D1"})
    assert e1 == e2
