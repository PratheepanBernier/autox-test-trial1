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
    carrier = CarrierInfo(carrier_name="", mc_number=None, phone=" ", email=None)
    assert carrier.carrier_name == ""
    assert carrier.mc_number is None
    assert carrier.phone == " "
    assert carrier.email is None

def test_driver_info_boundary_conditions():
    driver = DriverInfo(driver_name="A", cell_number="1", truck_number="T"*100, trailer_number=None)
    assert driver.driver_name == "A"
    assert driver.cell_number == "1"
    assert driver.truck_number == "T"*100
    assert driver.trailer_number is None

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

def test_shipment_data_full_happy_path():
    shipment = ShipmentData(
        reference_id="REF123",
        load_id="LOAD456",
        po_number="PO789",
        shipper="Shipper Inc.",
        consignee="Consignee LLC",
        carrier=CarrierInfo(carrier_name="CarrierX", mc_number="MC123", phone="555-1234", email="carrier@example.com"),
        driver=DriverInfo(driver_name="John Doe", cell_number="555-5678", truck_number="TRK123", trailer_number="TRL456"),
        pickup=Location(name="Origin", address="1 Origin St", city="Start", state="ST", zip_code="00001", country="USA"),
        drop=Location(name="Destination", address="2 Dest Ave", city="End", state="EN", zip_code="99999", country="USA"),
        shipping_date="2024-06-01",
        delivery_date="2024-06-02",
        created_on="2024-05-31",
        booking_date="2024-05-30",
        equipment_type="Van",
        equipment_size="53",
        load_type="FTL",
        commodities=[
            CommodityItem(commodity_name="Widgets", weight="10000 lbs", quantity="1000 units", description="Blue widgets"),
            CommodityItem(commodity_name="Gadgets", weight="5000 lbs", quantity="500 units", description="Red gadgets")
        ],
        rate_info=RateInfo(total_rate=1500.0, currency="USD", rate_breakdown={"base": 1400.0, "fuel": 100.0}),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No weekend delivery",
        dispatcher_name="Jane Smith",
        dispatcher_phone="555-9999",
        dispatcher_email="dispatcher@example.com",
        additional_data={"custom_field": "custom_value"}
    )
    assert shipment.reference_id == "REF123"
    assert shipment.carrier.carrier_name == "CarrierX"
    assert shipment.commodities[0].commodity_name == "Widgets"
    assert shipment.rate_info.total_rate == 1500.0
    assert shipment.additional_data["custom_field"] == "custom_value"

def test_shipment_data_minimal_fields():
    shipment = ShipmentData()
    for field in shipment.__fields__:
        assert getattr(shipment, field) is None

def test_shipment_data_commodities_empty_list():
    shipment = ShipmentData(commodities=[])
    assert shipment.commodities == []

def test_shipment_data_additional_data_various_types():
    shipment = ShipmentData(additional_data={"int": 1, "float": 2.5, "list": [1,2], "dict": {"a": "b"}})
    assert shipment.additional_data["int"] == 1
    assert shipment.additional_data["float"] == 2.5
    assert shipment.additional_data["list"] == [1,2]
    assert shipment.additional_data["dict"] == {"a": "b"}

def test_extraction_response_happy_path():
    shipment = ShipmentData(reference_id="REF1")
    response = ExtractionResponse(data=shipment, document_id="DOC123")
    assert response.data.reference_id == "REF1"
    assert response.document_id == "DOC123"

def test_extraction_response_document_id_none():
    shipment = ShipmentData(reference_id="REF2")
    response = ExtractionResponse(data=shipment)
    assert response.data.reference_id == "REF2"
    assert response.document_id is None

def test_extraction_response_invalid_data_type_raises():
    with pytest.raises(ValidationError):
        ExtractionResponse(data="not a shipment data object")

def test_shipment_data_invalid_commodities_type_raises():
    with pytest.raises(ValidationError):
        ShipmentData(commodities="not a list")

def test_rate_info_invalid_total_rate_type_raises():
    with pytest.raises(ValidationError):
        RateInfo(total_rate="not a float")

def test_location_extra_fields_are_ignored():
    loc = Location(name="A", extra_field="should be ignored")
    assert loc.name == "A"
    assert not hasattr(loc, "extra_field")

def test_shipment_data_extra_fields_are_ignored():
    shipment = ShipmentData(reference_id="REF", unknown_field="ignored")
    assert shipment.reference_id == "REF"
    assert not hasattr(shipment, "unknown_field")

def test_commodity_item_empty_string_fields():
    item = CommodityItem(commodity_name="", weight="", quantity="", description="")
    assert item.commodity_name == ""
    assert item.weight == ""
    assert item.quantity == ""
    assert item.description == ""

def test_rate_info_rate_breakdown_empty_dict():
    rate = RateInfo(rate_breakdown={})
    assert rate.rate_breakdown == {}

def test_shipment_data_none_nested_models():
    shipment = ShipmentData(carrier=None, driver=None, pickup=None, drop=None, rate_info=None)
    assert shipment.carrier is None
    assert shipment.driver is None
    assert shipment.pickup is None
    assert shipment.drop is None
    assert shipment.rate_info is None

def test_shipment_data_nested_models_partial():
    shipment = ShipmentData(
        carrier=CarrierInfo(carrier_name="CarrierOnly"),
        driver=DriverInfo(),
        pickup=Location(city="OnlyCity"),
        drop=None,
        rate_info=RateInfo(currency="EUR")
    )
    assert shipment.carrier.carrier_name == "CarrierOnly"
    assert shipment.driver.driver_name is None
    assert shipment.pickup.city == "OnlyCity"
    assert shipment.drop is None
    assert shipment.rate_info.currency == "EUR"

def test_shipment_data_reconciliation_equivalent_paths():
    # Test that ShipmentData with equivalent nested dicts and models produce the same output
    dict_input = {
        "reference_id": "REFX",
        "carrier": {"carrier_name": "CarrierY"},
        "commodities": [{"commodity_name": "Item1"}],
    }
    model_input = ShipmentData(
        reference_id="REFX",
        carrier=CarrierInfo(carrier_name="CarrierY"),
        commodities=[CommodityItem(commodity_name="Item1")]
    )
    shipment_from_dict = ShipmentData(**dict_input)
    assert shipment_from_dict == model_input

def test_shipment_data_additional_data_none_and_empty():
    shipment_none = ShipmentData(additional_data=None)
    shipment_empty = ShipmentData(additional_data={})
    assert shipment_none.additional_data is None
    assert shipment_empty.additional_data == {}

def test_location_zip_code_boundary():
    loc = Location(zip_code="00000")
    assert loc.zip_code == "00000"
    loc2 = Location(zip_code="99999")
    assert loc2.zip_code == "99999"

def test_location_country_case_sensitivity():
    loc = Location(country="usa")
    assert loc.country == "usa"
    loc2 = Location(country="USA")
    assert loc2.country == "USA"
