# source_hash: c0f4f32a02e6e1f2
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

def make_full_shipment_data():
    return ShipmentData(
        reference_id="REF123",
        load_id="LOAD456",
        po_number="PO789",
        shipper="Shipper Inc.",
        consignee="Consignee LLC",
        carrier=CarrierInfo(
            carrier_name="Carrier Co.",
            mc_number="MC12345",
            phone="555-1234",
            email="carrier@example.com"
        ),
        driver=DriverInfo(
            driver_name="John Doe",
            cell_number="555-5678",
            truck_number="TRK123",
            trailer_number="TRL456"
        ),
        pickup=Location(
            name="Warehouse A",
            address="123 Main St",
            city="Metropolis",
            state="NY",
            zip_code="10001",
            country="USA",
            appointment_time="2024-06-01T09:00:00Z",
            po_number="PO789"
        ),
        drop=Location(
            name="Warehouse B",
            address="456 Elm St",
            city="Gotham",
            state="NJ",
            zip_code="07001",
            country="USA",
            appointment_time="2024-06-02T10:00:00Z",
            po_number="PO789"
        ),
        shipping_date="2024-06-01",
        delivery_date="2024-06-02",
        created_on="2024-05-31",
        booking_date="2024-05-30",
        equipment_type="Flatbed",
        equipment_size="53",
        load_type="FTL",
        commodities=[
            CommodityItem(
                commodity_name="Steel",
                weight="56000.00 lbs",
                quantity="10000 units",
                description="Steel rods"
            ),
            CommodityItem(
                commodity_name="Copper",
                weight="12000.00 lbs",
                quantity="2000 units",
                description="Copper wires"
            )
        ],
        rate_info=RateInfo(
            total_rate=2500.0,
            currency="USD",
            rate_breakdown={"base": 2000.0, "fuel": 500.0}
        ),
        special_instructions="Handle with care.",
        shipper_instructions="Call before arrival.",
        carrier_instructions="No overnight parking.",
        dispatcher_name="Alice Smith",
        dispatcher_phone="555-9999",
        dispatcher_email="alice@dispatch.com",
        additional_data={"custom_field": "custom_value"}
    )

def make_minimal_shipment_data():
    return ShipmentData()

def test_extraction_response_happy_path_and_equivalent_paths():
    # Happy path: full data
    shipment_data = make_full_shipment_data()
    response1 = ExtractionResponse(data=shipment_data, document_id="DOC123")
    response2 = ExtractionResponse(data=shipment_data, document_id="DOC123")
    assert response1 == response2
    assert response1.data.reference_id == "REF123"
    assert response1.document_id == "DOC123"
    # Reconciliation: serialization/deserialization equivalence
    serialized = response1.json()
    deserialized = ExtractionResponse.parse_raw(serialized)
    assert deserialized == response1

def test_extraction_response_minimal_and_equivalent_paths():
    # Minimal data: only required field (data)
    shipment_data = make_minimal_shipment_data()
    response1 = ExtractionResponse(data=shipment_data)
    response2 = ExtractionResponse(data=shipment_data, document_id=None)
    assert response1 == response2
    assert response1.data.reference_id is None
    # Reconciliation: dict and model equivalence
    as_dict = response1.dict()
    from_dict = ExtractionResponse(**as_dict)
    assert from_dict == response1

def test_location_model_equivalence_and_edge_cases():
    # All fields None
    loc1 = Location()
    loc2 = Location()
    assert loc1 == loc2
    # Partial fields
    loc3 = Location(name="A", city="B")
    loc4 = Location(name="A", city="B")
    assert loc3 == loc4
    # Reconciliation: dict and model
    as_dict = loc3.dict()
    from_dict = Location(**as_dict)
    assert from_dict == loc3

def test_commodity_item_equivalence_and_boundary_conditions():
    # Empty
    item1 = CommodityItem()
    item2 = CommodityItem()
    assert item1 == item2
    # Boundary: long strings
    long_str = "x" * 1000
    item3 = CommodityItem(commodity_name=long_str, weight=long_str, quantity=long_str, description=long_str)
    item4 = CommodityItem(commodity_name=long_str, weight=long_str, quantity=long_str, description=long_str)
    assert item3 == item4
    # Reconciliation: dict and model
    as_dict = item3.dict()
    from_dict = CommodityItem(**as_dict)
    assert from_dict == item3

def test_carrier_info_and_driver_info_equivalence():
    carrier = CarrierInfo(carrier_name="Carrier", mc_number="MC", phone="123", email="a@b.com")
    driver = DriverInfo(driver_name="Driver", cell_number="456", truck_number="TRK", trailer_number="TRL")
    # Reconciliation: dict and model
    assert CarrierInfo(**carrier.dict()) == carrier
    assert DriverInfo(**driver.dict()) == driver

def test_rate_info_equivalence_and_edge_cases():
    # All None
    rate1 = RateInfo()
    rate2 = RateInfo()
    assert rate1 == rate2
    # Only total_rate
    rate3 = RateInfo(total_rate=0.0)
    rate4 = RateInfo(total_rate=0.0)
    assert rate3 == rate4
    # rate_breakdown with empty dict
    rate5 = RateInfo(rate_breakdown={})
    rate6 = RateInfo(rate_breakdown={})
    assert rate5 == rate6
    # Reconciliation: dict and model
    assert RateInfo(**rate3.dict()) == rate3

def test_shipment_data_equivalence_and_additional_data_edge_cases():
    # All None
    data1 = ShipmentData()
    data2 = ShipmentData()
    assert data1 == data2
    # additional_data with nested dict
    data3 = ShipmentData(additional_data={"foo": {"bar": 1}})
    data4 = ShipmentData(additional_data={"foo": {"bar": 1}})
    assert data3 == data4
    # Reconciliation: dict and model
    assert ShipmentData(**data3.dict()) == data3

def test_extraction_response_error_handling_missing_data():
    # data is required
    with pytest.raises(ValidationError):
        ExtractionResponse()

def test_shipment_data_error_handling_invalid_types():
    # Invalid type for commodities
    with pytest.raises(ValidationError):
        ShipmentData(commodities="notalist")
    # Invalid type for rate_info
    with pytest.raises(ValidationError):
        ShipmentData(rate_info="notadict")
    # Invalid type for additional_data
    with pytest.raises(ValidationError):
        ShipmentData(additional_data="notadict")

def test_location_field_boundary_conditions():
    # zip_code: empty string vs None
    loc1 = Location(zip_code="")
    loc2 = Location(zip_code=None)
    assert loc1 != loc2
    # country: whitespace
    loc3 = Location(country=" ")
    loc4 = Location(country=" ")
    assert loc3 == loc4

def test_rate_info_currency_boundary_conditions():
    # currency: empty string vs None
    rate1 = RateInfo(currency="")
    rate2 = RateInfo(currency=None)
    assert rate1 != rate2

def test_commodity_item_quantity_weight_edge_cases():
    # quantity and weight as numbers (should fail)
    with pytest.raises(ValidationError):
        CommodityItem(quantity=100, weight=200)
    # quantity and weight as strings (should pass)
    item = CommodityItem(quantity="100 units", weight="200 lbs")
    assert item.quantity == "100 units"
    assert item.weight == "200 lbs"

def test_shipment_data_commodities_empty_list_vs_none():
    # commodities=None vs commodities=[]
    data1 = ShipmentData(commodities=None)
    data2 = ShipmentData(commodities=[])
    assert data1 != data2
    # But both are valid

def test_extraction_response_document_id_none_vs_missing():
    shipment_data = make_full_shipment_data()
    resp1 = ExtractionResponse(data=shipment_data)
    resp2 = ExtractionResponse(data=shipment_data, document_id=None)
    assert resp1 == resp2

def test_shipment_data_partial_fields_and_equivalence():
    # Only a few fields set
    data1 = ShipmentData(reference_id="A", shipper="B")
    data2 = ShipmentData(reference_id="A", shipper="B")
    assert data1 == data2
    # Reconciliation: dict and model
    assert ShipmentData(**data1.dict()) == data1
