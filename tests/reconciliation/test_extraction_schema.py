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
            mc_number="MC123456",
            phone="555-1234",
            email="carrier@example.com"
        ),
        driver=DriverInfo(
            driver_name="John Doe",
            cell_number="555-5678",
            truck_number="TRK100",
            trailer_number="TRL200"
        ),
        pickup=Location(
            name="Warehouse A",
            address="123 Main St",
            city="Metropolis",
            state="NY",
            zip_code="10001",
            country="USA",
            appointment_time="2024-06-01T08:00:00Z",
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
        equipment_type="Van",
        equipment_size="53",
        load_type="FTL",
        commodities=[
            CommodityItem(
                commodity_name="Widgets",
                weight="56000.00 lbs",
                quantity="10000 units",
                description="Blue widgets"
            ),
            CommodityItem(
                commodity_name="Gadgets",
                weight="12000.00 lbs",
                quantity="2000 units",
                description="Red gadgets"
            )
        ],
        rate_info=RateInfo(
            total_rate=2500.00,
            currency="USD",
            rate_breakdown={"base": 2000.00, "fuel": 500.00}
        ),
        special_instructions="Handle with care.",
        shipper_instructions="Call before arrival.",
        carrier_instructions="No partial loads.",
        dispatcher_name="Alice Smith",
        dispatcher_phone="555-9999",
        dispatcher_email="alice@dispatch.com",
        additional_data={"custom_field": "custom_value"}
    )

def make_minimal_shipment_data():
    return ShipmentData()

def test_extraction_response_happy_path_and_equivalent_paths():
    # Happy path: all fields populated
    shipment_data = make_full_shipment_data()
    response1 = ExtractionResponse(data=shipment_data, document_id="DOC123")
    response2 = ExtractionResponse(data=shipment_data, document_id="DOC123")
    # Reconciliation: outputs should be equivalent for same input
    assert response1 == response2
    assert response1.data == response2.data
    assert response1.document_id == response2.document_id

def test_extraction_response_minimal_and_equivalent_paths():
    # Minimal path: only required fields (none required, so all optional)
    shipment_data = make_minimal_shipment_data()
    response1 = ExtractionResponse(data=shipment_data)
    response2 = ExtractionResponse(data=shipment_data, document_id=None)
    # Reconciliation: outputs should be equivalent for explicit None and omitted
    assert response1 == response2
    assert response1.data == response2.data
    assert response1.document_id == response2.document_id

def test_shipment_data_partial_fields_and_equivalent_paths():
    # Only a subset of fields populated
    shipment_data1 = ShipmentData(
        reference_id="REF1",
        shipper="Shipper Only",
        pickup=Location(city="Smallville"),
        commodities=[CommodityItem(commodity_name="Thing")]
    )
    shipment_data2 = ShipmentData(
        reference_id="REF1",
        shipper="Shipper Only",
        pickup=Location(city="Smallville"),
        commodities=[CommodityItem(commodity_name="Thing")]
    )
    # Reconciliation: outputs should be equivalent for same partial input
    assert shipment_data1 == shipment_data2

def test_location_and_equivalent_paths():
    loc1 = Location(name="Loc", city="City", country="USA")
    loc2 = Location(name="Loc", city="City", country="USA")
    # Reconciliation: outputs should be equivalent for same input
    assert loc1 == loc2

def test_commodity_item_and_equivalent_paths():
    item1 = CommodityItem(commodity_name="Widget", weight="100 lbs")
    item2 = CommodityItem(commodity_name="Widget", weight="100 lbs")
    # Reconciliation: outputs should be equivalent for same input
    assert item1 == item2

def test_carrier_info_and_equivalent_paths():
    carrier1 = CarrierInfo(carrier_name="Carrier", mc_number="MC1")
    carrier2 = CarrierInfo(carrier_name="Carrier", mc_number="MC1")
    # Reconciliation: outputs should be equivalent for same input
    assert carrier1 == carrier2

def test_driver_info_and_equivalent_paths():
    driver1 = DriverInfo(driver_name="Driver", cell_number="555-0000")
    driver2 = DriverInfo(driver_name="Driver", cell_number="555-0000")
    # Reconciliation: outputs should be equivalent for same input
    assert driver1 == driver2

def test_rate_info_and_equivalent_paths():
    rate1 = RateInfo(total_rate=100.0, currency="USD", rate_breakdown={"base": 80.0, "fuel": 20.0})
    rate2 = RateInfo(total_rate=100.0, currency="USD", rate_breakdown={"base": 80.0, "fuel": 20.0})
    # Reconciliation: outputs should be equivalent for same input
    assert rate1 == rate2

def test_shipment_data_boundary_conditions_empty_lists_and_dicts():
    # Edge: commodities as empty list, rate_breakdown as empty dict
    shipment_data1 = ShipmentData(
        commodities=[],
        rate_info=RateInfo(rate_breakdown={})
    )
    shipment_data2 = ShipmentData(
        commodities=[],
        rate_info=RateInfo(rate_breakdown={})
    )
    assert shipment_data1 == shipment_data2

def test_shipment_data_boundary_conditions_large_numbers():
    # Edge: very large numbers in rate and quantities
    shipment_data1 = ShipmentData(
        commodities=[CommodityItem(quantity="999999999 units")],
        rate_info=RateInfo(total_rate=1e12)
    )
    shipment_data2 = ShipmentData(
        commodities=[CommodityItem(quantity="999999999 units")],
        rate_info=RateInfo(total_rate=1e12)
    )
    assert shipment_data1 == shipment_data2

def test_shipment_data_error_handling_invalid_types():
    # Error: invalid type for total_rate (should be float)
    with pytest.raises(ValidationError):
        RateInfo(total_rate="not_a_float")

    # Error: invalid type for commodities (should be list)
    with pytest.raises(ValidationError):
        ShipmentData(commodities="not_a_list")

    # Error: invalid type for rate_breakdown (should be dict)
    with pytest.raises(ValidationError):
        RateInfo(rate_breakdown="not_a_dict")

def test_shipment_data_error_handling_missing_required_data():
    # Error: ExtractionResponse requires data
    with pytest.raises(ValidationError):
        ExtractionResponse()

def test_equivalent_paths_with_none_and_missing_fields():
    # Reconciliation: None and missing fields should be equivalent for optional fields
    loc1 = Location()
    loc2 = Location(name=None, address=None, city=None, state=None, zip_code=None, country=None, appointment_time=None, po_number=None)
    assert loc1 == loc2

    item1 = CommodityItem()
    item2 = CommodityItem(commodity_name=None, weight=None, quantity=None, description=None)
    assert item1 == item2

    carrier1 = CarrierInfo()
    carrier2 = CarrierInfo(carrier_name=None, mc_number=None, phone=None, email=None)
    assert carrier1 == carrier2

    driver1 = DriverInfo()
    driver2 = DriverInfo(driver_name=None, cell_number=None, truck_number=None, trailer_number=None)
    assert driver1 == driver2

    rate1 = RateInfo()
    rate2 = RateInfo(total_rate=None, currency=None, rate_breakdown=None)
    assert rate1 == rate2

def test_shipment_data_additional_data_equivalent_paths():
    # Reconciliation: additional_data as None and missing should be equivalent
    shipment1 = ShipmentData()
    shipment2 = ShipmentData(additional_data=None)
    assert shipment1 == shipment2

def test_shipment_data_additional_data_content_equivalence():
    # Reconciliation: additional_data with same content
    shipment1 = ShipmentData(additional_data={"foo": "bar"})
    shipment2 = ShipmentData(additional_data={"foo": "bar"})
    assert shipment1 == shipment2

def test_extraction_response_document_id_equivalence():
    # Reconciliation: document_id None and missing
    shipment = make_full_shipment_data()
    resp1 = ExtractionResponse(data=shipment)
    resp2 = ExtractionResponse(data=shipment, document_id=None)
    assert resp1 == resp2

def test_shipment_data_commutative_equality():
    # Regression: order of fields does not affect equality
    shipment1 = ShipmentData(reference_id="A", load_id="B")
    shipment2 = ShipmentData(load_id="B", reference_id="A")
    assert shipment1 == shipment2

def test_shipment_data_repr_and_dict_equivalence():
    # Reconciliation: dict and repr outputs for equivalent objects
    shipment = make_full_shipment_data()
    shipment2 = make_full_shipment_data()
    assert shipment.dict() == shipment2.dict()
    assert repr(shipment) == repr(shipment2)
