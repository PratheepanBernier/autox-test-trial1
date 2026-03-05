# source_hash: 88361007730f06bf
import pytest
from unittest.mock import patch, MagicMock, create_autospec
from backend.src.services.extraction import ExtractionService
from models.extraction_schema import ShipmentData, ExtractionResponse
from models.schemas import Chunk, DocumentMetadata
import logging

@pytest.fixture(autouse=True)
def patch_logger(monkeypatch):
    # Prevent actual logging during tests
    monkeypatch.setattr(logging, "getLogger", lambda name=None: MagicMock())

@pytest.fixture
def mock_settings(monkeypatch):
    class DummySettings:
        QA_MODEL = "test-model"
        GROQ_API_KEY = "test-key"
    monkeypatch.setattr("core.config.settings", DummySettings())

@pytest.fixture
def mock_llm(monkeypatch):
    # Patch ChatGroq and its usage in ExtractionService
    mock_llm_instance = MagicMock()
    monkeypatch.setattr("langchain_groq.ChatGroq", lambda **kwargs: mock_llm_instance)
    return mock_llm_instance

@pytest.fixture
def mock_prompt(monkeypatch):
    mock_prompt_instance = MagicMock()
    monkeypatch.setattr("langchain_core.prompts.ChatPromptTemplate.from_template", lambda template: mock_prompt_instance)
    return mock_prompt_instance

@pytest.fixture
def mock_parser(monkeypatch):
    mock_parser_instance = MagicMock()
    mock_parser_instance.get_format_instructions.return_value = "FORMAT"
    monkeypatch.setattr("langchain_core.output_parsers.PydanticOutputParser", lambda pydantic_object: mock_parser_instance)
    return mock_parser_instance

@pytest.fixture
def extraction_service(mock_settings, mock_llm, mock_prompt, mock_parser):
    return ExtractionService()

def make_shipment_data(**kwargs):
    # Helper to create ShipmentData with defaults
    data = ShipmentData()
    for k, v in kwargs.items():
        setattr(data, k, v)
    return data

def test_extract_data_happy_path(extraction_service, mock_prompt, mock_llm, mock_parser):
    # Setup
    shipment_data = make_shipment_data(reference_id="REF123", load_id="LOAD456")
    mock_parser.invoke.return_value = shipment_data
    chain = MagicMock()
    chain.invoke.return_value = shipment_data
    # Compose the chain: prompt | llm | parser
    mock_prompt.__or__.return_value = mock_llm
    mock_llm.__or__.return_value = mock_parser
    mock_parser.__or__.side_effect = None  # Not used
    # Patch chain.invoke
    mock_parser.invoke = chain.invoke

    # Act
    result = extraction_service.extract_data("test text", filename="doc1.txt")

    # Assert
    assert isinstance(result, ExtractionResponse)
    assert result.data.reference_id == "REF123"
    assert result.data.load_id == "LOAD456"
    assert result.document_id == "doc1.txt"

def test_extract_data_empty_text_returns_empty_shipment(extraction_service, mock_prompt, mock_llm, mock_parser):
    # Setup: simulate LLM returning empty/None
    mock_parser.invoke.side_effect = Exception("No data extracted")
    mock_prompt.__or__.return_value = mock_llm
    mock_llm.__or__.return_value = mock_parser

    # Act
    result = extraction_service.extract_data("", filename="empty.txt")

    # Assert: should return empty ShipmentData
    assert isinstance(result, ExtractionResponse)
    assert isinstance(result.data, ShipmentData)
    assert result.document_id == "empty.txt"
    # All fields should be default/None
    for field in result.data.__fields__:
        assert getattr(result.data, field) in (None, [], {})

def test_extract_data_handles_llm_exception_and_returns_empty(extraction_service, mock_prompt, mock_llm, mock_parser):
    # Setup: simulate LLM raising error
    mock_parser.invoke.side_effect = Exception("LLM error")
    mock_prompt.__or__.return_value = mock_llm
    mock_llm.__or__.return_value = mock_parser

    # Act
    result = extraction_service.extract_data("irrelevant", filename="fail.txt")

    # Assert
    assert isinstance(result, ExtractionResponse)
    assert isinstance(result.data, ShipmentData)
    assert result.document_id == "fail.txt"
    for field in result.data.__fields__:
        assert getattr(result.data, field) in (None, [], {})

def test_format_extraction_as_text_full_fields(extraction_service):
    # Setup: fill all fields
    from models.extraction_schema import Carrier, Driver, Location, Commodity, RateInfo
    data = ShipmentData(
        reference_id="REF1",
        load_id="LID2",
        po_number="PO3",
        shipper="ShipperX",
        consignee="ConsigneeY",
        carrier=Carrier(carrier_name="CarrierZ", mc_number="MC123", phone="555-1111"),
        driver=Driver(driver_name="John Doe", cell_number="555-2222", truck_number="TRK9"),
        pickup=Location(name="Warehouse A", address="123 St", city="CityA", state="ST", zip="12345", appointment_time="2024-01-01T09:00"),
        drop=Location(name="Store B", address="456 Ave", city="CityB", state="TS", zip="67890"),
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Van",
        equipment_size=53,
        load_type="Full",
        commodities=[Commodity(commodity_name="Widgets", weight="1000 lbs", quantity="10", description="Blue widgets")],
        rate_info=RateInfo(total_rate=1200, currency="USD", rate_breakdown={"linehaul": 1000, "fuel": 200}),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No tarps",
        dispatcher_name="Jane Smith",
        dispatcher_phone="555-3333"
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")

    # Act
    text = extraction_service.format_extraction_as_text(extraction)

    # Assert
    assert "Reference ID: REF1" in text
    assert "Load ID: LID2" in text
    assert "PO Number: PO3" in text
    assert "Shipper: ShipperX" in text
    assert "Consignee: ConsigneeY" in text
    assert "Carrier Name: CarrierZ" in text
    assert "MC Number: MC123" in text
    assert "Carrier Phone: 555-1111" in text
    assert "Driver Name: John Doe" in text
    assert "Driver Phone: 555-2222" in text
    assert "Truck Number: TRK9" in text
    assert "Pickup Location: Warehouse A" in text
    assert "Pickup City: CityA, ST" in text
    assert "Pickup Appointment: 2024-01-01T09:00" in text
    assert "Drop Location: Store B" in text
    assert "Drop City: CityB, TS" in text
    assert "Shipping Date: 2024-01-01" in text
    assert "Delivery Date: 2024-01-02" in text
    assert "Equipment Type: Van" in text
    assert "Equipment Size: 53 feet" in text
    assert "Load Type: Full" in text
    assert "Commodities:" in text
    assert "1. Widgets" in text
    assert "Weight: 1000 lbs" in text
    assert "Quantity: 10" in text
    assert "Total Rate: $1200 USD" in text
    assert "Rate Breakdown:" in text
    assert "Special Instructions: Handle with care" in text
    assert "Shipper Instructions: Call before arrival" in text
    assert "Carrier Instructions: No tarps" in text
    assert "Dispatcher: Jane Smith" in text
    assert "Dispatcher Phone: 555-3333" in text

def test_format_extraction_as_text_minimal_fields(extraction_service):
    # Only one field set
    data = ShipmentData(reference_id="ONLYREF")
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Reference ID: ONLYREF" in text
    # Should not contain other fields
    assert "Load ID:" not in text
    assert "Shipper:" not in text
    assert "Carrier Name:" not in text
    assert "Commodities:" not in text

def test_format_extraction_as_text_handles_nulls_and_lists(extraction_service):
    # No fields set, commodities is empty list
    data = ShipmentData()
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    # Should only contain the header
    assert text.strip() == "=== EXTRACTED STRUCTURED DATA ==="

def test_create_structured_chunk_produces_expected_chunk(extraction_service):
    # Setup
    data = ShipmentData(reference_id="R1")
    extraction = ExtractionResponse(data=data, document_id="file.txt")
    chunk = extraction_service.create_structured_chunk(extraction, filename="file.txt")
    assert isinstance(chunk, Chunk)
    assert isinstance(chunk.metadata, DocumentMetadata)
    assert chunk.metadata.filename == "file.txt"
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.source == "file.txt - Extracted Data"
    assert chunk.metadata.chunk_type == "structured_data"
    assert "Reference ID: R1" in chunk.text

def test_extraction_service_init_success(monkeypatch, mock_settings):
    # Patch all dependencies to not raise
    monkeypatch.setattr("langchain_groq.ChatGroq", lambda **kwargs: MagicMock())
    monkeypatch.setattr("langchain_core.prompts.ChatPromptTemplate.from_template", lambda template: MagicMock())
    monkeypatch.setattr("langchain_core.output_parsers.PydanticOutputParser", lambda pydantic_object: MagicMock())
    # Should not raise
    service = ExtractionService()
    assert hasattr(service, "llm")
    assert hasattr(service, "parser")
    assert hasattr(service, "extraction_prompt")

def test_extraction_service_init_failure(monkeypatch, mock_settings):
    # Simulate failure in LLM init
    def fail_llm(**kwargs):
        raise RuntimeError("fail")
    monkeypatch.setattr("langchain_groq.ChatGroq", fail_llm)
    monkeypatch.setattr("langchain_core.prompts.ChatPromptTemplate.from_template", lambda template: MagicMock())
    monkeypatch.setattr("langchain_core.output_parsers.PydanticOutputParser", lambda pydantic_object: MagicMock())
    with pytest.raises(RuntimeError):
        ExtractionService()

def test_format_extraction_as_text_boundary_equipment_size(extraction_service):
    # Test boundary value for equipment_size (0 and very large)
    data = ShipmentData(equipment_size=0)
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Equipment Size: 0 feet" in text

    data2 = ShipmentData(equipment_size=1000)
    extraction2 = ExtractionResponse(data=data2, document_id="doc.txt")
    text2 = extraction_service.format_extraction_as_text(extraction2)
    assert "Equipment Size: 1000 feet" in text2

def test_format_extraction_as_text_multiple_commodities(extraction_service):
    from models.extraction_schema import Commodity
    data = ShipmentData(
        commodities=[
            Commodity(commodity_name="A", weight="10", quantity="1"),
            Commodity(commodity_name="B", weight="20", quantity="2"),
        ]
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "1. A" in text
    assert "2. B" in text
    assert "Weight: 10" in text
    assert "Weight: 20" in text
    assert "Quantity: 1" in text
    assert "Quantity: 2" in text

def test_format_extraction_as_text_pickup_and_drop_fallbacks(extraction_service):
    from models.extraction_schema import Location
    # Only address set for pickup, only address set for drop
    data = ShipmentData(
        pickup=Location(address="123 Main St"),
        drop=Location(address="456 Elm St")
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Pickup Location: 123 Main St" in text
    assert "Drop Location: 456 Elm St" in text

def test_format_extraction_as_text_pickup_and_drop_na(extraction_service):
    from models.extraction_schema import Location
    # No name or address
    data = ShipmentData(
        pickup=Location(),
        drop=Location()
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Pickup Location: N/A" in text
    assert "Drop Location: N/A" in text

def test_format_extraction_as_text_rate_info_defaults(extraction_service):
    from models.extraction_schema import RateInfo
    # Only total_rate set, no currency or breakdown
    data = ShipmentData(rate_info=RateInfo(total_rate=500))
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Total Rate: $500 USD" in text
    assert "Rate Breakdown:" not in text

def test_format_extraction_as_text_rate_info_with_breakdown(extraction_service):
    from models.extraction_schema import RateInfo
    data = ShipmentData(rate_info=RateInfo(total_rate=700, currency="EUR", rate_breakdown={"base": 600, "fuel": 100}))
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Total Rate: $700 EUR" in text
    assert "Rate Breakdown:" in text
    assert '"base": 600' in text
    assert '"fuel": 100' in text

def test_format_extraction_as_text_instructions_variants(extraction_service):
    data = ShipmentData(
        special_instructions="Spec",
        shipper_instructions="Ship",
        carrier_instructions="Carr"
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Special Instructions: Spec" in text
    assert "Shipper Instructions: Ship" in text
    assert "Carrier Instructions: Carr" in text

def test_format_extraction_as_text_dispatcher_variants(extraction_service):
    data = ShipmentData(
        dispatcher_name="Disp",
        dispatcher_phone="123-456"
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Dispatcher: Disp" in text
    assert "Dispatcher Phone: 123-456" in text

def test_format_extraction_as_text_no_dispatcher_phone(extraction_service):
    data = ShipmentData(
        dispatcher_name="Disp"
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Dispatcher: Disp" in text
    assert "Dispatcher Phone:" not in text
