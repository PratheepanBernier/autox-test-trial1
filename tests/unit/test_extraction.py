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
        QA_MODEL = "dummy-model"
        GROQ_API_KEY = "dummy-key"
    monkeypatch.setattr("core.config.settings", DummySettings())

@pytest.fixture
def mock_groq(monkeypatch):
    mock_llm = MagicMock()
    monkeypatch.setattr("langchain_groq.ChatGroq", lambda **kwargs: mock_llm)
    return mock_llm

@pytest.fixture
def mock_prompt(monkeypatch):
    mock_prompt = MagicMock()
    monkeypatch.setattr("langchain_core.prompts.ChatPromptTemplate.from_template", lambda template: mock_prompt)
    return mock_prompt

@pytest.fixture
def mock_parser(monkeypatch):
    mock_parser = MagicMock()
    mock_parser.get_format_instructions.return_value = "FORMAT"
    monkeypatch.setattr("langchain_core.output_parsers.PydanticOutputParser", lambda pydantic_object: mock_parser)
    return mock_parser

@pytest.fixture
def extraction_service(mock_settings, mock_groq, mock_prompt, mock_parser):
    return ExtractionService()

def make_shipment_data(**kwargs):
    # Helper to create a ShipmentData with defaults and overrides
    data = ShipmentData()
    for k, v in kwargs.items():
        setattr(data, k, v)
    return data

def test_extract_data_happy_path(extraction_service, mock_groq, mock_parser):
    # Arrange
    fake_result = make_shipment_data(reference_id="REF123", load_id="LID456")
    mock_parser.get_format_instructions.return_value = "FORMAT"
    chain = MagicMock()
    chain.invoke.return_value = fake_result
    # Compose the chain: prompt | llm | parser
    extraction_service.extraction_prompt.__or__ = lambda self, other: chain if other is extraction_service.llm else NotImplemented
    extraction_service.llm.__or__ = lambda self, other: chain if other is extraction_service.parser else NotImplemented
    # Act
    result = extraction_service.extract_data("test text", filename="file1.txt")
    # Assert
    assert isinstance(result, ExtractionResponse)
    assert result.data.reference_id == "REF123"
    assert result.data.load_id == "LID456"
    assert result.document_id == "file1.txt"

def test_extract_data_empty_text(extraction_service, mock_groq, mock_parser):
    # Arrange
    fake_result = make_shipment_data()
    mock_parser.get_format_instructions.return_value = "FORMAT"
    chain = MagicMock()
    chain.invoke.return_value = fake_result
    extraction_service.extraction_prompt.__or__ = lambda self, other: chain if other is extraction_service.llm else NotImplemented
    extraction_service.llm.__or__ = lambda self, other: chain if other is extraction_service.parser else NotImplemented
    # Act
    result = extraction_service.extract_data("", filename="empty.txt")
    # Assert
    assert isinstance(result, ExtractionResponse)
    assert result.data.reference_id is None
    assert result.document_id == "empty.txt"

def test_extract_data_chain_raises_exception_returns_empty(extraction_service, mock_groq, mock_parser):
    # Arrange
    mock_parser.get_format_instructions.return_value = "FORMAT"
    chain = MagicMock()
    chain.invoke.side_effect = Exception("Chain failed")
    extraction_service.extraction_prompt.__or__ = lambda self, other: chain if other is extraction_service.llm else NotImplemented
    extraction_service.llm.__or__ = lambda self, other: chain if other is extraction_service.parser else NotImplemented
    # Act
    result = extraction_service.extract_data("bad input", filename="fail.txt")
    # Assert
    assert isinstance(result, ExtractionResponse)
    assert isinstance(result.data, ShipmentData)
    assert result.data.reference_id is None
    assert result.document_id == "fail.txt"

def test_format_extraction_as_text_full_fields(extraction_service):
    # Arrange
    data = make_shipment_data(
        reference_id="REF1",
        load_id="LID2",
        po_number="PO3",
        shipper="ShipperX",
        consignee="ConsigneeY",
        carrier=type("Carrier", (), {"carrier_name": "CarrierZ", "mc_number": "MC123", "phone": "555-1111"})(),
        driver=type("Driver", (), {"driver_name": "John Doe", "cell_number": "555-2222", "truck_number": "TRK9"})(),
        pickup=type("Loc", (), {"name": "Warehouse", "address": "123 St", "city": "CityA", "state": "ST", "appointment_time": "10:00"})(),
        drop=type("Loc", (), {"name": "Store", "address": "456 Ave", "city": "CityB", "state": "TS", "appointment_time": None})(),
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Van",
        equipment_size=53,
        load_type="Full",
        commodities=[
            type("Commodity", (), {"commodity_name": "Widgets", "weight": "1000lb", "quantity": 10})(),
            type("Commodity", (), {"commodity_name": "Gadgets", "weight": None, "quantity": None})()
        ],
        rate_info=type("Rate", (), {"total_rate": 1200, "currency": "USD", "rate_breakdown": {"linehaul": 1000, "fuel": 200}})(),
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
    assert "Pickup Location: Warehouse" in text
    assert "Pickup City: CityA, ST" in text
    assert "Pickup Appointment: 10:00" in text
    assert "Drop Location: Store" in text
    assert "Drop City: CityB, TS" in text
    assert "Shipping Date: 2024-01-01" in text
    assert "Delivery Date: 2024-01-02" in text
    assert "Equipment Type: Van" in text
    assert "Equipment Size: 53 feet" in text
    assert "Load Type: Full" in text
    assert "Commodities:" in text
    assert "1. Widgets" in text
    assert "Weight: 1000lb" in text
    assert "Quantity: 10" in text
    assert "2. Gadgets" in text
    assert "Total Rate: $1200 USD" in text
    assert "Rate Breakdown:" in text
    assert "Special Instructions: Handle with care" in text
    assert "Shipper Instructions: Call before arrival" in text
    assert "Carrier Instructions: No tarps" in text
    assert "Dispatcher: Jane Smith" in text
    assert "Dispatcher Phone: 555-3333" in text

def test_format_extraction_as_text_minimal_fields(extraction_service):
    # Arrange
    data = make_shipment_data()
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert
    assert "EXTRACTED STRUCTURED DATA" in text
    # Should not contain any field lines except the header
    assert text.strip() == "=== EXTRACTED STRUCTURED DATA ==="

def test_format_extraction_as_text_partial_fields(extraction_service):
    # Arrange
    data = make_shipment_data(reference_id="REFX", shipper="ShipperY")
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert
    assert "Reference ID: REFX" in text
    assert "Shipper: ShipperY" in text
    assert "Load ID:" not in text
    assert "Consignee:" not in text

def test_create_structured_chunk_returns_chunk_with_expected_metadata_and_text(extraction_service):
    # Arrange
    data = make_shipment_data(reference_id="RID")
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    # Act
    chunk = extraction_service.create_structured_chunk(extraction, filename="doc.txt")
    # Assert
    assert isinstance(chunk, Chunk)
    assert isinstance(chunk.metadata, DocumentMetadata)
    assert chunk.metadata.filename == "doc.txt"
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.source == "doc.txt - Extracted Data"
    assert chunk.metadata.chunk_type == "structured_data"
    assert "EXTRACTED STRUCTURED DATA" in chunk.text
    assert "Reference ID: RID" in chunk.text

def test_extraction_service_init_success(monkeypatch, mock_settings, mock_groq, mock_prompt, mock_parser):
    # Should not raise
    ExtractionService()

def test_extraction_service_init_failure(monkeypatch):
    # Simulate failure in LLM init
    def fail_llm(*args, **kwargs):
        raise RuntimeError("fail")
    monkeypatch.setattr("langchain_groq.ChatGroq", fail_llm)
    with pytest.raises(RuntimeError):
        ExtractionService()

def test_format_extraction_as_text_handles_null_and_empty_fields(extraction_service):
    # Arrange: All fields set to None or empty
    data = make_shipment_data(
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
        equipment_type=None,
        equipment_size=None,
        load_type=None,
        commodities=None,
        rate_info=None,
        special_instructions=None,
        shipper_instructions=None,
        carrier_instructions=None,
        dispatcher_name=None,
        dispatcher_phone=None
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert: Only header present
    assert text.strip() == "=== EXTRACTED STRUCTURED DATA ==="

def test_format_extraction_as_text_boundary_equipment_size(extraction_service):
    # Arrange: equipment_size = 0 and negative
    data_zero = make_shipment_data(equipment_size=0)
    extraction_zero = ExtractionResponse(data=data_zero, document_id="doc.txt")
    text_zero = extraction_service.format_extraction_as_text(extraction_zero)
    # Should include "Equipment Size: 0 feet"
    assert "Equipment Size: 0 feet" in text_zero

    data_neg = make_shipment_data(equipment_size=-53)
    extraction_neg = ExtractionResponse(data=data_neg, document_id="doc.txt")
    text_neg = extraction_service.format_extraction_as_text(extraction_neg)
    assert "Equipment Size: -53 feet" in text_neg

def test_format_extraction_as_text_handles_multiple_commodities(extraction_service):
    # Arrange
    commodities = [
        type("Commodity", (), {"commodity_name": "A", "weight": "10", "quantity": 1})(),
        type("Commodity", (), {"commodity_name": "B", "weight": "20", "quantity": 2})(),
    ]
    data = make_shipment_data(commodities=commodities)
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert
    assert "1. A" in text
    assert "2. B" in text
    assert "Weight: 10" in text
    assert "Weight: 20" in text
    assert "Quantity: 1" in text
    assert "Quantity: 2" in text

def test_format_extraction_as_text_rate_info_missing_fields(extraction_service):
    # Arrange: rate_info with missing currency and breakdown
    rate_info = type("Rate", (), {"total_rate": 500, "currency": None, "rate_breakdown": None})()
    data = make_shipment_data(rate_info=rate_info)
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert
    assert "Total Rate: $500 USD" in text
    assert "Rate Breakdown" not in text

def test_format_extraction_as_text_pickup_drop_fallbacks(extraction_service):
    # Arrange: pickup/drop with only address
    pickup = type("Loc", (), {"name": None, "address": "123 Main", "city": None, "state": None, "appointment_time": None})()
    drop = type("Loc", (), {"name": None, "address": "456 Elm", "city": None, "state": None, "appointment_time": None})()
    data = make_shipment_data(pickup=pickup, drop=drop)
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Pickup Location: 123 Main" in text
    assert "Drop Location: 456 Elm" in text

def test_format_extraction_as_text_pickup_drop_all_missing(extraction_service):
    # Arrange: pickup/drop with no name or address
    pickup = type("Loc", (), {"name": None, "address": None, "city": None, "state": None, "appointment_time": None})()
    drop = type("Loc", (), {"name": None, "address": None, "city": None, "state": None, "appointment_time": None})()
    data = make_shipment_data(pickup=pickup, drop=drop)
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Pickup Location: N/A" in text
    assert "Drop Location: N/A" in text

def test_create_structured_chunk_text_and_metadata_consistency(extraction_service):
    # Arrange
    data = make_shipment_data(reference_id="RID")
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    chunk = extraction_service.create_structured_chunk(extraction, filename="doc.txt")
    # Act
    formatted_text = extraction_service.format_extraction_as_text(extraction)
    # Assert
    assert chunk.text == formatted_text
    assert chunk.metadata.filename == "doc.txt"
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.source == "doc.txt - Extracted Data"
    assert chunk.metadata.chunk_type == "structured_data"
