# source_hash: 88361007730f06bf
import pytest
from unittest.mock import patch, MagicMock, create_autospec
from backend.src.services.extraction import ExtractionService
from models.extraction_schema import ShipmentData, ExtractionResponse
from models.schemas import Chunk, DocumentMetadata
import logging

@pytest.fixture(autouse=True)
def patch_logger(monkeypatch):
    # Silence logger for test output
    monkeypatch.setattr(logging, "getLogger", lambda name=None: MagicMock())

@pytest.fixture
def mock_settings(monkeypatch):
    class DummySettings:
        QA_MODEL = "test-model"
        GROQ_API_KEY = "test-key"
    monkeypatch.setattr("core.config.settings", DummySettings())

@pytest.fixture
def mock_groq(monkeypatch):
    mock_llm = MagicMock()
    monkeypatch.setattr("langchain_groq.ChatGroq", lambda **kwargs: mock_llm)
    return mock_llm

@pytest.fixture
def mock_prompt(monkeypatch):
    mock_prompt = MagicMock()
    monkeypatch.setattr("langchain_core.prompts.ChatPromptTemplate.from_template", lambda *a, **k: mock_prompt)
    return mock_prompt

@pytest.fixture
def mock_parser(monkeypatch):
    mock_parser = MagicMock()
    mock_parser.get_format_instructions.return_value = "FORMAT"
    monkeypatch.setattr("langchain_core.output_parsers.PydanticOutputParser", lambda **kwargs: mock_parser)
    return mock_parser

@pytest.fixture
def extraction_service(mock_settings, mock_groq, mock_prompt, mock_parser):
    return ExtractionService()

def make_shipment_data(**kwargs):
    # Helper to create ShipmentData with arbitrary fields
    data = ShipmentData()
    for k, v in kwargs.items():
        setattr(data, k, v)
    return data

def test_extract_data_happy_path(extraction_service, mock_groq, mock_parser):
    # Arrange
    shipment_data = make_shipment_data(reference_id="REF123", load_id="L456")
    mock_parser.get_format_instructions.return_value = "FORMAT"
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = shipment_data
    # Compose the chain: prompt | llm | parser
    extraction_service.extraction_prompt.__or__.return_value = mock_chain
    # Act
    result = extraction_service.extract_data("test text", filename="doc1.txt")
    # Assert
    assert isinstance(result, ExtractionResponse)
    assert result.data.reference_id == "REF123"
    assert result.data.load_id == "L456"
    assert result.document_id == "doc1.txt"

def test_extract_data_empty_text_returns_empty_shipment(extraction_service, mock_groq, mock_parser):
    # Arrange
    mock_parser.get_format_instructions.return_value = "FORMAT"
    mock_chain = MagicMock()
    # Simulate LLM returns empty ShipmentData
    mock_chain.invoke.return_value = ShipmentData()
    extraction_service.extraction_prompt.__or__.return_value = mock_chain
    # Act
    result = extraction_service.extract_data("", filename="empty.txt")
    # Assert
    assert isinstance(result, ExtractionResponse)
    assert result.data.reference_id is None
    assert result.document_id == "empty.txt"

def test_extract_data_llm_raises_exception_returns_empty(extraction_service, mock_groq, mock_parser):
    # Arrange
    mock_parser.get_format_instructions.return_value = "FORMAT"
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception("LLM error")
    extraction_service.extraction_prompt.__or__.return_value = mock_chain
    # Act
    result = extraction_service.extract_data("bad input", filename="fail.txt")
    # Assert
    assert isinstance(result, ExtractionResponse)
    assert isinstance(result.data, ShipmentData)
    assert result.data.reference_id is None
    assert result.document_id == "fail.txt"

def test_format_extraction_as_text_full_fields(extraction_service):
    # Arrange
    class DummyCarrier:
        carrier_name = "CarrierX"
        mc_number = "MC123"
        phone = "555-1234"
    class DummyDriver:
        driver_name = "John Doe"
        cell_number = "555-5678"
        truck_number = "TRK999"
    class DummyLocation:
        name = "Warehouse A"
        address = "123 Main St"
        city = "Metropolis"
        state = "NY"
        zip = "10001"
        appointment_time = "2024-06-01T10:00"
    class DummyDrop:
        name = "Store B"
        address = "456 Elm St"
        city = "Gotham"
        state = "NJ"
        zip = "07001"
        appointment_time = None
    class DummyCommodity:
        commodity_name = "Widgets"
        weight = "1000 lbs"
        quantity = "50"
        description = "Blue widgets"
    class DummyRateInfo:
        total_rate = 1200.0
        currency = "USD"
        rate_breakdown = {"linehaul": 1000, "fuel": 200}
    shipment = make_shipment_data(
        reference_id="REF123",
        load_id="L456",
        po_number="PO789",
        shipper="Shipper Inc.",
        consignee="Consignee LLC",
        carrier=DummyCarrier(),
        driver=DummyDriver(),
        pickup=DummyLocation(),
        drop=DummyDrop(),
        shipping_date="2024-06-01",
        delivery_date="2024-06-02",
        equipment_type="Van",
        equipment_size=53,
        load_type="Full",
        commodities=[DummyCommodity()],
        rate_info=DummyRateInfo(),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No tarps",
        dispatcher_name="Dispatch Dan",
        dispatcher_phone="555-9999"
    )
    extraction = ExtractionResponse(data=shipment, document_id="doc.txt")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert
    assert "Reference ID: REF123" in text
    assert "Load ID: L456" in text
    assert "PO Number: PO789" in text
    assert "Shipper: Shipper Inc." in text
    assert "Consignee: Consignee LLC" in text
    assert "Carrier Name: CarrierX" in text
    assert "MC Number: MC123" in text
    assert "Carrier Phone: 555-1234" in text
    assert "Driver Name: John Doe" in text
    assert "Driver Phone: 555-5678" in text
    assert "Truck Number: TRK999" in text
    assert "Pickup Location: Warehouse A" in text
    assert "Pickup City: Metropolis, NY" in text
    assert "Pickup Appointment: 2024-06-01T10:00" in text
    assert "Drop Location: Store B" in text
    assert "Drop City: Gotham, NJ" in text
    assert "Shipping Date: 2024-06-01" in text
    assert "Delivery Date: 2024-06-02" in text
    assert "Equipment Type: Van" in text
    assert "Equipment Size: 53 feet" in text
    assert "Load Type: Full" in text
    assert "Commodities:" in text
    assert "1. Widgets" in text
    assert "Weight: 1000 lbs" in text
    assert "Quantity: 50" in text
    assert "Total Rate: $1200.0 USD" in text
    assert "Rate Breakdown:" in text
    assert "Special Instructions: Handle with care" in text
    assert "Shipper Instructions: Call before arrival" in text
    assert "Carrier Instructions: No tarps" in text
    assert "Dispatcher: Dispatch Dan" in text
    assert "Dispatcher Phone: 555-9999" in text

def test_format_extraction_as_text_minimal_fields(extraction_service):
    # Arrange
    shipment = make_shipment_data()
    extraction = ExtractionResponse(data=shipment, document_id="doc.txt")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert
    assert "EXTRACTED STRUCTURED DATA" in text
    # Should not contain any field lines except the header
    assert text.strip() == "=== EXTRACTED STRUCTURED DATA ==="

def test_format_extraction_as_text_partial_fields(extraction_service):
    # Arrange
    class DummyCarrier:
        carrier_name = "CarrierY"
        mc_number = None
        phone = None
    shipment = make_shipment_data(
        reference_id="REF999",
        carrier=DummyCarrier(),
        shipping_date="2024-01-01"
    )
    extraction = ExtractionResponse(data=shipment, document_id="doc.txt")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert
    assert "Reference ID: REF999" in text
    assert "Carrier Name: CarrierY" in text
    assert "Shipping Date: 2024-01-01" in text
    assert "MC Number" not in text
    assert "Carrier Phone" not in text

def test_create_structured_chunk_returns_chunk_with_expected_metadata(extraction_service):
    # Arrange
    shipment = make_shipment_data(reference_id="R1")
    extraction = ExtractionResponse(data=shipment, document_id="doc.txt")
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

def test_create_structured_chunk_with_empty_extraction(extraction_service):
    # Arrange
    extraction = ExtractionResponse(data=ShipmentData(), document_id="doc.txt")
    # Act
    chunk = extraction_service.create_structured_chunk(extraction, filename="doc.txt")
    # Assert
    assert isinstance(chunk, Chunk)
    assert "EXTRACTED STRUCTURED DATA" in chunk.text

def test_extraction_service_init_failure(monkeypatch):
    # Simulate ChatGroq raises exception
    monkeypatch.setattr("langchain_groq.ChatGroq", lambda **kwargs: (_ for _ in ()).throw(Exception("fail")))
    with pytest.raises(Exception):
        ExtractionService()

def test_format_extraction_as_text_handles_none_fields(extraction_service):
    # Arrange
    class DummyCarrier:
        carrier_name = None
        mc_number = None
        phone = None
    shipment = make_shipment_data(
        reference_id=None,
        carrier=DummyCarrier(),
        shipping_date=None
    )
    extraction = ExtractionResponse(data=shipment, document_id="doc.txt")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert
    assert "Carrier Name" not in text
    assert "Reference ID" not in text
    assert "Shipping Date" not in text

def test_format_extraction_as_text_commodity_missing_fields(extraction_service):
    # Arrange
    class DummyCommodity:
        commodity_name = None
        weight = None
        quantity = None
        description = None
    shipment = make_shipment_data(
        commodities=[DummyCommodity()]
    )
    extraction = ExtractionResponse(data=shipment, document_id="doc.txt")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert
    assert "Commodities:" in text
    assert "Unknown" in text
    assert "Weight:" not in text
    assert "Quantity:" not in text

def test_format_extraction_as_text_rate_info_missing_fields(extraction_service):
    # Arrange
    class DummyRateInfo:
        total_rate = 500.0
        currency = None
        rate_breakdown = None
    shipment = make_shipment_data(
        rate_info=DummyRateInfo()
    )
    extraction = ExtractionResponse(data=shipment, document_id="doc.txt")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert
    assert "Total Rate: $500.0 USD" in text
    assert "Rate Breakdown" not in text

def test_format_extraction_as_text_pickup_and_drop_missing_fields(extraction_service):
    # Arrange
    class DummyLocation:
        name = None
        address = None
        city = None
        state = None
        zip = None
        appointment_time = None
    shipment = make_shipment_data(
        pickup=DummyLocation(),
        drop=DummyLocation()
    )
    extraction = ExtractionResponse(data=shipment, document_id="doc.txt")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert
    assert "Pickup Location: N/A" in text
    assert "Drop Location: N/A" in text
    assert "Pickup City" not in text
    assert "Drop City" not in text
    assert "Pickup Appointment" not in text

def test_format_extraction_as_text_equipment_size_boundary(extraction_service):
    # Arrange
    shipment = make_shipment_data(
        equipment_size=0
    )
    extraction = ExtractionResponse(data=shipment, document_id="doc.txt")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert
    # 0 is a valid boundary, should be shown as "0 feet"
    assert "Equipment Size: 0 feet" in text

def test_format_extraction_as_text_multiple_commodities(extraction_service):
    # Arrange
    class DummyCommodity1:
        commodity_name = "A"
        weight = "10"
        quantity = "1"
        description = None
    class DummyCommodity2:
        commodity_name = "B"
        weight = None
        quantity = None
        description = None
    shipment = make_shipment_data(
        commodities=[DummyCommodity1(), DummyCommodity2()]
    )
    extraction = ExtractionResponse(data=shipment, document_id="doc.txt")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert
    assert "1. A" in text
    assert "2. B" in text
    assert "Weight: 10" in text
    assert "Quantity: 1" in text

def test_format_extraction_as_text_instructions_fields(extraction_service):
    # Arrange
    shipment = make_shipment_data(
        special_instructions="Spec",
        shipper_instructions="Ship",
        carrier_instructions="Carr"
    )
    extraction = ExtractionResponse(data=shipment, document_id="doc.txt")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert
    assert "Special Instructions: Spec" in text
    assert "Shipper Instructions: Ship" in text
    assert "Carrier Instructions: Carr" in text

def test_format_extraction_as_text_dispatcher_fields(extraction_service):
    # Arrange
    shipment = make_shipment_data(
        dispatcher_name="DName",
        dispatcher_phone="DPhone"
    )
    extraction = ExtractionResponse(data=shipment, document_id="doc.txt")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert
    assert "Dispatcher: DName" in text
    assert "Dispatcher Phone: DPhone" in text

def test_format_extraction_as_text_dispatcher_name_only(extraction_service):
    # Arrange
    shipment = make_shipment_data(
        dispatcher_name="DName"
    )
    extraction = ExtractionResponse(data=shipment, document_id="doc.txt")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert
    assert "Dispatcher: DName" in text
    assert "Dispatcher Phone" not in text
