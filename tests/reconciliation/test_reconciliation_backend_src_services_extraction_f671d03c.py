# source_hash: 88361007730f06bf
# import_target: backend.src.services.extraction
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import pytest
from unittest.mock import patch, MagicMock
from backend.src.services.extraction import ExtractionService
from models.extraction_schema import ShipmentData, ExtractionResponse
from models.schemas import Chunk, DocumentMetadata

@pytest.fixture
def mock_settings(monkeypatch):
    monkeypatch.setattr("backend.src.services.extraction.settings.QA_MODEL", "mock-model")
    monkeypatch.setattr("backend.src.services.extraction.settings.GROQ_API_KEY", "mock-key")

@pytest.fixture
def mock_llm(monkeypatch):
    mock_llm_instance = MagicMock()
    monkeypatch.setattr("backend.src.services.extraction.ChatGroq", MagicMock(return_value=mock_llm_instance))
    return mock_llm_instance

@pytest.fixture
def mock_parser(monkeypatch):
    mock_parser_instance = MagicMock()
    mock_parser_instance.get_format_instructions.return_value = "FORMAT"
    monkeypatch.setattr("backend.src.services.extraction.PydanticOutputParser", MagicMock(return_value=mock_parser_instance))
    return mock_parser_instance

@pytest.fixture
def mock_prompt(monkeypatch):
    mock_prompt_instance = MagicMock()
    monkeypatch.setattr("backend.src.services.extraction.ChatPromptTemplate.from_template", MagicMock(return_value=mock_prompt_instance))
    return mock_prompt_instance

@pytest.fixture
def extraction_service_instance(mock_settings, mock_llm, mock_parser, mock_prompt):
    return ExtractionService()

def test_extract_data_happy_path(extraction_service_instance, mock_parser, mock_prompt, mock_llm):
    # Setup
    shipment_data = ShipmentData(reference_id="REF123")
    extraction_response = ExtractionResponse(data=shipment_data, document_id="file.txt")
    # Chain mock: prompt | llm | parser
    class ChainMock:
        def invoke(self, args):
            return shipment_data
    chain_mock = ChainMock()
    # Compose chain
    mock_prompt.__or__.return_value = chain_mock
    # Patch __or__ for chaining
    mock_prompt.__or__ = lambda self, other: chain_mock
    # Patch parser.get_format_instructions
    mock_parser.get_format_instructions.return_value = "FORMAT"
    # Patch extraction_service_instance.extraction_prompt to use our mock
    extraction_service_instance.extraction_prompt = mock_prompt
    extraction_service_instance.llm = mock_llm
    extraction_service_instance.parser = mock_parser

    # Act
    result = extraction_service_instance.extract_data("test text", filename="file.txt")

    # Assert
    assert isinstance(result, ExtractionResponse)
    assert result.data.reference_id == "REF123"
    assert result.document_id == "file.txt"

def test_extract_data_error_returns_empty_structured(extraction_service_instance, mock_parser, mock_prompt, mock_llm):
    # Chain mock that raises
    class ChainMock:
        def invoke(self, args):
            raise Exception("fail")
    chain_mock = ChainMock()
    mock_prompt.__or__ = lambda self, other: chain_mock
    extraction_service_instance.extraction_prompt = mock_prompt
    extraction_service_instance.llm = mock_llm
    extraction_service_instance.parser = mock_parser

    result = extraction_service_instance.extract_data("bad text", filename="fail.txt")
    assert isinstance(result, ExtractionResponse)
    assert isinstance(result.data, ShipmentData)
    # All fields should be default/empty
    for field in result.data.__fields__:
        assert getattr(result.data, field) is None or getattr(result.data, field) == []

def test_extract_data_empty_text(extraction_service_instance, mock_parser, mock_prompt, mock_llm):
    shipment_data = ShipmentData()
    class ChainMock:
        def invoke(self, args):
            return shipment_data
    chain_mock = ChainMock()
    mock_prompt.__or__ = lambda self, other: chain_mock
    extraction_service_instance.extraction_prompt = mock_prompt
    extraction_service_instance.llm = mock_llm
    extraction_service_instance.parser = mock_parser

    result = extraction_service_instance.extract_data("", filename="empty.txt")
    assert isinstance(result, ExtractionResponse)
    assert result.document_id == "empty.txt"
    assert isinstance(result.data, ShipmentData)

def test_format_extraction_as_text_happy_path(extraction_service_instance):
    # Fill all fields for maximal coverage
    from models.extraction_schema import Carrier, Driver, Location, Commodity, RateInfo
    data = ShipmentData(
        reference_id="REF",
        load_id="LOAD",
        po_number="PO",
        shipper="Shipper Inc.",
        consignee="Consignee LLC",
        carrier=Carrier(carrier_name="CarrierX", mc_number="MC123", phone="555-1234"),
        driver=Driver(driver_name="John Doe", cell_number="555-5678", truck_number="TRK42"),
        pickup=Location(name="Warehouse", address="123 Main", city="Metropolis", state="NY", zip="10001", appointment_time="2024-01-01T08:00"),
        drop=Location(name="Store", address="456 Elm", city="Gotham", state="NJ", zip="07001"),
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Van",
        equipment_size=53,
        load_type="Full",
        commodities=[Commodity(commodity_name="Widgets", weight="1000 lbs", quantity="100", description="Blue widgets")],
        rate_info=RateInfo(total_rate=1500, currency="USD", rate_breakdown={"linehaul": 1200, "fuel": 300}),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No tarps",
        dispatcher_name="Dispatch Dan",
        dispatcher_phone="555-9999"
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service_instance.format_extraction_as_text(extraction)
    assert "Reference ID: REF" in text
    assert "Load ID: LOAD" in text
    assert "PO Number: PO" in text
    assert "Shipper: Shipper Inc." in text
    assert "Consignee: Consignee LLC" in text
    assert "Carrier Name: CarrierX" in text
    assert "MC Number: MC123" in text
    assert "Carrier Phone: 555-1234" in text
    assert "Driver Name: John Doe" in text
    assert "Driver Phone: 555-5678" in text
    assert "Truck Number: TRK42" in text
    assert "Pickup Location: Warehouse" in text
    assert "Pickup City: Metropolis, NY" in text
    assert "Pickup Appointment: 2024-01-01T08:00" in text
    assert "Drop Location: Store" in text
    assert "Drop City: Gotham, NJ" in text
    assert "Shipping Date: 2024-01-01" in text
    assert "Delivery Date: 2024-01-02" in text
    assert "Equipment Type: Van" in text
    assert "Equipment Size: 53 feet" in text
    assert "Load Type: Full" in text
    assert "Commodities:" in text
    assert "1. Widgets" in text
    assert "Weight: 1000 lbs" in text
    assert "Quantity: 100" in text
    assert "Total Rate: $1500 USD" in text
    assert "Rate Breakdown:" in text
    assert "Special Instructions: Handle with care" in text
    assert "Shipper Instructions: Call before arrival" in text
    assert "Carrier Instructions: No tarps" in text
    assert "Dispatcher: Dispatch Dan" in text
    assert "Dispatcher Phone: 555-9999" in text

def test_format_extraction_as_text_minimal(extraction_service_instance):
    data = ShipmentData()
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service_instance.format_extraction_as_text(extraction)
    assert "EXTRACTED STRUCTURED DATA" in text
    # Should not contain any field lines except header
    assert text.strip().count("\n") <= 1

def test_format_extraction_as_text_partial_fields(extraction_service_instance):
    from models.extraction_schema import Carrier
    data = ShipmentData(
        reference_id="REF",
        carrier=Carrier(carrier_name="CarrierX"),
        dispatcher_name="Dispatch Dan"
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service_instance.format_extraction_as_text(extraction)
    assert "Reference ID: REF" in text
    assert "Carrier Name: CarrierX" in text
    assert "Dispatcher: Dispatch Dan" in text

def test_create_structured_chunk_output(extraction_service_instance):
    from models.extraction_schema import ShipmentData
    extraction = ExtractionResponse(data=ShipmentData(reference_id="R1"), document_id="doc.txt")
    chunk = extraction_service_instance.create_structured_chunk(extraction, filename="doc.txt")
    assert isinstance(chunk, Chunk)
    assert isinstance(chunk.metadata, DocumentMetadata)
    assert chunk.metadata.filename == "doc.txt"
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.chunk_type == "structured_data"
    assert "EXTRACTED STRUCTURED DATA" in chunk.text

def test_create_structured_chunk_with_empty_data(extraction_service_instance):
    extraction = ExtractionResponse(data=ShipmentData(), document_id="doc.txt")
    chunk = extraction_service_instance.create_structured_chunk(extraction, filename="doc.txt")
    assert isinstance(chunk, Chunk)
    assert "EXTRACTED STRUCTURED DATA" in chunk.text

def test_format_extraction_as_text_handles_missing_nested_fields(extraction_service_instance):
    from models.extraction_schema import Carrier
    data = ShipmentData(
        carrier=Carrier(carrier_name=None, mc_number=None, phone=None)
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service_instance.format_extraction_as_text(extraction)
    # Should not raise or include carrier fields with None values
    assert "Carrier Name:" not in text

def test_extract_data_equivalent_paths(monkeypatch, mock_settings, mock_llm, mock_parser, mock_prompt):
    # Simulate two equivalent chains producing the same output
    shipment_data = ShipmentData(reference_id="EQ1")
    class ChainMock:
        def invoke(self, args):
            return shipment_data
    chain_mock = ChainMock()
    mock_prompt.__or__ = lambda self, other: chain_mock
    service1 = ExtractionService()
    service1.extraction_prompt = mock_prompt
    service1.llm = mock_llm
    service1.parser = mock_parser

    service2 = ExtractionService()
    service2.extraction_prompt = mock_prompt
    service2.llm = mock_llm
    service2.parser = mock_parser

    result1 = service1.extract_data("text", filename="f1.txt")
    result2 = service2.extract_data("text", filename="f1.txt")
    assert result1.data.reference_id == result2.data.reference_id
    assert result1.document_id == result2.document_id

def test_format_extraction_as_text_equivalent_for_same_data(extraction_service_instance):
    from models.extraction_schema import ShipmentData
    data = ShipmentData(reference_id="SAME")
    extraction1 = ExtractionResponse(data=data, document_id="doc.txt")
    extraction2 = ExtractionResponse(data=data, document_id="doc.txt")
    text1 = extraction_service_instance.format_extraction_as_text(extraction1)
    text2 = extraction_service_instance.format_extraction_as_text(extraction2)
    assert text1 == text2

def test_create_structured_chunk_equivalent_for_same_input(extraction_service_instance):
    from models.extraction_schema import ShipmentData
    extraction = ExtractionResponse(data=ShipmentData(reference_id="EQ2"), document_id="doc.txt")
    chunk1 = extraction_service_instance.create_structured_chunk(extraction, filename="doc.txt")
    chunk2 = extraction_service_instance.create_structured_chunk(extraction, filename="doc.txt")
    assert chunk1.text == chunk2.text
    assert chunk1.metadata.filename == chunk2.metadata.filename
    assert chunk1.metadata.chunk_id == chunk2.metadata.chunk_id
    assert chunk1.metadata.chunk_type == chunk2.metadata.chunk_type
