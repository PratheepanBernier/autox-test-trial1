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
    monkeypatch.setattr("backend.src.services.extraction.settings.GROQ_API_KEY", "mock-api-key")

@pytest.fixture
def extraction_service_instance(mock_settings):
    # Patch ChatGroq, PydanticOutputParser, ChatPromptTemplate
    with patch("backend.src.services.extraction.ChatGroq") as mock_llm, \
         patch("backend.src.services.extraction.PydanticOutputParser") as mock_parser, \
         patch("backend.src.services.extraction.ChatPromptTemplate") as mock_prompt:

        mock_llm.return_value = MagicMock()
        mock_parser.return_value = MagicMock()
        mock_prompt.from_template.return_value = MagicMock()
        yield ExtractionService()

def test_extract_data_happy_path(extraction_service_instance):
    service = extraction_service_instance

    # Patch the chain and parser
    fake_result = ShipmentData(reference_id="REF123", load_id="LOAD456", po_number="PO789")
    fake_response = ExtractionResponse(data=fake_result, document_id="testfile.txt")

    chain_mock = MagicMock()
    chain_mock.invoke.return_value = fake_result

    service.extraction_prompt.__or__.return_value = service.extraction_prompt
    service.extraction_prompt.__or__.side_effect = lambda other: chain_mock if hasattr(other, "invoke") else service.extraction_prompt

    service.llm.__or__.return_value = chain_mock
    service.parser.get_format_instructions.return_value = "FORMAT"

    with patch.object(service, "extraction_prompt", service.extraction_prompt), \
         patch.object(service, "llm", service.llm), \
         patch.object(service, "parser", service.parser):

        result = service.extract_data("Sample text", filename="testfile.txt")
        assert isinstance(result, ExtractionResponse)
        assert result.data.reference_id == "REF123"
        assert result.document_id == "testfile.txt"

def test_extract_data_empty_text_returns_empty_shipment(extraction_service_instance):
    service = extraction_service_instance

    # Patch the chain to raise an exception
    chain_mock = MagicMock()
    chain_mock.invoke.side_effect = Exception("Extraction failed")

    service.extraction_prompt.__or__.return_value = service.extraction_prompt
    service.extraction_prompt.__or__.side_effect = lambda other: chain_mock if hasattr(other, "invoke") else service.extraction_prompt

    service.llm.__or__.return_value = chain_mock
    service.parser.get_format_instructions.return_value = "FORMAT"

    with patch.object(service, "extraction_prompt", service.extraction_prompt), \
         patch.object(service, "llm", service.llm), \
         patch.object(service, "parser", service.parser):

        result = service.extract_data("", filename="empty.txt")
        assert isinstance(result, ExtractionResponse)
        assert isinstance(result.data, ShipmentData)
        # All fields should be default/None
        assert result.data.reference_id is None
        assert result.document_id == "empty.txt"

def test_extract_data_handles_parser_error(extraction_service_instance):
    service = extraction_service_instance

    # Patch the chain to raise an exception
    chain_mock = MagicMock()
    chain_mock.invoke.side_effect = Exception("Parser error")

    service.extraction_prompt.__or__.return_value = service.extraction_prompt
    service.extraction_prompt.__or__.side_effect = lambda other: chain_mock if hasattr(other, "invoke") else service.extraction_prompt

    service.llm.__or__.return_value = chain_mock
    service.parser.get_format_instructions.return_value = "FORMAT"

    with patch.object(service, "extraction_prompt", service.extraction_prompt), \
         patch.object(service, "llm", service.llm), \
         patch.object(service, "parser", service.parser):

        result = service.extract_data("Some text", filename="fail.txt")
        assert isinstance(result, ExtractionResponse)
        assert isinstance(result.data, ShipmentData)
        assert result.data.reference_id is None
        assert result.document_id == "fail.txt"

def test_format_extraction_as_text_full_fields(extraction_service_instance):
    service = extraction_service_instance

    # Fill all fields for ShipmentData
    from models.extraction_schema import Carrier, Driver, Location, Commodity, RateInfo

    data = ShipmentData(
        reference_id="REFID",
        load_id="LOADID",
        po_number="PO123",
        shipper="Shipper Inc.",
        consignee="Consignee LLC",
        carrier=Carrier(carrier_name="CarrierName", mc_number="MC123", phone="555-1234"),
        driver=Driver(driver_name="John Doe", cell_number="555-5678", truck_number="TRK123"),
        pickup=Location(name="Warehouse A", address="123 Main St", city="CityA", state="ST", zip="12345", appointment_time="2024-01-01T08:00"),
        drop=Location(name="Warehouse B", address="456 Elm St", city="CityB", state="ST", zip="67890", appointment_time="2024-01-02T09:00"),
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Van",
        equipment_size=53,
        load_type="Full",
        commodities=[Commodity(commodity_name="Widgets", weight="1000 lbs", quantity="10", description="Blue widgets")],
        rate_info=RateInfo(total_rate=1200.0, currency="USD", rate_breakdown={"linehaul": 1000, "fuel": 200}),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No tarps required",
        dispatcher_name="Jane Smith",
        dispatcher_phone="555-9999"
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = service.format_extraction_as_text(extraction)
    assert "Reference ID: REFID" in text
    assert "Shipper: Shipper Inc." in text
    assert "Carrier Name: CarrierName" in text
    assert "Driver Name: John Doe" in text
    assert "Pickup Location: Warehouse A" in text
    assert "Drop Location: Warehouse B" in text
    assert "Shipping Date: 2024-01-01" in text
    assert "Total Rate: $1200.0 USD" in text
    assert "Special Instructions: Handle with care" in text
    assert "Dispatcher: Jane Smith" in text

def test_format_extraction_as_text_minimal_fields(extraction_service_instance):
    service = extraction_service_instance
    data = ShipmentData()
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = service.format_extraction_as_text(extraction)
    assert "EXTRACTED STRUCTURED DATA" in text
    # Should not contain any field lines except the header
    assert text.strip() == "=== EXTRACTED STRUCTURED DATA ==="

def test_create_structured_chunk_returns_chunk_with_metadata(extraction_service_instance):
    service = extraction_service_instance
    data = ShipmentData(reference_id="RID")
    extraction = ExtractionResponse(data=data, document_id="file.pdf")
    chunk = service.create_structured_chunk(extraction, filename="file.pdf")
    assert isinstance(chunk, Chunk)
    assert isinstance(chunk.metadata, DocumentMetadata)
    assert chunk.metadata.filename == "file.pdf"
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.chunk_type == "structured_data"
    assert "EXTRACTED STRUCTURED DATA" in chunk.text

def test_extraction_service_init_error(monkeypatch):
    # Simulate ChatGroq raising an exception
    with patch("backend.src.services.extraction.ChatGroq", side_effect=Exception("LLM init fail")), \
         patch("backend.src.services.extraction.PydanticOutputParser"), \
         patch("backend.src.services.extraction.ChatPromptTemplate"):
        with pytest.raises(Exception) as excinfo:
            ExtractionService()
        assert "LLM init fail" in str(excinfo.value)

def test_format_extraction_as_text_handles_partial_data(extraction_service_instance):
    service = extraction_service_instance
    from models.extraction_schema import Carrier
    data = ShipmentData(
        reference_id="RID",
        carrier=Carrier(carrier_name="CarrierX", mc_number=None, phone=None),
        pickup=None,
        drop=None,
        commodities=[]
    )
    extraction = ExtractionResponse(data=data, document_id="partial.txt")
    text = service.format_extraction_as_text(extraction)
    assert "Reference ID: RID" in text
    assert "Carrier Name: CarrierX" in text
    assert "Pickup Location" not in text
    assert "Drop Location" not in text
    assert "Commodities:" not in text

def test_format_extraction_as_text_handles_multiple_commodities(extraction_service_instance):
    service = extraction_service_instance
    from models.extraction_schema import Commodity
    data = ShipmentData(
        commodities=[
            Commodity(commodity_name="Item1", weight="100", quantity="1", description="Desc1"),
            Commodity(commodity_name="Item2", weight="200", quantity="2", description="Desc2"),
        ]
    )
    extraction = ExtractionResponse(data=data, document_id="commodities.txt")
    text = service.format_extraction_as_text(extraction)
    assert "Commodities:" in text
    assert "1. Item1" in text
    assert "2. Item2" in text
    assert "Weight: 100" in text
    assert "Weight: 200" in text

def test_format_extraction_as_text_handles_rate_breakdown_none(extraction_service_instance):
    service = extraction_service_instance
    from models.extraction_schema import RateInfo
    data = ShipmentData(
        rate_info=RateInfo(total_rate=500.0, currency=None, rate_breakdown=None)
    )
    extraction = ExtractionResponse(data=data, document_id="rate.txt")
    text = service.format_extraction_as_text(extraction)
    assert "Total Rate: $500.0 USD" in text
    assert "Rate Breakdown" not in text
