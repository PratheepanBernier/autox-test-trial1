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
from unittest.mock import patch, MagicMock, create_autospec
from backend.src.services.extraction import ExtractionService
from models.extraction_schema import ShipmentData, ExtractionResponse
from models.schemas import Chunk, DocumentMetadata

@pytest.fixture
def mock_settings(monkeypatch):
    monkeypatch.setattr("backend.src.services.extraction.settings.QA_MODEL", "mock-model")
    monkeypatch.setattr("backend.src.services.extraction.settings.GROQ_API_KEY", "mock-key")

@pytest.fixture
def mock_logger(monkeypatch):
    mock_log = MagicMock()
    monkeypatch.setattr("backend.src.services.extraction.logger", mock_log)
    return mock_log

@pytest.fixture
def extraction_service(mock_settings):
    with patch("backend.src.services.extraction.ChatGroq") as mock_llm, \
         patch("backend.src.services.extraction.PydanticOutputParser") as mock_parser, \
         patch("backend.src.services.extraction.ChatPromptTemplate") as mock_prompt:
        mock_llm.return_value = MagicMock()
        mock_parser.return_value = MagicMock()
        mock_prompt.from_template.return_value = MagicMock()
        return ExtractionService()

def test_extractionservice_init_success(mock_settings, mock_logger):
    with patch("backend.src.services.extraction.ChatGroq") as mock_llm, \
         patch("backend.src.services.extraction.PydanticOutputParser") as mock_parser, \
         patch("backend.src.services.extraction.ChatPromptTemplate") as mock_prompt:
        mock_llm.return_value = MagicMock()
        mock_parser.return_value = MagicMock()
        mock_prompt.from_template.return_value = MagicMock()
        service = ExtractionService()
        assert hasattr(service, "llm")
        assert hasattr(service, "parser")
        assert hasattr(service, "extraction_prompt")
        mock_logger.info.assert_called_with("ExtractionService initialized successfully.")

def test_extractionservice_init_failure(mock_settings, mock_logger):
    with patch("backend.src.services.extraction.ChatGroq", side_effect=Exception("fail")):
        with pytest.raises(Exception) as excinfo:
            ExtractionService()
        assert "fail" in str(excinfo.value)
        assert mock_logger.error.called

def test_extract_data_happy_path(extraction_service, mock_logger):
    fake_result = ShipmentData(reference_id="REF123")
    fake_parser = extraction_service.parser
    fake_parser.get_format_instructions.return_value = "FORMAT"
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = fake_result
    extraction_service.extraction_prompt.__or__.return_value = extraction_service.llm
    extraction_service.llm.__or__.return_value = extraction_service.parser
    extraction_service.parser.__or__ = lambda other: chain_mock
    with patch.object(extraction_service, "extraction_prompt", create_autospec(extraction_service.extraction_prompt)), \
         patch.object(extraction_service, "llm", create_autospec(extraction_service.llm)), \
         patch.object(extraction_service, "parser", fake_parser):
        chain = MagicMock()
        chain.invoke.return_value = fake_result
        with patch("backend.src.services.extraction.ChatPromptTemplate.from_template", return_value=MagicMock()), \
             patch("backend.src.services.extraction.PydanticOutputParser", return_value=fake_parser):
            extraction_service.extraction_prompt.__or__ = lambda self, other: extraction_service.llm
            extraction_service.llm.__or__ = lambda self, other: chain
            result = extraction_service.extract_data("some text", filename="file.txt")
            assert isinstance(result, ExtractionResponse)
            assert result.data.reference_id == "REF123"
            assert result.document_id == "file.txt"
            mock_logger.info.assert_any_call("Successfully extracted data.")

def test_extract_data_handles_exception(extraction_service, mock_logger):
    fake_parser = extraction_service.parser
    fake_parser.get_format_instructions.return_value = "FORMAT"
    chain_mock = MagicMock()
    chain_mock.invoke.side_effect = Exception("invoke error")
    extraction_service.extraction_prompt.__or__.return_value = extraction_service.llm
    extraction_service.llm.__or__.return_value = extraction_service.parser
    extraction_service.parser.__or__ = lambda other: chain_mock
    with patch.object(extraction_service, "extraction_prompt", create_autospec(extraction_service.extraction_prompt)), \
         patch.object(extraction_service, "llm", create_autospec(extraction_service.llm)), \
         patch.object(extraction_service, "parser", fake_parser):
        extraction_service.extraction_prompt.__or__ = lambda self, other: extraction_service.llm
        extraction_service.llm.__or__ = lambda self, other: chain_mock
        result = extraction_service.extract_data("bad text", filename="fail.txt")
        assert isinstance(result, ExtractionResponse)
        assert isinstance(result.data, ShipmentData)
        assert result.document_id == "fail.txt"
        mock_logger.error.assert_any_call("Extraction error: invoke error", exc_info=True)

def test_format_extraction_as_text_full_fields(extraction_service):
    class Carrier:
        carrier_name = "CarrierX"
        mc_number = "MC123"
        phone = "555-1234"
    class Driver:
        driver_name = "John Doe"
        cell_number = "555-5678"
        truck_number = "TRK123"
    class Location:
        name = "Warehouse"
        address = "123 Main St"
        city = "Metropolis"
        state = "NY"
        zip = "10001"
        appointment_time = "2024-01-01T10:00"
    class Commodity:
        commodity_name = "Widgets"
        weight = "1000"
        quantity = "10"
        description = "Blue widgets"
    class RateInfo:
        total_rate = "2000"
        currency = "USD"
        rate_breakdown = {"linehaul": 1800, "fuel": 200}
    data = ShipmentData(
        reference_id="REF1",
        load_id="LOAD1",
        po_number="PO1",
        shipper="Shipper Inc.",
        consignee="Consignee LLC",
        carrier=Carrier(),
        driver=Driver(),
        pickup=Location(),
        drop=Location(),
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Van",
        equipment_size="53",
        load_type="Full",
        commodities=[Commodity()],
        rate_info=RateInfo(),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No tarps",
        dispatcher_name="Dispatch Dave",
        dispatcher_phone="555-9999"
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Reference ID: REF1" in text
    assert "Carrier Name: CarrierX" in text
    assert "Driver Name: John Doe" in text
    assert "Pickup Location: Warehouse" in text
    assert "Drop Location: Warehouse" in text
    assert "Shipping Date: 2024-01-01" in text
    assert "Equipment Type: Van" in text
    assert "Commodities:" in text
    assert "Total Rate: $2000 USD" in text
    assert "Special Instructions: Handle with care" in text
    assert "Dispatcher: Dispatch Dave" in text

def test_format_extraction_as_text_minimal_fields(extraction_service):
    data = ShipmentData()
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "EXTRACTED STRUCTURED DATA" in text
    # Should not contain any field lines except header
    assert text.strip() == "=== EXTRACTED STRUCTURED DATA ==="

def test_format_extraction_as_text_partial_fields(extraction_service):
    class Carrier:
        carrier_name = "CarrierY"
        mc_number = None
        phone = None
    data = ShipmentData(
        reference_id="REF2",
        carrier=Carrier(),
        pickup=None,
        drop=None
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Reference ID: REF2" in text
    assert "Carrier Name: CarrierY" in text
    assert "MC Number" not in text
    assert "Pickup Location" not in text

def test_create_structured_chunk_returns_chunk(extraction_service):
    extraction = ExtractionResponse(
        data=ShipmentData(reference_id="REF3"),
        document_id="doc3.txt"
    )
    chunk = extraction_service.create_structured_chunk(extraction, filename="doc3.txt")
    assert isinstance(chunk, Chunk)
    assert isinstance(chunk.metadata, DocumentMetadata)
    assert chunk.metadata.filename == "doc3.txt"
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.chunk_type == "structured_data"
    assert "EXTRACTED STRUCTURED DATA" in chunk.text

def test_create_structured_chunk_with_empty_data(extraction_service):
    extraction = ExtractionResponse(
        data=ShipmentData(),
        document_id="empty.txt"
    )
    chunk = extraction_service.create_structured_chunk(extraction, filename="empty.txt")
    assert isinstance(chunk, Chunk)
    assert chunk.metadata.filename == "empty.txt"
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.chunk_type == "structured_data"
    assert chunk.text.strip() == "=== EXTRACTED STRUCTURED DATA ==="

def test_format_extraction_as_text_handles_none_and_empty_lists(extraction_service):
    class RateInfo:
        total_rate = None
        currency = None
        rate_breakdown = None
    data = ShipmentData(
        commodities=[],
        rate_info=RateInfo()
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Commodities:" not in text
    assert "Total Rate" not in text

def test_format_extraction_as_text_handles_missing_nested_fields(extraction_service):
    class Carrier:
        carrier_name = None
        mc_number = None
        phone = None
    data = ShipmentData(
        carrier=Carrier()
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    # Should not print carrier name if None
    assert "Carrier Name" not in text
    assert "MC Number" not in text
    assert "Carrier Phone" not in text
