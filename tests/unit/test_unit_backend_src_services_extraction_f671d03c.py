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
from backend.src.models.extraction_schema import ShipmentData, ExtractionResponse
from backend.src.models.schemas import Chunk, DocumentMetadata

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
def extraction_service(mock_settings, mock_llm, mock_parser, mock_prompt):
    return ExtractionService()

def make_shipment_data(**kwargs):
    # Helper to create ShipmentData with arbitrary fields
    data = ShipmentData()
    for k, v in kwargs.items():
        setattr(data, k, v)
    return data

def make_extraction_response(data=None, document_id="file.txt"):
    if data is None:
        data = ShipmentData()
    return ExtractionResponse(data=data, document_id=document_id)

def test_extraction_service_init_success(mock_settings, mock_llm, mock_parser, mock_prompt):
    service = ExtractionService()
    assert hasattr(service, "llm")
    assert hasattr(service, "parser")
    assert hasattr(service, "extraction_prompt")

def test_extraction_service_init_failure(monkeypatch):
    # Simulate failure in ChatGroq
    monkeypatch.setattr("backend.src.services.extraction.ChatGroq", MagicMock(side_effect=Exception("fail")))
    with pytest.raises(Exception):
        ExtractionService()

def test_extract_data_happy_path(extraction_service, mock_parser, mock_prompt):
    # Setup chain mock
    mock_chain = MagicMock()
    expected_data = make_shipment_data(reference_id="REF123")
    mock_chain.invoke.return_value = expected_data
    # Compose chain: prompt | llm | parser
    extraction_service.extraction_prompt.__or__.return_value = mock_chain
    mock_parser.get_format_instructions.return_value = "FORMAT"
    # Patch __or__ for chaining
    extraction_service.llm.__or__.return_value = mock_parser
    extraction_service.extraction_prompt.__or__.return_value = extraction_service.llm
    # Patch chain.invoke to return expected_data
    with patch.object(extraction_service, "extraction_prompt") as mock_prompt:
        mock_prompt.__or__.return_value = extraction_service.llm
        extraction_service.llm.__or__.return_value = mock_parser
        mock_parser.__or__.return_value = mock_chain
        mock_chain.invoke.return_value = expected_data
        # Patch chain to be mock_chain
        with patch("backend.src.services.extraction.ExtractionService.extract_data.__globals__", wraps=extraction_service.extract_data.__globals__) as g:
            g["logger"] = MagicMock()
            # Actually call
            result = extraction_service.extract_data("some text", filename="file.txt")
            assert isinstance(result, ExtractionResponse)
            assert result.data.reference_id == "REF123"
            assert result.document_id == "file.txt"

def test_extract_data_error_returns_empty(extraction_service, mock_parser, mock_prompt):
    # Setup chain mock to raise
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception("fail")
    extraction_service.extraction_prompt.__or__.return_value = extraction_service.llm
    extraction_service.llm.__or__.return_value = mock_parser
    mock_parser.__or__.return_value = mock_chain
    with patch.object(extraction_service, "extraction_prompt") as mock_prompt:
        mock_prompt.__or__.return_value = extraction_service.llm
        extraction_service.llm.__or__.return_value = mock_parser
        mock_parser.__or__.return_value = mock_chain
        mock_chain.invoke.side_effect = Exception("fail")
        with patch("backend.src.services.extraction.ExtractionService.extract_data.__globals__", wraps=extraction_service.extract_data.__globals__) as g:
            g["logger"] = MagicMock()
            result = extraction_service.extract_data("bad text", filename="fail.txt")
            assert isinstance(result, ExtractionResponse)
            # Should be empty ShipmentData
            assert isinstance(result.data, ShipmentData)
            assert result.document_id == "fail.txt"

def test_format_extraction_as_text_full_fields(extraction_service):
    # All fields present
    class DummyCarrier:
        carrier_name = "CarrierX"
        mc_number = "MC123"
        phone = "555-1111"
    class DummyDriver:
        driver_name = "John Doe"
        cell_number = "555-2222"
        truck_number = "TRK123"
    class DummyLocation:
        name = "Warehouse"
        address = "123 Main"
        city = "Metropolis"
        state = "NY"
        appointment_time = "2024-01-01 10:00"
    class DummyDrop:
        name = "Store"
        address = "456 Elm"
        city = "Gotham"
        state = "NJ"
        appointment_time = None
    class DummyCommodity:
        commodity_name = "Widgets"
        weight = "1000 lbs"
        quantity = "10"
        description = "Blue widgets"
    class DummyRateInfo:
        total_rate = 1500
        currency = "USD"
        rate_breakdown = {"linehaul": 1200, "fuel": 300}
    data = make_shipment_data(
        reference_id="REF",
        load_id="LOAD",
        po_number="PO123",
        shipper="Shipper Inc.",
        consignee="Consignee LLC",
        carrier=DummyCarrier(),
        driver=DummyDriver(),
        pickup=DummyLocation(),
        drop=DummyDrop(),
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Van",
        equipment_size="53",
        load_type="Full",
        commodities=[DummyCommodity()],
        rate_info=DummyRateInfo(),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No tarps",
        dispatcher_name="Dispatch Dan",
        dispatcher_phone="555-3333"
    )
    extraction = make_extraction_response(data)
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Reference ID: REF" in text
    assert "Load ID: LOAD" in text
    assert "PO Number: PO123" in text
    assert "Shipper: Shipper Inc." in text
    assert "Consignee: Consignee LLC" in text
    assert "Carrier Name: CarrierX" in text
    assert "MC Number: MC123" in text
    assert "Carrier Phone: 555-1111" in text
    assert "Driver Name: John Doe" in text
    assert "Driver Phone: 555-2222" in text
    assert "Truck Number: TRK123" in text
    assert "Pickup Location: Warehouse" in text
    assert "Pickup City: Metropolis, NY" in text
    assert "Pickup Appointment: 2024-01-01 10:00" in text
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
    assert "Quantity: 10" in text
    assert "Total Rate: $1500 USD" in text
    assert "Rate Breakdown:" in text
    assert "Special Instructions: Handle with care" in text
    assert "Shipper Instructions: Call before arrival" in text
    assert "Carrier Instructions: No tarps" in text
    assert "Dispatcher: Dispatch Dan" in text
    assert "Dispatcher Phone: 555-3333" in text

def test_format_extraction_as_text_minimal_fields(extraction_service):
    # Only reference_id present
    data = make_shipment_data(reference_id="REFMIN")
    extraction = make_extraction_response(data)
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Reference ID: REFMIN" in text
    # Should not contain other fields
    assert "Load ID:" not in text
    assert "Shipper:" not in text
    assert "Carrier Name:" not in text
    assert "Commodities:" not in text

def test_format_extraction_as_text_empty(extraction_service):
    # All fields empty
    data = make_shipment_data()
    extraction = make_extraction_response(data)
    text = extraction_service.format_extraction_as_text(extraction)
    assert "EXTRACTED STRUCTURED DATA" in text
    # Should not contain any field labels except header
    assert text.strip() == "=== EXTRACTED STRUCTURED DATA ==="

def test_format_extraction_as_text_edge_cases(extraction_service):
    # Pickup/drop with only address
    class DummyLocation:
        name = None
        address = "789 Oak"
        city = None
        state = None
        appointment_time = None
    data = make_shipment_data(
        pickup=DummyLocation(),
        drop=DummyLocation()
    )
    extraction = make_extraction_response(data)
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Pickup Location: 789 Oak" in text
    assert "Drop Location: 789 Oak" in text

def test_create_structured_chunk_happy_path(extraction_service):
    data = make_shipment_data(reference_id="R1")
    extraction = make_extraction_response(data, document_id="doc1.txt")
    chunk = extraction_service.create_structured_chunk(extraction, filename="doc1.txt")
    assert isinstance(chunk, Chunk)
    assert isinstance(chunk.metadata, DocumentMetadata)
    assert chunk.metadata.filename == "doc1.txt"
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.chunk_type == "structured_data"
    assert "EXTRACTED STRUCTURED DATA" in chunk.text

def test_create_structured_chunk_empty(extraction_service):
    extraction = make_extraction_response()
    chunk = extraction_service.create_structured_chunk(extraction, filename="empty.txt")
    assert isinstance(chunk, Chunk)
    assert chunk.metadata.filename == "empty.txt"
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.chunk_type == "structured_data"
    assert "EXTRACTED STRUCTURED DATA" in chunk.text

def test_format_extraction_as_text_rate_info_partial(extraction_service):
    # Only total_rate present
    class DummyRateInfo:
        total_rate = 500
        currency = None
        rate_breakdown = None
    data = make_shipment_data(rate_info=DummyRateInfo())
    extraction = make_extraction_response(data)
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Total Rate: $500 USD" in text
    assert "Rate Breakdown:" not in text

def test_format_extraction_as_text_commodities_partial(extraction_service):
    class DummyCommodity:
        commodity_name = None
        weight = None
        quantity = None
        description = None
    data = make_shipment_data(commodities=[DummyCommodity()])
    extraction = make_extraction_response(data)
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Commodities:" in text
    assert "1. Unknown" in text

def test_format_extraction_as_text_dispatcher_partial(extraction_service):
    data = make_shipment_data(dispatcher_name="Jane", dispatcher_phone=None)
    extraction = make_extraction_response(data)
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Dispatcher: Jane" in text
    assert "Dispatcher Phone:" not in text

def test_format_extraction_as_text_pickup_drop_missing_city_state(extraction_service):
    class DummyLocation:
        name = "Loc"
        address = None
        city = None
        state = None
        appointment_time = None
    data = make_shipment_data(pickup=DummyLocation(), drop=DummyLocation())
    extraction = make_extraction_response(data)
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Pickup Location: Loc" in text
    assert "Drop Location: Loc" in text
    assert "Pickup City:" not in text
    assert "Drop City:" not in text

def test_format_extraction_as_text_pickup_drop_with_city_state(extraction_service):
    class DummyLocation:
        name = None
        address = None
        city = "Springfield"
        state = "IL"
        appointment_time = None
    data = make_shipment_data(pickup=DummyLocation(), drop=DummyLocation())
    extraction = make_extraction_response(data)
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Pickup City: Springfield, IL" in text
    assert "Drop City: Springfield, IL" in text
