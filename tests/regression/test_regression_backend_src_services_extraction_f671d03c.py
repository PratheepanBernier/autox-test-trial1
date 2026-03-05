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

def test_extraction_service_init_success(mock_settings, mock_llm, mock_parser, mock_prompt):
    service = ExtractionService()
    assert hasattr(service, "llm")
    assert hasattr(service, "parser")
    assert hasattr(service, "extraction_prompt")

def test_extraction_service_init_failure(monkeypatch):
    monkeypatch.setattr("backend.src.services.extraction.ChatGroq", MagicMock(side_effect=Exception("fail")))
    with pytest.raises(Exception):
        ExtractionService()

def test_extract_data_happy_path(extraction_service, mock_parser, mock_prompt):
    # Setup
    fake_result = ShipmentData(reference_id="REF123")
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = fake_result
    # Compose chain: prompt | llm | parser
    extraction_service.extraction_prompt.__or__.return_value = extraction_service.llm
    extraction_service.llm.__or__.return_value = extraction_service.parser
    extraction_service.parser.__or__.return_value = chain_mock
    # Patch chain
    with patch.object(extraction_service, "extraction_prompt", create_autospec(extraction_service.extraction_prompt, instance=True)):
        with patch.object(extraction_service, "llm", create_autospec(extraction_service.llm, instance=True)):
            with patch.object(extraction_service, "parser", create_autospec(extraction_service.parser, instance=True)):
                # Compose chain
                chain = MagicMock()
                chain.invoke.return_value = fake_result
                extraction_service.extraction_prompt.__or__.return_value = extraction_service.llm
                extraction_service.llm.__or__.return_value = extraction_service.parser
                extraction_service.parser.__or__.return_value = chain
                extraction_service.parser.get_format_instructions.return_value = "FORMAT"
                # Test
                text = "Sample document text"
                filename = "doc1.txt"
                resp = extraction_service.extract_data(text, filename)
                assert isinstance(resp, ExtractionResponse)
                assert resp.data.reference_id == "REF123"
                assert resp.document_id == filename

def test_extract_data_empty_text(extraction_service, mock_parser, mock_prompt):
    fake_result = ShipmentData()
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = fake_result
    extraction_service.extraction_prompt.__or__.return_value = extraction_service.llm
    extraction_service.llm.__or__.return_value = extraction_service.parser
    extraction_service.parser.__or__.return_value = chain_mock
    extraction_service.parser.get_format_instructions.return_value = "FORMAT"
    with patch.object(extraction_service, "extraction_prompt", create_autospec(extraction_service.extraction_prompt, instance=True)):
        with patch.object(extraction_service, "llm", create_autospec(extraction_service.llm, instance=True)):
            with patch.object(extraction_service, "parser", create_autospec(extraction_service.parser, instance=True)):
                chain = MagicMock()
                chain.invoke.return_value = fake_result
                extraction_service.extraction_prompt.__or__.return_value = extraction_service.llm
                extraction_service.llm.__or__.return_value = extraction_service.parser
                extraction_service.parser.__or__.return_value = chain
                extraction_service.parser.get_format_instructions.return_value = "FORMAT"
                resp = extraction_service.extract_data("", "empty.txt")
                assert isinstance(resp, ExtractionResponse)
                assert resp.data.reference_id is None
                assert resp.document_id == "empty.txt"

def test_extract_data_chain_raises_exception_returns_empty(extraction_service, mock_parser, mock_prompt):
    chain_mock = MagicMock()
    chain_mock.invoke.side_effect = Exception("fail")
    extraction_service.extraction_prompt.__or__.return_value = extraction_service.llm
    extraction_service.llm.__or__.return_value = extraction_service.parser
    extraction_service.parser.__or__.return_value = chain_mock
    extraction_service.parser.get_format_instructions.return_value = "FORMAT"
    with patch.object(extraction_service, "extraction_prompt", create_autospec(extraction_service.extraction_prompt, instance=True)):
        with patch.object(extraction_service, "llm", create_autospec(extraction_service.llm, instance=True)):
            with patch.object(extraction_service, "parser", create_autospec(extraction_service.parser, instance=True)):
                chain = MagicMock()
                chain.invoke.side_effect = Exception("fail")
                extraction_service.extraction_prompt.__or__.return_value = extraction_service.llm
                extraction_service.llm.__or__.return_value = extraction_service.parser
                extraction_service.parser.__or__.return_value = chain
                extraction_service.parser.get_format_instructions.return_value = "FORMAT"
                resp = extraction_service.extract_data("bad input", "fail.txt")
                assert isinstance(resp, ExtractionResponse)
                assert isinstance(resp.data, ShipmentData)
                assert resp.document_id == "fail.txt"

def test_format_extraction_as_text_full_fields(extraction_service):
    # Fill all fields
    from models.extraction_schema import Carrier, Driver, Location, Commodity, RateInfo
    data = ShipmentData(
        reference_id="REF1",
        load_id="LOAD1",
        po_number="PO1",
        shipper="Shipper Inc.",
        consignee="Consignee LLC",
        carrier=Carrier(carrier_name="CarrierX", mc_number="MC123", phone="555-1111"),
        driver=Driver(driver_name="John Doe", cell_number="555-2222", truck_number="TRK123"),
        pickup=Location(name="Warehouse", address="123 St", city="CityA", state="ST", zip="12345", appointment_time="2024-01-01 10:00"),
        drop=Location(name="Store", address="456 Ave", city="CityB", state="TS", zip="67890"),
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Van",
        equipment_size=53,
        load_type="Full",
        commodities=[Commodity(commodity_name="Widgets", weight="1000 lbs", quantity="100")],
        rate_info=RateInfo(total_rate=1200, currency="USD", rate_breakdown={"linehaul": 1000, "fuel": 200}),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No partial loads",
        dispatcher_name="Jane Smith",
        dispatcher_phone="555-3333"
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Reference ID: REF1" in text
    assert "Load ID: LOAD1" in text
    assert "PO Number: PO1" in text
    assert "Shipper: Shipper Inc." in text
    assert "Consignee: Consignee LLC" in text
    assert "Carrier Name: CarrierX" in text
    assert "MC Number: MC123" in text
    assert "Carrier Phone: 555-1111" in text
    assert "Driver Name: John Doe" in text
    assert "Driver Phone: 555-2222" in text
    assert "Truck Number: TRK123" in text
    assert "Pickup Location: Warehouse" in text
    assert "Pickup City: CityA, ST" in text
    assert "Pickup Appointment: 2024-01-01 10:00" in text
    assert "Drop Location: Store" in text
    assert "Drop City: CityB, TS" in text
    assert "Shipping Date: 2024-01-01" in text
    assert "Delivery Date: 2024-01-02" in text
    assert "Equipment Type: Van" in text
    assert "Equipment Size: 53 feet" in text
    assert "Load Type: Full" in text
    assert "Commodities:" in text
    assert "1. Widgets" in text
    assert "Weight: 1000 lbs" in text
    assert "Quantity: 100" in text
    assert "Total Rate: $1200 USD" in text
    assert "Rate Breakdown:" in text
    assert "Special Instructions: Handle with care" in text
    assert "Shipper Instructions: Call before arrival" in text
    assert "Carrier Instructions: No partial loads" in text
    assert "Dispatcher: Jane Smith" in text
    assert "Dispatcher Phone: 555-3333" in text

def test_format_extraction_as_text_minimal_fields(extraction_service):
    data = ShipmentData()
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "EXTRACTED STRUCTURED DATA" in text
    # Should not contain any field values
    assert "Reference ID:" not in text
    assert "Shipper:" not in text
    assert "Carrier Name:" not in text
    assert "Driver Name:" not in text
    assert "Pickup Location:" not in text
    assert "Drop Location:" not in text
    assert "Shipping Date:" not in text
    assert "Equipment Type:" not in text
    assert "Commodities:" not in text
    assert "Total Rate:" not in text
    assert "Special Instructions:" not in text
    assert "Dispatcher:" not in text

def test_format_extraction_as_text_partial_fields(extraction_service):
    from models.extraction_schema import Carrier
    data = ShipmentData(
        reference_id="REF2",
        carrier=Carrier(carrier_name="CarrierY"),
        commodities=[]
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Reference ID: REF2" in text
    assert "Carrier Name: CarrierY" in text
    assert "MC Number:" not in text
    assert "Driver Name:" not in text
    assert "Commodities:" not in text

def test_create_structured_chunk_happy_path(extraction_service):
    from models.extraction_schema import Carrier
    data = ShipmentData(reference_id="REF3", carrier=Carrier(carrier_name="CarrierZ"))
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    chunk = extraction_service.create_structured_chunk(extraction, "doc.txt")
    assert isinstance(chunk, Chunk)
    assert isinstance(chunk.metadata, DocumentMetadata)
    assert chunk.metadata.filename == "doc.txt"
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.chunk_type == "structured_data"
    assert "EXTRACTED STRUCTURED DATA" in chunk.text
    assert "Reference ID: REF3" in chunk.text
    assert "Carrier Name: CarrierZ" in chunk.text

def test_create_structured_chunk_empty_data(extraction_service):
    data = ShipmentData()
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    chunk = extraction_service.create_structured_chunk(extraction, "doc.txt")
    assert isinstance(chunk, Chunk)
    assert isinstance(chunk.metadata, DocumentMetadata)
    assert chunk.metadata.filename == "doc.txt"
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.chunk_type == "structured_data"
    assert "EXTRACTED STRUCTURED DATA" in chunk.text
    # No fields should be present
    assert "Reference ID:" not in chunk.text
    assert "Carrier Name:" not in chunk.text

def test_format_extraction_as_text_handles_null_and_zero_values(extraction_service):
    from models.extraction_schema import RateInfo
    data = ShipmentData(
        reference_id=None,
        rate_info=RateInfo(total_rate=0, currency=None, rate_breakdown=None)
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    # Should show zero rate, but not currency if None
    assert "Total Rate: $0 USD" in text

def test_format_extraction_as_text_handles_multiple_commodities(extraction_service):
    from models.extraction_schema import Commodity
    data = ShipmentData(
        commodities=[
            Commodity(commodity_name="Item1", weight="10", quantity="1"),
            Commodity(commodity_name="Item2", weight="20", quantity="2"),
        ]
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "1. Item1" in text
    assert "2. Item2" in text
    assert "Weight: 10" in text
    assert "Weight: 20" in text
    assert "Quantity: 1" in text
    assert "Quantity: 2" in text

def test_format_extraction_as_text_pickup_and_drop_fallbacks(extraction_service):
    from models.extraction_schema import Location
    # Only address for pickup, only address for drop
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
