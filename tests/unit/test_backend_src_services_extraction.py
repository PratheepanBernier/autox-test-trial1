import pytest
from unittest.mock import patch, MagicMock, create_autospec
from backend.src.services.extraction import ExtractionService
from models.extraction_schema import ShipmentData, ExtractionResponse
from models.schemas import Chunk, DocumentMetadata
import logging

@pytest.fixture(autouse=True)
def patch_logger():
    with patch("backend.src.services.extraction.logger", autospec=logging.getLoggerClass()) as mock_logger:
        yield mock_logger

@pytest.fixture
def mock_settings(monkeypatch):
    class DummySettings:
        QA_MODEL = "test-model"
        GROQ_API_KEY = "test-key"
    monkeypatch.setattr("core.config.settings", DummySettings())

@pytest.fixture
def mock_llm(monkeypatch):
    mock_llm = MagicMock()
    monkeypatch.setattr("backend.src.services.extraction.ChatGroq", lambda **kwargs: mock_llm)
    return mock_llm

@pytest.fixture
def mock_parser(monkeypatch):
    mock_parser = MagicMock()
    mock_parser.get_format_instructions.return_value = "FORMAT"
    monkeypatch.setattr("backend.src.services.extraction.PydanticOutputParser", lambda pydantic_object: mock_parser)
    return mock_parser

@pytest.fixture
def mock_prompt(monkeypatch):
    mock_prompt = MagicMock()
    monkeypatch.setattr("backend.src.services.extraction.ChatPromptTemplate", MagicMock(from_template=lambda t: mock_prompt))
    return mock_prompt

@pytest.fixture
def extraction_service(mock_settings, mock_llm, mock_parser, mock_prompt):
    # Re-import to ensure patches are in effect
    from backend.src.services.extraction import ExtractionService
    return ExtractionService()

def test_extraction_service_init_success(mock_settings, mock_llm, mock_parser, mock_prompt):
    from backend.src.services.extraction import ExtractionService
    service = ExtractionService()
    assert hasattr(service, "llm")
    assert hasattr(service, "parser")
    assert hasattr(service, "extraction_prompt")

def test_extraction_service_init_failure(monkeypatch):
    # Simulate failure in ChatGroq
    monkeypatch.setattr("backend.src.services.extraction.ChatGroq", lambda **kwargs: (_ for _ in ()).throw(Exception("fail")))
    with pytest.raises(Exception) as excinfo:
        from backend.src.services.extraction import ExtractionService
        ExtractionService()
    assert "fail" in str(excinfo.value)

def test_extract_data_happy_path(extraction_service, mock_parser, mock_prompt):
    # Setup
    expected_data = ShipmentData(reference_id="REF123")
    mock_parser.get_format_instructions.return_value = "FORMAT"
    mock_parser.__or__.return_value = mock_parser
    mock_prompt.__or__.return_value = mock_parser
    # Chain: prompt | llm | parser
    chain = MagicMock()
    chain.invoke.return_value = expected_data
    # Patch the chain
    extraction_service.extraction_prompt.__or__ = lambda self, other: chain
    extraction_service.llm.__or__ = lambda self, other: chain
    extraction_service.parser.__or__ = lambda self, other: chain
    # Patch chain.invoke
    chain.invoke.return_value = expected_data
    # Patch the extraction_service to use this chain
    def fake_chain_invoke(args):
        return expected_data
    chain.invoke = fake_chain_invoke
    # Actually call
    result = extraction_service.extract_data("test text", filename="file1.txt")
    assert isinstance(result, ExtractionResponse)
    assert result.data.reference_id == "REF123"
    assert result.document_id == "file1.txt"

def test_extract_data_error_returns_empty(extraction_service, mock_parser, mock_prompt):
    # Chain that raises
    chain = MagicMock()
    chain.invoke.side_effect = Exception("fail extraction")
    extraction_service.extraction_prompt.__or__ = lambda self, other: chain
    extraction_service.llm.__or__ = lambda self, other: chain
    extraction_service.parser.__or__ = lambda self, other: chain
    result = extraction_service.extract_data("bad text", filename="fail.txt")
    assert isinstance(result, ExtractionResponse)
    # Should be empty ShipmentData
    assert isinstance(result.data, ShipmentData)
    assert result.document_id == "fail.txt"
    # All fields should be default/None
    assert result.data.reference_id is None

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
        pickup=Location(name="Warehouse A", address="123 Main", city="CityA", state="ST", zip="12345", appointment_time="2024-01-01T10:00"),
        drop=Location(name="Store B", address="456 Elm", city="CityB", state="TS", zip="67890"),
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Van",
        equipment_size=53,
        load_type="Full",
        commodities=[Commodity(commodity_name="Widgets", weight="1000 lbs", quantity="10", description="Blue widgets")],
        rate_info=RateInfo(total_rate=1200.0, currency="USD", rate_breakdown={"linehaul": 1000, "fuel": 200}),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No tarps",
        dispatcher_name="Jane Smith",
        dispatcher_phone="555-3333"
    )
    extraction = ExtractionResponse(data=data, document_id="doc1.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    # Check for presence of all major fields
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
    assert "Pickup Location: Warehouse A" in text
    assert "Pickup City: CityA, ST" in text
    assert "Pickup Appointment: 2024-01-01T10:00" in text
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
    assert "Total Rate: $1200.0 USD" in text
    assert "Rate Breakdown:" in text
    assert "Special Instructions: Handle with care" in text
    assert "Shipper Instructions: Call before arrival" in text
    assert "Carrier Instructions: No tarps" in text
    assert "Dispatcher: Jane Smith" in text
    assert "Dispatcher Phone: 555-3333" in text

def test_format_extraction_as_text_minimal_fields(extraction_service):
    # Only one field set
    data = ShipmentData(reference_id="ONLYREF")
    extraction = ExtractionResponse(data=data, document_id="doc2.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Reference ID: ONLYREF" in text
    # Should not contain other fields
    assert "Load ID:" not in text
    assert "Shipper:" not in text

def test_format_extraction_as_text_empty(extraction_service):
    data = ShipmentData()
    extraction = ExtractionResponse(data=data, document_id="doc3.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    # Only header should be present
    assert text.strip().startswith("=== EXTRACTED STRUCTURED DATA ===")

def test_format_extraction_as_text_commodities_edge(extraction_service):
    from models.extraction_schema import Commodity
    data = ShipmentData(commodities=[Commodity()])
    extraction = ExtractionResponse(data=data, document_id="doc4.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Commodities:" in text
    assert "Unknown" in text

def test_create_structured_chunk_happy_path(extraction_service):
    data = ShipmentData(reference_id="R1")
    extraction = ExtractionResponse(data=data, document_id="doc5.txt")
    chunk = extraction_service.create_structured_chunk(extraction, filename="doc5.txt")
    assert isinstance(chunk, Chunk)
    assert isinstance(chunk.metadata, DocumentMetadata)
    assert chunk.metadata.filename == "doc5.txt"
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.chunk_type == "structured_data"
    assert "EXTRACTED STRUCTURED DATA" in chunk.text

def test_create_structured_chunk_empty(extraction_service):
    data = ShipmentData()
    extraction = ExtractionResponse(data=data, document_id="doc6.txt")
    chunk = extraction_service.create_structured_chunk(extraction, filename="doc6.txt")
    assert isinstance(chunk, Chunk)
    assert chunk.metadata.filename == "doc6.txt"
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.chunk_type == "structured_data"
    assert "EXTRACTED STRUCTURED DATA" in chunk.text

def test_format_extraction_as_text_boundary_equipment_size(extraction_service):
    # Test boundary value for equipment_size (0 and very large)
    data = ShipmentData(equipment_size=0)
    extraction = ExtractionResponse(data=data, document_id="doc7.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    # Should not print "Equipment Size: 0 feet"
    assert "Equipment Size:" not in text

    data2 = ShipmentData(equipment_size=1000)
    extraction2 = ExtractionResponse(data=data2, document_id="doc8.txt")
    text2 = extraction_service.format_extraction_as_text(extraction2)
    assert "Equipment Size: 1000 feet" in text2

def test_format_extraction_as_text_null_fields(extraction_service):
    # All fields explicitly set to None
    data = ShipmentData(
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
    extraction = ExtractionResponse(data=data, document_id="doc9.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    # Only header should be present
    assert text.strip().startswith("=== EXTRACTED STRUCTURED DATA ===")
    assert "Reference ID:" not in text
    assert "Load ID:" not in text
    assert "Shipper:" not in text
    assert "Commodities:" not in text
    assert "Dispatcher:" not in text
