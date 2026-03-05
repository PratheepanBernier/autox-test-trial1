# source_hash: 88361007730f06bf
import pytest
from unittest.mock import patch, MagicMock
from backend.src.services.extraction import ExtractionService
from models.extraction_schema import ShipmentData, ExtractionResponse
from models.schemas import Chunk, DocumentMetadata

@pytest.fixture
def extraction_service():
    # Patch ChatGroq, ChatPromptTemplate, PydanticOutputParser, and settings for isolation
    with patch("backend.src.services.extraction.ChatGroq") as mock_llm, \
         patch("backend.src.services.extraction.ChatPromptTemplate") as mock_prompt, \
         patch("backend.src.services.extraction.PydanticOutputParser") as mock_parser, \
         patch("backend.src.services.extraction.settings") as mock_settings:
        # Mock settings
        mock_settings.QA_MODEL = "mock-model"
        mock_settings.GROQ_API_KEY = "mock-key"
        # Mock prompt template
        mock_prompt.from_template.return_value = MagicMock()
        # Mock parser
        parser_instance = MagicMock()
        parser_instance.get_format_instructions.return_value = "FORMAT"
        mock_parser.return_value = parser_instance
        # Mock LLM
        mock_llm.return_value = MagicMock()
        yield ExtractionService()

@pytest.fixture
def minimal_shipment_data():
    return ShipmentData()

@pytest.fixture
def full_shipment_data():
    # Fill all fields with non-null values for a comprehensive test
    from models.extraction_schema import Carrier, Driver, Location, Commodity, RateInfo
    return ShipmentData(
        reference_id="REF123",
        load_id="LOAD456",
        po_number="PO789",
        shipper="Acme Shipper",
        consignee="Beta Consignee",
        carrier=Carrier(carrier_name="CarrierX", mc_number="MC123", phone="555-1111"),
        driver=Driver(driver_name="John Doe", cell_number="555-2222", truck_number="TRK123"),
        pickup=Location(name="Warehouse A", address="123 Main St", city="Metropolis", state="NY", zip="10001", appointment_time="2024-01-01T08:00:00"),
        drop=Location(name="Store B", address="456 Elm St", city="Gotham", state="NJ", zip="07001", appointment_time="2024-01-02T09:00:00"),
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Van",
        equipment_size=53,
        load_type="Full",
        commodities=[Commodity(commodity_name="Widgets", weight="1000 lbs", quantity="100", description="Blue widgets")],
        rate_info=RateInfo(total_rate=1200.0, currency="USD", rate_breakdown={"linehaul": 1000, "fuel": 200}),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No partial loads",
        dispatcher_name="Jane Smith",
        dispatcher_phone="555-3333",
        dispatcher_email="jane@dispatch.com"
    )

@pytest.fixture
def extraction_response_full(full_shipment_data):
    return ExtractionResponse(data=full_shipment_data, document_id="doc1.pdf")

@pytest.fixture
def extraction_response_minimal(minimal_shipment_data):
    return ExtractionResponse(data=minimal_shipment_data, document_id="doc2.pdf")

def test_extract_data_happy_path(extraction_service, full_shipment_data):
    # Patch the chain.invoke to return a full_shipment_data instance
    with patch.object(extraction_service, "extraction_prompt") as mock_prompt, \
         patch.object(extraction_service, "llm") as mock_llm, \
         patch.object(extraction_service, "parser") as mock_parser:
        # Compose the chain: prompt | llm | parser
        chain = MagicMock()
        chain.invoke.return_value = full_shipment_data
        # Simulate the chain composition
        mock_prompt.__or__.return_value = chain
        chain.__or__.return_value = chain
        # parser.get_format_instructions
        mock_parser.get_format_instructions.return_value = "FORMAT"
        # Actually call extract_data
        result = extraction_service.extract_data("some text", filename="doc1.pdf")
        assert isinstance(result, ExtractionResponse)
        assert result.data == full_shipment_data
        assert result.document_id == "doc1.pdf"

def test_extract_data_returns_empty_on_exception(extraction_service, minimal_shipment_data):
    with patch.object(extraction_service, "extraction_prompt") as mock_prompt, \
         patch.object(extraction_service, "llm") as mock_llm, \
         patch.object(extraction_service, "parser") as mock_parser:
        chain = MagicMock()
        chain.invoke.side_effect = Exception("LLM failure")
        mock_prompt.__or__.return_value = chain
        chain.__or__.return_value = chain
        mock_parser.get_format_instructions.return_value = "FORMAT"
        result = extraction_service.extract_data("bad input", filename="fail.pdf")
        assert isinstance(result, ExtractionResponse)
        # Should be empty ShipmentData
        assert result.data == minimal_shipment_data
        assert result.document_id == "fail.pdf"

def test_extract_data_boundary_empty_text(extraction_service, minimal_shipment_data):
    with patch.object(extraction_service, "extraction_prompt") as mock_prompt, \
         patch.object(extraction_service, "llm") as mock_llm, \
         patch.object(extraction_service, "parser") as mock_parser:
        chain = MagicMock()
        chain.invoke.return_value = minimal_shipment_data
        mock_prompt.__or__.return_value = chain
        chain.__or__.return_value = chain
        mock_parser.get_format_instructions.return_value = "FORMAT"
        result = extraction_service.extract_data("", filename="empty.pdf")
        assert isinstance(result, ExtractionResponse)
        assert result.data == minimal_shipment_data
        assert result.document_id == "empty.pdf"

def test_format_extraction_as_text_full(extraction_service, extraction_response_full):
    text = extraction_service.format_extraction_as_text(extraction_response_full)
    # Check that all major fields are present in the output
    assert "Reference ID: REF123" in text
    assert "Load ID: LOAD456" in text
    assert "PO Number: PO789" in text
    assert "Shipper: Acme Shipper" in text
    assert "Consignee: Beta Consignee" in text
    assert "Carrier Name: CarrierX" in text
    assert "MC Number: MC123" in text
    assert "Carrier Phone: 555-1111" in text
    assert "Driver Name: John Doe" in text
    assert "Driver Phone: 555-2222" in text
    assert "Truck Number: TRK123" in text
    assert "Pickup Location: Warehouse A" in text
    assert "Pickup City: Metropolis, NY" in text
    assert "Pickup Appointment: 2024-01-01T08:00:00" in text
    assert "Drop Location: Store B" in text
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
    assert "Total Rate: $1200.0 USD" in text
    assert "Rate Breakdown:" in text
    assert "Special Instructions: Handle with care" in text
    assert "Shipper Instructions: Call before arrival" in text
    assert "Carrier Instructions: No partial loads" in text
    assert "Dispatcher: Jane Smith" in text
    assert "Dispatcher Phone: 555-3333" in text

def test_format_extraction_as_text_minimal(extraction_service, extraction_response_minimal):
    text = extraction_service.format_extraction_as_text(extraction_response_minimal)
    # Should only contain the header line
    assert text.strip() == "=== EXTRACTED STRUCTURED DATA ==="

def test_format_extraction_as_text_edge_cases(extraction_service):
    # Some fields present, some missing, some empty strings
    from models.extraction_schema import ShipmentData, Carrier, Driver, Location, Commodity, RateInfo
    data = ShipmentData(
        reference_id="",
        load_id=None,
        po_number="PO999",
        shipper=None,
        consignee="",
        carrier=Carrier(carrier_name="", mc_number=None, phone=""),
        driver=None,
        pickup=Location(name=None, address=None, city=None, state=None, zip=None, appointment_time=None),
        drop=None,
        shipping_date=None,
        delivery_date="",
        equipment_type="",
        equipment_size=None,
        load_type=None,
        commodities=[],
        rate_info=None,
        special_instructions=None,
        shipper_instructions="",
        carrier_instructions=None,
        dispatcher_name=None,
        dispatcher_phone=None,
        dispatcher_email=None
    )
    extraction = ExtractionResponse(data=data, document_id="edge.pdf")
    text = extraction_service.format_extraction_as_text(extraction)
    # Only PO Number should appear
    assert "PO Number: PO999" in text
    # No other fields should be present
    assert "Reference ID:" not in text
    assert "Load ID:" not in text
    assert "Shipper:" not in text
    assert "Consignee:" not in text
    assert "Carrier Name:" not in text
    assert "Driver Name:" not in text
    assert "Pickup Location:" not in text
    assert "Drop Location:" not in text
    assert "Shipping Date:" not in text
    assert "Delivery Date:" not in text
    assert "Equipment Type:" not in text
    assert "Commodities:" not in text
    assert "Total Rate:" not in text
    assert "Special Instructions:" not in text
    assert "Dispatcher:" not in text

def test_create_structured_chunk_reconciles_with_format_text(extraction_service, extraction_response_full):
    # Reconciliation: create_structured_chunk should use format_extraction_as_text output
    chunk = extraction_service.create_structured_chunk(extraction_response_full, filename="doc1.pdf")
    formatted_text = extraction_service.format_extraction_as_text(extraction_response_full)
    assert isinstance(chunk, Chunk)
    assert chunk.text == formatted_text
    assert isinstance(chunk.metadata, DocumentMetadata)
    assert chunk.metadata.filename == "doc1.pdf"
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.source == "doc1.pdf - Extracted Data"
    assert chunk.metadata.chunk_type == "structured_data"

def test_create_structured_chunk_minimal(extraction_service, extraction_response_minimal):
    chunk = extraction_service.create_structured_chunk(extraction_response_minimal, filename="doc2.pdf")
    formatted_text = extraction_service.format_extraction_as_text(extraction_response_minimal)
    assert chunk.text == formatted_text
    assert chunk.metadata.filename == "doc2.pdf"
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.source == "doc2.pdf - Extracted Data"
    assert chunk.metadata.chunk_type == "structured_data"

def test_format_extraction_as_text_and_create_structured_chunk_consistency(extraction_service, extraction_response_full):
    # Reconciliation: Both methods should produce consistent output for the same input
    text = extraction_service.format_extraction_as_text(extraction_response_full)
    chunk = extraction_service.create_structured_chunk(extraction_response_full, filename="doc1.pdf")
    assert chunk.text == text

def test_extract_data_and_format_extraction_as_text_reconciliation(extraction_service, full_shipment_data):
    # Reconciliation: extract_data output should be compatible with format_extraction_as_text
    with patch.object(extraction_service, "extraction_prompt") as mock_prompt, \
         patch.object(extraction_service, "llm") as mock_llm, \
         patch.object(extraction_service, "parser") as mock_parser:
        chain = MagicMock()
        chain.invoke.return_value = full_shipment_data
        mock_prompt.__or__.return_value = chain
        chain.__or__.return_value = chain
        mock_parser.get_format_instructions.return_value = "FORMAT"
        extraction_response = extraction_service.extract_data("test text", filename="doc3.pdf")
        text = extraction_service.format_extraction_as_text(extraction_response)
        # Should contain at least one known field
        assert "Reference ID: REF123" in text
        assert "Load ID: LOAD456" in text
        assert extraction_response.document_id == "doc3.pdf"
