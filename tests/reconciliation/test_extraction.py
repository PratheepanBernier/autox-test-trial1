import pytest
from unittest.mock import patch, MagicMock
from backend.src.services.extraction import ExtractionService
from models.extraction_schema import ShipmentData, ExtractionResponse
from models.schemas import Chunk, DocumentMetadata

@pytest.fixture
def extraction_service():
    # Patch ChatGroq, ChatPromptTemplate, PydanticOutputParser, and settings
    with patch("backend.src.services.extraction.ChatGroq") as mock_llm, \
         patch("backend.src.services.extraction.ChatPromptTemplate") as mock_prompt, \
         patch("backend.src.services.extraction.PydanticOutputParser") as mock_parser, \
         patch("backend.src.services.extraction.settings") as mock_settings:
        # Mock settings
        mock_settings.QA_MODEL = "test-model"
        mock_settings.GROQ_API_KEY = "test-key"
        # Mock LLM
        mock_llm.return_value = MagicMock()
        # Mock Prompt
        mock_prompt.from_template.return_value = MagicMock()
        # Mock Parser
        parser_instance = MagicMock()
        parser_instance.get_format_instructions.return_value = "FORMAT"
        mock_parser.return_value = parser_instance
        yield ExtractionService()

@pytest.fixture
def minimal_shipment_data():
    return ShipmentData(
        reference_id="REF123",
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
        commodities=[],
        rate_info=None,
        special_instructions=None,
        shipper_instructions=None,
        carrier_instructions=None,
        dispatcher_name=None,
        dispatcher_phone=None
    )

@pytest.fixture
def full_shipment_data():
    from models.extraction_schema import Carrier, Driver, Location, Commodity, RateInfo
    return ShipmentData(
        reference_id="REF123",
        load_id="LOAD456",
        po_number="PO789",
        shipper="Shipper Inc.",
        consignee="Consignee LLC",
        carrier=Carrier(carrier_name="CarrierX", mc_number="MC123", phone="555-1111", email="c@x.com"),
        driver=Driver(driver_name="John Doe", cell_number="555-2222", truck_number="TRK123", trailer_number="TRL456"),
        pickup=Location(name="Warehouse A", address="123 Main St", city="CityA", state="ST", zip="12345", appointment_time="2024-01-01T08:00:00"),
        drop=Location(name="Warehouse B", address="456 Elm St", city="CityB", state="ST", zip="67890", appointment_time="2024-01-02T09:00:00"),
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Van",
        equipment_size=53,
        load_type="Full",
        commodities=[
            Commodity(commodity_name="Widgets", weight="1000 lbs", quantity="10", description="Blue widgets"),
            Commodity(commodity_name="Gadgets", weight="500 lbs", quantity="5", description="Red gadgets")
        ],
        rate_info=RateInfo(total_rate=2500, currency="USD", rate_breakdown={"base": 2000, "fuel": 500}),
        special_instructions="Handle with care.",
        shipper_instructions="Call before arrival.",
        carrier_instructions="No partial loads.",
        dispatcher_name="Jane Smith",
        dispatcher_phone="555-3333"
    )

def test_extract_data_happy_path(extraction_service, minimal_shipment_data):
    # Patch the chain.invoke to return minimal_shipment_data
    with patch.object(extraction_service, "extraction_prompt") as mock_prompt, \
         patch.object(extraction_service, "llm") as mock_llm, \
         patch.object(extraction_service, "parser") as mock_parser:
        # Compose the chain
        chain = MagicMock()
        chain.invoke.return_value = minimal_shipment_data
        # Compose the chain using | operator
        mock_prompt.__or__.return_value = chain
        chain2 = MagicMock()
        chain2.__or__.return_value = chain
        mock_llm.__or__.return_value = chain
        # parser.get_format_instructions
        mock_parser.get_format_instructions.return_value = "FORMAT"
        # Actually call extract_data
        resp = extraction_service.extract_data("test text", filename="doc1.txt")
        assert isinstance(resp, ExtractionResponse)
        assert resp.document_id == "doc1.txt"
        assert resp.data.reference_id == "REF123"
        assert resp.data.load_id is None

def test_extract_data_error_returns_empty(extraction_service):
    # Patch the chain.invoke to raise an exception
    with patch.object(extraction_service, "extraction_prompt") as mock_prompt, \
         patch.object(extraction_service, "llm") as mock_llm, \
         patch.object(extraction_service, "parser") as mock_parser:
        chain = MagicMock()
        chain.invoke.side_effect = Exception("LLM error")
        mock_prompt.__or__.return_value = chain
        chain2 = MagicMock()
        chain2.__or__.return_value = chain
        mock_llm.__or__.return_value = chain
        mock_parser.get_format_instructions.return_value = "FORMAT"
        resp = extraction_service.extract_data("bad text", filename="fail.txt")
        assert isinstance(resp, ExtractionResponse)
        assert resp.document_id == "fail.txt"
        # Should be empty ShipmentData
        assert isinstance(resp.data, ShipmentData)
        # All fields should be None or empty
        assert resp.data.reference_id is None
        assert resp.data.commodities == []

def test_format_extraction_as_text_minimal(extraction_service, minimal_shipment_data):
    extraction = ExtractionResponse(data=minimal_shipment_data, document_id="doc1.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Reference ID: REF123" in text
    assert "Load ID:" not in text
    assert "=== EXTRACTED STRUCTURED DATA ===" in text

def test_format_extraction_as_text_full(extraction_service, full_shipment_data):
    extraction = ExtractionResponse(data=full_shipment_data, document_id="doc2.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    # Check for presence of all major fields
    assert "Reference ID: REF123" in text
    assert "Load ID: LOAD456" in text
    assert "PO Number: PO789" in text
    assert "Shipper: Shipper Inc." in text
    assert "Consignee: Consignee LLC" in text
    assert "Carrier Name: CarrierX" in text
    assert "MC Number: MC123" in text
    assert "Driver Name: John Doe" in text
    assert "Pickup Location: Warehouse A" in text
    assert "Drop Location: Warehouse B" in text
    assert "Shipping Date: 2024-01-01" in text
    assert "Delivery Date: 2024-01-02" in text
    assert "Equipment Type: Van" in text
    assert "Equipment Size: 53 feet" in text
    assert "Load Type: Full" in text
    assert "Commodities:" in text
    assert "1. Widgets" in text
    assert "2. Gadgets" in text
    assert "Total Rate: $2500 USD" in text
    assert "Special Instructions: Handle with care." in text
    assert "Shipper Instructions: Call before arrival." in text
    assert "Carrier Instructions: No partial loads." in text
    assert "Dispatcher: Jane Smith" in text
    assert "Dispatcher Phone: 555-3333" in text

def test_format_extraction_as_text_edge_cases(extraction_service):
    # All fields None/empty
    empty_data = ShipmentData()
    extraction = ExtractionResponse(data=empty_data, document_id="doc3.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    # Only header should be present
    assert text.strip() == "=== EXTRACTED STRUCTURED DATA ==="

def test_create_structured_chunk_output(extraction_service, minimal_shipment_data):
    extraction = ExtractionResponse(data=minimal_shipment_data, document_id="doc4.txt")
    chunk = extraction_service.create_structured_chunk(extraction, filename="doc4.txt")
    assert isinstance(chunk, Chunk)
    assert isinstance(chunk.metadata, DocumentMetadata)
    assert chunk.metadata.filename == "doc4.txt"
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.chunk_type == "structured_data"
    assert "Reference ID: REF123" in chunk.text

def test_create_structured_chunk_and_format_consistency(extraction_service, full_shipment_data):
    extraction = ExtractionResponse(data=full_shipment_data, document_id="doc5.txt")
    # Reconciliation: output of format_extraction_as_text should match chunk.text
    formatted = extraction_service.format_extraction_as_text(extraction)
    chunk = extraction_service.create_structured_chunk(extraction, filename="doc5.txt")
    assert chunk.text == formatted

def test_format_extraction_as_text_boundary_conditions(extraction_service, full_shipment_data):
    # Remove optional fields to test boundaries
    full_shipment_data.commodities = []
    full_shipment_data.rate_info = None
    full_shipment_data.special_instructions = None
    full_shipment_data.dispatcher_name = None
    extraction = ExtractionResponse(data=full_shipment_data, document_id="doc6.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Commodities:" not in text
    assert "Total Rate:" not in text
    assert "Special Instructions:" not in text
    assert "Dispatcher:" not in text

def test_extract_data_equivalent_paths_consistency(extraction_service, minimal_shipment_data):
    # Reconciliation: extracting the same text twice yields equivalent ExtractionResponse
    with patch.object(extraction_service, "extraction_prompt") as mock_prompt, \
         patch.object(extraction_service, "llm") as mock_llm, \
         patch.object(extraction_service, "parser") as mock_parser:
        chain = MagicMock()
        chain.invoke.return_value = minimal_shipment_data
        mock_prompt.__or__.return_value = chain
        chain2 = MagicMock()
        chain2.__or__.return_value = chain
        mock_llm.__or__.return_value = chain
        mock_parser.get_format_instructions.return_value = "FORMAT"
        resp1 = extraction_service.extract_data("same text", filename="doc7.txt")
        resp2 = extraction_service.extract_data("same text", filename="doc7.txt")
        assert resp1.data == resp2.data
        assert resp1.document_id == resp2.document_id
