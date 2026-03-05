import pytest
from unittest.mock import patch, MagicMock
from backend.src.services.extraction import ExtractionService
from models.extraction_schema import ShipmentData, ExtractionResponse
from models.schemas import Chunk, DocumentMetadata

@pytest.fixture
def extraction_service():
    # Patch ChatGroq, PydanticOutputParser, ChatPromptTemplate for all tests
    with patch("backend.src.services.extraction.ChatGroq") as mock_llm, \
         patch("backend.src.services.extraction.PydanticOutputParser") as mock_parser, \
         patch("backend.src.services.extraction.ChatPromptTemplate") as mock_prompt:
        # Mock LLM
        mock_llm.return_value = MagicMock()
        # Mock parser
        mock_parser_inst = MagicMock()
        mock_parser_inst.get_format_instructions.return_value = "FORMAT"
        mock_parser.return_value = mock_parser_inst
        # Mock prompt
        mock_prompt_inst = MagicMock()
        mock_prompt.from_template.return_value = mock_prompt_inst
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
        dispatcher_phone=None,
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
        carrier=Carrier(carrier_name="CarrierX", mc_number="MC111", phone="555-1111", email="carrier@example.com"),
        driver=Driver(driver_name="John Doe", cell_number="555-2222", truck_number="TRK123", trailer_number="TRL456"),
        pickup=Location(name="Warehouse A", address="123 Main St", city="CityA", state="ST", zip="12345", appointment_time="2024-01-01T08:00:00"),
        drop=Location(name="Warehouse B", address="456 Elm St", city="CityB", state="ST", zip="67890", appointment_time="2024-01-02T09:00:00"),
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Van",
        equipment_size=53,
        load_type="Full",
        commodities=[
            Commodity(commodity_name="Widgets", weight="1000 lbs", quantity="100", description="Blue widgets"),
            Commodity(commodity_name="Gadgets", weight="500 lbs", quantity="50", description="Red gadgets"),
        ],
        rate_info=RateInfo(total_rate=2500.0, currency="USD", rate_breakdown={"linehaul": 2000, "fuel": 500}),
        special_instructions="Handle with care.",
        shipper_instructions="Call before arrival.",
        carrier_instructions="No partial loads.",
        dispatcher_name="Jane Smith",
        dispatcher_phone="555-3333",
    )

def test_extract_data_happy_path(extraction_service, minimal_shipment_data):
    # Patch the chain to return minimal_shipment_data
    with patch.object(extraction_service, "extraction_prompt") as mock_prompt, \
         patch.object(extraction_service, "llm") as mock_llm, \
         patch.object(extraction_service, "parser") as mock_parser:
        # Compose the chain
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = minimal_shipment_data
        # Simulate chain composition
        mock_prompt.__or__.return_value = mock_chain
        mock_chain.__or__.return_value = mock_chain
        mock_parser.__or__.return_value = mock_chain
        # Patch parser.get_format_instructions
        mock_parser.get_format_instructions.return_value = "FORMAT"
        # Patch chain.invoke
        mock_chain.invoke.return_value = minimal_shipment_data

        # Compose the chain as in the code
        extraction_service.extraction_prompt = mock_prompt
        extraction_service.llm = mock_llm
        extraction_service.parser = mock_parser

        result = extraction_service.extract_data("test text", filename="doc1.txt")
        assert isinstance(result, ExtractionResponse)
        assert result.data.reference_id == "REF123"
        assert result.document_id == "doc1.txt"

def test_extract_data_error_returns_empty(extraction_service):
    # Patch the chain to raise an exception
    with patch.object(extraction_service, "extraction_prompt") as mock_prompt, \
         patch.object(extraction_service, "llm") as mock_llm, \
         patch.object(extraction_service, "parser") as mock_parser:
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("LLM error")
        mock_prompt.__or__.return_value = mock_chain
        mock_chain.__or__.return_value = mock_chain
        mock_parser.__or__.return_value = mock_chain
        mock_parser.get_format_instructions.return_value = "FORMAT"
        extraction_service.extraction_prompt = mock_prompt
        extraction_service.llm = mock_llm
        extraction_service.parser = mock_parser

        result = extraction_service.extract_data("bad text", filename="fail.txt")
        assert isinstance(result, ExtractionResponse)
        # Should be empty ShipmentData
        assert result.data.reference_id is None
        assert result.document_id == "fail.txt"

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
    assert "MC Number: MC111" in text
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
    assert "Total Rate: $2500.0 USD" in text
    assert "Special Instructions: Handle with care." in text
    assert "Shipper Instructions: Call before arrival." in text
    assert "Carrier Instructions: No partial loads." in text
    assert "Dispatcher: Jane Smith" in text
    assert "Dispatcher Phone: 555-3333" in text

def test_format_extraction_as_text_edge_cases(extraction_service):
    # All fields None/empty
    empty_data = ShipmentData()
    extraction = ExtractionResponse(data=empty_data, document_id="empty.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    # Only header should be present
    assert text.strip() == "=== EXTRACTED STRUCTURED DATA ==="

def test_create_structured_chunk_output(extraction_service, minimal_shipment_data):
    extraction = ExtractionResponse(data=minimal_shipment_data, document_id="doc1.txt")
    chunk = extraction_service.create_structured_chunk(extraction, filename="doc1.txt")
    assert isinstance(chunk, Chunk)
    assert isinstance(chunk.metadata, DocumentMetadata)
    assert chunk.metadata.filename == "doc1.txt"
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.chunk_type == "structured_data"
    assert "Reference ID: REF123" in chunk.text

def test_create_structured_chunk_reconciliation(extraction_service, full_shipment_data):
    extraction = ExtractionResponse(data=full_shipment_data, document_id="doc2.txt")
    # Path 1: via create_structured_chunk
    chunk1 = extraction_service.create_structured_chunk(extraction, filename="doc2.txt")
    # Path 2: manual format + manual Chunk
    formatted_text = extraction_service.format_extraction_as_text(extraction)
    metadata = DocumentMetadata(
        filename="doc2.txt",
        chunk_id=9999,
        source="doc2.txt - Extracted Data",
        chunk_type="structured_data"
    )
    chunk2 = Chunk(text=formatted_text, metadata=metadata)
    # Reconciliation: outputs should be equivalent
    assert chunk1.text == chunk2.text
    assert chunk1.metadata.filename == chunk2.metadata.filename
    assert chunk1.metadata.chunk_id == chunk2.metadata.chunk_id
    assert chunk1.metadata.chunk_type == chunk2.metadata.chunk_type
    assert chunk1.metadata.source == chunk2.metadata.source

def test_format_extraction_as_text_boundary_conditions(extraction_service):
    # Boundary: Only one commodity, only pickup, no drop, only dispatcher name
    from models.extraction_schema import Commodity, Location
    data = ShipmentData(
        reference_id=None,
        load_id=None,
        po_number=None,
        shipper=None,
        consignee=None,
        carrier=None,
        driver=None,
        pickup=Location(name=None, address="789 Oak St", city="CityC", state="ST", zip="24680", appointment_time=None),
        drop=None,
        shipping_date=None,
        delivery_date=None,
        equipment_type=None,
        equipment_size=None,
        load_type=None,
        commodities=[Commodity(commodity_name="Thingamajigs", weight=None, quantity=None, description=None)],
        rate_info=None,
        special_instructions=None,
        shipper_instructions=None,
        carrier_instructions=None,
        dispatcher_name="Sam Dispatcher",
        dispatcher_phone=None,
    )
    extraction = ExtractionResponse(data=data, document_id="boundary.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Pickup Location: 789 Oak St" in text
    assert "Drop Location:" not in text
    assert "Commodities:" in text
    assert "1. Thingamajigs" in text
    assert "Dispatcher: Sam Dispatcher" in text
    assert "Dispatcher Phone:" not in text

def test_extract_data_equivalent_paths_reconciliation(extraction_service, minimal_shipment_data):
    # Patch the chain to return minimal_shipment_data
    with patch.object(extraction_service, "extraction_prompt") as mock_prompt, \
         patch.object(extraction_service, "llm") as mock_llm, \
         patch.object(extraction_service, "parser") as mock_parser:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = minimal_shipment_data
        mock_prompt.__or__.return_value = mock_chain
        mock_chain.__or__.return_value = mock_chain
        mock_parser.__or__.return_value = mock_chain
        mock_parser.get_format_instructions.return_value = "FORMAT"
        extraction_service.extraction_prompt = mock_prompt
        extraction_service.llm = mock_llm
        extraction_service.parser = mock_parser

        # Path 1: extract_data
        result1 = extraction_service.extract_data("test text", filename="doc1.txt")
        # Path 2: direct ExtractionResponse
        result2 = ExtractionResponse(data=minimal_shipment_data, document_id="doc1.txt")
        # Reconciliation: data and document_id should match
        assert result1.data == result2.data
        assert result1.document_id == result2.document_id
