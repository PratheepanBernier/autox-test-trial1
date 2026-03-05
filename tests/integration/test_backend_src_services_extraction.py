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
        parser_instance = MagicMock()
        parser_instance.get_format_instructions.return_value = "FORMAT"
        parser_instance.pydantic_object = ShipmentData
        mock_parser.return_value = parser_instance
        # Mock prompt
        prompt_instance = MagicMock()
        mock_prompt.from_template.return_value = prompt_instance
        yield ExtractionService()

def test_extract_data_happy_path(extraction_service):
    # Arrange
    text = "Reference ID: 12345\nShipper: ACME Corp\nPickup: New York"
    filename = "doc1.txt"
    shipment_data = ShipmentData(reference_id="12345", shipper="ACME Corp", pickup=None)
    expected_response = ExtractionResponse(data=shipment_data, document_id=filename)

    # Patch the chain to return shipment_data
    with patch.object(extraction_service, "extraction_prompt") as mock_prompt, \
         patch.object(extraction_service, "llm") as mock_llm, \
         patch.object(extraction_service, "parser") as mock_parser:
        # Compose a fake chain with .invoke
        class FakeChain:
            def invoke(self, args):
                return shipment_data
        mock_prompt.__or__.return_value = mock_llm
        mock_llm.__or__.return_value = mock_parser
        mock_parser.__or__.return_value = FakeChain()
        mock_parser.get_format_instructions.return_value = "FORMAT"
        # Act
        result = extraction_service.extract_data(text, filename)
        # Assert
        assert isinstance(result, ExtractionResponse)
        assert result.document_id == filename
        assert result.data.reference_id == "12345"
        assert result.data.shipper == "ACME Corp"

def test_extract_data_returns_empty_on_exception(extraction_service):
    text = "Some text"
    filename = "fail.txt"
    # Patch the chain to raise an exception
    with patch.object(extraction_service, "extraction_prompt") as mock_prompt, \
         patch.object(extraction_service, "llm") as mock_llm, \
         patch.object(extraction_service, "parser") as mock_parser:
        class FakeChain:
            def invoke(self, args):
                raise RuntimeError("LLM failure")
        mock_prompt.__or__.return_value = mock_llm
        mock_llm.__or__.return_value = mock_parser
        mock_parser.__or__.return_value = FakeChain()
        mock_parser.get_format_instructions.return_value = "FORMAT"
        result = extraction_service.extract_data(text, filename)
        assert isinstance(result, ExtractionResponse)
        assert result.document_id == filename
        # Should be empty ShipmentData
        assert isinstance(result.data, ShipmentData)
        # All fields should be None or empty
        for field in result.data.__fields__:
            assert getattr(result.data, field) in (None, [], {})

def test_format_extraction_as_text_full_fields(extraction_service):
    # All fields populated
    from models.extraction_schema import Carrier, Driver, Location, Commodity, RateInfo
    shipment_data = ShipmentData(
        reference_id="REF123",
        load_id="LOAD456",
        po_number="PO789",
        shipper="Shipper Inc",
        consignee="Consignee LLC",
        carrier=Carrier(carrier_name="CarrierX", mc_number="MC123", phone="555-1111"),
        driver=Driver(driver_name="John Doe", cell_number="555-2222", truck_number="TRK123"),
        pickup=Location(name="Warehouse A", address="123 Main", city="Metropolis", state="NY", zip="10001", appointment_time="2024-01-01 08:00"),
        drop=Location(name="Store B", address="456 Elm", city="Gotham", state="NJ", zip="07001", appointment_time=None),
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Van",
        equipment_size=53,
        load_type="Full",
        commodities=[Commodity(commodity_name="Widgets", weight="1000 lbs", quantity="100", description="Blue widgets")],
        rate_info=RateInfo(total_rate=2000, currency="USD", rate_breakdown={"linehaul": 1800, "fuel": 200}),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No tarps",
        dispatcher_name="Jane Smith",
        dispatcher_phone="555-3333"
    )
    extraction = ExtractionResponse(data=shipment_data, document_id="doc2.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    # Check that key fields are present in the output
    assert "Reference ID: REF123" in text
    assert "Load ID: LOAD456" in text
    assert "PO Number: PO789" in text
    assert "Shipper: Shipper Inc" in text
    assert "Consignee: Consignee LLC" in text
    assert "Carrier Name: CarrierX" in text
    assert "MC Number: MC123" in text
    assert "Driver Name: John Doe" in text
    assert "Pickup Location: Warehouse A" in text
    assert "Drop Location: Store B" in text
    assert "Shipping Date: 2024-01-01" in text
    assert "Delivery Date: 2024-01-02" in text
    assert "Equipment Type: Van" in text
    assert "Equipment Size: 53 feet" in text
    assert "Load Type: Full" in text
    assert "Commodities:" in text
    assert "Widgets" in text
    assert "Weight: 1000 lbs" in text
    assert "Quantity: 100" in text
    assert "Total Rate: $2000 USD" in text
    assert "Rate Breakdown:" in text
    assert "Special Instructions: Handle with care" in text
    assert "Shipper Instructions: Call before arrival" in text
    assert "Carrier Instructions: No tarps" in text
    assert "Dispatcher: Jane Smith" in text
    assert "Dispatcher Phone: 555-3333" in text

def test_format_extraction_as_text_minimal_fields(extraction_service):
    # Only reference_id and shipper
    shipment_data = ShipmentData(reference_id="REFMIN", shipper="MiniShipper")
    extraction = ExtractionResponse(data=shipment_data, document_id="doc3.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Reference ID: REFMIN" in text
    assert "Shipper: MiniShipper" in text
    # Should not contain fields that are None
    assert "Load ID:" not in text
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

def test_format_extraction_as_text_edge_cases(extraction_service):
    # Edge: empty commodities, zero rate, empty strings
    from models.extraction_schema import RateInfo
    shipment_data = ShipmentData(
        reference_id="",
        shipper="",
        commodities=[],
        rate_info=RateInfo(total_rate=0, currency="", rate_breakdown={})
    )
    extraction = ExtractionResponse(data=shipment_data, document_id="doc4.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    # Should not print empty fields
    assert "Reference ID:" not in text
    assert "Shipper:" not in text
    # Should print zero rate if present
    assert "Total Rate: $0 " in text

def test_create_structured_chunk_happy_path(extraction_service):
    shipment_data = ShipmentData(reference_id="CHUNK1", shipper="ChunkShipper")
    extraction = ExtractionResponse(data=shipment_data, document_id="doc5.txt")
    chunk = extraction_service.create_structured_chunk(extraction, "doc5.txt")
    assert isinstance(chunk, Chunk)
    assert isinstance(chunk.metadata, DocumentMetadata)
    assert chunk.metadata.filename == "doc5.txt"
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.source == "doc5.txt - Extracted Data"
    assert chunk.metadata.chunk_type == "structured_data"
    assert "Reference ID: CHUNK1" in chunk.text
    assert "Shipper: ChunkShipper" in chunk.text

def test_create_structured_chunk_with_empty_extraction(extraction_service):
    shipment_data = ShipmentData()
    extraction = ExtractionResponse(data=shipment_data, document_id="doc6.txt")
    chunk = extraction_service.create_structured_chunk(extraction, "doc6.txt")
    assert isinstance(chunk, Chunk)
    assert chunk.metadata.filename == "doc6.txt"
    # Should still have the header
    assert "=== EXTRACTED STRUCTURED DATA ===" in chunk.text
    # Should not have any field lines
    assert chunk.text.strip() == "=== EXTRACTED STRUCTURED DATA ==="

def test_extraction_service_init_error(monkeypatch):
    # Simulate error in __init__ (e.g., settings missing)
    with patch("backend.src.services.extraction.ChatGroq", side_effect=Exception("fail")), \
         pytest.raises(Exception) as excinfo:
        from backend.src.services import extraction
        extraction.ExtractionService()
    assert "fail" in str(excinfo.value)
