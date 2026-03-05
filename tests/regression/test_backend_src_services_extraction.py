import pytest
from unittest.mock import patch, MagicMock
from backend.src.services.extraction import ExtractionService
from models.extraction_schema import ShipmentData, ExtractionResponse
from models.schemas import Chunk, DocumentMetadata

@pytest.fixture
def extraction_service():
    # Patch ChatGroq, PydanticOutputParser, ChatPromptTemplate, and settings for isolation
    with patch("backend.src.services.extraction.ChatGroq") as mock_llm, \
         patch("backend.src.services.extraction.PydanticOutputParser") as mock_parser_cls, \
         patch("backend.src.services.extraction.ChatPromptTemplate") as mock_prompt_cls, \
         patch("backend.src.services.extraction.settings") as mock_settings:
        # Mock settings
        mock_settings.QA_MODEL = "test-model"
        mock_settings.GROQ_API_KEY = "test-key"
        # Mock LLM
        mock_llm.return_value = MagicMock()
        # Mock parser
        mock_parser = MagicMock()
        mock_parser.get_format_instructions.return_value = "FORMAT"
        mock_parser_cls.return_value = mock_parser
        # Mock prompt
        mock_prompt = MagicMock()
        mock_prompt_cls.from_template.return_value = mock_prompt
        # Chain: prompt | llm | parser
        def chain_or(other):
            return mock_prompt
        mock_prompt.__or__.side_effect = chain_or
        # The chain's invoke will be set per test
        yield ExtractionService()

def make_shipment_data(**kwargs):
    # Helper to create ShipmentData with defaults
    data = ShipmentData.construct()
    for k, v in kwargs.items():
        setattr(data, k, v)
    return data

def make_extraction_response(data=None, document_id="file1.txt"):
    if data is None:
        data = make_shipment_data()
    return ExtractionResponse(data=data, document_id=document_id)

def test_extract_data_happy_path(extraction_service):
    # Patch the chain to return a populated ShipmentData
    shipment_data = make_shipment_data(reference_id="REF123", load_id="LID456", po_number="PO789")
    with patch.object(extraction_service, "extraction_prompt") as mock_prompt, \
         patch.object(extraction_service, "llm") as mock_llm, \
         patch.object(extraction_service, "parser") as mock_parser:
        # Compose the chain: prompt | llm | parser
        class Chain:
            def invoke(self, args):
                assert args["text"] == "test document"
                assert "format_instructions" in args
                return shipment_data
        chain = Chain()
        # Simulate prompt | llm | parser
        mock_prompt.__or__.return_value = mock_llm
        mock_llm.__or__.return_value = mock_parser
        mock_parser.__or__.return_value = chain
        # Actually, the code does: prompt | llm | parser, so we need to chain __or__ calls
        # We'll patch __or__ on prompt and llm to return objects with __or__ as needed
        mock_prompt.__or__.side_effect = lambda other: mock_llm
        mock_llm.__or__.side_effect = lambda other: chain
        mock_parser.get_format_instructions.return_value = "FORMAT"
        # Run
        resp = extraction_service.extract_data("test document", filename="file1.txt")
        assert isinstance(resp, ExtractionResponse)
        assert resp.data.reference_id == "REF123"
        assert resp.data.load_id == "LID456"
        assert resp.data.po_number == "PO789"
        assert resp.document_id == "file1.txt"

def test_extract_data_error_returns_empty(extraction_service):
    # Patch the chain to raise an exception
    with patch.object(extraction_service, "extraction_prompt") as mock_prompt, \
         patch.object(extraction_service, "llm") as mock_llm, \
         patch.object(extraction_service, "parser") as mock_parser:
        class Chain:
            def invoke(self, args):
                raise RuntimeError("LLM error")
        chain = Chain()
        mock_prompt.__or__.side_effect = lambda other: mock_llm
        mock_llm.__or__.side_effect = lambda other: chain
        mock_parser.get_format_instructions.return_value = "FORMAT"
        resp = extraction_service.extract_data("bad input", filename="fail.txt")
        assert isinstance(resp, ExtractionResponse)
        # Should be empty ShipmentData
        assert isinstance(resp.data, ShipmentData)
        # All fields should be None or empty
        for field in resp.data.__fields__:
            assert getattr(resp.data, field) in (None, [], {}, "")

def test_format_extraction_as_text_full_fields(extraction_service):
    # All fields populated
    class Carrier:
        carrier_name = "CarrierX"
        mc_number = "MC123"
        phone = "555-1111"
    class Driver:
        driver_name = "John Doe"
        cell_number = "555-2222"
        truck_number = "TRK123"
    class Location:
        name = "Warehouse"
        address = "123 Main St"
        city = "Metropolis"
        state = "NY"
        zip = "10001"
        appointment_time = "2024-06-01T10:00"
    class Commodity:
        commodity_name = "Widgets"
        weight = "1000 lbs"
        quantity = "50"
        description = "Blue widgets"
    class RateInfo:
        total_rate = 1200.0
        currency = "USD"
        rate_breakdown = {"linehaul": 1000, "fuel": 200}
    data = make_shipment_data(
        reference_id="REF1",
        load_id="LID2",
        po_number="PO3",
        shipper="Shipper Inc.",
        consignee="Consignee LLC",
        carrier=Carrier(),
        driver=Driver(),
        pickup=Location(),
        drop=Location(),
        shipping_date="2024-06-01",
        delivery_date="2024-06-02",
        equipment_type="Van",
        equipment_size=53,
        load_type="Full",
        commodities=[Commodity()],
        rate_info=RateInfo(),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No pallets",
        dispatcher_name="Dispatch Dan",
        dispatcher_phone="555-3333"
    )
    extraction = make_extraction_response(data)
    text = extraction_service.format_extraction_as_text(extraction)
    # Check for key fields in output
    assert "Reference ID: REF1" in text
    assert "Load ID: LID2" in text
    assert "PO Number: PO3" in text
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
    assert "Pickup Appointment: 2024-06-01T10:00" in text
    assert "Drop Location: Warehouse" in text
    assert "Drop City: Metropolis, NY" in text
    assert "Shipping Date: 2024-06-01" in text
    assert "Delivery Date: 2024-06-02" in text
    assert "Equipment Type: Van" in text
    assert "Equipment Size: 53 feet" in text
    assert "Load Type: Full" in text
    assert "Commodities:" in text
    assert "1. Widgets" in text
    assert "Weight: 1000 lbs" in text
    assert "Quantity: 50" in text
    assert "Total Rate: $1200.0 USD" in text
    assert "Rate Breakdown:" in text
    assert "Special Instructions: Handle with care" in text
    assert "Shipper Instructions: Call before arrival" in text
    assert "Carrier Instructions: No pallets" in text
    assert "Dispatcher: Dispatch Dan" in text
    assert "Dispatcher Phone: 555-3333" in text

def test_format_extraction_as_text_minimal_fields(extraction_service):
    # Only reference_id and shipper
    data = make_shipment_data(reference_id="REFMIN", shipper="Shipper Only")
    extraction = make_extraction_response(data)
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Reference ID: REFMIN" in text
    assert "Shipper: Shipper Only" in text
    # Should not contain fields not present
    assert "Load ID:" not in text
    assert "Consignee:" not in text
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

def test_format_extraction_as_text_edge_cases(extraction_service):
    # Null/empty fields, pickup/drop with only address
    class Location:
        name = None
        address = "456 Side St"
        city = None
        state = None
        zip = None
        appointment_time = None
    data = make_shipment_data(
        pickup=Location(),
        drop=Location(),
        commodities=[],
        rate_info=None,
        dispatcher_name=None,
        dispatcher_phone=None
    )
    extraction = make_extraction_response(data)
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Pickup Location: 456 Side St" in text
    assert "Drop Location: 456 Side St" in text
    # Should not error on missing fields
    assert "Pickup City:" not in text
    assert "Drop City:" not in text
    assert "Commodities:" not in text
    assert "Total Rate:" not in text
    assert "Dispatcher:" not in text

def test_create_structured_chunk_returns_chunk_with_expected_metadata_and_text(extraction_service):
    data = make_shipment_data(reference_id="REFCHUNK", shipper="Chunk Shipper")
    extraction = make_extraction_response(data, document_id="doc1.pdf")
    chunk = extraction_service.create_structured_chunk(extraction, filename="doc1.pdf")
    assert isinstance(chunk, Chunk)
    assert "EXTRACTED STRUCTURED DATA" in chunk.text
    assert "Reference ID: REFCHUNK" in chunk.text
    assert "Shipper: Chunk Shipper" in chunk.text
    assert isinstance(chunk.metadata, DocumentMetadata)
    assert chunk.metadata.filename == "doc1.pdf"
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.source == "doc1.pdf - Extracted Data"
    assert chunk.metadata.chunk_type == "structured_data"

def test_extraction_service_init_error(monkeypatch):
    # Simulate error in __init__ (e.g., ChatGroq raises)
    with patch("backend.src.services.extraction.ChatGroq", side_effect=RuntimeError("fail")), \
         patch("backend.src.services.extraction.PydanticOutputParser"), \
         patch("backend.src.services.extraction.ChatPromptTemplate"), \
         patch("backend.src.services.extraction.settings"):
        with pytest.raises(RuntimeError):
            ExtractionService()
