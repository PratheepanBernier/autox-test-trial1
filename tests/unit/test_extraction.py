# source_hash: 88361007730f06bf
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
def mock_settings():
    with patch("backend.src.services.extraction.settings") as mock_settings:
        mock_settings.QA_MODEL = "test-model"
        mock_settings.GROQ_API_KEY = "test-key"
        yield mock_settings

@pytest.fixture
def mock_llm_chain(monkeypatch):
    # Patch ChatGroq, ChatPromptTemplate, PydanticOutputParser
    mock_llm = MagicMock()
    mock_parser = MagicMock()
    mock_prompt = MagicMock()
    mock_parser.get_format_instructions.return_value = "FORMAT"
    mock_parser.pydantic_object = ShipmentData
    mock_prompt.from_template.return_value = mock_prompt

    # Chain: prompt | llm | parser
    class FakeChain:
        def invoke(self, args):
            return ShipmentData(reference_id="REF123", load_id="LOAD456", po_number="PO789")

    monkeypatch.setattr("backend.src.services.extraction.ChatGroq", lambda **kwargs: mock_llm)
    monkeypatch.setattr("backend.src.services.extraction.PydanticOutputParser", lambda pydantic_object: mock_parser)
    monkeypatch.setattr("backend.src.services.extraction.ChatPromptTemplate", MagicMock(from_template=lambda t: mock_prompt))
    # The | operator is used to chain, so we need to support it
    def chain_or(self, other):
        return FakeChain()
    mock_prompt.__or__ = chain_or.__get__(mock_prompt)
    mock_llm.__or__ = chain_or.__get__(mock_llm)
    mock_parser.__or__ = chain_or.__get__(mock_parser)
    return mock_llm, mock_parser, mock_prompt

@pytest.fixture
def extraction_service(mock_settings, mock_llm_chain):
    # Re-import to ensure patches are in effect
    from backend.src.services.extraction import ExtractionService
    return ExtractionService()

def test_extraction_service_init_success(mock_settings):
    # Should not raise and should log info
    with patch("backend.src.services.extraction.ChatGroq") as mock_llm, \
         patch("backend.src.services.extraction.PydanticOutputParser") as mock_parser, \
         patch("backend.src.services.extraction.ChatPromptTemplate") as mock_prompt:
        instance = ExtractionService()
        assert hasattr(instance, "llm")
        assert hasattr(instance, "parser")
        assert hasattr(instance, "extraction_prompt")

def test_extraction_service_init_failure(monkeypatch):
    # Simulate failure in ChatGroq
    def raise_exc(*a, **kw):
        raise RuntimeError("fail")
    monkeypatch.setattr("backend.src.services.extraction.ChatGroq", raise_exc)
    with pytest.raises(RuntimeError):
        ExtractionService()

def test_extract_data_happy_path(extraction_service, monkeypatch):
    # Patch the chain to return a ShipmentData instance
    fake_result = ShipmentData(reference_id="R1", load_id="L1", po_number="P1")
    class FakeChain:
        def invoke(self, args):
            return fake_result
    extraction_service.extraction_prompt.__or__ = lambda self, other: FakeChain()
    extraction_service.llm.__or__ = lambda self, other: FakeChain()
    extraction_service.parser.__or__ = lambda self, other: FakeChain()
    extraction_service.parser.get_format_instructions = MagicMock(return_value="FORMAT")
    resp = extraction_service.extract_data("test text", filename="file1.txt")
    assert isinstance(resp, ExtractionResponse)
    assert resp.data.reference_id == "R1"
    assert resp.document_id == "file1.txt"

def test_extract_data_error_returns_empty(monkeypatch, extraction_service):
    # Patch the chain to raise
    class FakeChain:
        def invoke(self, args):
            raise ValueError("fail")
    extraction_service.extraction_prompt.__or__ = lambda self, other: FakeChain()
    extraction_service.llm.__or__ = lambda self, other: FakeChain()
    extraction_service.parser.__or__ = lambda self, other: FakeChain()
    extraction_service.parser.get_format_instructions = MagicMock(return_value="FORMAT")
    resp = extraction_service.extract_data("bad text", filename="badfile.txt")
    assert isinstance(resp, ExtractionResponse)
    # Should be empty ShipmentData
    assert resp.data == ShipmentData()
    assert resp.document_id == "badfile.txt"

def test_extract_data_empty_text(monkeypatch, extraction_service):
    # Should still call chain and return result
    fake_result = ShipmentData(reference_id=None)
    class FakeChain:
        def invoke(self, args):
            return fake_result
    extraction_service.extraction_prompt.__or__ = lambda self, other: FakeChain()
    extraction_service.llm.__or__ = lambda self, other: FakeChain()
    extraction_service.parser.__or__ = lambda self, other: FakeChain()
    extraction_service.parser.get_format_instructions = MagicMock(return_value="FORMAT")
    resp = extraction_service.extract_data("", filename="empty.txt")
    assert isinstance(resp, ExtractionResponse)
    assert resp.data.reference_id is None
    assert resp.document_id == "empty.txt"

def test_format_extraction_as_text_all_fields(extraction_service):
    # Fill all fields
    from models.extraction_schema import Carrier, Driver, Location, Commodity, RateInfo
    data = ShipmentData(
        reference_id="REF",
        load_id="LOAD",
        po_number="PO",
        shipper="Shipper Inc.",
        consignee="Consignee LLC",
        carrier=Carrier(carrier_name="CarrierX", mc_number="MC123", phone="555-1234"),
        driver=Driver(driver_name="John Doe", cell_number="555-5678", truck_number="TRK1"),
        pickup=Location(name="Warehouse", address="123 St", city="CityA", state="ST", zip="12345", appointment_time="10:00"),
        drop=Location(name="Store", address="456 Ave", city="CityB", state="ST", zip="67890"),
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Van",
        equipment_size=53,
        load_type="Full",
        commodities=[Commodity(commodity_name="Widgets", weight="1000", quantity="10")],
        rate_info=RateInfo(total_rate=2000, currency="USD", rate_breakdown={"linehaul": 1800, "fuel": 200}),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No partial loads",
        dispatcher_name="Dispatch Dan",
        dispatcher_phone="555-9999"
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    # Check for presence of key fields
    assert "Reference ID: REF" in text
    assert "Load ID: LOAD" in text
    assert "PO Number: PO" in text
    assert "Shipper: Shipper Inc." in text
    assert "Consignee: Consignee LLC" in text
    assert "Carrier Name: CarrierX" in text
    assert "MC Number: MC123" in text
    assert "Driver Name: John Doe" in text
    assert "Pickup Location: Warehouse" in text
    assert "Drop Location: Store" in text
    assert "Shipping Date: 2024-01-01" in text
    assert "Delivery Date: 2024-01-02" in text
    assert "Equipment Type: Van" in text
    assert "Equipment Size: 53 feet" in text
    assert "Load Type: Full" in text
    assert "Commodities:" in text
    assert "Widgets" in text
    assert "Weight: 1000" in text
    assert "Quantity: 10" in text
    assert "Total Rate: $2000 USD" in text
    assert "Rate Breakdown:" in text
    assert "Special Instructions: Handle with care" in text
    assert "Shipper Instructions: Call before arrival" in text
    assert "Carrier Instructions: No partial loads" in text
    assert "Dispatcher: Dispatch Dan" in text
    assert "Dispatcher Phone: 555-9999" in text

def test_format_extraction_as_text_minimal_fields(extraction_service):
    # Only reference_id and document_id
    data = ShipmentData(reference_id="REF")
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Reference ID: REF" in text
    # Should not contain fields not present
    assert "Load ID:" not in text
    assert "Shipper:" not in text
    assert "Carrier Name:" not in text
    assert "Commodities:" not in text

def test_format_extraction_as_text_empty(extraction_service):
    # All fields empty
    data = ShipmentData()
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    # Only header should be present
    assert text.strip().startswith("=== EXTRACTED STRUCTURED DATA ===")

def test_format_extraction_as_text_edge_cases(extraction_service):
    # Some nested fields partially filled
    from models.extraction_schema import Carrier, Driver, Location, Commodity, RateInfo
    data = ShipmentData(
        carrier=Carrier(carrier_name="CarrierX"),
        driver=Driver(driver_name="Jane"),
        pickup=Location(name=None, address=None, city="CityA", state=None, zip=None, appointment_time=None),
        drop=Location(name=None, address=None, city=None, state=None, zip=None),
        commodities=[Commodity(commodity_name=None, weight=None, quantity=None)],
        rate_info=RateInfo(total_rate=None, currency=None, rate_breakdown=None)
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Carrier Name: CarrierX" in text
    assert "Driver Name: Jane" in text
    assert "Pickup Location: N/A" in text
    assert "Drop Location: N/A" in text
    assert "Commodities:" in text

def test_create_structured_chunk_happy_path(extraction_service):
    # Use a filled ExtractionResponse
    from models.extraction_schema import ShipmentData
    extraction = ExtractionResponse(data=ShipmentData(reference_id="R1"), document_id="file.txt")
    chunk = extraction_service.create_structured_chunk(extraction, filename="file.txt")
    assert isinstance(chunk, Chunk)
    assert isinstance(chunk.metadata, DocumentMetadata)
    assert chunk.metadata.filename == "file.txt"
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.chunk_type == "structured_data"
    assert "EXTRACTED STRUCTURED DATA" in chunk.text

def test_create_structured_chunk_empty(extraction_service):
    extraction = ExtractionResponse(data=ShipmentData(), document_id="file.txt")
    chunk = extraction_service.create_structured_chunk(extraction, filename="file.txt")
    assert isinstance(chunk, Chunk)
    assert chunk.metadata.filename == "file.txt"
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.chunk_type == "structured_data"
    assert "EXTRACTED STRUCTURED DATA" in chunk.text

def test_format_extraction_as_text_boundary_equipment_size(extraction_service):
    # Test boundary value for equipment_size (e.g., 0, negative, very large)
    from models.extraction_schema import ShipmentData
    for size in [0, -1, 1000]:
        data = ShipmentData(equipment_size=size)
        extraction = ExtractionResponse(data=data, document_id="doc.txt")
        text = extraction_service.format_extraction_as_text(extraction)
        if size > 0:
            assert f"Equipment Size: {size} feet" in text
        else:
            assert "Equipment Size:" in text  # Should still print, but value may be odd

def test_format_extraction_as_text_multiple_commodities(extraction_service):
    from models.extraction_schema import Commodity
    data = ShipmentData(
        commodities=[
            Commodity(commodity_name="A", weight="10", quantity="1"),
            Commodity(commodity_name="B", weight="20", quantity="2"),
        ]
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "1. A" in text
    assert "2. B" in text
    assert "Weight: 10" in text
    assert "Weight: 20" in text

def test_format_extraction_as_text_rate_info_missing_fields(extraction_service):
    from models.extraction_schema import RateInfo
    data = ShipmentData(rate_info=RateInfo(total_rate=None, currency=None, rate_breakdown=None))
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    # Should not print total rate if total_rate is None
    assert "Total Rate:" not in text

def test_format_extraction_as_text_instructions_only(extraction_service):
    data = ShipmentData(
        special_instructions="Fragile",
        shipper_instructions="Arrive early",
        carrier_instructions="No detours"
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Special Instructions: Fragile" in text
    assert "Shipper Instructions: Arrive early" in text
    assert "Carrier Instructions: No detours" in text

def test_format_extraction_as_text_dispatcher_only(extraction_service):
    data = ShipmentData(
        dispatcher_name="Alice",
        dispatcher_phone="555-0000"
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Dispatcher: Alice" in text
    assert "Dispatcher Phone: 555-0000" in text

def test_format_extraction_as_text_pickup_and_drop_partial(extraction_service):
    from models.extraction_schema import Location
    data = ShipmentData(
        pickup=Location(name="Warehouse", address=None, city=None, state=None, zip=None, appointment_time=None),
        drop=Location(name=None, address="456 Ave", city="CityB", state="ST", zip=None)
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Pickup Location: Warehouse" in text
    assert "Drop Location: 456 Ave" in text
    assert "Drop City: CityB, ST" in text
