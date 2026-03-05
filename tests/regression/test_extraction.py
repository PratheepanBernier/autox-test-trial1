import pytest
from unittest.mock import patch, MagicMock, create_autospec
from backend.src.services.extraction import ExtractionService
from models.extraction_schema import ShipmentData, ExtractionResponse
from models.schemas import Chunk, DocumentMetadata
import logging

@pytest.fixture(autouse=True)
def patch_logger(monkeypatch):
    # Silence logger for test output
    monkeypatch.setattr(logging, "getLogger", lambda name=None: MagicMock())

@pytest.fixture
def mock_settings(monkeypatch):
    class DummySettings:
        QA_MODEL = "test-model"
        GROQ_API_KEY = "test-key"
    monkeypatch.setattr("core.config.settings", DummySettings())

@pytest.fixture
def mock_groq(monkeypatch):
    mock_llm = MagicMock()
    monkeypatch.setattr("langchain_groq.ChatGroq", lambda **kwargs: mock_llm)
    return mock_llm

@pytest.fixture
def mock_prompt(monkeypatch):
    mock_prompt = MagicMock()
    monkeypatch.setattr("langchain_core.prompts.ChatPromptTemplate.from_template", lambda template: mock_prompt)
    return mock_prompt

@pytest.fixture
def mock_parser(monkeypatch):
    mock_parser = MagicMock()
    mock_parser.get_format_instructions.return_value = "FORMAT"
    monkeypatch.setattr("langchain_core.output_parsers.PydanticOutputParser", lambda pydantic_object: mock_parser)
    return mock_parser

@pytest.fixture
def extraction_service(mock_settings, mock_groq, mock_prompt, mock_parser):
    return ExtractionService()

def test_extractionservice_init_success(mock_settings, mock_groq, mock_prompt, mock_parser):
    # Should not raise
    service = ExtractionService()
    assert hasattr(service, "llm")
    assert hasattr(service, "parser")
    assert hasattr(service, "extraction_prompt")

def test_extractionservice_init_failure(monkeypatch, mock_settings):
    # Simulate failure in LLM init
    monkeypatch.setattr("langchain_groq.ChatGroq", lambda **kwargs: (_ for _ in ()).throw(Exception("fail")))
    with pytest.raises(Exception) as exc:
        ExtractionService()
    assert "fail" in str(exc.value)

def test_extract_data_happy_path(extraction_service, mock_groq, mock_prompt, mock_parser):
    # Setup chain.invoke to return a ShipmentData instance
    shipment_data = ShipmentData(reference_id="REF123")
    mock_parser.get_format_instructions.return_value = "FORMAT"
    chain = MagicMock()
    chain.invoke.return_value = shipment_data
    # Compose the chain: prompt | llm | parser
    mock_prompt.__or__.return_value = chain
    # Call extract_data
    resp = extraction_service.extract_data("test text", filename="file1.txt")
    assert isinstance(resp, ExtractionResponse)
    assert resp.data.reference_id == "REF123"
    assert resp.document_id == "file1.txt"

def test_extract_data_handles_exception(extraction_service, mock_groq, mock_prompt, mock_parser):
    # Setup chain.invoke to raise
    chain = MagicMock()
    chain.invoke.side_effect = Exception("invoke fail")
    mock_prompt.__or__.return_value = chain
    # Call extract_data
    resp = extraction_service.extract_data("bad text", filename="fail.txt")
    assert isinstance(resp, ExtractionResponse)
    # Should return empty ShipmentData
    assert isinstance(resp.data, ShipmentData)
    assert resp.document_id == "fail.txt"

def test_extract_data_empty_text(extraction_service, mock_groq, mock_prompt, mock_parser):
    # Should still call chain.invoke with empty text
    shipment_data = ShipmentData()
    chain = MagicMock()
    chain.invoke.return_value = shipment_data
    mock_prompt.__or__.return_value = chain
    resp = extraction_service.extract_data("", filename="empty.txt")
    assert isinstance(resp, ExtractionResponse)
    assert resp.document_id == "empty.txt"

def test_format_extraction_as_text_all_fields(extraction_service):
    # Fill all fields for maximal output
    from models.extraction_schema import Carrier, Driver, Location, Commodity, RateInfo
    data = ShipmentData(
        reference_id="REF1",
        load_id="LID2",
        po_number="PO3",
        shipper="ShipperX",
        consignee="ConsigneeY",
        carrier=Carrier(carrier_name="CarrierZ", mc_number="MC123", phone="555-1111"),
        driver=Driver(driver_name="John Doe", cell_number="555-2222", truck_number="TRK9"),
        pickup=Location(name="Warehouse A", address="123 St", city="City1", state="ST", zip="12345", appointment_time="10:00"),
        drop=Location(name="Store B", address="456 Ave", city="City2", state="TS", zip="67890"),
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Van",
        equipment_size=53,
        load_type="Full",
        commodities=[Commodity(commodity_name="Widgets", weight="1000 lbs", quantity="10")],
        rate_info=RateInfo(total_rate=1200, currency="USD", rate_breakdown={"linehaul": 1000, "fuel": 200}),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No tarps",
        dispatcher_name="Jane Smith",
        dispatcher_phone="555-3333"
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    # Check for presence of all major fields
    assert "Reference ID: REF1" in text
    assert "Load ID: LID2" in text
    assert "PO Number: PO3" in text
    assert "Shipper: ShipperX" in text
    assert "Consignee: ConsigneeY" in text
    assert "Carrier Name: CarrierZ" in text
    assert "MC Number: MC123" in text
    assert "Carrier Phone: 555-1111" in text
    assert "Driver Name: John Doe" in text
    assert "Driver Phone: 555-2222" in text
    assert "Truck Number: TRK9" in text
    assert "Pickup Location: Warehouse A" in text
    assert "Pickup City: City1, ST" in text
    assert "Pickup Appointment: 10:00" in text
    assert "Drop Location: Store B" in text
    assert "Drop City: City2, TS" in text
    assert "Shipping Date: 2024-01-01" in text
    assert "Delivery Date: 2024-01-02" in text
    assert "Equipment Type: Van" in text
    assert "Equipment Size: 53 feet" in text
    assert "Load Type: Full" in text
    assert "Commodities:" in text
    assert "1. Widgets" in text
    assert "Weight: 1000 lbs" in text
    assert "Quantity: 10" in text
    assert "Total Rate: $1200 USD" in text
    assert "Rate Breakdown:" in text
    assert "Special Instructions: Handle with care" in text
    assert "Shipper Instructions: Call before arrival" in text
    assert "Carrier Instructions: No tarps" in text
    assert "Dispatcher: Jane Smith" in text
    assert "Dispatcher Phone: 555-3333" in text

def test_format_extraction_as_text_minimal_fields(extraction_service):
    # Only one field set
    data = ShipmentData(reference_id="REFMIN")
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Reference ID: REFMIN" in text
    # Should not contain other fields
    assert "Load ID:" not in text
    assert "Shipper:" not in text or text.count("Shipper:") == 1  # Only in header

def test_format_extraction_as_text_empty(extraction_service):
    # All fields empty
    data = ShipmentData()
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    # Only header should be present
    assert text.strip() == "=== EXTRACTED STRUCTURED DATA ==="

def test_format_extraction_as_text_edge_cases(extraction_service):
    # Some fields are None, some are empty strings, some are zero
    from models.extraction_schema import Carrier, Driver, Location, Commodity, RateInfo
    data = ShipmentData(
        reference_id="",
        load_id=None,
        po_number="",
        shipper=None,
        consignee="",
        carrier=Carrier(carrier_name="", mc_number=None, phone=""),
        driver=None,
        pickup=Location(name=None, address=None, city=None, state=None, zip=None, appointment_time=None),
        drop=None,
        shipping_date=None,
        delivery_date="",
        equipment_type="",
        equipment_size=0,
        load_type=None,
        commodities=[],
        rate_info=None,
        special_instructions=None,
        shipper_instructions="",
        carrier_instructions=None,
        dispatcher_name="",
        dispatcher_phone=None
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    # Should not include empty/None fields
    assert "Reference ID:" not in text
    assert "Load ID:" not in text
    assert "PO Number:" not in text
    assert "Shipper:" not in text
    assert "Consignee:" not in text
    assert "Carrier Name:" not in text
    assert "Driver Name:" not in text
    assert "Pickup Location:" not in text or "N/A" in text
    assert "Drop Location:" not in text
    assert "Shipping Date:" not in text
    assert "Delivery Date:" not in text
    assert "Equipment Type:" not in text
    assert "Equipment Size:" not in text
    assert "Load Type:" not in text
    assert "Commodities:" not in text
    assert "Total Rate:" not in text
    assert "Special Instructions:" not in text
    assert "Dispatcher:" not in text

def test_create_structured_chunk_happy_path(extraction_service):
    # Use a filled ExtractionResponse
    data = ShipmentData(reference_id="CHUNKREF")
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    chunk = extraction_service.create_structured_chunk(extraction, filename="doc.txt")
    assert isinstance(chunk, Chunk)
    assert isinstance(chunk.metadata, DocumentMetadata)
    assert chunk.metadata.filename == "doc.txt"
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.chunk_type == "structured_data"
    assert "CHUNKREF" in chunk.text

def test_create_structured_chunk_empty(extraction_service):
    # Empty ExtractionResponse
    data = ShipmentData()
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    chunk = extraction_service.create_structured_chunk(extraction, filename="doc.txt")
    assert isinstance(chunk, Chunk)
    assert "EXTRACTED STRUCTURED DATA" in chunk.text
    assert chunk.metadata.chunk_id == 9999

def test_create_structured_chunk_filename_variants(extraction_service):
    # Test with different filenames
    data = ShipmentData(reference_id="REFX")
    extraction = ExtractionResponse(data=data, document_id="docA.txt")
    chunk = extraction_service.create_structured_chunk(extraction, filename="docA.txt")
    assert chunk.metadata.filename == "docA.txt"
    assert chunk.metadata.source.startswith("docA.txt")

def test_format_extraction_as_text_commodity_edge(extraction_service):
    # Commodity with missing fields
    from models.extraction_schema import Commodity
    data = ShipmentData(commodities=[Commodity()])
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Commodities:" in text
    assert "Unknown" in text

def test_format_extraction_as_text_rateinfo_edge(extraction_service):
    # RateInfo with only total_rate
    from models.extraction_schema import RateInfo
    data = ShipmentData(rate_info=RateInfo(total_rate=500))
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Total Rate: $500 USD" in text

def test_format_extraction_as_text_pickup_drop_fallback(extraction_service):
    # Pickup/drop with only address
    from models.extraction_schema import Location
    data = ShipmentData(
        pickup=Location(name=None, address="123 Main", city=None, state=None, zip=None, appointment_time=None),
        drop=Location(name=None, address="456 Elm", city=None, state=None, zip=None)
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Pickup Location: 123 Main" in text
    assert "Drop Location: 456 Elm" in text

def test_format_extraction_as_text_equipment_size_zero(extraction_service):
    # equipment_size=0 should not be shown
    data = ShipmentData(equipment_size=0)
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Equipment Size:" not in text

def test_format_extraction_as_text_rateinfo_breakdown_none(extraction_service):
    # RateInfo with no breakdown
    from models.extraction_schema import RateInfo
    data = ShipmentData(rate_info=RateInfo(total_rate=1000, currency="USD", rate_breakdown=None))
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Total Rate: $1000 USD" in text
    assert "Rate Breakdown:" not in text

def test_format_extraction_as_text_dispatcher_partial(extraction_service):
    # Dispatcher name only
    data = ShipmentData(dispatcher_name="DispatchX")
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Dispatcher: DispatchX" in text
    assert "Dispatcher Phone:" not in text

def test_format_extraction_as_text_instructions_partial(extraction_service):
    # Only carrier_instructions
    data = ShipmentData(carrier_instructions="Only for carrier")
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Carrier Instructions: Only for carrier" in text
    assert "Special Instructions:" not in text
    assert "Shipper Instructions:" not in text

def test_extract_data_filename_default(extraction_service, mock_groq, mock_prompt, mock_parser):
    # Should use "unknown" as default filename
    shipment_data = ShipmentData(reference_id="DEF")
    chain = MagicMock()
    chain.invoke.return_value = shipment_data
    mock_prompt.__or__.return_value = chain
    resp = extraction_service.extract_data("text")
    assert resp.document_id == "unknown"
    assert resp.data.reference_id == "DEF"
