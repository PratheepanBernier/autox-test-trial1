import pytest
from unittest.mock import patch, MagicMock, create_autospec
from backend.src.services.extraction import ExtractionService
from models.extraction_schema import ShipmentData, ExtractionResponse
from models.schemas import Chunk, DocumentMetadata
import logging

@pytest.fixture(autouse=True)
def patch_logger(monkeypatch):
    # Silence logger for test output
    monkeypatch.setattr(logging, "getLogger", lambda *a, **k: MagicMock())

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
    monkeypatch.setattr("langchain_core.prompts.ChatPromptTemplate.from_template", lambda *a, **k: mock_prompt)
    return mock_prompt

@pytest.fixture
def mock_parser(monkeypatch):
    mock_parser = MagicMock()
    mock_parser.get_format_instructions.return_value = "FORMAT"
    monkeypatch.setattr("langchain_core.output_parsers.PydanticOutputParser", lambda **kwargs: mock_parser)
    return mock_parser

@pytest.fixture
def extraction_service(mock_settings, mock_groq, mock_prompt, mock_parser):
    return ExtractionService()

def test_extract_data_happy_path(extraction_service, mock_groq, mock_parser):
    # Setup
    shipment_data = ShipmentData(reference_id="REF123", load_id="LOAD456")
    mock_parser.get_format_instructions.return_value = "FORMAT"
    mock_parser.pydantic_object = ShipmentData
    mock_groq.__or__.return_value = mock_groq
    # Simulate chain.invoke returning ShipmentData
    chain = MagicMock()
    chain.invoke.return_value = shipment_data
    # Compose the chain
    extraction_service.extraction_prompt.__or__.return_value = extraction_service.llm
    extraction_service.llm.__or__.return_value = extraction_service.parser
    extraction_service.parser.__or__.return_value = chain
    # Patch the chain in extract_data
    with patch.object(extraction_service, "extraction_prompt", create_autospec(extraction_service.extraction_prompt)) as prompt:
        prompt.__or__.return_value = extraction_service.llm
        extraction_service.llm.__or__.return_value = extraction_service.parser
        extraction_service.parser.__or__.return_value = chain
        # Act
        result = extraction_service.extract_data("test text", filename="file1.txt")
    # Assert
    assert isinstance(result, ExtractionResponse)
    assert result.data.reference_id == "REF123"
    assert result.data.load_id == "LOAD456"
    assert result.document_id == "file1.txt"

def test_extract_data_empty_text_returns_empty_shipment(extraction_service, mock_groq, mock_parser):
    # Simulate chain.invoke raising an exception
    chain = MagicMock()
    chain.invoke.side_effect = Exception("LLM error")
    extraction_service.extraction_prompt.__or__.return_value = extraction_service.llm
    extraction_service.llm.__or__.return_value = extraction_service.parser
    extraction_service.parser.__or__.return_value = chain
    with patch.object(extraction_service, "extraction_prompt", create_autospec(extraction_service.extraction_prompt)) as prompt:
        prompt.__or__.return_value = extraction_service.llm
        extraction_service.llm.__or__.return_value = extraction_service.parser
        extraction_service.parser.__or__.return_value = chain
        result = extraction_service.extract_data("", filename="empty.txt")
    assert isinstance(result, ExtractionResponse)
    # Should be empty ShipmentData
    assert isinstance(result.data, ShipmentData)
    assert result.document_id == "empty.txt"

def test_extract_data_handles_llm_exception_and_returns_empty(extraction_service, mock_groq, mock_parser):
    # Simulate chain.invoke raising an exception
    chain = MagicMock()
    chain.invoke.side_effect = RuntimeError("Some LLM error")
    extraction_service.extraction_prompt.__or__.return_value = extraction_service.llm
    extraction_service.llm.__or__.return_value = extraction_service.parser
    extraction_service.parser.__or__.return_value = chain
    with patch.object(extraction_service, "extraction_prompt", create_autospec(extraction_service.extraction_prompt)) as prompt:
        prompt.__or__.return_value = extraction_service.llm
        extraction_service.llm.__or__.return_value = extraction_service.parser
        extraction_service.parser.__or__.return_value = chain
        result = extraction_service.extract_data("irrelevant", filename="fail.txt")
    assert isinstance(result, ExtractionResponse)
    assert isinstance(result.data, ShipmentData)
    assert result.document_id == "fail.txt"

def test_format_extraction_as_text_full_fields(extraction_service):
    # All fields filled
    from models.extraction_schema import Carrier, Driver, Location, Commodity, RateInfo
    data = ShipmentData(
        reference_id="R1",
        load_id="L1",
        po_number="PO1",
        shipper="ShipperX",
        consignee="ConsigneeY",
        carrier=Carrier(carrier_name="CarrierZ", mc_number="MC123", phone="555-1111"),
        driver=Driver(driver_name="John Doe", cell_number="555-2222", truck_number="TRK123"),
        pickup=Location(name="Warehouse A", address="123 Main", city="CityA", state="ST", zip="12345", appointment_time="2024-01-01T09:00"),
        drop=Location(name="Store B", address="456 Elm", city="CityB", state="TS", zip="67890"),
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Van",
        equipment_size=53,
        load_type="Full",
        commodities=[Commodity(commodity_name="Widgets", weight="1000", quantity="10", description="Blue widgets")],
        rate_info=RateInfo(total_rate=1200.0, currency="USD", rate_breakdown={"linehaul": 1000, "fuel": 200}),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No partial loads",
        dispatcher_name="Jane Smith",
        dispatcher_phone="555-3333"
    )
    extraction = ExtractionResponse(data=data, document_id="doc1.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Reference ID: R1" in text
    assert "Load ID: L1" in text
    assert "PO Number: PO1" in text
    assert "Shipper: ShipperX" in text
    assert "Consignee: ConsigneeY" in text
    assert "Carrier Name: CarrierZ" in text
    assert "MC Number: MC123" in text
    assert "Carrier Phone: 555-1111" in text
    assert "Driver Name: John Doe" in text
    assert "Driver Phone: 555-2222" in text
    assert "Truck Number: TRK123" in text
    assert "Pickup Location: Warehouse A" in text
    assert "Pickup City: CityA, ST" in text
    assert "Pickup Appointment: 2024-01-01T09:00" in text
    assert "Drop Location: Store B" in text
    assert "Drop City: CityB, TS" in text
    assert "Shipping Date: 2024-01-01" in text
    assert "Delivery Date: 2024-01-02" in text
    assert "Equipment Type: Van" in text
    assert "Equipment Size: 53 feet" in text
    assert "Load Type: Full" in text
    assert "Commodities:" in text
    assert "1. Widgets" in text
    assert "Weight: 1000" in text
    assert "Quantity: 10" in text
    assert "Total Rate: $1200.0 USD" in text
    assert "Rate Breakdown:" in text
    assert "Special Instructions: Handle with care" in text
    assert "Shipper Instructions: Call before arrival" in text
    assert "Carrier Instructions: No partial loads" in text
    assert "Dispatcher: Jane Smith" in text
    assert "Dispatcher Phone: 555-3333" in text

def test_format_extraction_as_text_minimal_fields(extraction_service):
    # Only document_id and empty ShipmentData
    extraction = ExtractionResponse(data=ShipmentData(), document_id="doc2.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    # Should contain header, but no data fields
    assert "EXTRACTED STRUCTURED DATA" in text
    # Should not contain any field labels
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
    # Only a few fields
    from models.extraction_schema import Carrier
    data = ShipmentData(reference_id="R2", carrier=Carrier(carrier_name="CarrierOnly"))
    extraction = ExtractionResponse(data=data, document_id="doc3.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Reference ID: R2" in text
    assert "Carrier Name: CarrierOnly" in text
    assert "MC Number:" not in text  # Not present
    assert "Driver Name:" not in text

def test_create_structured_chunk_produces_valid_chunk(extraction_service):
    # Use a simple ExtractionResponse
    from models.extraction_schema import ShipmentData
    extraction = ExtractionResponse(data=ShipmentData(reference_id="R3"), document_id="doc4.txt")
    chunk = extraction_service.create_structured_chunk(extraction, filename="doc4.txt")
    assert isinstance(chunk, Chunk)
    assert isinstance(chunk.metadata, DocumentMetadata)
    assert chunk.metadata.filename == "doc4.txt"
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.source == "doc4.txt - Extracted Data"
    assert chunk.metadata.chunk_type == "structured_data"
    assert "EXTRACTED STRUCTURED DATA" in chunk.text

def test_extraction_service_init_handles_llm_init_failure(monkeypatch, mock_settings):
    # Simulate ChatGroq raising exception
    monkeypatch.setattr("langchain_groq.ChatGroq", lambda **kwargs: (_ for _ in ()).throw(Exception("fail")))
    with pytest.raises(Exception) as excinfo:
        ExtractionService()
    assert "fail" in str(excinfo.value)

def test_extraction_service_init_handles_prompt_failure(monkeypatch, mock_settings):
    # Simulate prompt template raising exception
    monkeypatch.setattr("langchain_groq.ChatGroq", lambda **kwargs: MagicMock())
    monkeypatch.setattr("langchain_core.prompts.ChatPromptTemplate.from_template", lambda *a, **k: (_ for _ in ()).throw(Exception("prompt fail")))
    with pytest.raises(Exception) as excinfo:
        ExtractionService()
    assert "prompt fail" in str(excinfo.value)

def test_format_extraction_as_text_handles_null_fields(extraction_service):
    # All fields explicitly set to None
    from models.extraction_schema import ShipmentData
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
    extraction = ExtractionResponse(data=data, document_id="doc5.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    # Should not raise and should only contain header
    assert "EXTRACTED STRUCTURED DATA" in text
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

def test_format_extraction_as_text_commodity_edge_cases(extraction_service):
    # Commodities with missing fields
    from models.extraction_schema import Commodity
    data = ShipmentData(
        commodities=[
            Commodity(commodity_name=None, weight=None, quantity=None, description=None),
            Commodity(commodity_name="Gadgets", weight="500", quantity=None, description=None)
        ]
    )
    extraction = ExtractionResponse(data=data, document_id="doc6.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Commodities:" in text
    assert "1. Unknown" in text
    assert "2. Gadgets" in text
    assert "Weight: 500" in text
    assert "Quantity:" not in text  # Only present if not None

def test_format_extraction_as_text_rateinfo_edge_cases(extraction_service):
    # RateInfo with only total_rate
    from models.extraction_schema import RateInfo
    data = ShipmentData(
        rate_info=RateInfo(total_rate=999.99, currency=None, rate_breakdown=None)
    )
    extraction = ExtractionResponse(data=data, document_id="doc7.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Total Rate: $999.99 USD" in text  # Default currency USD
    assert "Rate Breakdown:" not in text

def test_format_extraction_as_text_pickup_drop_fallbacks(extraction_service):
    # Pickup/drop with only address
    from models.extraction_schema import Location
    data = ShipmentData(
        pickup=Location(name=None, address="789 Oak", city=None, state=None, zip=None, appointment_time=None),
        drop=Location(name=None, address=None, city=None, state=None, zip=None)
    )
    extraction = ExtractionResponse(data=data, document_id="doc8.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Pickup Location: 789 Oak" in text
    assert "Drop Location: N/A" in text

def test_format_extraction_as_text_equipment_size_boundary(extraction_service):
    # Equipment size at boundary (0 and very large)
    data = ShipmentData(equipment_size=0)
    extraction = ExtractionResponse(data=data, document_id="doc9.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Equipment Size: 0 feet" in text
    data2 = ShipmentData(equipment_size=1000)
    extraction2 = ExtractionResponse(data=data2, document_id="doc10.txt")
    text2 = extraction_service.format_extraction_as_text(extraction2)
    assert "Equipment Size: 1000 feet" in text2
