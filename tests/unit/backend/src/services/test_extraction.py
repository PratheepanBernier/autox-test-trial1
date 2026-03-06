import pytest
from unittest.mock import patch, MagicMock

from backend.src.services.extraction import ExtractionService
from backend.src.models.extraction_schema import ShipmentData, ExtractionResponse
from backend.src.models.schemas import Chunk, DocumentMetadata

@pytest.fixture
def mock_settings(monkeypatch):
    monkeypatch.setattr("backend.src.core.config.settings.QA_MODEL", "mock-model")
    monkeypatch.setattr("backend.src.core.config.settings.GROQ_API_KEY", "mock-key")

@pytest.fixture
def minimal_shipment_data():
    return ShipmentData()

@pytest.fixture
def full_shipment_data():
    from backend.src.models.extraction_schema import Carrier, Driver, Location, Commodity, RateInfo
    return ShipmentData(
        reference_id="REF123",
        load_id="LOAD456",
        po_number="PO789",
        shipper="Shipper Inc.",
        consignee="Consignee LLC",
        carrier=Carrier(carrier_name="CarrierX", mc_number="MC001", phone="555-1234"),
        driver=Driver(driver_name="John Doe", cell_number="555-5678", truck_number="TRK123"),
        pickup=Location(name="Warehouse A", address="123 Main St", city="Metropolis", state="NY", zip="10001", appointment_time="2024-06-01T09:00:00"),
        drop=Location(name="Store B", address="456 Elm St", city="Gotham", state="NJ", zip="07001"),
        shipping_date="2024-06-01",
        delivery_date="2024-06-02",
        equipment_type="Van",
        equipment_size=53,
        load_type="Full",
        commodities=[Commodity(commodity_name="Widgets", weight="1000 lbs", quantity="100")],
        rate_info=RateInfo(total_rate=1500.0, currency="USD", rate_breakdown={"linehaul": 1200, "fuel": 300}),
        special_instructions="Handle with care.",
        shipper_instructions="Call before arrival.",
        carrier_instructions="No partial loads.",
        dispatcher_name="Jane Smith",
        dispatcher_phone="555-9999"
    )

@pytest.fixture
def extraction_response_minimal(minimal_shipment_data):
    return ExtractionResponse(data=minimal_shipment_data, document_id="file.txt")

@pytest.fixture
def extraction_response_full(full_shipment_data):
    return ExtractionResponse(data=full_shipment_data, document_id="file.txt")

@pytest.fixture
def mock_langchain(monkeypatch):
    # Patch langchain_groq.ChatGroq, langchain_core.output_parsers.PydanticOutputParser, langchain_core.prompts.ChatPromptTemplate
    chat_groq_mock = MagicMock()
    parser_mock = MagicMock()
    prompt_template_mock = MagicMock()
    prompt_template_mock.from_template.return_value = prompt_template_mock

    monkeypatch.setattr("backend.src.services.extraction.ChatGroq", lambda **kwargs: chat_groq_mock)
    monkeypatch.setattr("backend.src.services.extraction.PydanticOutputParser", lambda pydantic_object: parser_mock)
    monkeypatch.setattr("backend.src.services.extraction.ChatPromptTemplate", prompt_template_mock)
    return chat_groq_mock, parser_mock, prompt_template_mock

def test_extraction_service_init_success(mock_settings, mock_langchain):
    # Should initialize without error and set up attributes
    service = ExtractionService()
    assert hasattr(service, "llm")
    assert hasattr(service, "parser")
    assert hasattr(service, "extraction_prompt")

def test_extraction_service_init_failure(monkeypatch):
    # Simulate import error
    monkeypatch.setattr("backend.src.services.extraction.ChatGroq", lambda **kwargs: (_ for _ in ()).throw(ImportError("fail")))
    with pytest.raises(Exception):
        ExtractionService()

def test_extract_data_happy_path(mock_settings, mock_langchain):
    chat_groq_mock, parser_mock, prompt_template_mock = mock_langchain
    # Simulate chain: prompt | llm | parser
    chain_mock = MagicMock()
    prompt_template_mock.__or__.side_effect = lambda other: chain_mock if other is chat_groq_mock else NotImplemented
    chain_mock.__or__.side_effect = lambda other: chain_mock if other is parser_mock else NotImplemented
    # parser.get_format_instructions returns a string
    parser_mock.get_format_instructions.return_value = "FORMAT"
    # chain.invoke returns a ShipmentData instance
    shipment_data = ShipmentData(reference_id="ABC")
    chain_mock.invoke.return_value = shipment_data

    service = ExtractionService()
    result = service.extract_data("test text", filename="doc1.txt")
    assert isinstance(result, ExtractionResponse)
    assert result.data.reference_id == "ABC"
    assert result.document_id == "doc1.txt"

def test_extract_data_error_returns_empty(mock_settings, mock_langchain):
    chat_groq_mock, parser_mock, prompt_template_mock = mock_langchain
    chain_mock = MagicMock()
    prompt_template_mock.__or__.side_effect = lambda other: chain_mock if other is chat_groq_mock else NotImplemented
    chain_mock.__or__.side_effect = lambda other: chain_mock if other is parser_mock else NotImplemented
    parser_mock.get_format_instructions.return_value = "FORMAT"
    # chain.invoke raises error
    chain_mock.invoke.side_effect = Exception("fail")

    service = ExtractionService()
    result = service.extract_data("bad text", filename="fail.txt")
    assert isinstance(result, ExtractionResponse)
    # Should be empty ShipmentData
    assert result.data == ShipmentData()
    assert result.document_id == "fail.txt"

def test_format_extraction_as_text_minimal(extraction_response_minimal):
    service = ExtractionService.__new__(ExtractionService)  # bypass __init__
    text = service.format_extraction_as_text(extraction_response_minimal)
    assert "EXTRACTED STRUCTURED DATA" in text
    # Should not contain any field values
    assert "Reference ID:" not in text
    assert "Shipper:" not in text
    assert "Carrier Name:" not in text

def test_format_extraction_as_text_full(extraction_response_full):
    service = ExtractionService.__new__(ExtractionService)
    text = service.format_extraction_as_text(extraction_response_full)
    # Check for presence of all major fields
    assert "Reference ID: REF123" in text
    assert "Load ID: LOAD456" in text
    assert "PO Number: PO789" in text
    assert "Shipper: Shipper Inc." in text
    assert "Consignee: Consignee LLC" in text
    assert "Carrier Name: CarrierX" in text
    assert "MC Number: MC001" in text
    assert "Carrier Phone: 555-1234" in text
    assert "Driver Name: John Doe" in text
    assert "Driver Phone: 555-5678" in text
    assert "Truck Number: TRK123" in text
    assert "Pickup Location: Warehouse A" in text
    assert "Pickup City: Metropolis, NY" in text
    assert "Pickup Appointment: 2024-06-01T09:00:00" in text
    assert "Drop Location: Store B" in text
    assert "Drop City: Gotham, NJ" in text
    assert "Shipping Date: 2024-06-01" in text
    assert "Delivery Date: 2024-06-02" in text
    assert "Equipment Type: Van" in text
    assert "Equipment Size: 53 feet" in text
    assert "Load Type: Full" in text
    assert "Commodities:" in text
    assert "1. Widgets" in text
    assert "Weight: 1000 lbs" in text
    assert "Quantity: 100" in text
    assert "Total Rate: $1500.0 USD" in text
    assert "Rate Breakdown:" in text
    assert "Special Instructions: Handle with care." in text
    assert "Shipper Instructions: Call before arrival." in text
    assert "Carrier Instructions: No partial loads." in text
    assert "Dispatcher: Jane Smith" in text
    assert "Dispatcher Phone: 555-9999" in text

def test_format_extraction_as_text_edge_cases():
    # Test with partial data and nulls
    from backend.src.models.extraction_schema import ShipmentData, Carrier, Driver, Location, Commodity, RateInfo
    data = ShipmentData(
        reference_id=None,
        load_id="LOAD999",
        po_number=None,
        shipper=None,
        consignee="Consignee Only",
        carrier=Carrier(carrier_name="CarrierY", mc_number=None, phone=None),
        driver=None,
        pickup=Location(name=None, address=None, city=None, state=None, zip=None, appointment_time=None),
        drop=None,
        shipping_date=None,
        delivery_date=None,
        equipment_type=None,
        equipment_size=None,
        load_type=None,
        commodities=[Commodity(commodity_name=None, weight=None, quantity=None)],
        rate_info=None,
        special_instructions=None,
        shipper_instructions=None,
        carrier_instructions=None,
        dispatcher_name=None,
        dispatcher_phone=None
    )
    extraction = ExtractionResponse(data=data, document_id="file.txt")
    service = ExtractionService.__new__(ExtractionService)
    text = service.format_extraction_as_text(extraction)
    assert "Load ID: LOAD999" in text
    assert "Consignee: Consignee Only" in text
    assert "Carrier Name: CarrierY" in text
    assert "Commodities:" in text
    assert "1. Unknown" in text

def test_create_structured_chunk(extraction_response_full):
    service = ExtractionService.__new__(ExtractionService)
    chunk = service.create_structured_chunk(extraction_response_full, filename="doc2.pdf")
    assert isinstance(chunk, Chunk)
    assert isinstance(chunk.metadata, DocumentMetadata)
    assert chunk.metadata.filename == "doc2.pdf"
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.source == "doc2.pdf - Extracted Data"
    assert chunk.metadata.chunk_type == "structured_data"
    assert "EXTRACTED STRUCTURED DATA" in chunk.text

def test_create_structured_chunk_minimal(extraction_response_minimal):
    service = ExtractionService.__new__(ExtractionService)
    chunk = service.create_structured_chunk(extraction_response_minimal, filename="empty.doc")
    assert isinstance(chunk, Chunk)
    assert chunk.metadata.filename == "empty.doc"
    assert "EXTRACTED STRUCTURED DATA" in chunk.text
    # Should not contain any field values
    assert "Reference ID:" not in chunk.text
    assert "Shipper:" not in chunk.text
    assert "Carrier Name:" not in chunk.text
