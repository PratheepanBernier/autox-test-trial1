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
        QA_MODEL = "dummy-model"
        GROQ_API_KEY = "dummy-key"
    monkeypatch.setattr("core.config.settings", DummySettings())

@pytest.fixture
def mock_llm(monkeypatch):
    mock_llm = MagicMock(name="ChatGroq")
    monkeypatch.setattr("backend.src.services.extraction.ChatGroq", lambda **kwargs: mock_llm)
    return mock_llm

@pytest.fixture
def mock_parser(monkeypatch):
    mock_parser = MagicMock(name="PydanticOutputParser")
    mock_parser.get_format_instructions.return_value = "FORMAT"
    monkeypatch.setattr("backend.src.services.extraction.PydanticOutputParser", lambda pydantic_object: mock_parser)
    return mock_parser

@pytest.fixture
def mock_prompt(monkeypatch):
    mock_prompt = MagicMock(name="ChatPromptTemplate")
    mock_prompt.from_template.return_value = mock_prompt
    monkeypatch.setattr("backend.src.services.extraction.ChatPromptTemplate", mock_prompt)
    return mock_prompt

@pytest.fixture
def extraction_service(mock_settings, mock_llm, mock_parser, mock_prompt):
    # Re-import to ensure patches are in effect
    from backend.src.services.extraction import ExtractionService
    return ExtractionService()

def test_extract_data_happy_path(extraction_service, mock_llm, mock_parser):
    # Arrange
    dummy_result = ShipmentData(reference_id="REF123")
    mock_parser.get_format_instructions.return_value = "FORMAT"
    chain = MagicMock()
    chain.invoke.return_value = dummy_result
    # Compose the chain: prompt | llm | parser
    extraction_service.extraction_prompt.__or__.side_effect = lambda other: chain if other is extraction_service.llm else NotImplemented
    extraction_service.llm.__or__.side_effect = lambda other: chain if other is extraction_service.parser else NotImplemented
    # Act
    response = extraction_service.extract_data("test text", filename="file1.txt")
    # Assert
    assert isinstance(response, ExtractionResponse)
    assert response.data.reference_id == "REF123"
    assert response.document_id == "file1.txt"
    chain.invoke.assert_called_once()
    args = chain.invoke.call_args[0][0]
    assert args["text"] == "test text"
    assert args["format_instructions"] == "FORMAT"

def test_extract_data_error_returns_empty(extraction_service, mock_llm, mock_parser):
    # Arrange
    chain = MagicMock()
    chain.invoke.side_effect = Exception("llm error")
    extraction_service.extraction_prompt.__or__.side_effect = lambda other: chain if other is extraction_service.llm else NotImplemented
    extraction_service.llm.__or__.side_effect = lambda other: chain if other is extraction_service.parser else NotImplemented
    # Act
    response = extraction_service.extract_data("bad text", filename="fail.txt")
    # Assert
    assert isinstance(response, ExtractionResponse)
    assert isinstance(response.data, ShipmentData)
    assert response.document_id == "fail.txt"
    # Should be empty ShipmentData
    for field in response.data.__fields__:
        assert getattr(response.data, field) is None or getattr(response.data, field) == []

def test_format_extraction_as_text_full_fields(extraction_service):
    # Arrange
    from models.extraction_schema import Carrier, Driver, Location, Commodity, RateInfo
    data = ShipmentData(
        reference_id="REF1",
        load_id="LOAD1",
        po_number="PO1",
        shipper="Shipper Inc.",
        consignee="Consignee LLC",
        carrier=Carrier(carrier_name="CarrierX", mc_number="MC123", phone="555-1111"),
        driver=Driver(driver_name="John Doe", cell_number="555-2222", truck_number="TRK123"),
        pickup=Location(name="Warehouse A", address="123 Main", city="CityA", state="ST", zip="12345", appointment_time="2024-01-01T09:00"),
        drop=Location(name="Warehouse B", address="456 Elm", city="CityB", state="TS", zip="67890"),
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Van",
        equipment_size=53,
        load_type="Full",
        commodities=[Commodity(commodity_name="Widgets", weight="1000 lbs", quantity="100", description="Blue widgets")],
        rate_info=RateInfo(total_rate=1200, currency="USD", rate_breakdown={"linehaul": 1000, "fuel": 200}),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No tarps",
        dispatcher_name="Jane Smith",
        dispatcher_phone="555-3333"
    )
    extraction = ExtractionResponse(data=data, document_id="doc1")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert
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
    assert "Pickup Appointment: 2024-01-01T09:00" in text
    assert "Drop Location: Warehouse B" in text
    assert "Drop City: CityB, TS" in text
    assert "Shipping Date: 2024-01-01" in text
    assert "Delivery Date: 2024-01-02" in text
    assert "Equipment Type: Van" in text
    assert "Equipment Size: 53 feet" in text
    assert "Load Type: Full" in text
    assert "Commodities:" in text
    assert "1. Widgets" in text
    assert "Weight: 1000 lbs" in text
    assert "Quantity: 100" in text
    assert "Total Rate: $1200 USD" in text
    assert "Rate Breakdown:" in text
    assert "Special Instructions: Handle with care" in text
    assert "Shipper Instructions: Call before arrival" in text
    assert "Carrier Instructions: No tarps" in text
    assert "Dispatcher: Jane Smith" in text
    assert "Dispatcher Phone: 555-3333" in text

def test_format_extraction_as_text_minimal_fields(extraction_service):
    # Arrange
    data = ShipmentData()
    extraction = ExtractionResponse(data=data, document_id="doc2")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert
    # Only the header should be present
    assert text.strip() == "=== EXTRACTED STRUCTURED DATA ==="

def test_format_extraction_as_text_partial_fields(extraction_service):
    # Arrange
    from models.extraction_schema import Carrier
    data = ShipmentData(
        reference_id="REF2",
        carrier=Carrier(carrier_name="CarrierY")
    )
    extraction = ExtractionResponse(data=data, document_id="doc3")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert
    assert "Reference ID: REF2" in text
    assert "Carrier Name: CarrierY" in text
    # No MC Number or Phone
    assert "MC Number:" not in text
    assert "Carrier Phone:" not in text

def test_create_structured_chunk_creates_chunk_correctly(extraction_service):
    # Arrange
    data = ShipmentData(reference_id="REF3")
    extraction = ExtractionResponse(data=data, document_id="doc4")
    filename = "file4.txt"
    # Act
    chunk = extraction_service.create_structured_chunk(extraction, filename)
    # Assert
    assert isinstance(chunk, Chunk)
    assert chunk.metadata.filename == filename
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.source == f"{filename} - Extracted Data"
    assert chunk.metadata.chunk_type == "structured_data"
    assert "=== EXTRACTED STRUCTURED DATA ===" in chunk.text
    assert "Reference ID: REF3" in chunk.text

def test_extraction_service_init_success(monkeypatch):
    # Arrange
    dummy_llm = MagicMock()
    dummy_parser = MagicMock()
    dummy_prompt = MagicMock()
    class DummySettings:
        QA_MODEL = "model"
        GROQ_API_KEY = "key"
    monkeypatch.setattr("backend.src.services.extraction.ChatGroq", lambda **kwargs: dummy_llm)
    monkeypatch.setattr("backend.src.services.extraction.PydanticOutputParser", lambda pydantic_object: dummy_parser)
    dummy_prompt.from_template.return_value = dummy_prompt
    monkeypatch.setattr("backend.src.services.extraction.ChatPromptTemplate", dummy_prompt)
    monkeypatch.setattr("core.config.settings", DummySettings())
    # Act
    service = ExtractionService()
    # Assert
    assert service.llm is dummy_llm
    assert service.parser is dummy_parser
    assert service.extraction_prompt is dummy_prompt

def test_extraction_service_init_failure(monkeypatch):
    # Arrange
    def fail_llm(**kwargs):
        raise RuntimeError("fail")
    monkeypatch.setattr("backend.src.services.extraction.ChatGroq", fail_llm)
    class DummySettings:
        QA_MODEL = "model"
        GROQ_API_KEY = "key"
    monkeypatch.setattr("core.config.settings", DummySettings())
    # Act / Assert
    with pytest.raises(RuntimeError):
        ExtractionService()
