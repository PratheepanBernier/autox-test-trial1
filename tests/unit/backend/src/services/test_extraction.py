import pytest
from unittest.mock import patch, MagicMock, create_autospec
from backend.src.services.extraction import ExtractionService
from backend.src.models.extraction_schema import ShipmentData, ExtractionResponse
from backend.src.models.schemas import Chunk, DocumentMetadata

@pytest.fixture
def mock_settings(monkeypatch):
    monkeypatch.setattr("backend.src.core.config.settings.QA_MODEL", "mock-model")
    monkeypatch.setattr("backend.src.core.config.settings.GROQ_API_KEY", "mock-api-key")

@pytest.fixture
def mock_langchain(monkeypatch):
    # Patch langchain_groq.ChatGroq
    mock_llm = MagicMock(name="ChatGroq")
    monkeypatch.setattr("langchain_groq.ChatGroq", lambda **kwargs: mock_llm)
    # Patch langchain_core.output_parsers.PydanticOutputParser
    mock_parser = MagicMock(name="PydanticOutputParser")
    mock_parser.get_format_instructions.return_value = "FORMAT"
    monkeypatch.setattr("langchain_core.output_parsers.PydanticOutputParser", lambda pydantic_object: mock_parser)
    # Patch langchain_core.prompts.ChatPromptTemplate
    mock_prompt = MagicMock(name="ChatPromptTemplate")
    mock_prompt.from_template = MagicMock(return_value=mock_prompt)
    monkeypatch.setattr("langchain_core.prompts.ChatPromptTemplate", mock_prompt)
    return mock_llm, mock_parser, mock_prompt

@pytest.fixture
def extraction_service(mock_settings, mock_langchain):
    # Patch import inside __init__ by patching sys.modules
    import sys
    sys.modules["langchain_groq"] = MagicMock()
    sys.modules["langchain_core.output_parsers"] = MagicMock()
    sys.modules["langchain_core.prompts"] = MagicMock()
    return ExtractionService()

def test_extractionservice_init_success(monkeypatch):
    # Arrange
    monkeypatch.setattr("backend.src.core.config.settings.QA_MODEL", "mock-model")
    monkeypatch.setattr("backend.src.core.config.settings.GROQ_API_KEY", "mock-api-key")
    sys_modules_backup = dict(__import__("sys").modules)
    import sys
    sys.modules["langchain_groq"] = MagicMock()
    sys.modules["langchain_core.output_parsers"] = MagicMock()
    sys.modules["langchain_core.prompts"] = MagicMock()
    # Act
    service = ExtractionService()
    # Assert
    assert hasattr(service, "llm")
    assert hasattr(service, "parser")
    assert hasattr(service, "extraction_prompt")
    # Cleanup
    sys.modules.clear()
    sys.modules.update(sys_modules_backup)

def test_extractionservice_init_failure(monkeypatch):
    # Arrange: cause import error
    monkeypatch.setattr("backend.src.core.config.settings.QA_MODEL", "mock-model")
    monkeypatch.setattr("backend.src.core.config.settings.GROQ_API_KEY", "mock-api-key")
    import sys
    sys_modules_backup = dict(sys.modules)
    sys.modules["langchain_groq"] = None
    sys.modules["langchain_core.output_parsers"] = None
    sys.modules["langchain_core.prompts"] = None
    # Act & Assert
    with pytest.raises(Exception):
        ExtractionService()
    # Cleanup
    sys.modules.clear()
    sys.modules.update(sys_modules_backup)

def test_extract_data_happy_path(monkeypatch, extraction_service):
    # Arrange
    mock_chain = MagicMock()
    expected_data = ShipmentData(reference_id="REF123")
    expected_response = ExtractionResponse(data=expected_data, document_id="file.txt")
    # Patch the chain and parser
    extraction_service.extraction_prompt.__or__ = lambda self, other: mock_chain
    extraction_service.llm.__or__ = lambda self, other: mock_chain
    extraction_service.parser.__or__ = lambda self, other: mock_chain
    extraction_service.parser.get_format_instructions = MagicMock(return_value="FORMAT")
    mock_chain.invoke = MagicMock(return_value=expected_data)
    # Act
    result = extraction_service.extract_data("test text", filename="file.txt")
    # Assert
    assert isinstance(result, ExtractionResponse)
    assert result.data.reference_id == "REF123"
    assert result.document_id == "file.txt"

def test_extract_data_error(monkeypatch, extraction_service):
    # Arrange
    mock_chain = MagicMock()
    extraction_service.extraction_prompt.__or__ = lambda self, other: mock_chain
    extraction_service.llm.__or__ = lambda self, other: mock_chain
    extraction_service.parser.__or__ = lambda self, other: mock_chain
    extraction_service.parser.get_format_instructions = MagicMock(return_value="FORMAT")
    mock_chain.invoke = MagicMock(side_effect=Exception("fail"))
    # Act
    result = extraction_service.extract_data("bad text", filename="fail.txt")
    # Assert: returns empty ShipmentData
    assert isinstance(result, ExtractionResponse)
    assert isinstance(result.data, ShipmentData)
    assert result.document_id == "fail.txt"
    # All fields should be default/None
    for field in result.data.__fields__:
        assert getattr(result.data, field) is None or getattr(result.data, field) == []

def test_extract_data_empty_text(monkeypatch, extraction_service):
    # Arrange
    mock_chain = MagicMock()
    expected_data = ShipmentData()
    extraction_service.extraction_prompt.__or__ = lambda self, other: mock_chain
    extraction_service.llm.__or__ = lambda self, other: mock_chain
    extraction_service.parser.__or__ = lambda self, other: mock_chain
    extraction_service.parser.get_format_instructions = MagicMock(return_value="FORMAT")
    mock_chain.invoke = MagicMock(return_value=expected_data)
    # Act
    result = extraction_service.extract_data("", filename="empty.txt")
    # Assert
    assert isinstance(result, ExtractionResponse)
    assert result.document_id == "empty.txt"
    assert isinstance(result.data, ShipmentData)

def test_format_extraction_as_text_full_fields(extraction_service):
    # Arrange
    data = ShipmentData(
        reference_id="REF1",
        load_id="LOAD1",
        po_number="PO1",
        shipper="ShipperX",
        consignee="ConsigneeY",
        carrier=type("Carrier", (), {"carrier_name": "CarrierZ", "mc_number": "MC123", "phone": "555-1234"})(),
        driver=type("Driver", (), {"driver_name": "John Doe", "cell_number": "555-5678", "truck_number": "TRK42"})(),
        pickup=type("Loc", (), {"name": "Warehouse", "address": "123 St", "city": "CityA", "state": "ST", "appointment_time": "2024-01-01T10:00"})(),
        drop=type("Loc", (), {"name": "Store", "address": "456 Ave", "city": "CityB", "state": "ST", "appointment_time": None})(),
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Van",
        equipment_size=53,
        load_type="Full",
        commodities=[
            type("Commodity", (), {"commodity_name": "Widgets", "weight": 1000, "quantity": 10})(),
            type("Commodity", (), {"commodity_name": "Gadgets", "weight": 500, "quantity": 5})(),
        ],
        rate_info=type("RateInfo", (), {"total_rate": 2000, "currency": "USD", "rate_breakdown": {"linehaul": 1500, "fuel": 500}})(),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No tarps",
        dispatcher_name="Dispatch Dan",
        dispatcher_phone="555-9999"
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert
    assert "Reference ID: REF1" in text
    assert "Load ID: LOAD1" in text
    assert "PO Number: PO1" in text
    assert "Shipper: ShipperX" in text
    assert "Consignee: ConsigneeY" in text
    assert "Carrier Name: CarrierZ" in text
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
    assert "1. Widgets" in text
    assert "2. Gadgets" in text
    assert "Total Rate: $2000 USD" in text
    assert "Rate Breakdown:" in text
    assert "Special Instructions: Handle with care" in text
    assert "Shipper Instructions: Call before arrival" in text
    assert "Carrier Instructions: No tarps" in text
    assert "Dispatcher: Dispatch Dan" in text
    assert "Dispatcher Phone: 555-9999" in text

def test_format_extraction_as_text_minimal_fields(extraction_service):
    # Arrange: Only one field set
    data = ShipmentData(reference_id="REFMIN")
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert
    assert "Reference ID: REFMIN" in text
    assert "Load ID:" not in text
    assert "Shipper:" not in text
    assert "Consignee:" not in text

def test_format_extraction_as_text_empty(extraction_service):
    # Arrange: All fields None/empty
    data = ShipmentData()
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert
    assert "EXTRACTED STRUCTURED DATA" in text
    # Should not contain any field lines except header
    assert text.strip() == "=== EXTRACTED STRUCTURED DATA ==="

def test_format_extraction_as_text_edge_cases(extraction_service):
    # Arrange: Some fields empty, some zero/false
    data = ShipmentData(
        reference_id="",
        load_id=None,
        po_number="",
        shipper=None,
        consignee="",
        carrier=None,
        driver=None,
        pickup=None,
        drop=None,
        shipping_date=None,
        delivery_date=None,
        equipment_type="",
        equipment_size=0,
        load_type=None,
        commodities=[],
        rate_info=None,
        special_instructions="",
        shipper_instructions=None,
        carrier_instructions=None,
        dispatcher_name=None,
        dispatcher_phone=None
    )
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert: Only header should be present
    assert text.strip() == "=== EXTRACTED STRUCTURED DATA ==="

def test_create_structured_chunk(extraction_service):
    # Arrange
    data = ShipmentData(reference_id="REFCHUNK")
    extraction = ExtractionResponse(data=data, document_id="doc.txt")
    filename = "doc.txt"
    # Act
    chunk = extraction_service.create_structured_chunk(extraction, filename)
    # Assert
    assert isinstance(chunk, Chunk)
    assert isinstance(chunk.metadata, DocumentMetadata)
    assert chunk.metadata.filename == filename
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.chunk_type == "structured_data"
    assert "EXTRACTED STRUCTURED DATA" in chunk.text
    assert "Reference ID: REFCHUNK" in chunk.text

def test_create_structured_chunk_with_empty_data(extraction_service):
    # Arrange
    extraction = ExtractionResponse(data=ShipmentData(), document_id="doc.txt")
    filename = "doc.txt"
    # Act
    chunk = extraction_service.create_structured_chunk(extraction, filename)
    # Assert
    assert isinstance(chunk, Chunk)
    assert chunk.metadata.filename == filename
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.chunk_type == "structured_data"
    assert chunk.text.strip() == "=== EXTRACTED STRUCTURED DATA ==="
