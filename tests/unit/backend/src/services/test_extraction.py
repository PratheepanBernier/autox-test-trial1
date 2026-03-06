import pytest
from unittest.mock import patch, MagicMock, create_autospec

from backend.src.services.extraction import ExtractionService
from backend.src.models.extraction_schema import ShipmentData, ExtractionResponse
from backend.src.models.schemas import Chunk, DocumentMetadata

@pytest.fixture
def mock_settings(monkeypatch):
    monkeypatch.setattr("backend.src.core.config.settings.QA_MODEL", "mock-model")
    monkeypatch.setattr("backend.src.core.config.settings.GROQ_API_KEY", "mock-key")

@pytest.fixture
def mock_langchain(monkeypatch):
    # Patch langchain_groq.ChatGroq
    chatgroq_mock = MagicMock()
    monkeypatch.setattr("langchain_groq.ChatGroq", lambda **kwargs: chatgroq_mock)
    # Patch langchain_core.output_parsers.PydanticOutputParser
    parser_mock = MagicMock()
    parser_mock.get_format_instructions.return_value = "FORMAT"
    parser_mock.pydantic_object = ShipmentData
    monkeypatch.setattr("langchain_core.output_parsers.PydanticOutputParser", lambda **kwargs: parser_mock)
    # Patch langchain_core.prompts.ChatPromptTemplate
    prompt_mock = MagicMock()
    prompt_mock.from_template = MagicMock(return_value=prompt_mock)
    monkeypatch.setattr("langchain_core.prompts.ChatPromptTemplate", prompt_mock)
    return chatgroq_mock, parser_mock, prompt_mock

@pytest.fixture
def extraction_service(mock_settings, mock_langchain):
    # Patch the import inside __init__ by pre-importing the modules
    return ExtractionService()

def test_extractionservice_init_success(monkeypatch, mock_settings):
    # Arrange/Act
    with patch("langchain_groq.ChatGroq") as chatgroq_patch, \
         patch("langchain_core.output_parsers.PydanticOutputParser") as parser_patch, \
         patch("langchain_core.prompts.ChatPromptTemplate") as prompt_patch:
        chatgroq_patch.return_value = MagicMock()
        parser_patch.return_value = MagicMock()
        prompt_patch.from_template.return_value = MagicMock()
        # Assert
        service = ExtractionService()
        assert hasattr(service, "llm")
        assert hasattr(service, "parser")
        assert hasattr(service, "extraction_prompt")

def test_extractionservice_init_failure(monkeypatch):
    # Arrange: Patch import to raise ImportError
    with patch("langchain_groq.ChatGroq", side_effect=ImportError("fail")):
        # Act/Assert
        with pytest.raises(Exception):
            ExtractionService()

def test_extract_data_happy_path(extraction_service, mock_langchain):
    chatgroq_mock, parser_mock, prompt_mock = mock_langchain
    # Arrange
    fake_result = ShipmentData(reference_id="REF123")
    parser_mock.get_format_instructions.return_value = "FORMAT"
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = fake_result
    # Compose the chain: prompt | llm | parser
    extraction_service.extraction_prompt.__or__.side_effect = lambda other: chain_mock if other is extraction_service.llm else MagicMock()
    extraction_service.llm.__or__.side_effect = lambda other: chain_mock if other is extraction_service.parser else MagicMock()
    # Act
    with patch.object(chain_mock, "invoke", return_value=fake_result) as invoke_patch:
        response = extraction_service.extract_data("test text", filename="file1.txt")
    # Assert
    assert isinstance(response, ExtractionResponse)
    assert response.data.reference_id == "REF123"
    assert response.document_id == "file1.txt"
    invoke_patch.assert_called_once()

def test_extract_data_chain_raises_exception_returns_empty(extraction_service, mock_langchain):
    chatgroq_mock, parser_mock, prompt_mock = mock_langchain
    # Arrange
    parser_mock.get_format_instructions.return_value = "FORMAT"
    chain_mock = MagicMock()
    chain_mock.invoke.side_effect = Exception("fail")
    extraction_service.extraction_prompt.__or__.side_effect = lambda other: chain_mock if other is extraction_service.llm else MagicMock()
    extraction_service.llm.__or__.side_effect = lambda other: chain_mock if other is extraction_service.parser else MagicMock()
    # Act
    response = extraction_service.extract_data("bad text", filename="fail.txt")
    # Assert
    assert isinstance(response, ExtractionResponse)
    assert isinstance(response.data, ShipmentData)
    assert response.document_id == "fail.txt"
    # All fields should be default/None
    for field in response.data.__fields__:
        assert getattr(response.data, field) is None or getattr(response.data, field) == []

def test_format_extraction_as_text_full_fields(extraction_service):
    # Arrange
    from backend.src.models.extraction_schema import Carrier, Driver, Location, Commodity, RateInfo
    data = ShipmentData(
        reference_id="REF1",
        load_id="LOAD1",
        po_number="PO1",
        shipper="Shipper Inc.",
        consignee="Consignee LLC",
        carrier=Carrier(carrier_name="CarrierX", mc_number="MC123", phone="555-1111"),
        driver=Driver(driver_name="John Doe", cell_number="555-2222", truck_number="TRK123"),
        pickup=Location(name="Warehouse A", address="123 Main", city="Metropolis", state="NY", zip="10001", appointment_time="2024-01-01T09:00"),
        drop=Location(name="Store B", address="456 Elm", city="Gotham", state="NJ", zip="07001"),
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Van",
        equipment_size=53,
        load_type="Full",
        commodities=[Commodity(commodity_name="Widgets", weight="1000 lbs", quantity="100")],
        rate_info=RateInfo(total_rate=1200, currency="USD", rate_breakdown={"linehaul": 1000, "fuel": 200}),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No tarps",
        dispatcher_name="Jane Smith",
        dispatcher_phone="555-3333"
    )
    extraction = ExtractionResponse(data=data, document_id="doc1.txt")
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
    assert "Total Rate: $1200 USD" in text
    assert "Rate Breakdown:" in text
    assert "Special Instructions: Handle with care" in text
    assert "Shipper Instructions: Call before arrival" in text
    assert "Carrier Instructions: No tarps" in text
    assert "Dispatcher: Jane Smith" in text
    assert "Dispatcher Phone: 555-3333" in text

def test_format_extraction_as_text_minimal_fields(extraction_service):
    # Arrange: Only reference_id and empty ShipmentData
    data = ShipmentData(reference_id="REFMIN")
    extraction = ExtractionResponse(data=data, document_id="doc2.txt")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert
    assert "Reference ID: REFMIN" in text
    # Should not contain fields that are None
    assert "Load ID:" not in text
    assert "Shipper:" not in text
    assert "Carrier Name:" not in text
    assert "Pickup Location:" not in text
    assert "Commodities:" not in text
    assert "Total Rate:" not in text

def test_format_extraction_as_text_edge_cases(extraction_service):
    # Arrange: Some fields are empty strings, zero, or empty lists
    from backend.src.models.extraction_schema import Carrier, Driver, Location, Commodity, RateInfo
    data = ShipmentData(
        reference_id="",
        load_id=None,
        po_number="",
        shipper=None,
        consignee="",
        carrier=Carrier(carrier_name="", mc_number=None, phone=None),
        driver=None,
        pickup=None,
        drop=Location(name=None, address=None, city=None, state=None, zip=None),
        shipping_date=None,
        delivery_date="",
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
    extraction = ExtractionResponse(data=data, document_id="doc3.txt")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert: Should not include empty/None fields
    assert "Reference ID:" not in text
    assert "Load ID:" not in text
    assert "PO Number:" not in text
    assert "Shipper:" not in text
    assert "Consignee:" not in text
    assert "Carrier Name:" not in text
    assert "Driver Name:" not in text
    assert "Pickup Location:" not in text
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
    # Arrange
    from backend.src.models.extraction_schema import ShipmentData
    extraction = ExtractionResponse(
        data=ShipmentData(reference_id="CHUNKREF"),
        document_id="chunkfile.txt"
    )
    # Act
    chunk = extraction_service.create_structured_chunk(extraction, filename="chunkfile.txt")
    # Assert
    assert isinstance(chunk, Chunk)
    assert isinstance(chunk.metadata, DocumentMetadata)
    assert chunk.metadata.filename == "chunkfile.txt"
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.source == "chunkfile.txt - Extracted Data"
    assert chunk.metadata.chunk_type == "structured_data"
    assert "CHUNKREF" in chunk.text

def test_create_structured_chunk_empty_data(extraction_service):
    # Arrange
    extraction = ExtractionResponse(
        data=ShipmentData(),
        document_id="emptyfile.txt"
    )
    # Act
    chunk = extraction_service.create_structured_chunk(extraction, filename="emptyfile.txt")
    # Assert
    assert isinstance(chunk, Chunk)
    assert isinstance(chunk.metadata, DocumentMetadata)
    assert chunk.metadata.filename == "emptyfile.txt"
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.source == "emptyfile.txt - Extracted Data"
    assert chunk.metadata.chunk_type == "structured_data"
    # Should contain the header line
    assert "EXTRACTED STRUCTURED DATA" in chunk.text
    # Should not contain any field lines
    assert chunk.text.strip() == "=== EXTRACTED STRUCTURED DATA ==="
