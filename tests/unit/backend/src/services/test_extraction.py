import pytest
from unittest.mock import patch, MagicMock, create_autospec
from backend.src.services.extraction import ExtractionService
from backend.src.models.extraction_schema import ShipmentData, ExtractionResponse
from backend.src.models.schemas import Chunk, DocumentMetadata
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
    monkeypatch.setattr("backend.src.core.config.settings", DummySettings())

@pytest.fixture
def mock_langchain(monkeypatch):
    # Patch langchain_groq.ChatGroq, langchain_core.output_parsers.PydanticOutputParser, langchain_core.prompts.ChatPromptTemplate
    chat_groq = MagicMock(name="ChatGroq")
    parser = MagicMock(name="PydanticOutputParser")
    prompt_template = MagicMock(name="ChatPromptTemplate")
    prompt_template.from_template.return_value = MagicMock(name="PromptInstance")
    monkeypatch.setattr("langchain_groq.ChatGroq", lambda **kwargs: chat_groq)
    monkeypatch.setattr("langchain_core.output_parsers.PydanticOutputParser", lambda pydantic_object: parser)
    monkeypatch.setattr("langchain_core.prompts.ChatPromptTemplate", MagicMock(from_template=prompt_template.from_template))
    return chat_groq, parser, prompt_template

@pytest.fixture
def extraction_service(mock_settings, mock_langchain):
    # Re-import ExtractionService to ensure patches are in effect
    return ExtractionService()

def test_extractionservice_init_success(mock_settings, mock_langchain):
    # Should initialize without error and log success
    service = ExtractionService()
    assert hasattr(service, "llm")
    assert hasattr(service, "parser")
    assert hasattr(service, "extraction_prompt")

def test_extractionservice_init_failure(monkeypatch):
    # Simulate import error in __init__
    monkeypatch.setitem(__import__("sys").modules, "langchain_groq", None)
    monkeypatch.setitem(__import__("sys").modules, "langchain_core.output_parsers", None)
    monkeypatch.setitem(__import__("sys").modules, "langchain_core.prompts", None)
    with patch("backend.src.services.extraction.logger") as mock_logger:
        with pytest.raises(Exception):
            ExtractionService()
        assert mock_logger.error.called

def test_extract_data_happy_path(extraction_service, mock_langchain):
    chat_groq, parser, _ = mock_langchain
    # Arrange
    fake_result = ShipmentData(reference_id="REF123", load_id="LID456", po_number="PO789")
    parser.get_format_instructions.return_value = "FORMAT"
    chain = MagicMock()
    chain.invoke.return_value = fake_result
    # Compose the chain: prompt | llm | parser
    extraction_service.extraction_prompt.__or__.side_effect = lambda other: chain if other is extraction_service.llm else NotImplemented
    extraction_service.llm.__or__.side_effect = lambda other: chain if other is extraction_service.parser else NotImplemented
    # Act
    with patch.object(chain, "invoke", return_value=fake_result) as mock_invoke:
        response = extraction_service.extract_data("test text", filename="file1.txt")
    # Assert
    assert isinstance(response, ExtractionResponse)
    assert response.data.reference_id == "REF123"
    assert response.document_id == "file1.txt"
    mock_invoke.assert_called_once()
    parser.get_format_instructions.assert_called_once()

def test_extract_data_exception_returns_empty(extraction_service, mock_langchain):
    chat_groq, parser, _ = mock_langchain
    parser.get_format_instructions.return_value = "FORMAT"
    chain = MagicMock()
    chain.invoke.side_effect = Exception("fail")
    extraction_service.extraction_prompt.__or__.side_effect = lambda other: chain if other is extraction_service.llm else NotImplemented
    extraction_service.llm.__or__.side_effect = lambda other: chain if other is extraction_service.parser else NotImplemented
    with patch.object(chain, "invoke", side_effect=Exception("fail")):
        response = extraction_service.extract_data("bad text", filename="fail.txt")
    assert isinstance(response, ExtractionResponse)
    assert isinstance(response.data, ShipmentData)
    assert response.document_id == "fail.txt"
    # Should be empty ShipmentData (all fields None/empty)

def test_format_extraction_as_text_full_fields(extraction_service):
    # Arrange
    from backend.src.models.extraction_schema import Carrier, Driver, Location, Commodity, RateInfo
    data = ShipmentData(
        reference_id="REF1",
        load_id="LID2",
        po_number="PO3",
        shipper="Shipper Inc.",
        consignee="Consignee LLC",
        carrier=Carrier(carrier_name="CarrierX", mc_number="MC123", phone="555-1234"),
        driver=Driver(driver_name="John Doe", cell_number="555-5678", truck_number="TRK9"),
        pickup=Location(name="Warehouse A", address="123 Main", city="Metropolis", state="NY", zip="10001", appointment_time="2024-01-01T10:00"),
        drop=Location(name="Store B", address="456 Elm", city="Gotham", state="NJ", zip="07001"),
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Van",
        equipment_size=53,
        load_type="Full",
        commodities=[Commodity(commodity_name="Widgets", weight="1000", quantity="10", description="Blue widgets")],
        rate_info=RateInfo(total_rate=1200.0, currency="USD", rate_breakdown={"linehaul": 1000, "fuel": 200}),
        special_instructions="Handle with care",
        shipper_instructions="Call before arrival",
        carrier_instructions="No tarps",
        dispatcher_name="Dispatch Dan",
        dispatcher_phone="555-9999"
    )
    extraction = ExtractionResponse(data=data, document_id="doc1.txt")
    # Act
    text = extraction_service.format_extraction_as_text(extraction)
    # Assert
    assert "Reference ID: REF1" in text
    assert "Load ID: LID2" in text
    assert "PO Number: PO3" in text
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
    assert "1. Widgets" in text
    assert "Weight: 1000" in text
    assert "Quantity: 10" in text
    assert "Total Rate: $1200.0 USD" in text
    assert "Rate Breakdown:" in text
    assert "Special Instructions: Handle with care" in text
    assert "Shipper Instructions: Call before arrival" in text
    assert "Carrier Instructions: No tarps" in text
    assert "Dispatcher: Dispatch Dan" in text
    assert "Dispatcher Phone: 555-9999" in text

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
    assert "Commodities:" not in text
    assert "Dispatcher:" not in text

def test_format_extraction_as_text_edge_cases(extraction_service):
    # Arrange: Some fields are empty strings or zero
    from backend.src.models.extraction_schema import Carrier, Driver, Location, Commodity, RateInfo
    data = ShipmentData(
        reference_id="",
        load_id=None,
        po_number="",
        shipper="",
        consignee=None,
        carrier=Carrier(carrier_name="", mc_number=None, phone=None),
        driver=None,
        pickup=Location(name=None, address=None, city=None, state=None, zip=None, appointment_time=None),
        drop=None,
        shipping_date=None,
        delivery_date=None,
        equipment_type=None,
        equipment_size=0,
        load_type="",
        commodities=[],
        rate_info=None,
        special_instructions=None,
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
    assert "Carrier Name:" in text  # carrier object exists, but name is empty
    assert "MC Number:" not in text
    assert "Driver Name:" not in text
    assert "Pickup Location: N/A" in text  # fallback to N/A
    assert "Commodities:" not in text
    assert "Equipment Size:" not in text  # 0 is falsy

def test_create_structured_chunk(extraction_service):
    # Arrange
    data = ShipmentData(reference_id="R1", load_id="L2")
    extraction = ExtractionResponse(data=data, document_id="doc4.txt")
    filename = "doc4.txt"
    # Act
    chunk = extraction_service.create_structured_chunk(extraction, filename)
    # Assert
    assert isinstance(chunk, Chunk)
    assert chunk.metadata.filename == filename
    assert chunk.metadata.chunk_id == 9999
    assert chunk.metadata.source == f"{filename} - Extracted Data"
    assert chunk.metadata.chunk_type == "structured_data"
    assert "EXTRACTED STRUCTURED DATA" in chunk.text
    assert "Reference ID: R1" in chunk.text

def test_create_structured_chunk_with_empty_data(extraction_service):
    # Arrange
    data = ShipmentData()
    extraction = ExtractionResponse(data=data, document_id="doc5.txt")
    filename = "doc5.txt"
    # Act
    chunk = extraction_service.create_structured_chunk(extraction, filename)
    # Assert
    assert isinstance(chunk, Chunk)
    assert chunk.metadata.filename == filename
    assert "EXTRACTED STRUCTURED DATA" in chunk.text
    # Should not contain any field lines except header

def test_format_extraction_as_text_multiple_commodities(extraction_service):
    from backend.src.models.extraction_schema import Commodity
    data = ShipmentData(
        commodities=[
            Commodity(commodity_name="A", weight="10", quantity="1", description="descA"),
            Commodity(commodity_name="B", weight=None, quantity=None, description=None),
        ]
    )
    extraction = ExtractionResponse(data=data, document_id="doc6.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Commodities:" in text
    assert "1. A" in text
    assert "2. B" in text
    assert "Weight: 10" in text
    assert "Quantity: 1" in text

def test_format_extraction_as_text_rateinfo_partial(extraction_service):
    from backend.src.models.extraction_schema import RateInfo
    data = ShipmentData(
        rate_info=RateInfo(total_rate=500.0, currency=None, rate_breakdown=None)
    )
    extraction = ExtractionResponse(data=data, document_id="doc7.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Total Rate: $500.0 USD" in text
    assert "Rate Breakdown:" not in text

def test_format_extraction_as_text_dispatcher_partial(extraction_service):
    data = ShipmentData(
        dispatcher_name="Jane",
        dispatcher_phone=None
    )
    extraction = ExtractionResponse(data=data, document_id="doc8.txt")
    text = extraction_service.format_extraction_as_text(extraction)
    assert "Dispatcher: Jane" in text
    assert "Dispatcher Phone:" not in text
