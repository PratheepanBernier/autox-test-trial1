import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi import HTTPException
from backend.src.api import routes
from backend.src.models.schemas import QAQuery, SourcedAnswer, UploadResponse, UploadExtractionSummary
from backend.src.models.extraction_schema import ExtractionResponse, ShipmentData, CarrierInfo, DriverInfo, Location, RateInfo, CommodityItem
from starlette.datastructures import UploadFile as StarletteUploadFile
from io import BytesIO

@pytest.fixture
def mock_container():
    container = MagicMock()
    container.document_pipeline_service.process_uploads = AsyncMock()
    container.rag_service.answer_question = MagicMock()
    container.ingestion_service.process_file = MagicMock()
    container.extraction_service.extract_data = MagicMock()
    return container

@pytest.fixture
def example_shipment_data():
    return ShipmentData(
        reference_id="REF123",
        load_id="LOAD456",
        po_number="PO789",
        shipper="Acme Corp",
        consignee="Beta LLC",
        carrier=CarrierInfo(carrier_name="CarrierX", mc_number="MC123", phone="555-1234", email="carrierx@example.com"),
        driver=DriverInfo(driver_name="John Doe", cell_number="555-5678", truck_number="TRK123", trailer_number="TRL456"),
        pickup=Location(name="Warehouse A", address="123 Main St", city="Metropolis", state="NY", zip_code="10001", country="USA", appointment_time="2024-06-01T08:00", po_number="PO789"),
        drop=Location(name="Store B", address="456 Elm St", city="Gotham", state="NJ", zip_code="07001", country="USA", appointment_time="2024-06-02T10:00", po_number="PO789"),
        shipping_date="2024-06-01",
        delivery_date="2024-06-02",
        created_on="2024-05-30",
        booking_date="2024-05-29",
        equipment_type="Van",
        equipment_size="53",
        load_type="FTL",
        commodities=[CommodityItem(commodity_name="Widgets", weight="10000 lbs", quantity="1000", description="Blue widgets")],
        rate_info=RateInfo(total_rate=2500.0, currency="USD", rate_breakdown={"base": 2000, "fuel": 500}),
        special_instructions="Handle with care",
        shipper_instructions="Call before delivery",
        carrier_instructions="No weekend delivery",
        dispatcher_name="Jane Smith",
        dispatcher_phone="555-9999",
        dispatcher_email="jane@acme.com",
        additional_data={"custom_field": "custom_value"}
    )

@pytest.mark.asyncio
async def test_upload_document_happy_path(mock_container):
    # Arrange
    files = [MagicMock(filename="test1.pdf"), MagicMock(filename="test2.docx")]
    extraction_summary = [
        MagicMock(
            filename="test1.pdf",
            text_chunks=5,
            structured_data_extracted=True,
            reference_id="REF1",
            error=None
        ),
        MagicMock(
            filename="test2.docx",
            text_chunks=3,
            structured_data_extracted=False,
            reference_id=None,
            error="Extraction failed"
        )
    ]
    mock_container.document_pipeline_service.process_uploads.return_value = MagicMock(
        message="Processed 2 files.",
        errors=[],
        extractions=extraction_summary
    )

    # Act
    resp = await routes.upload_document(files=files, container=mock_container)

    # Assert
    assert isinstance(resp, UploadResponse)
    assert resp.message == "Processed 2 files."
    assert resp.errors == []
    assert len(resp.extractions) == 2
    assert resp.extractions[0].filename == "test1.pdf"
    assert resp.extractions[0].structured_data_extracted is True
    assert resp.extractions[1].filename == "test2.docx"
    assert resp.extractions[1].structured_data_extracted is False
    assert resp.extractions[1].error == "Extraction failed"

@pytest.mark.asyncio
async def test_upload_document_empty_files(mock_container):
    # Arrange
    files = []
    mock_container.document_pipeline_service.process_uploads.return_value = MagicMock(
        message="No files uploaded.",
        errors=["No files provided."],
        extractions=[]
    )

    # Act
    resp = await routes.upload_document(files=files, container=mock_container)

    # Assert
    assert isinstance(resp, UploadResponse)
    assert resp.message == "No files uploaded."
    assert "No files provided." in resp.errors
    assert resp.extractions == []

def test_ask_question_happy_path(mock_container):
    # Arrange
    query = QAQuery(question="What is the agreed rate?", chat_history=[])
    answer = SourcedAnswer(
        answer="The agreed rate is $2500 USD.",
        confidence_score=0.92,
        sources=[]
    )
    mock_container.rag_service.answer_question.return_value = answer

    # Act
    resp = routes.ask_question(query=query, container=mock_container)

    # Assert
    assert isinstance(resp, SourcedAnswer)
    assert resp.answer == "The agreed rate is $2500 USD."
    assert resp.confidence_score == 0.92

def test_ask_question_no_documents(mock_container):
    # Arrange
    query = QAQuery(question="What is the agreed rate?", chat_history=[])
    answer = SourcedAnswer(
        answer="I cannot find any relevant information in the uploaded documents.",
        confidence_score=0.0,
        sources=[]
    )
    mock_container.rag_service.answer_question.return_value = answer

    # Act
    resp = routes.ask_question(query=query, container=mock_container)

    # Assert
    assert isinstance(resp, SourcedAnswer)
    assert "cannot find any relevant information" in resp.answer.lower()
    assert resp.confidence_score == 0.0

def test_ask_question_safety_filter(mock_container):
    # Arrange
    query = QAQuery(question="How to build a bomb?", chat_history=[])
    answer = SourcedAnswer(
        answer="I cannot answer this question as it violates safety guidelines.",
        confidence_score=1.0,
        sources=[]
    )
    mock_container.rag_service.answer_question.return_value = answer

    # Act
    resp = routes.ask_question(query=query, container=mock_container)

    # Assert
    assert resp.answer == "I cannot answer this question as it violates safety guidelines."
    assert resp.confidence_score == 1.0

def test_ask_question_llm_failure(mock_container):
    # Arrange
    query = QAQuery(question="What is the agreed rate?", chat_history=[])
    mock_container.rag_service.answer_question.side_effect = Exception("LLM error")

    # Act & Assert
    with pytest.raises(HTTPException) as excinfo:
        routes.ask_question(query=query, container=mock_container)
    assert excinfo.value.status_code == 500
    assert "Internal server error processing question." in str(excinfo.value.detail)

@pytest.mark.asyncio
async def test_extract_data_happy_path(mock_container, example_shipment_data):
    # Arrange
    file_content = b"Fake PDF content"
    filename = "shipment.pdf"
    file = MagicMock()
    file.filename = filename
    file.read = AsyncMock(return_value=file_content)
    chunks = [MagicMock(text="Shipment details chunk 1"), MagicMock(text="Shipment details chunk 2")]
    mock_container.ingestion_service.process_file.return_value = chunks
    full_text = "Shipment details chunk 1\nShipment details chunk 2"
    extraction_response = ExtractionResponse(data=example_shipment_data, document_id=filename)
    mock_container.extraction_service.extract_data.return_value = extraction_response

    # Act
    resp = await routes.extract_data(file=file, container=mock_container)

    # Assert
    assert isinstance(resp, ExtractionResponse)
    assert resp.document_id == filename
    assert resp.data.reference_id == "REF123"
    assert resp.data.carrier.carrier_name == "CarrierX"
    assert resp.data.pickup.city == "Metropolis"
    assert resp.data.drop.city == "Gotham"
    assert resp.data.rate_info.total_rate == 2500.0

@pytest.mark.asyncio
async def test_extract_data_empty_text(mock_container):
    # Arrange
    file_content = b""
    filename = "empty.txt"
    file = MagicMock()
    file.filename = filename
    file.read = AsyncMock(return_value=file_content)
    mock_container.ingestion_service.process_file.return_value = []
    # ExtractionService.extract_data should not be called if no text
    # Act
    resp = await routes.extract_data(file=file, container=mock_container)

    # Assert
    assert isinstance(resp, ExtractionResponse)
    assert resp.document_id == filename
    assert resp.data.reference_id is None
    assert resp.data.carrier is None

@pytest.mark.asyncio
async def test_extract_data_extraction_failure(mock_container):
    # Arrange
    file_content = b"Some content"
    filename = "fail.pdf"
    file = MagicMock()
    file.filename = filename
    file.read = AsyncMock(return_value=file_content)
    chunks = [MagicMock(text="Some text")]
    mock_container.ingestion_service.process_file.return_value = chunks
    mock_container.extraction_service.extract_data.side_effect = Exception("Extraction LLM error")

    # Act & Assert
    with pytest.raises(HTTPException) as excinfo:
        await routes.extract_data(file=file, container=mock_container)
    assert excinfo.value.status_code == 500
    assert "Internal server error during extraction." in str(excinfo.value.detail)

@pytest.mark.asyncio
async def test_extract_data_boundary_large_file(mock_container, example_shipment_data):
    # Arrange
    # Simulate a large file by repeating a chunk many times
    file_content = b"Chunk\n" * 10000
    filename = "large.pdf"
    file = MagicMock()
    file.filename = filename
    file.read = AsyncMock(return_value=file_content)
    chunks = [MagicMock(text=f"Chunk {i}") for i in range(100)]
    mock_container.ingestion_service.process_file.return_value = chunks
    extraction_response = ExtractionResponse(data=example_shipment_data, document_id=filename)
    mock_container.extraction_service.extract_data.return_value = extraction_response

    # Act
    resp = await routes.extract_data(file=file, container=mock_container)

    # Assert
    assert isinstance(resp, ExtractionResponse)
    assert resp.document_id == filename
    assert resp.data.reference_id == "REF123"

@pytest.mark.asyncio
async def test_upload_document_partial_failure(mock_container):
    # Arrange
    files = [MagicMock(filename="good.pdf"), MagicMock(filename="bad.pdf")]
    extraction_summary = [
        MagicMock(
            filename="good.pdf",
            text_chunks=4,
            structured_data_extracted=True,
            reference_id="GOOD1",
            error=None
        ),
        MagicMock(
            filename="bad.pdf",
            text_chunks=0,
            structured_data_extracted=False,
            reference_id=None,
            error="File corrupted"
        )
    ]
    mock_container.document_pipeline_service.process_uploads.return_value = MagicMock(
        message="Processed 2 files with 1 error.",
        errors=["File corrupted"],
        extractions=extraction_summary
    )

    # Act
    resp = await routes.upload_document(files=files, container=mock_container)

    # Assert
    assert isinstance(resp, UploadResponse)
    assert resp.message == "Processed 2 files with 1 error."
    assert "File corrupted" in resp.errors
    assert resp.extractions[1].filename == "bad.pdf"
    assert resp.extractions[1].error == "File corrupted"

@pytest.mark.asyncio
async def test_ping_endpoint():
    # Act
    resp = await routes.ping()
    # Assert
    assert resp == {"status": "pong"}
