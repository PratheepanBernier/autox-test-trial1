import io
import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from backend.src.api import routes
from backend.src.models.extraction_schema import ExtractionResponse, ShipmentData
from backend.src.models.schemas import (
    QAQuery,
    SourcedAnswer,
    UploadResponse,
    UploadExtractionSummary,
)

@pytest.fixture
def mock_container():
    container = MagicMock()
    container.document_pipeline_service.process_uploads = AsyncMock()
    container.rag_service.answer_question = MagicMock()
    container.ingestion_service.process_file = MagicMock()
    container.extraction_service.extract_data = MagicMock()
    return container

@pytest.fixture
def test_client():
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)

@pytest.mark.asyncio
async def test_upload_document_happy_path(monkeypatch, mock_container):
    files = [
        UploadFile(filename="doc1.pdf", file=io.BytesIO(b"file1")),
        UploadFile(filename="doc2.pdf", file=io.BytesIO(b"file2")),
    ]
    mock_result = MagicMock()
    mock_result.message = "Upload successful"
    mock_result.errors = []
    mock_result.extractions = [
        MagicMock(
            filename="doc1.pdf",
            text_chunks=5,
            structured_data_extracted=True,
            reference_id="abc123",
            error=None,
        ),
        MagicMock(
            filename="doc2.pdf",
            text_chunks=3,
            structured_data_extracted=False,
            reference_id="def456",
            error="Parse error",
        ),
    ]
    monkeypatch.setattr(routes, "get_container", lambda: mock_container)
    mock_container.document_pipeline_service.process_uploads.return_value = mock_result

    response = await routes.upload_document(files=files, container=mock_container)
    assert isinstance(response, UploadResponse)
    assert response.message == "Upload successful"
    assert response.errors == []
    assert len(response.extractions) == 2
    assert response.extractions[0].filename == "doc1.pdf"
    assert response.extractions[1].error == "Parse error"

@pytest.mark.asyncio
async def test_upload_document_empty_files(monkeypatch, mock_container):
    files = []
    mock_result = MagicMock()
    mock_result.message = "No files uploaded"
    mock_result.errors = ["No files provided"]
    mock_result.extractions = []
    monkeypatch.setattr(routes, "get_container", lambda: mock_container)
    mock_container.document_pipeline_service.process_uploads.return_value = mock_result

    response = await routes.upload_document(files=files, container=mock_container)
    assert isinstance(response, UploadResponse)
    assert response.message == "No files uploaded"
    assert response.errors == ["No files provided"]
    assert response.extractions == []

@pytest.mark.asyncio
async def test_upload_document_service_error(monkeypatch, mock_container):
    files = [UploadFile(filename="doc1.pdf", file=io.BytesIO(b"file1"))]
    monkeypatch.setattr(routes, "get_container", lambda: mock_container)
    mock_container.document_pipeline_service.process_uploads.side_effect = Exception("Service down")
    with pytest.raises(Exception) as exc:
        await routes.upload_document(files=files, container=mock_container)
    assert "Service down" in str(exc.value)

def test_ask_question_happy_path(monkeypatch, mock_container):
    query = QAQuery(question="What is the shipment date?")
    answer = SourcedAnswer(
        answer="2024-01-01",
        confidence_score=0.98,
        sources=["doc1.pdf"],
    )
    monkeypatch.setattr(routes, "get_container", lambda: mock_container)
    mock_container.rag_service.answer_question.return_value = answer

    response = routes.ask_question(query=query, container=mock_container)
    assert response == answer
    mock_container.rag_service.answer_question.assert_called_once_with(query)

def test_ask_question_service_error(monkeypatch, mock_container):
    query = QAQuery(question="What is the shipment date?")
    monkeypatch.setattr(routes, "get_container", lambda: mock_container)
    mock_container.rag_service.answer_question.side_effect = Exception("RAG failure")
    with pytest.raises(HTTPException) as exc:
        routes.ask_question(query=query, container=mock_container)
    assert exc.value.status_code == 500
    assert "Internal server error processing question." in exc.value.detail

@pytest.mark.asyncio
async def test_extract_data_happy_path(monkeypatch, mock_container):
    file_content = b"shipment details"
    file = UploadFile(filename="shipment.pdf", file=io.BytesIO(file_content))
    chunks = [MagicMock(text="chunk1"), MagicMock(text="chunk2")]
    extraction_result = ExtractionResponse(
        data=ShipmentData(order_id="ORD123"),
        document_id="shipment.pdf",
    )
    monkeypatch.setattr(routes, "get_container", lambda: mock_container)
    mock_container.ingestion_service.process_file.return_value = chunks
    mock_container.extraction_service.extract_data.return_value = extraction_result

    # Patch file.read to return file_content
    async def fake_read():
        return file_content
    file.read = fake_read

    response = await routes.extract_data(file=file, container=mock_container)
    assert isinstance(response, ExtractionResponse)
    assert response.data.order_id == "ORD123"
    assert response.document_id == "shipment.pdf"
    mock_container.ingestion_service.process_file.assert_called_once()
    mock_container.extraction_service.extract_data.assert_called_once()

@pytest.mark.asyncio
async def test_extract_data_no_text_extracted(monkeypatch, mock_container):
    file_content = b""
    file = UploadFile(filename="empty.pdf", file=io.BytesIO(file_content))
    monkeypatch.setattr(routes, "get_container", lambda: mock_container)
    mock_container.ingestion_service.process_file.return_value = []
    # Patch file.read to return file_content
    async def fake_read():
        return file_content
    file.read = fake_read

    response = await routes.extract_data(file=file, container=mock_container)
    assert isinstance(response, ExtractionResponse)
    assert response.data == ShipmentData()
    assert response.document_id == "empty.pdf"
    mock_container.ingestion_service.process_file.assert_called_once()
    mock_container.extraction_service.extract_data.assert_not_called()

@pytest.mark.asyncio
async def test_extract_data_extraction_service_error(monkeypatch, mock_container):
    file_content = b"shipment details"
    file = UploadFile(filename="shipment.pdf", file=io.BytesIO(file_content))
    chunks = [MagicMock(text="chunk1")]
    monkeypatch.setattr(routes, "get_container", lambda: mock_container)
    mock_container.ingestion_service.process_file.return_value = chunks
    mock_container.extraction_service.extract_data.side_effect = Exception("Extraction failed")
    # Patch file.read to return file_content
    async def fake_read():
        return file_content
    file.read = fake_read

    with pytest.raises(HTTPException) as exc:
        await routes.extract_data(file=file, container=mock_container)
    assert exc.value.status_code == 500
    assert "Internal server error during extraction." in exc.value.detail

def test_ping_endpoint(test_client):
    response = test_client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "pong"}
