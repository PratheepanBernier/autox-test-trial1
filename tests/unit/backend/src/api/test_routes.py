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
def test_app():
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(routes.router)
    return app

@pytest.mark.asyncio
async def test_upload_document_happy_path(monkeypatch, mock_container):
    files = [
        UploadFile(filename="doc1.pdf", file=io.BytesIO(b"file1")),
        UploadFile(filename="doc2.pdf", file=io.BytesIO(b"file2")),
    ]
    process_result = MagicMock()
    process_result.message = "Upload successful"
    process_result.errors = []
    process_result.extractions = [
        MagicMock(
            filename="doc1.pdf",
            text_chunks=5,
            structured_data_extracted=True,
            reference_id="ref1",
            error=None,
        ),
        MagicMock(
            filename="doc2.pdf",
            text_chunks=3,
            structured_data_extracted=False,
            reference_id="ref2",
            error="Parse error",
        ),
    ]
    mock_container.document_pipeline_service.process_uploads.return_value = process_result

    monkeypatch.setattr(routes, "get_container", lambda: mock_container)

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
    process_result = MagicMock()
    process_result.message = "No files uploaded"
    process_result.errors = ["No files"]
    process_result.extractions = []
    mock_container.document_pipeline_service.process_uploads.return_value = process_result

    monkeypatch.setattr(routes, "get_container", lambda: mock_container)

    response = await routes.upload_document(files=files, container=mock_container)
    assert response.message == "No files uploaded"
    assert response.errors == ["No files"]
    assert response.extractions == []

@pytest.mark.asyncio
async def test_upload_document_service_error(monkeypatch, mock_container):
    files = [UploadFile(filename="doc.pdf", file=io.BytesIO(b"file"))]
    mock_container.document_pipeline_service.process_uploads.side_effect = Exception("Service down")
    monkeypatch.setattr(routes, "get_container", lambda: mock_container)
    with pytest.raises(Exception):
        await routes.upload_document(files=files, container=mock_container)

def test_ask_question_happy_path(monkeypatch, mock_container):
    query = QAQuery(question="What is the shipment date?")
    answer = SourcedAnswer(
        answer="2024-01-01",
        sources=["doc1.pdf"],
        confidence_score=0.98,
    )
    mock_container.rag_service.answer_question.return_value = answer
    monkeypatch.setattr(routes, "get_container", lambda: mock_container)
    response = routes.ask_question(query=query, container=mock_container)
    assert response == answer
    mock_container.rag_service.answer_question.assert_called_once_with(query)

def test_ask_question_error(monkeypatch, mock_container):
    query = QAQuery(question="What is the shipment date?")
    mock_container.rag_service.answer_question.side_effect = Exception("RAG failure")
    monkeypatch.setattr(routes, "get_container", lambda: mock_container)
    with pytest.raises(HTTPException) as exc:
        routes.ask_question(query=query, container=mock_container)
    assert exc.value.status_code == 500
    assert "Internal server error" in exc.value.detail

@pytest.mark.asyncio
async def test_extract_data_happy_path(monkeypatch, mock_container):
    file_content = b"shipment details"
    file = UploadFile(filename="shipment.pdf", file=io.BytesIO(file_content))
    chunks = [MagicMock(text="chunk1"), MagicMock(text="chunk2")]
    mock_container.ingestion_service.process_file.return_value = chunks
    extraction_result = ExtractionResponse(
        data=ShipmentData(field1="value1"),
        document_id="shipment.pdf",
    )
    mock_container.extraction_service.extract_data.return_value = extraction_result
    monkeypatch.setattr(routes, "get_container", lambda: mock_container)
    file.file.seek(0)
    response = await routes.extract_data(file=file, container=mock_container)
    assert response == extraction_result
    mock_container.ingestion_service.process_file.assert_called_once()
    mock_container.extraction_service.extract_data.assert_called_once()

@pytest.mark.asyncio
async def test_extract_data_no_text(monkeypatch, mock_container):
    file_content = b""
    file = UploadFile(filename="empty.pdf", file=io.BytesIO(file_content))
    mock_container.ingestion_service.process_file.return_value = []
    monkeypatch.setattr(routes, "get_container", lambda: mock_container)
    file.file.seek(0)
    response = await routes.extract_data(file=file, container=mock_container)
    assert isinstance(response, ExtractionResponse)
    assert response.data == ShipmentData()
    assert response.document_id == "empty.pdf"

@pytest.mark.asyncio
async def test_extract_data_extraction_error(monkeypatch, mock_container):
    file_content = b"shipment details"
    file = UploadFile(filename="shipment.pdf", file=io.BytesIO(file_content))
    chunks = [MagicMock(text="chunk1")]
    mock_container.ingestion_service.process_file.return_value = chunks
    mock_container.extraction_service.extract_data.side_effect = Exception("Extraction failed")
    monkeypatch.setattr(routes, "get_container", lambda: mock_container)
    file.file.seek(0)
    with pytest.raises(HTTPException) as exc:
        await routes.extract_data(file=file, container=mock_container)
    assert exc.value.status_code == 500
    assert "Internal server error" in exc.value.detail

@pytest.mark.asyncio
async def test_extract_data_file_read_error(monkeypatch, mock_container):
    file = UploadFile(filename="corrupt.pdf", file=io.BytesIO(b""))
    async def bad_read():
        raise IOError("Read failed")
    file.read = bad_read
    monkeypatch.setattr(routes, "get_container", lambda: mock_container)
    with pytest.raises(HTTPException) as exc:
        await routes.extract_data(file=file, container=mock_container)
    assert exc.value.status_code == 500
    assert "Internal server error" in exc.value.detail

def test_ping_endpoint(test_app):
    client = TestClient(test_app)
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert resp.json() == {"status": "pong"}
