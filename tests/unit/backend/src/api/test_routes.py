import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile as StarletteUploadFile
from io import BytesIO

from backend.src.api import routes
from backend.src.models.extraction_schema import ExtractionResponse, ShipmentData
from backend.src.models.schemas import QAQuery, SourcedAnswer, UploadResponse, UploadExtractionSummary

import types

@pytest.fixture
def mock_container():
    container = MagicMock()
    container.document_pipeline_service.process_uploads = AsyncMock()
    container.rag_service.answer_question = MagicMock()
    container.ingestion_service.process_file = MagicMock()
    container.extraction_service.extract_data = MagicMock()
    return container

@pytest.fixture
def test_app(monkeypatch, mock_container):
    # Patch get_container to always return our mock_container
    monkeypatch.setattr("backend.src.api.routes.get_container", lambda: mock_container)
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(routes.router)
    return app

@pytest.mark.asyncio
async def test_upload_document_happy_path(monkeypatch, mock_container):
    # Arrange
    files = [
        MagicMock(spec=UploadFile, filename="file1.pdf"),
        MagicMock(spec=UploadFile, filename="file2.pdf"),
    ]
    extraction_summaries = [
        MagicMock(
            filename="file1.pdf",
            text_chunks=5,
            structured_data_extracted=True,
            reference_id="ref1",
            error=None,
        ),
        MagicMock(
            filename="file2.pdf",
            text_chunks=3,
            structured_data_extracted=False,
            reference_id="ref2",
            error="Parse error",
        ),
    ]
    process_result = MagicMock(
        message="Processed 2 files",
        errors=[],
        extractions=extraction_summaries,
    )
    mock_container.document_pipeline_service.process_uploads.return_value = process_result

    # Act
    response = await routes.upload_document(files=files, container=mock_container)

    # Assert
    assert isinstance(response, UploadResponse)
    assert response.message == "Processed 2 files"
    assert response.errors == []
    assert len(response.extractions) == 2
    assert response.extractions[0].filename == "file1.pdf"
    assert response.extractions[1].filename == "file2.pdf"
    assert response.extractions[1].error == "Parse error"
    mock_container.document_pipeline_service.process_uploads.assert_awaited_once_with(files)

@pytest.mark.asyncio
async def test_upload_document_empty_files(monkeypatch, mock_container):
    # Arrange
    files = []
    process_result = MagicMock(
        message="No files uploaded",
        errors=["No files provided"],
        extractions=[],
    )
    mock_container.document_pipeline_service.process_uploads.return_value = process_result

    # Act
    response = await routes.upload_document(files=files, container=mock_container)

    # Assert
    assert isinstance(response, UploadResponse)
    assert response.message == "No files uploaded"
    assert response.errors == ["No files provided"]
    assert response.extractions == []
    mock_container.document_pipeline_service.process_uploads.assert_awaited_once_with(files)

def test_ask_question_happy_path(mock_container):
    # Arrange
    query = QAQuery(question="What is the shipment date?")
    answer = SourcedAnswer(
        answer="2024-01-01",
        sources=["doc1.pdf"],
        confidence_score=0.98,
    )
    mock_container.rag_service.answer_question.return_value = answer

    # Act
    response = routes.ask_question(query=query, container=mock_container)

    # Assert
    assert response == answer
    mock_container.rag_service.answer_question.assert_called_once_with(query)

def test_ask_question_exception(monkeypatch, mock_container):
    # Arrange
    query = QAQuery(question="What is the shipment date?")
    mock_container.rag_service.answer_question.side_effect = Exception("RAG failure")

    # Act & Assert
    with pytest.raises(HTTPException) as excinfo:
        routes.ask_question(query=query, container=mock_container)
    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Internal server error processing question."
    mock_container.rag_service.answer_question.assert_called_once_with(query)

@pytest.mark.asyncio
async def test_extract_data_happy_path(monkeypatch, mock_container):
    # Arrange
    file_content = b"file content"
    filename = "shipment.pdf"
    file = MagicMock(spec=UploadFile)
    file.filename = filename
    file.read = AsyncMock(return_value=file_content)
    chunks = [MagicMock(text="chunk1"), MagicMock(text="chunk2")]
    mock_container.ingestion_service.process_file.return_value = chunks
    full_text = "chunk1\nchunk2"
    extraction_result = ExtractionResponse(
        data=ShipmentData(shipment_id="123"),
        document_id=filename,
    )
    mock_container.extraction_service.extract_data.return_value = extraction_result

    # Act
    response = await routes.extract_data(file=file, container=mock_container)

    # Assert
    file.read.assert_awaited_once()
    mock_container.ingestion_service.process_file.assert_called_once_with(file_content, filename)
    mock_container.extraction_service.extract_data.assert_called_once_with(full_text, filename)
    assert response == extraction_result

@pytest.mark.asyncio
async def test_extract_data_no_text_extracted(monkeypatch, mock_container):
    # Arrange
    file_content = b"empty"
    filename = "empty.pdf"
    file = MagicMock(spec=UploadFile)
    file.filename = filename
    file.read = AsyncMock(return_value=file_content)
    mock_container.ingestion_service.process_file.return_value = []
    # Act
    response = await routes.extract_data(file=file, container=mock_container)
    # Assert
    file.read.assert_awaited_once()
    mock_container.ingestion_service.process_file.assert_called_once_with(file_content, filename)
    assert isinstance(response, ExtractionResponse)
    assert response.data == ShipmentData()
    assert response.document_id == filename
    mock_container.extraction_service.extract_data.assert_not_called()

@pytest.mark.asyncio
async def test_extract_data_exception(monkeypatch, mock_container):
    # Arrange
    file_content = b"bad"
    filename = "bad.pdf"
    file = MagicMock(spec=UploadFile)
    file.filename = filename
    file.read = AsyncMock(return_value=file_content)
    mock_container.ingestion_service.process_file.side_effect = Exception("Processing failed")
    # Act & Assert
    with pytest.raises(HTTPException) as excinfo:
        await routes.extract_data(file=file, container=mock_container)
    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Internal server error during extraction."
    file.read.assert_awaited_once()
    mock_container.ingestion_service.process_file.assert_called_once_with(file_content, filename)
    mock_container.extraction_service.extract_data.assert_not_called()

@pytest.mark.asyncio
async def test_extract_data_empty_file(monkeypatch, mock_container):
    # Arrange
    file_content = b""
    filename = "emptyfile.pdf"
    file = MagicMock(spec=UploadFile)
    file.filename = filename
    file.read = AsyncMock(return_value=file_content)
    mock_container.ingestion_service.process_file.return_value = []
    # Act
    response = await routes.extract_data(file=file, container=mock_container)
    # Assert
    file.read.assert_awaited_once()
    mock_container.ingestion_service.process_file.assert_called_once_with(file_content, filename)
    assert isinstance(response, ExtractionResponse)
    assert response.data == ShipmentData()
    assert response.document_id == filename
    mock_container.extraction_service.extract_data.assert_not_called()

def test_ping_endpoint(test_app):
    # Arrange
    client = TestClient(test_app)
    # Act
    resp = client.get("/ping")
    # Assert
    assert resp.status_code == 200
    assert resp.json() == {"status": "pong"}

@pytest.mark.asyncio
async def test_upload_document_error_from_service(monkeypatch, mock_container):
    # Arrange
    files = [MagicMock(spec=UploadFile, filename="file1.pdf")]
    process_result = MagicMock(
        message="Processed with errors",
        errors=["File corrupted"],
        extractions=[
            MagicMock(
                filename="file1.pdf",
                text_chunks=0,
                structured_data_extracted=False,
                reference_id=None,
                error="File corrupted",
            )
        ],
    )
    mock_container.document_pipeline_service.process_uploads.return_value = process_result
    # Act
    response = await routes.upload_document(files=files, container=mock_container)
    # Assert
    assert response.message == "Processed with errors"
    assert response.errors == ["File corrupted"]
    assert len(response.extractions) == 1
    assert response.extractions[0].error == "File corrupted"

@pytest.mark.asyncio
async def test_upload_document_service_exception(monkeypatch, mock_container):
    # Arrange
    files = [MagicMock(spec=UploadFile, filename="file1.pdf")]
    mock_container.document_pipeline_service.process_uploads.side_effect = Exception("Service down")
    # Act & Assert
    with pytest.raises(Exception) as excinfo:
        await routes.upload_document(files=files, container=mock_container)
    assert "Service down" in str(excinfo.value)
    mock_container.document_pipeline_service.process_uploads.assert_awaited_once_with(files)

def test_ask_question_empty_question(mock_container):
    # Arrange
    query = QAQuery(question="")
    answer = SourcedAnswer(
        answer="",
        sources=[],
        confidence_score=0.0,
    )
    mock_container.rag_service.answer_question.return_value = answer
    # Act
    response = routes.ask_question(query=query, container=mock_container)
    # Assert
    assert response == answer
    mock_container.rag_service.answer_question.assert_called_once_with(query)

@pytest.mark.asyncio
async def test_extract_data_boundary_long_text(monkeypatch, mock_container):
    # Arrange
    file_content = b"x" * 10_000
    filename = "long.pdf"
    file = MagicMock(spec=UploadFile)
    file.filename = filename
    file.read = AsyncMock(return_value=file_content)
    chunks = [MagicMock(text="x" * 10_000)]
    mock_container.ingestion_service.process_file.return_value = chunks
    extraction_result = ExtractionResponse(
        data=ShipmentData(shipment_id="longid"),
        document_id=filename,
    )
    mock_container.extraction_service.extract_data.return_value = extraction_result
    # Act
    response = await routes.extract_data(file=file, container=mock_container)
    # Assert
    file.read.assert_awaited_once()
    mock_container.ingestion_service.process_file.assert_called_once_with(file_content, filename)
    mock_container.extraction_service.extract_data.assert_called_once_with("x" * 10_000, filename)
    assert response == extraction_result
