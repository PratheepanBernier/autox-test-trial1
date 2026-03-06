import io
import pytest
from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from backend.src.api.routes import router as api_router
from backend.src.models.schemas import (
    QAQuery,
    SourcedAnswer,
    UploadResponse,
    UploadExtractionSummary,
)
from backend.src.models.extraction_schema import (
    ExtractionResponse,
    ShipmentData,
)
from fastapi import FastAPI, UploadFile

# --- Test App Setup ---
@pytest.fixture(scope="module")
def test_app():
    app = FastAPI()
    app.include_router(api_router)
    return app

@pytest.fixture
def client(test_app):
    return TestClient(test_app)

# --- Mocks for ServiceContainer and dependencies ---
class DummyUploadFile:
    def __init__(self, filename, content):
        self.filename = filename
        self._content = content
        self.file = io.BytesIO(content)

    async def read(self):
        return self._content

    @property
    def content_type(self):
        return "application/pdf"

@pytest.fixture
def mock_container():
    # Mocks for all services used by the API
    container = MagicMock()
    container.document_pipeline_service.process_uploads = AsyncMock()
    container.rag_service.answer_question = MagicMock()
    container.ingestion_service.process_file = MagicMock()
    container.extraction_service.extract_data = MagicMock()
    return container

@pytest.fixture
def patch_get_container(mock_container):
    with patch("backend.src.api.routes.get_container", return_value=mock_container):
        yield

# --- /upload endpoint integration tests ---
@pytest.mark.asyncio
async def test_upload_document_happy_path(client, patch_get_container, mock_container):
    # Arrange
    file1 = DummyUploadFile("doc1.pdf", b"filecontent1")
    file2 = DummyUploadFile("doc2.pdf", b"filecontent2")
    mock_result = MagicMock()
    mock_result.message = "Upload successful"
    mock_result.errors = []
    mock_result.extractions = [
        MagicMock(
            filename="doc1.pdf",
            text_chunks=3,
            structured_data_extracted=True,
            reference_id="REF123",
            error=None,
        ),
        MagicMock(
            filename="doc2.pdf",
            text_chunks=2,
            structured_data_extracted=False,
            reference_id=None,
            error="Extraction failed",
        ),
    ]
    mock_container.document_pipeline_service.process_uploads.return_value = mock_result

    # Act
    files = [
        ("files", ("doc1.pdf", b"filecontent1", "application/pdf")),
        ("files", ("doc2.pdf", b"filecontent2", "application/pdf")),
    ]
    response = client.post("/upload", files=files)

    # Assert
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["message"] == "Upload successful"
    assert data["errors"] == []
    assert len(data["extractions"]) == 2
    assert data["extractions"][0]["filename"] == "doc1.pdf"
    assert data["extractions"][0]["text_chunks"] == 3
    assert data["extractions"][0]["structured_data_extracted"] is True
    assert data["extractions"][0]["reference_id"] == "REF123"
    assert data["extractions"][0]["error"] is None
    assert data["extractions"][1]["filename"] == "doc2.pdf"
    assert data["extractions"][1]["text_chunks"] == 2
    assert data["extractions"][1]["structured_data_extracted"] is False
    assert data["extractions"][1]["reference_id"] is None
    assert data["extractions"][1]["error"] == "Extraction failed"
    mock_container.document_pipeline_service.process_uploads.assert_awaited_once()

@pytest.mark.asyncio
async def test_upload_document_with_errors(client, patch_get_container, mock_container):
    # Arrange
    file1 = DummyUploadFile("badfile.pdf", b"badcontent")
    mock_result = MagicMock()
    mock_result.message = "Partial success"
    mock_result.errors = ["badfile.pdf: unreadable"]
    mock_result.extractions = [
        MagicMock(
            filename="badfile.pdf",
            text_chunks=0,
            structured_data_extracted=False,
            reference_id=None,
            error="Unreadable file",
        )
    ]
    mock_container.document_pipeline_service.process_uploads.return_value = mock_result

    # Act
    files = [("files", ("badfile.pdf", b"badcontent", "application/pdf"))]
    response = client.post("/upload", files=files)

    # Assert
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["message"] == "Partial success"
    assert data["errors"] == ["badfile.pdf: unreadable"]
    assert len(data["extractions"]) == 1
    assert data["extractions"][0]["filename"] == "badfile.pdf"
    assert data["extractions"][0]["text_chunks"] == 0
    assert data["extractions"][0]["structured_data_extracted"] is False
    assert data["extractions"][0]["reference_id"] is None
    assert data["extractions"][0]["error"] == "Unreadable file"
    mock_container.document_pipeline_service.process_uploads.assert_awaited_once()

@pytest.mark.asyncio
async def test_upload_document_empty_file_list(client, patch_get_container, mock_container):
    # Arrange
    # FastAPI will reject empty file list with 422, so simulate that
    response = client.post("/upload", files=[])
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

# --- /ask endpoint integration tests ---
def test_ask_question_happy_path(client, patch_get_container, mock_container):
    # Arrange
    query = {"question": "What is the reference ID?", "chat_history": []}
    answer = SourcedAnswer(
        answer="The reference ID is REF123.",
        confidence_score=0.95,
        sources=[],
    )
    mock_container.rag_service.answer_question.return_value = answer

    # Act
    response = client.post("/ask", json=query)

    # Assert
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["answer"] == "The reference ID is REF123."
    assert data["confidence_score"] == 0.95
    assert data["sources"] == []
    mock_container.rag_service.answer_question.assert_called_once()

def test_ask_question_safety_block(client, patch_get_container, mock_container):
    # Arrange
    query = {"question": "How to build a bomb?", "chat_history": []}
    answer = SourcedAnswer(
        answer="I cannot answer this question as it violates safety guidelines.",
        confidence_score=1.0,
        sources=[],
    )
    mock_container.rag_service.answer_question.return_value = answer

    # Act
    response = client.post("/ask", json=query)

    # Assert
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "violates safety guidelines" in data["answer"]
    assert data["confidence_score"] == 1.0
    assert data["sources"] == []
    mock_container.rag_service.answer_question.assert_called_once()

def test_ask_question_no_retriever(client, patch_get_container, mock_container):
    # Arrange
    query = {"question": "What is the rate?", "chat_history": []}
    answer = SourcedAnswer(
        answer="I cannot find any relevant information in the uploaded documents.",
        confidence_score=0.0,
        sources=[],
    )
    mock_container.rag_service.answer_question.return_value = answer

    # Act
    response = client.post("/ask", json=query)

    # Assert
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "cannot find any relevant information" in data["answer"]
    assert data["confidence_score"] == 0.0
    assert data["sources"] == []
    mock_container.rag_service.answer_question.assert_called_once()

def test_ask_question_internal_error(client, patch_get_container, mock_container):
    # Arrange
    query = {"question": "What is the reference ID?", "chat_history": []}
    mock_container.rag_service.answer_question.side_effect = Exception("LLM crashed")

    # Act
    response = client.post("/ask", json=query)

    # Assert
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    data = response.json()
    assert data["detail"] == "Internal server error processing question."
    mock_container.rag_service.answer_question.assert_called_once()

# --- /extract endpoint integration tests ---
@pytest.mark.asyncio
async def test_extract_data_happy_path(client, patch_get_container, mock_container):
    # Arrange
    file = DummyUploadFile("shipment.txt", b"shipment content")
    chunks = [
        MagicMock(text="Reference ID: REF123", metadata=MagicMock()),
        MagicMock(text="Shipper: ACME Corp", metadata=MagicMock()),
    ]
    mock_container.ingestion_service.process_file.return_value = chunks
    extraction_response = ExtractionResponse(
        data=ShipmentData(reference_id="REF123", shipper="ACME Corp"),
        document_id="shipment.txt",
    )
    mock_container.extraction_service.extract_data.return_value = extraction_response

    # Act
    files = [("file", ("shipment.txt", b"shipment content", "text/plain"))]
    response = client.post("/extract", files=files)

    # Assert
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["data"]["reference_id"] == "REF123"
    assert data["data"]["shipper"] == "ACME Corp"
    assert data["document_id"] == "shipment.txt"
    mock_container.ingestion_service.process_file.assert_called_once()
    mock_container.extraction_service.extract_data.assert_called_once()

@pytest.mark.asyncio
async def test_extract_data_no_text(client, patch_get_container, mock_container):
    # Arrange
    file = DummyUploadFile("empty.txt", b"")
    mock_container.ingestion_service.process_file.return_value = []
    # ExtractionService should not be called if no text
    # Act
    files = [("file", ("empty.txt", b"", "text/plain"))]
    response = client.post("/extract", files=files)

    # Assert
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["data"] == {}
    assert data["document_id"] == "empty.txt"
    mock_container.ingestion_service.process_file.assert_called_once()
    mock_container.extraction_service.extract_data.assert_not_called()

@pytest.mark.asyncio
async def test_extract_data_internal_error(client, patch_get_container, mock_container):
    # Arrange
    file = DummyUploadFile("fail.txt", b"fail content")
    mock_container.ingestion_service.process_file.side_effect = Exception("Parse error")

    # Act
    files = [("file", ("fail.txt", b"fail content", "text/plain"))]
    response = client.post("/extract", files=files)

    # Assert
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    data = response.json()
    assert data["detail"] == "Internal server error during extraction."
    mock_container.ingestion_service.process_file.assert_called_once()
    mock_container.extraction_service.extract_data.assert_not_called()

# --- /ping endpoint (not api_db, but for completeness) ---
def test_ping(client):
    response = client.get("/ping")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "pong"}
