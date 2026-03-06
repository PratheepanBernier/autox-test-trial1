import io
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile as StarletteUploadFile
from unittest.mock import AsyncMock, MagicMock, patch

from backend.src.api.routes import router as api_router
from backend.src.models.schemas import (
    UploadResponse,
    UploadExtractionSummary,
    QAQuery,
    SourcedAnswer,
)
from backend.src.models.extraction_schema import (
    ExtractionResponse,
    ShipmentData,
)
from backend.src.dependencies import get_container

@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(api_router)
    return app

@pytest.fixture
def client(app):
    return TestClient(app)

class DummyUploadFile(StarletteUploadFile):
    def __init__(self, filename, content):
        super().__init__(filename=filename, file=io.BytesIO(content))

@pytest.fixture
def dummy_files():
    return [
        DummyUploadFile("test1.txt", b"Test file content 1"),
        DummyUploadFile("test2.txt", b"Test file content 2"),
    ]

@pytest.fixture
def mock_container():
    # Mock the service container and its services
    container = MagicMock()
    # Mock document_pipeline_service.process_uploads
    process_uploads_result = MagicMock()
    process_uploads_result.message = "Upload successful"
    process_uploads_result.errors = []
    process_uploads_result.extractions = [
        MagicMock(
            filename="test1.txt",
            text_chunks=3,
            structured_data_extracted=True,
            reference_id="REF123",
            error=None,
        ),
        MagicMock(
            filename="test2.txt",
            text_chunks=2,
            structured_data_extracted=False,
            reference_id=None,
            error="Extraction failed",
        ),
    ]
    container.document_pipeline_service.process_uploads = AsyncMock(return_value=process_uploads_result)

    # Mock rag_service.answer_question
    sourced_answer = SourcedAnswer(
        answer="The answer is 42.",
        confidence_score=0.92,
        sources=[],
    )
    container.rag_service.answer_question = MagicMock(return_value=sourced_answer)

    # Mock ingestion_service.process_file
    from backend.src.models.schemas import Chunk, DocumentMetadata
    dummy_chunks = [
        Chunk(
            text="Chunk 1 text",
            metadata=DocumentMetadata(
                filename="test3.txt",
                chunk_id=0,
                source="test3.txt - General",
                chunk_type="text",
            ),
        ),
        Chunk(
            text="Chunk 2 text",
            metadata=DocumentMetadata(
                filename="test3.txt",
                chunk_id=1,
                source="test3.txt - General",
                chunk_type="text",
            ),
        ),
    ]
    container.ingestion_service.process_file = MagicMock(return_value=dummy_chunks)

    # Mock extraction_service.extract_data
    extraction_response = ExtractionResponse(
        data=ShipmentData(reference_id="REF999", shipper="ACME Inc."),
        document_id="test3.txt",
    )
    container.extraction_service.extract_data = MagicMock(return_value=extraction_response)

    return container

@pytest.fixture(autouse=True)
def override_container_dependency(app, mock_container):
    app.dependency_overrides[get_container] = lambda: mock_container
    yield
    app.dependency_overrides.clear()

def test_upload_document_integration(client, dummy_files):
    # Simulate multipart upload
    files = [
        ("files", (f.filename, f.file, "text/plain"))
        for f in dummy_files
    ]
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Upload successful"
    assert data["errors"] == []
    assert len(data["extractions"]) == 2
    assert data["extractions"][0]["filename"] == "test1.txt"
    assert data["extractions"][0]["text_chunks"] == 3
    assert data["extractions"][0]["structured_data_extracted"] is True
    assert data["extractions"][0]["reference_id"] == "REF123"
    assert data["extractions"][0]["error"] is None
    assert data["extractions"][1]["filename"] == "test2.txt"
    assert data["extractions"][1]["text_chunks"] == 2
    assert data["extractions"][1]["structured_data_extracted"] is False
    assert data["extractions"][1]["reference_id"] is None
    assert data["extractions"][1]["error"] == "Extraction failed"

def test_ask_question_integration(client):
    query = {"question": "What is the answer to life?"}
    response = client.post("/ask", json=query)
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The answer is 42."
    assert data["confidence_score"] == pytest.approx(0.92)
    assert isinstance(data["sources"], list)

def test_extract_data_integration(client):
    # Simulate file upload for extraction
    file_content = b"Sample document for extraction"
    files = {"file": ("test3.txt", io.BytesIO(file_content), "text/plain")}
    response = client.post("/extract", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert data["document_id"] == "test3.txt"
    assert data["data"]["reference_id"] == "REF999"
    assert data["data"]["shipper"] == "ACME Inc."

def test_extract_data_empty_text(client, mock_container):
    # Simulate ingestion_service.process_file returning empty list
    mock_container.ingestion_service.process_file.return_value = []
    file_content = b""
    files = {"file": ("empty.txt", io.BytesIO(file_content), "text/plain")}
    response = client.post("/extract", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "empty.txt"
    # All fields should be None/empty in ShipmentData
    assert data["data"]["reference_id"] is None
    assert data["data"]["shipper"] is None

def test_ask_question_error_handling(client, mock_container):
    # Simulate rag_service.answer_question raising an exception
    mock_container.rag_service.answer_question.side_effect = Exception("RAG error")
    query = {"question": "Trigger error"}
    response = client.post("/ask", json=query)
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error processing question."

def test_extract_data_error_handling(client, mock_container):
    # Simulate extraction_service.extract_data raising an exception
    mock_container.extraction_service.extract_data.side_effect = Exception("Extraction error")
    file_content = b"Some content"
    files = {"file": ("fail.txt", io.BytesIO(file_content), "text/plain")}
    response = client.post("/extract", files=files)
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error during extraction."
