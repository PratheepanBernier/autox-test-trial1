# source_hash: 7f49f55d5565bc86
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
from backend.src.api.routes import router
from fastapi import FastAPI, UploadFile
from models.schemas import SourcedAnswer, QAQuery
from models.extraction_schema import ExtractionResponse, ShipmentData

app = FastAPI()
app.include_router(router)

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_ingestion_service():
    with patch("services.ingestion.ingestion_service") as mock:
        yield mock

@pytest.fixture
def mock_vector_store_service():
    with patch("services.vector_store.vector_store_service") as mock:
        yield mock

@pytest.fixture
def mock_extraction_service():
    with patch("services.extraction.extraction_service") as mock:
        yield mock

@pytest.fixture
def mock_rag_service():
    with patch("services.rag.rag_service") as mock:
        yield mock

def make_upload_file(filename, content):
    file = MagicMock(spec=UploadFile)
    file.filename = filename
    file.read = AsyncMock(return_value=content)
    return file

@pytest.mark.asyncio
async def test_upload_document_happy_path(client, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    # Setup
    file_content = b"test content"
    filename = "test.txt"
    chunk = MagicMock()
    chunk.text = "chunk text"
    mock_ingestion_service.process_file.return_value = [chunk]
    mock_vector_store_service.add_documents.return_value = None
    extraction_result = MagicMock()
    extraction_result.data.reference_id = "ref123"
    mock_extraction_service.extract_data.return_value = extraction_result
    structured_chunk = MagicMock()
    mock_extraction_service.create_structured_chunk.return_value = structured_chunk

    files = [("files", (filename, file_content, "text/plain"))]
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 1 documents."
    assert data["errors"] == []
    assert data["extractions"][0]["filename"] == filename
    assert data["extractions"][0]["text_chunks"] == 1
    assert data["extractions"][0]["structured_data_extracted"] is True
    assert data["extractions"][0]["reference_id"] == "ref123"

@pytest.mark.asyncio
async def test_upload_document_no_text_extracted(client, mock_ingestion_service, mock_vector_store_service):
    file_content = b"empty"
    filename = "empty.txt"
    mock_ingestion_service.process_file.return_value = []
    files = [("files", (filename, file_content, "text/plain"))]
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 0 documents."
    assert len(data["errors"]) == 1
    assert "No text extracted from" in data["errors"][0]
    assert data["extractions"] == []

@pytest.mark.asyncio
async def test_upload_document_extraction_failure(client, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    file_content = b"test"
    filename = "fail_extract.txt"
    chunk = MagicMock()
    chunk.text = "chunk text"
    mock_ingestion_service.process_file.return_value = [chunk]
    mock_vector_store_service.add_documents.return_value = None
    mock_extraction_service.extract_data.side_effect = Exception("Extraction failed")
    files = [("files", (filename, file_content, "text/plain"))]
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 1 documents."
    assert data["errors"] == []
    assert data["extractions"][0]["filename"] == filename
    assert data["extractions"][0]["text_chunks"] == 1
    assert data["extractions"][0]["structured_data_extracted"] is False
    assert "error" in data["extractions"][0]
    assert "Extraction failed" in data["extractions"][0]["error"]

@pytest.mark.asyncio
async def test_upload_document_file_processing_error(client, mock_ingestion_service):
    file_content = b"fail"
    filename = "fail.txt"
    mock_ingestion_service.process_file.side_effect = Exception("Processing error")
    files = [("files", (filename, file_content, "text/plain"))]
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 0 documents."
    assert len(data["errors"]) == 1
    assert "Error processing" in data["errors"][0]
    assert "Processing error" in data["errors"][0]
    assert data["extractions"] == []

@pytest.mark.asyncio
async def test_upload_document_multiple_files_mixed_results(client, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    # First file: success, second file: no text, third file: extraction error
    chunk = MagicMock()
    chunk.text = "chunk text"
    extraction_result = MagicMock()
    extraction_result.data.reference_id = "refA"
    structured_chunk = MagicMock()
    mock_ingestion_service.process_file.side_effect = [
        [chunk], [], [chunk]
    ]
    mock_vector_store_service.add_documents.return_value = None
    mock_extraction_service.extract_data.side_effect = [extraction_result, Exception("Extraction failed")]
    mock_extraction_service.create_structured_chunk.return_value = structured_chunk

    files = [
        ("files", ("file1.txt", b"abc", "text/plain")),
        ("files", ("file2.txt", b"", "text/plain")),
        ("files", ("file3.txt", b"def", "text/plain")),
    ]
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 2 documents."
    assert len(data["errors"]) == 1
    assert "No text extracted from file2.txt" in data["errors"][0]
    assert len(data["extractions"]) == 2
    assert data["extractions"][0]["filename"] == "file1.txt"
    assert data["extractions"][0]["structured_data_extracted"] is True
    assert data["extractions"][1]["filename"] == "file3.txt"
    assert data["extractions"][1]["structured_data_extracted"] is False
    assert "Extraction failed" in data["extractions"][1]["error"]

def test_ask_question_happy_path(client, mock_rag_service):
    answer = SourcedAnswer(answer="42", confidence_score=0.99, sources=["doc1"])
    mock_rag_service.answer_question.return_value = answer
    payload = {"question": "What is the answer?", "top_k": 1}
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "42"
    assert data["confidence_score"] == 0.99
    assert data["sources"] == ["doc1"]

def test_ask_question_error(client, mock_rag_service):
    mock_rag_service.answer_question.side_effect = Exception("RAG error")
    payload = {"question": "fail?", "top_k": 1}
    response = client.post("/ask", json=payload)
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error processing question."

@pytest.mark.asyncio
async def test_extract_data_happy_path(client, mock_ingestion_service, mock_extraction_service):
    file_content = b"shipment"
    filename = "shipment.txt"
    chunk = MagicMock()
    chunk.text = "shipment text"
    mock_ingestion_service.process_file.return_value = [chunk]
    extraction_response = ExtractionResponse(data=ShipmentData(reference_id="refX"), document_id=filename)
    mock_extraction_service.extract_data.return_value = extraction_response
    files = [("file", (filename, file_content, "text/plain"))]
    response = client.post("/extract", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["reference_id"] == "refX"
    assert data["document_id"] == filename

@pytest.mark.asyncio
async def test_extract_data_no_text(client, mock_ingestion_service):
    file_content = b"empty"
    filename = "empty.txt"
    mock_ingestion_service.process_file.return_value = []
    files = [("file", (filename, file_content, "text/plain"))]
    response = client.post("/extract", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["data"] == {}
    assert data["document_id"] == filename

@pytest.mark.asyncio
async def test_extract_data_error(client, mock_ingestion_service):
    file_content = b"fail"
    filename = "fail.txt"
    mock_ingestion_service.process_file.side_effect = Exception("Extraction error")
    files = [("file", (filename, file_content, "text/plain"))]
    response = client.post("/extract", files=files)
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error during extraction."

def test_ping_endpoint(client):
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "pong"}
