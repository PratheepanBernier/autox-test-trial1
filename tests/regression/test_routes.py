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
def fake_upload_file(tmp_path):
    class DummyUploadFile:
        def __init__(self, filename, content):
            self.filename = filename
            self._content = content
            self.file = MagicMock()
        async def read(self):
            return self._content
    return DummyUploadFile

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

def make_chunk(text):
    chunk = MagicMock()
    chunk.text = text
    return chunk

# ---------------------- /upload ----------------------

@pytest.mark.asyncio
async def test_upload_document_happy_path(client, fake_upload_file, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    # Arrange
    file_content = b"test content"
    filename = "test.txt"
    chunks = [make_chunk("chunk1"), make_chunk("chunk2")]
    extraction_result = MagicMock()
    extraction_result.data.reference_id = "ref-123"
    structured_chunk = make_chunk("structured data")

    mock_ingestion_service.process_file.return_value = chunks
    mock_vector_store_service.add_documents.return_value = None
    mock_extraction_service.extract_data.return_value = extraction_result
    mock_extraction_service.create_structured_chunk.return_value = structured_chunk

    files = [("files", (filename, file_content, "text/plain"))]

    # Act
    response = client.post("/upload", files=files)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 1 documents."
    assert data["errors"] == []
    assert data["extractions"][0]["filename"] == filename
    assert data["extractions"][0]["text_chunks"] == 2
    assert data["extractions"][0]["structured_data_extracted"] is True
    assert data["extractions"][0]["reference_id"] == "ref-123"

@pytest.mark.asyncio
async def test_upload_document_multiple_files(client, fake_upload_file, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    # Arrange
    file1 = ("files", ("a.txt", b"abc", "text/plain"))
    file2 = ("files", ("b.txt", b"def", "text/plain"))
    chunks1 = [make_chunk("c1")]
    chunks2 = [make_chunk("c2"), make_chunk("c3")]
    extraction_result1 = MagicMock()
    extraction_result1.data.reference_id = "r1"
    extraction_result2 = MagicMock()
    extraction_result2.data.reference_id = "r2"
    structured_chunk1 = make_chunk("s1")
    structured_chunk2 = make_chunk("s2")

    def process_file_side_effect(content, filename):
        if filename == "a.txt":
            return chunks1
        else:
            return chunks2

    def extract_data_side_effect(full_text, filename):
        if filename == "a.txt":
            return extraction_result1
        else:
            return extraction_result2

    def create_structured_chunk_side_effect(result, filename):
        if filename == "a.txt":
            return structured_chunk1
        else:
            return structured_chunk2

    mock_ingestion_service.process_file.side_effect = process_file_side_effect
    mock_vector_store_service.add_documents.return_value = None
    mock_extraction_service.extract_data.side_effect = extract_data_side_effect
    mock_extraction_service.create_structured_chunk.side_effect = create_structured_chunk_side_effect

    files = [file1, file2]

    # Act
    response = client.post("/upload", files=files)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 2 documents."
    assert data["errors"] == []
    assert len(data["extractions"]) == 2
    assert {e["filename"] for e in data["extractions"]} == {"a.txt", "b.txt"}

@pytest.mark.asyncio
async def test_upload_document_no_text_extracted(client, fake_upload_file, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    # Arrange
    mock_ingestion_service.process_file.return_value = []
    files = [("files", ("empty.txt", b"", "text/plain"))]

    # Act
    response = client.post("/upload", files=files)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 0 documents."
    assert "No text extracted from empty.txt" in data["errors"][0]
    assert data["extractions"] == []

@pytest.mark.asyncio
async def test_upload_document_extraction_failure(client, fake_upload_file, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    # Arrange
    chunks = [make_chunk("chunk1")]
    mock_ingestion_service.process_file.return_value = chunks
    mock_vector_store_service.add_documents.return_value = None
    mock_extraction_service.extract_data.side_effect = Exception("Extraction failed")
    files = [("files", ("fail.txt", b"fail", "text/plain"))]

    # Act
    response = client.post("/upload", files=files)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 1 documents."
    assert data["errors"] == []
    assert data["extractions"][0]["filename"] == "fail.txt"
    assert data["extractions"][0]["structured_data_extracted"] is False
    assert "Extraction failed" in data["extractions"][0]["error"]

@pytest.mark.asyncio
async def test_upload_document_file_processing_error(client, fake_upload_file, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    # Arrange
    mock_ingestion_service.process_file.side_effect = Exception("File corrupt")
    files = [("files", ("bad.txt", b"bad", "text/plain"))]

    # Act
    response = client.post("/upload", files=files)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 0 documents."
    assert "Error processing bad.txt: File corrupt" in data["errors"][0]
    assert data["extractions"] == []

# ---------------------- /ask ----------------------

def test_ask_question_happy_path(client, mock_rag_service):
    # Arrange
    answer = SourcedAnswer(answer="42", confidence_score=0.99, sources=["doc1"])
    mock_rag_service.answer_question.return_value = answer
    payload = {"question": "What is the answer?", "top_k": 1}

    # Act
    response = client.post("/ask", json=payload)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "42"
    assert data["confidence_score"] == 0.99
    assert data["sources"] == ["doc1"]

def test_ask_question_error_handling(client, mock_rag_service):
    # Arrange
    mock_rag_service.answer_question.side_effect = Exception("RAG failed")
    payload = {"question": "fail?", "top_k": 1}

    # Act
    response = client.post("/ask", json=payload)

    # Assert
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error processing question."

# ---------------------- /extract ----------------------

@pytest.mark.asyncio
async def test_extract_data_happy_path(client, fake_upload_file, mock_ingestion_service, mock_extraction_service):
    # Arrange
    chunks = [make_chunk("shipment info")]
    mock_ingestion_service.process_file.return_value = chunks
    extraction_response = ExtractionResponse(data=ShipmentData(reference_id="abc"), document_id="doc.txt")
    mock_extraction_service.extract_data.return_value = extraction_response

    files = [("file", ("doc.txt", b"shipment", "text/plain"))]

    # Act
    response = client.post("/extract", files=files)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["reference_id"] == "abc"
    assert data["document_id"] == "doc.txt"

@pytest.mark.asyncio
async def test_extract_data_no_text(client, fake_upload_file, mock_ingestion_service, mock_extraction_service):
    # Arrange
    mock_ingestion_service.process_file.return_value = []
    files = [("file", ("empty.txt", b"", "text/plain"))]

    # Act
    response = client.post("/extract", files=files)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["data"] == {}
    assert data["document_id"] == "empty.txt"

@pytest.mark.asyncio
async def test_extract_data_extraction_error(client, fake_upload_file, mock_ingestion_service, mock_extraction_service):
    # Arrange
    chunks = [make_chunk("shipment info")]
    mock_ingestion_service.process_file.return_value = chunks
    mock_extraction_service.extract_data.side_effect = Exception("Extraction error")
    files = [("file", ("fail.txt", b"fail", "text/plain"))]

    # Act
    response = client.post("/extract", files=files)

    # Assert
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error during extraction."

# ---------------------- /ping ----------------------

def test_ping_endpoint(client):
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "pong"}
