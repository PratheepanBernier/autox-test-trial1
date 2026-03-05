# source_hash: 7f49f55d5565bc86
# import_target: backend.src.api.routes
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

from backend.src.api.routes import router
from models.schemas import SourcedAnswer, QAQuery
from models.extraction_schema import ExtractionResponse, ShipmentData

from fastapi import FastAPI

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
    return (filename, io.BytesIO(content.encode("utf-8")), "text/plain")

def make_chunk(text):
    chunk = MagicMock()
    chunk.text = text
    return chunk

def make_structured_chunk():
    chunk = MagicMock()
    chunk.text = "structured data"
    return chunk

def make_extraction_result(reference_id="ref-123"):
    result = MagicMock()
    result.data.reference_id = reference_id
    return result

def make_extraction_response(document_id="doc-1"):
    return ExtractionResponse(data=ShipmentData(), document_id=document_id)

def make_sourced_answer():
    return SourcedAnswer(answer="42", sources=["doc1"], confidence_score=0.99)

@pytest.mark.asyncio
def test_upload_document_happy_path(client, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    file_content = "This is a test document."
    filename = "test.txt"
    chunks = [make_chunk("chunk1"), make_chunk("chunk2")]
    structured_chunk = make_structured_chunk()
    extraction_result = make_extraction_result("ref-abc")

    mock_ingestion_service.process_file.return_value = chunks
    mock_vector_store_service.add_documents.return_value = None
    mock_extraction_service.extract_data.return_value = extraction_result
    mock_extraction_service.create_structured_chunk.return_value = structured_chunk

    files = [make_upload_file(filename, file_content)]
    response = client.post("/upload", files={"files": files})

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 1 documents."
    assert data["errors"] == []
    assert len(data["extractions"]) == 1
    extraction = data["extractions"][0]
    assert extraction["filename"] == filename
    assert extraction["text_chunks"] == 2
    assert extraction["structured_data_extracted"] is True
    assert extraction["reference_id"] == "ref-abc"

@pytest.mark.asyncio
def test_upload_document_no_text_extracted(client, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    filename = "empty.txt"
    mock_ingestion_service.process_file.return_value = []

    files = [make_upload_file(filename, "")]
    response = client.post("/upload", files={"files": files})

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 0 documents."
    assert len(data["errors"]) == 1
    assert "No text extracted from" in data["errors"][0]
    assert data["extractions"] == []

@pytest.mark.asyncio
def test_upload_document_extraction_failure(client, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    filename = "fail_extract.txt"
    chunks = [make_chunk("chunk1")]
    structured_chunk = make_structured_chunk()

    mock_ingestion_service.process_file.return_value = chunks
    mock_vector_store_service.add_documents.return_value = None
    mock_extraction_service.extract_data.side_effect = Exception("Extraction failed")
    mock_extraction_service.create_structured_chunk.return_value = structured_chunk

    files = [make_upload_file(filename, "content")]
    response = client.post("/upload", files={"files": files})

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 1 documents."
    assert data["errors"] == []
    assert len(data["extractions"]) == 1
    extraction = data["extractions"][0]
    assert extraction["filename"] == filename
    assert extraction["text_chunks"] == 1
    assert extraction["structured_data_extracted"] is False
    assert "error" in extraction
    assert "Extraction failed" in extraction["error"]

@pytest.mark.asyncio
def test_upload_document_processing_failure(client, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    filename = "badfile.txt"
    mock_ingestion_service.process_file.side_effect = Exception("Processing error")

    files = [make_upload_file(filename, "bad content")]
    response = client.post("/upload", files={"files": files})

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 0 documents."
    assert len(data["errors"]) == 1
    assert "Error processing" in data["errors"][0]
    assert "Processing error" in data["errors"][0]
    assert data["extractions"] == []

@pytest.mark.asyncio
def test_upload_document_multiple_files_mixed_results(client, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    files = [
        make_upload_file("good.txt", "good content"),
        make_upload_file("empty.txt", ""),
        make_upload_file("fail.txt", "fail content"),
    ]
    # First file: success
    chunks1 = [make_chunk("chunk1")]
    structured_chunk1 = make_structured_chunk()
    extraction_result1 = make_extraction_result("ref-1")
    # Second file: no text
    # Third file: extraction fails
    chunks3 = [make_chunk("chunk3")]
    structured_chunk3 = make_structured_chunk()

    def process_file_side_effect(content, filename):
        if filename == "good.txt":
            return chunks1
        elif filename == "empty.txt":
            return []
        elif filename == "fail.txt":
            return chunks3

    def extract_data_side_effect(full_text, filename):
        if filename == "good.txt":
            return extraction_result1
        elif filename == "fail.txt":
            raise Exception("Extraction failed")

    mock_ingestion_service.process_file.side_effect = process_file_side_effect
    mock_vector_store_service.add_documents.return_value = None
    mock_extraction_service.extract_data.side_effect = extract_data_side_effect
    mock_extraction_service.create_structured_chunk.side_effect = [structured_chunk1, structured_chunk3]

    response = client.post("/upload", files={"files": files})

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 2 documents."
    assert len(data["errors"]) == 1
    assert "No text extracted from empty.txt" in data["errors"][0]
    assert len(data["extractions"]) == 2
    filenames = {e["filename"] for e in data["extractions"]}
    assert "good.txt" in filenames
    assert "fail.txt" in filenames
    for extraction in data["extractions"]:
        if extraction["filename"] == "good.txt":
            assert extraction["structured_data_extracted"] is True
            assert extraction["reference_id"] == "ref-1"
        elif extraction["filename"] == "fail.txt":
            assert extraction["structured_data_extracted"] is False
            assert "Extraction failed" in extraction["error"]

@pytest.mark.asyncio
def test_ask_question_happy_path(client, mock_rag_service):
    answer = make_sourced_answer()
    mock_rag_service.answer_question.return_value = answer

    query = {"question": "What is the answer?", "top_k": 3}
    response = client.post("/ask", json=query)

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "42"
    assert data["sources"] == ["doc1"]
    assert data["confidence_score"] == 0.99

@pytest.mark.asyncio
def test_ask_question_error_handling(client, mock_rag_service):
    mock_rag_service.answer_question.side_effect = Exception("RAG error")

    query = {"question": "What is the answer?", "top_k": 3}
    response = client.post("/ask", json=query)

    assert response.status_code == 500
    data = response.json()
    assert data["detail"] == "Internal server error processing question."

@pytest.mark.asyncio
def test_extract_data_happy_path(client, mock_ingestion_service, mock_extraction_service):
    filename = "extract.txt"
    chunks = [make_chunk("chunk1"), make_chunk("chunk2")]
    extraction_response = make_extraction_response(document_id=filename)

    mock_ingestion_service.process_file.return_value = chunks
    mock_extraction_service.extract_data.return_value = extraction_response

    files = [make_upload_file(filename, "extract content")]
    response = client.post("/extract", files={"file": files[0]})

    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == filename
    assert "data" in data

@pytest.mark.asyncio
def test_extract_data_no_text_extracted(client, mock_ingestion_service):
    filename = "empty_extract.txt"
    mock_ingestion_service.process_file.return_value = []

    files = [make_upload_file(filename, "")]
    response = client.post("/extract", files={"file": files[0]})

    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == filename
    assert data["data"] == {}

@pytest.mark.asyncio
def test_extract_data_error_handling(client, mock_ingestion_service):
    filename = "fail_extract.txt"
    mock_ingestion_service.process_file.side_effect = Exception("Extraction error")

    files = [make_upload_file(filename, "fail content")]
    response = client.post("/extract", files={"file": files[0]})

    assert response.status_code == 500
    data = response.json()
    assert data["detail"] == "Internal server error during extraction."

def test_ping_endpoint(client):
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "pong"}
