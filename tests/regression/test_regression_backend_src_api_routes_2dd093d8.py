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

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
from backend.src.api.routes import router
from fastapi import FastAPI, UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile
import io

app = FastAPI()
app.include_router(router)

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def fake_upload_file():
    def _make(filename="test.txt", content=b"hello world"):
        file = StarletteUploadFile(filename=filename, file=io.BytesIO(content))
        return file
    return _make

@pytest.mark.asyncio
@patch("backend.src.api.routes.ingestion_service")
@patch("backend.src.api.routes.vector_store_service")
@patch("backend.src.api.routes.extraction_service")
def test_upload_document_happy_path(mock_extraction, mock_vector_store, mock_ingestion, client, tmp_path):
    # Setup mocks
    chunk_mock = MagicMock()
    chunk_mock.text = "chunk text"
    mock_ingestion.process_file.return_value = [chunk_mock]
    extraction_result_mock = MagicMock()
    extraction_result_mock.data.reference_id = "ref-123"
    mock_extraction.extract_data.return_value = extraction_result_mock
    structured_chunk_mock = MagicMock()
    mock_extraction.create_structured_chunk.return_value = structured_chunk_mock

    files = {'files': ('test1.txt', b"filecontent1")}
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["message"].startswith("Successfully processed")
    assert data["errors"] == []
    assert len(data["extractions"]) == 1
    assert data["extractions"][0]["filename"] == "test1.txt"
    assert data["extractions"][0]["structured_data_extracted"] is True
    assert data["extractions"][0]["reference_id"] == "ref-123"
    mock_vector_store.add_documents.assert_any_call([chunk_mock])
    mock_vector_store.add_documents.assert_any_call([structured_chunk_mock])

@pytest.mark.asyncio
@patch("backend.src.api.routes.ingestion_service")
@patch("backend.src.api.routes.vector_store_service")
@patch("backend.src.api.routes.extraction_service")
def test_upload_document_no_text_extracted(mock_extraction, mock_vector_store, mock_ingestion, client):
    mock_ingestion.process_file.return_value = []
    files = {'files': ('empty.txt', b"")}
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["message"].startswith("Successfully processed 0")
    assert len(data["errors"]) == 1
    assert "No text extracted from empty.txt" in data["errors"][0]
    assert data["extractions"] == []

@pytest.mark.asyncio
@patch("backend.src.api.routes.ingestion_service")
@patch("backend.src.api.routes.vector_store_service")
@patch("backend.src.api.routes.extraction_service")
def test_upload_document_extraction_failure(mock_extraction, mock_vector_store, mock_ingestion, client):
    chunk_mock = MagicMock()
    chunk_mock.text = "chunk text"
    mock_ingestion.process_file.return_value = [chunk_mock]
    mock_extraction.extract_data.side_effect = Exception("Extraction failed")
    files = {'files': ('fail.txt', b"failcontent")}
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["message"].startswith("Successfully processed 1")
    assert data["errors"] == []
    assert len(data["extractions"]) == 1
    assert data["extractions"][0]["filename"] == "fail.txt"
    assert data["extractions"][0]["structured_data_extracted"] is False
    assert "Extraction failed" in data["extractions"][0]["error"]

@pytest.mark.asyncio
@patch("backend.src.api.routes.ingestion_service")
@patch("backend.src.api.routes.vector_store_service")
@patch("backend.src.api.routes.extraction_service")
def test_upload_document_file_processing_error(mock_extraction, mock_vector_store, mock_ingestion, client):
    mock_ingestion.process_file.side_effect = Exception("Processing error")
    files = {'files': ('badfile.txt', b"bad")}
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["message"].startswith("Successfully processed 0")
    assert len(data["errors"]) == 1
    assert "Error processing badfile.txt" in data["errors"][0]
    assert data["extractions"] == []

@pytest.mark.asyncio
@patch("backend.src.api.routes.rag_service")
def test_ask_question_happy_path(mock_rag, client):
    answer_mock = MagicMock()
    answer_mock.confidence_score = 0.95
    answer_mock.dict.return_value = {
        "answer": "42",
        "confidence_score": 0.95,
        "sources": []
    }
    mock_rag.answer_question.return_value = answer_mock
    payload = {"question": "What is the answer?", "top_k": 1}
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    # Accept both dict and model serialization
    data = response.json()
    assert "confidence_score" in data
    assert data["confidence_score"] == 0.95

@pytest.mark.asyncio
@patch("backend.src.api.routes.rag_service")
def test_ask_question_error_handling(mock_rag, client):
    mock_rag.answer_question.side_effect = Exception("RAG error")
    payload = {"question": "fail?", "top_k": 1}
    response = client.post("/ask", json=payload)
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error processing question."

@pytest.mark.asyncio
@patch("backend.src.api.routes.ingestion_service")
@patch("backend.src.api.routes.extraction_service")
def test_extract_data_happy_path(mock_extraction, mock_ingestion, client):
    chunk_mock = MagicMock()
    chunk_mock.text = "chunk text"
    mock_ingestion.process_file.return_value = [chunk_mock]
    extraction_response_mock = MagicMock()
    extraction_response_mock.dict.return_value = {
        "data": {"reference_id": "abc"},
        "document_id": "doc1.txt"
    }
    mock_extraction.extract_data.return_value = extraction_response_mock
    files = {'file': ('doc1.txt', b"doccontent")}
    response = client.post("/extract", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert data["document_id"] == "doc1.txt"

@pytest.mark.asyncio
@patch("backend.src.api.routes.ingestion_service")
@patch("backend.src.api.routes.extraction_service")
def test_extract_data_no_text(mock_extraction, mock_ingestion, client):
    mock_ingestion.process_file.return_value = []
    files = {'file': ('empty.txt', b"")}
    response = client.post("/extract", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert data["document_id"] == "empty.txt"

@pytest.mark.asyncio
@patch("backend.src.api.routes.ingestion_service")
@patch("backend.src.api.routes.extraction_service")
def test_extract_data_error_handling(mock_extraction, mock_ingestion, client):
    mock_ingestion.process_file.side_effect = Exception("Extraction error")
    files = {'file': ('fail.txt', b"fail")}
    response = client.post("/extract", files=files)
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error during extraction."

def test_ping_endpoint(client):
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "pong"}

@pytest.mark.asyncio
@patch("backend.src.api.routes.ingestion_service")
@patch("backend.src.api.routes.vector_store_service")
@patch("backend.src.api.routes.extraction_service")
def test_upload_multiple_files_mixed_results(mock_extraction, mock_vector_store, mock_ingestion, client):
    # First file: success
    chunk1 = MagicMock()
    chunk1.text = "chunk1"
    # Second file: no text
    # Third file: extraction error
    chunk3 = MagicMock()
    chunk3.text = "chunk3"
    def process_file_side_effect(content, filename):
        if filename == "file1.txt":
            return [chunk1]
        elif filename == "file2.txt":
            return []
        elif filename == "file3.txt":
            return [chunk3]
    mock_ingestion.process_file.side_effect = process_file_side_effect
    extraction_result_mock = MagicMock()
    extraction_result_mock.data.reference_id = "ref-1"
    mock_extraction.extract_data.side_effect = [extraction_result_mock, Exception("Extraction failed")]
    structured_chunk_mock = MagicMock()
    mock_extraction.create_structured_chunk.return_value = structured_chunk_mock

    files = [
        ('files', ('file1.txt', b"content1")),
        ('files', ('file2.txt', b"content2")),
        ('files', ('file3.txt', b"content3")),
    ]
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["message"].startswith("Successfully processed 2")
    assert len(data["errors"]) == 1
    assert "No text extracted from file2.txt" in data["errors"][0]
    assert len(data["extractions"]) == 2
    filenames = [e["filename"] for e in data["extractions"]]
    assert "file1.txt" in filenames
    assert "file3.txt" in filenames
    for e in data["extractions"]:
        if e["filename"] == "file1.txt":
            assert e["structured_data_extracted"] is True
            assert e["reference_id"] == "ref-1"
        elif e["filename"] == "file3.txt":
            assert e["structured_data_extracted"] is False
            assert "Extraction failed" in e["error"]
