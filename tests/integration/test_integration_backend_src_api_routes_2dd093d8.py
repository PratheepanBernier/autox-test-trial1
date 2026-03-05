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
from backend.src.models.schemas import QAQuery, SourcedAnswer
from backend.src.models.extraction_schema import ExtractionResponse, ShipmentData

from fastapi import FastAPI

app = FastAPI()
app.include_router(router)

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_ingestion_service():
    with patch("backend.src.api.routes.ingestion_service") as mock:
        yield mock

@pytest.fixture
def mock_vector_store_service():
    with patch("backend.src.api.routes.vector_store_service") as mock:
        yield mock

@pytest.fixture
def mock_extraction_service():
    with patch("backend.src.api.routes.extraction_service") as mock:
        yield mock

@pytest.fixture
def mock_rag_service():
    with patch("backend.src.api.routes.rag_service") as mock:
        yield mock

def make_upload_file(filename, content):
    return (
        filename,
        io.BytesIO(content.encode("utf-8")),
        "text/plain"
    )

def test_ping_returns_pong(client):
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert resp.json() == {"status": "pong"}

@pytest.mark.asyncio
def test_upload_document_happy_path(client, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    # Setup mocks
    class Chunk:
        def __init__(self, text):
            self.text = text

    chunks = [Chunk("chunk1"), Chunk("chunk2")]
    mock_ingestion_service.process_file.return_value = chunks

    extraction_result = MagicMock()
    extraction_result.data.reference_id = "ref-123"
    mock_extraction_service.extract_data.return_value = extraction_result

    structured_chunk = Chunk("structured")
    mock_extraction_service.create_structured_chunk.return_value = structured_chunk

    mock_vector_store_service.add_documents.return_value = None

    files = [
        make_upload_file("doc1.txt", "file content 1"),
        make_upload_file("doc2.txt", "file content 2"),
    ]

    resp = client.post(
        "/upload",
        files=[
            ("files", files[0]),
            ("files", files[1]),
        ]
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["message"] == "Successfully processed 2 documents."
    assert data["errors"] == []
    assert len(data["extractions"]) == 2
    for extraction in data["extractions"]:
        assert extraction["structured_data_extracted"] is True
        assert extraction["reference_id"] == "ref-123"
        assert extraction["text_chunks"] == 2

@pytest.mark.asyncio
def test_upload_document_no_text_extracted(client, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    mock_ingestion_service.process_file.return_value = []

    files = [
        make_upload_file("empty.txt", "   "),
    ]

    resp = client.post(
        "/upload",
        files=[("files", files[0])]
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["message"] == "Successfully processed 0 documents."
    assert len(data["errors"]) == 1
    assert "No text extracted from empty.txt" in data["errors"][0]
    assert data["extractions"] == []

@pytest.mark.asyncio
def test_upload_document_extraction_failure(client, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    class Chunk:
        def __init__(self, text):
            self.text = text

    chunks = [Chunk("chunk1")]
    mock_ingestion_service.process_file.return_value = chunks

    mock_vector_store_service.add_documents.return_value = None

    mock_extraction_service.extract_data.side_effect = Exception("Extraction failed")

    files = [
        make_upload_file("fail.txt", "fail content"),
    ]

    resp = client.post(
        "/upload",
        files=[("files", files[0])]
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["message"] == "Successfully processed 1 documents."
    assert data["errors"] == []
    assert len(data["extractions"]) == 1
    extraction = data["extractions"][0]
    assert extraction["filename"] == "fail.txt"
    assert extraction["structured_data_extracted"] is False
    assert "Extraction failed" in extraction["error"]

@pytest.mark.asyncio
def test_upload_document_file_processing_error(client, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    mock_ingestion_service.process_file.side_effect = Exception("File corrupted")

    files = [
        make_upload_file("corrupt.txt", "bad content"),
    ]

    resp = client.post(
        "/upload",
        files=[("files", files[0])]
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["message"] == "Successfully processed 0 documents."
    assert len(data["errors"]) == 1
    assert "Error processing corrupt.txt: File corrupted" in data["errors"][0]
    assert data["extractions"] == []

@pytest.mark.asyncio
def test_ask_question_happy_path(client, mock_rag_service):
    answer = MagicMock(spec=SourcedAnswer)
    answer.answer = "42"
    answer.confidence_score = 0.99
    answer.sources = ["doc1.txt"]
    mock_rag_service.answer_question.return_value = answer

    payload = {
        "question": "What is the answer?",
        "top_k": 3
    }
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "42"
    assert data["confidence_score"] == 0.99
    assert data["sources"] == ["doc1.txt"]

@pytest.mark.asyncio
def test_ask_question_error(client, mock_rag_service):
    mock_rag_service.answer_question.side_effect = Exception("RAG error")
    payload = {
        "question": "What is the answer?",
        "top_k": 3
    }
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 500
    assert resp.json()["detail"] == "Internal server error processing question."

@pytest.mark.asyncio
def test_extract_data_happy_path(client, mock_ingestion_service, mock_extraction_service):
    class Chunk:
        def __init__(self, text):
            self.text = text

    chunks = [Chunk("shipment info")]
    mock_ingestion_service.process_file.return_value = chunks

    extraction_response = ExtractionResponse(
        data=ShipmentData(reference_id="abc123"),
        document_id="shipment.pdf"
    )
    mock_extraction_service.extract_data.return_value = extraction_response

    file = make_upload_file("shipment.pdf", "shipment content")
    resp = client.post(
        "/extract",
        files=[("file", file)]
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["data"]["reference_id"] == "abc123"
    assert data["document_id"] == "shipment.pdf"

@pytest.mark.asyncio
def test_extract_data_no_text(client, mock_ingestion_service, mock_extraction_service):
    mock_ingestion_service.process_file.return_value = []

    file = make_upload_file("empty.pdf", "")
    resp = client.post(
        "/extract",
        files=[("file", file)]
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["data"] == {}
    assert data["document_id"] == "empty.pdf"

@pytest.mark.asyncio
def test_extract_data_error(client, mock_ingestion_service):
    mock_ingestion_service.process_file.side_effect = Exception("Extraction crash")

    file = make_upload_file("crash.pdf", "bad")
    resp = client.post(
        "/extract",
        files=[("file", file)]
    )
    assert resp.status_code == 500
    assert resp.json()["detail"] == "Internal server error during extraction."
