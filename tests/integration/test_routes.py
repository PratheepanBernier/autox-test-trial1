# source_hash: 7f49f55d5565bc86
import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
from backend.src.api.routes import router
from fastapi import FastAPI
from models.schemas import SourcedAnswer, QAQuery
from models.extraction_schema import ExtractionResponse, ShipmentData

app = FastAPI()
app.include_router(router)

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def fake_chunks():
    class Chunk:
        def __init__(self, text):
            self.text = text
    return [Chunk("chunk1 text"), Chunk("chunk2 text")]

@pytest.fixture
def fake_structured_chunk():
    class Chunk:
        def __init__(self, text):
            self.text = text
    return Chunk("structured data chunk")

@pytest.fixture
def fake_extraction_result():
    class Data:
        reference_id = "ref-123"
    class ExtractionResult:
        data = Data()
    return ExtractionResult()

@pytest.fixture
def fake_extraction_response():
    return ExtractionResponse(data=ShipmentData(reference_id="ref-123"), document_id="test.pdf")

@pytest.fixture
def fake_sourced_answer():
    return SourcedAnswer(
        answer="42",
        sources=["doc1.pdf"],
        confidence_score=0.99
    )

def make_upload_file(filename, content):
    return ("files", (filename, io.BytesIO(content.encode()), "application/pdf"))

def make_single_upload_file(filename, content):
    return ("file", (filename, io.BytesIO(content.encode()), "application/pdf"))

@patch("services.ingestion.ingestion_service.process_file", new_callable=MagicMock)
@patch("services.vector_store.vector_store_service.add_documents", new_callable=MagicMock)
@patch("services.extraction.extraction_service.extract_data", new_callable=MagicMock)
@patch("services.extraction.extraction_service.create_structured_chunk", new_callable=MagicMock)
def test_upload_document_happy_path(
    mock_create_structured_chunk,
    mock_extract_data,
    mock_add_documents,
    mock_process_file,
    client,
    fake_chunks,
    fake_structured_chunk,
    fake_extraction_result
):
    mock_process_file.return_value = fake_chunks
    mock_add_documents.return_value = None
    mock_extract_data.return_value = fake_extraction_result
    mock_create_structured_chunk.return_value = fake_structured_chunk

    files = [
        make_upload_file("doc1.pdf", "file content 1"),
        make_upload_file("doc2.pdf", "file content 2"),
    ]
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 2 documents."
    assert data["errors"] == []
    assert len(data["extractions"]) == 2
    for extraction in data["extractions"]:
        assert extraction["structured_data_extracted"] is True
        assert extraction["reference_id"] == "ref-123"
        assert extraction["text_chunks"] == 2

@patch("services.ingestion.ingestion_service.process_file", new_callable=MagicMock)
@patch("services.vector_store.vector_store_service.add_documents", new_callable=MagicMock)
@patch("services.extraction.extraction_service.extract_data", new_callable=MagicMock)
@patch("services.extraction.extraction_service.create_structured_chunk", new_callable=MagicMock)
def test_upload_document_no_text_extracted(
    mock_create_structured_chunk,
    mock_extract_data,
    mock_add_documents,
    mock_process_file,
    client
):
    # Simulate process_file returning empty list (no text extracted)
    mock_process_file.return_value = []
    files = [make_upload_file("empty.pdf", "")]
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 0 documents."
    assert len(data["errors"]) == 1
    assert "No text extracted from empty.pdf" in data["errors"][0]
    assert data["extractions"] == []

@patch("services.ingestion.ingestion_service.process_file", new_callable=MagicMock)
@patch("services.vector_store.vector_store_service.add_documents", new_callable=MagicMock)
@patch("services.extraction.extraction_service.extract_data", new_callable=MagicMock)
@patch("services.extraction.extraction_service.create_structured_chunk", new_callable=MagicMock)
def test_upload_document_extraction_failure(
    mock_create_structured_chunk,
    mock_extract_data,
    mock_add_documents,
    mock_process_file,
    client,
    fake_chunks
):
    mock_process_file.return_value = fake_chunks
    mock_add_documents.return_value = None
    mock_extract_data.side_effect = Exception("Extraction failed!")
    files = [make_upload_file("fail.pdf", "some content")]
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 1 documents."
    assert data["errors"] == []
    assert len(data["extractions"]) == 1
    extraction = data["extractions"][0]
    assert extraction["filename"] == "fail.pdf"
    assert extraction["structured_data_extracted"] is False
    assert "Extraction failed!" in extraction["error"]

@patch("services.ingestion.ingestion_service.process_file", new_callable=MagicMock)
@patch("services.vector_store.vector_store_service.add_documents", new_callable=MagicMock)
@patch("services.extraction.extraction_service.extract_data", new_callable=MagicMock)
@patch("services.extraction.extraction_service.create_structured_chunk", new_callable=MagicMock)
def test_upload_document_file_processing_error(
    mock_create_structured_chunk,
    mock_extract_data,
    mock_add_documents,
    mock_process_file,
    client
):
    mock_process_file.side_effect = Exception("File corrupted")
    files = [make_upload_file("corrupt.pdf", "bad content")]
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 0 documents."
    assert len(data["errors"]) == 1
    assert "Error processing corrupt.pdf: File corrupted" in data["errors"][0]
    assert data["extractions"] == []

@patch("services.rag.rag_service.answer_question", new_callable=MagicMock)
def test_ask_question_happy_path(mock_answer_question, client, fake_sourced_answer):
    mock_answer_question.return_value = fake_sourced_answer
    payload = {"question": "What is the answer?", "top_k": 3}
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "42"
    assert data["sources"] == ["doc1.pdf"]
    assert data["confidence_score"] == 0.99

@patch("services.rag.rag_service.answer_question", new_callable=MagicMock)
def test_ask_question_error_handling(mock_answer_question, client):
    mock_answer_question.side_effect = Exception("RAG failed")
    payload = {"question": "What is the answer?", "top_k": 3}
    response = client.post("/ask", json=payload)
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error processing question."

@patch("services.ingestion.ingestion_service.process_file", new_callable=MagicMock)
@patch("services.extraction.extraction_service.extract_data", new_callable=MagicMock)
def test_extract_data_happy_path(
    mock_extract_data,
    mock_process_file,
    client,
    fake_chunks,
    fake_extraction_response
):
    mock_process_file.return_value = fake_chunks
    mock_extract_data.return_value = fake_extraction_response
    files = [make_single_upload_file("extract.pdf", "extract content")]
    response = client.post("/extract", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["reference_id"] == "ref-123"
    assert data["document_id"] == "extract.pdf"

@patch("services.ingestion.ingestion_service.process_file", new_callable=MagicMock)
def test_extract_data_no_text_extracted(
    mock_process_file,
    client
):
    mock_process_file.return_value = []
    files = [make_single_upload_file("empty.pdf", "")]
    response = client.post("/extract", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["reference_id"] is None
    assert data["document_id"] == "empty.pdf"

@patch("services.ingestion.ingestion_service.process_file", new_callable=MagicMock)
@patch("services.extraction.extraction_service.extract_data", new_callable=MagicMock)
def test_extract_data_extraction_error(
    mock_extract_data,
    mock_process_file,
    client,
    fake_chunks
):
    mock_process_file.return_value = fake_chunks
    mock_extract_data.side_effect = Exception("Extraction error")
    files = [make_single_upload_file("fail.pdf", "fail content")]
    response = client.post("/extract", files=files)
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error during extraction."

def test_ping_endpoint(client):
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "pong"}

# Edge/boundary: upload with zero files
def test_upload_document_no_files(client):
    response = client.post("/upload", files=[])
    assert response.status_code == 422  # FastAPI validation error

# Edge: ask with missing question field
def test_ask_question_missing_field(client):
    payload = {"top_k": 3}
    response = client.post("/ask", json=payload)
    assert response.status_code == 422

# Edge: extract with no file
def test_extract_data_no_file(client):
    response = client.post("/extract", files=[])
    assert response.status_code == 422

# Reconciliation: /upload and /extract both process file, compare empty extraction
@patch("services.ingestion.ingestion_service.process_file", new_callable=MagicMock)
def test_upload_and_extract_empty_file_consistency(mock_process_file, client):
    mock_process_file.return_value = []
    files = [make_upload_file("empty.pdf", "")]
    upload_resp = client.post("/upload", files=files)
    extract_resp = client.post("/extract", files=[make_single_upload_file("empty.pdf", "")])
    assert upload_resp.status_code == 200
    assert extract_resp.status_code == 200
    upload_data = upload_resp.json()
    extract_data = extract_resp.json()
    # Both should indicate no text extracted and empty data
    assert upload_data["extractions"] == []
    assert extract_data["data"]["reference_id"] is None
    assert extract_data["document_id"] == "empty.pdf"
