import io
import pytest
from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

from backend.src.api.routes import router
from models.schemas import QAQuery, SourcedAnswer
from models.extraction_schema import ExtractionResponse, ShipmentData

from fastapi import FastAPI, UploadFile

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
        make_upload_file("doc1.pdf", "filecontent1"),
        make_upload_file("doc2.pdf", "filecontent2"),
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
    mock_extract_data.side_effect = Exception("Extraction failed")
    files = [make_upload_file("fail.pdf", "failcontent")]
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 1 documents."
    assert data["errors"] == []
    assert len(data["extractions"]) == 1
    extraction = data["extractions"][0]
    assert extraction["filename"] == "fail.pdf"
    assert extraction["structured_data_extracted"] is False
    assert "Extraction failed" in extraction["error"]

@patch("services.ingestion.ingestion_service.process_file", new_callable=MagicMock)
def test_upload_document_processing_exception(
    mock_process_file,
    client
):
    mock_process_file.side_effect = Exception("Processing error")
    files = [make_upload_file("bad.pdf", "badcontent")]
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 0 documents."
    assert len(data["errors"]) == 1
    assert "Error processing bad.pdf: Processing error" in data["errors"][0]
    assert data["extractions"] == []

@patch("services.rag.rag_service.answer_question", new_callable=MagicMock)
def test_ask_question_happy_path(mock_answer_question, client, fake_sourced_answer):
    mock_answer_question.return_value = fake_sourced_answer
    payload = {"question": "What is the answer?", "top_k": 1}
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "42"
    assert data["sources"] == ["doc1.pdf"]
    assert data["confidence_score"] == 0.99

@patch("services.rag.rag_service.answer_question", new_callable=MagicMock)
def test_ask_question_error_handling(mock_answer_question, client):
    mock_answer_question.side_effect = Exception("RAG error")
    payload = {"question": "What is the answer?", "top_k": 1}
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
    files = [make_single_upload_file("extract.pdf", "extractcontent")]
    response = client.post("/extract", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["reference_id"] == "ref-123"
    assert data["document_id"] == "test.pdf"

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
def test_extract_data_processing_exception(
    mock_process_file,
    client
):
    mock_process_file.side_effect = Exception("Extraction error")
    files = [make_single_upload_file("fail.pdf", "failcontent")]
    response = client.post("/extract", files=files)
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error during extraction."

def test_ping_endpoint(client):
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "pong"}
