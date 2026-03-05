import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from io import BytesIO

import sys
import types

# Patch FastAPI app and router for test import
from fastapi import FastAPI
from backend.src.api import routes

app = FastAPI()
app.include_router(routes.router)

client = TestClient(app)

@pytest.fixture(autouse=True)
def patch_services(monkeypatch):
    # Patch ingestion_service
    ingestion_service_mock = MagicMock()
    ingestion_service_mock.process_file = MagicMock()
    monkeypatch.setattr("services.ingestion.ingestion_service", ingestion_service_mock)
    # Patch vector_store_service
    vector_store_service_mock = MagicMock()
    vector_store_service_mock.add_documents = MagicMock()
    monkeypatch.setattr("services.vector_store.vector_store_service", vector_store_service_mock)
    # Patch extraction_service
    extraction_service_mock = MagicMock()
    extraction_service_mock.extract_data = MagicMock()
    extraction_service_mock.create_structured_chunk = MagicMock()
    monkeypatch.setattr("services.extraction.extraction_service", extraction_service_mock)
    # Patch rag_service
    rag_service_mock = MagicMock()
    rag_service_mock.answer_question = MagicMock()
    monkeypatch.setattr("services.rag.rag_service", rag_service_mock)
    return {
        "ingestion": ingestion_service_mock,
        "vector_store": vector_store_service_mock,
        "extraction": extraction_service_mock,
        "rag": rag_service_mock,
    }

def make_upload_file(filename, content):
    return (filename, BytesIO(content.encode("utf-8")), "text/plain")

def make_chunk(text):
    # Simulate a chunk object with .text attribute
    chunk = MagicMock()
    chunk.text = text
    return chunk

def make_extraction_result(reference_id="ref-123"):
    # Simulate extraction result with .data.reference_id
    data = MagicMock()
    data.reference_id = reference_id
    result = MagicMock()
    result.data = data
    return result

def make_structured_chunk():
    chunk = MagicMock()
    chunk.text = "structured data"
    return chunk

def make_sourced_answer():
    answer = MagicMock()
    answer.answer = "42"
    answer.confidence_score = 0.99
    answer.sources = ["doc1.pdf"]
    return answer

def make_extraction_response(document_id="doc1.pdf"):
    resp = MagicMock()
    resp.data = MagicMock()
    resp.document_id = document_id
    return resp

@pytest.mark.asyncio
@patch("services.ingestion.ingestion_service.process_file", new_callable=AsyncMock)
@patch("services.vector_store.vector_store_service.add_documents", new_callable=MagicMock)
@patch("services.extraction.extraction_service.extract_data", new_callable=MagicMock)
@patch("services.extraction.extraction_service.create_structured_chunk", new_callable=MagicMock)
def test_upload_document_happy_path(
    create_structured_chunk,
    extract_data,
    add_documents,
    process_file,
):
    # Setup
    process_file.side_effect = lambda content, filename: [make_chunk("chunk1"), make_chunk("chunk2")]
    extract_data.side_effect = lambda text, filename: make_extraction_result("ref-abc")
    create_structured_chunk.side_effect = lambda extraction_result, filename: make_structured_chunk()
    files = [
        make_upload_file("doc1.txt", "hello world"),
        make_upload_file("doc2.txt", "foo bar"),
    ]
    response = client.post("/upload", files=[
        ("files", files[0]),
        ("files", files[1]),
    ])
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 2 documents."
    assert data["errors"] == []
    assert len(data["extractions"]) == 2
    for extraction in data["extractions"]:
        assert extraction["structured_data_extracted"] is True
        assert extraction["text_chunks"] == 2
        assert extraction["reference_id"] == "ref-abc"

@pytest.mark.asyncio
@patch("services.ingestion.ingestion_service.process_file", new_callable=AsyncMock)
@patch("services.vector_store.vector_store_service.add_documents", new_callable=MagicMock)
@patch("services.extraction.extraction_service.extract_data", new_callable=MagicMock)
@patch("services.extraction.extraction_service.create_structured_chunk", new_callable=MagicMock)
def test_upload_document_no_text_extracted(
    create_structured_chunk,
    extract_data,
    add_documents,
    process_file,
):
    # Simulate process_file returns empty list (no text extracted)
    process_file.side_effect = lambda content, filename: []
    files = [make_upload_file("empty.txt", "")]
    response = client.post("/upload", files=[("files", files[0])])
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 0 documents."
    assert len(data["errors"]) == 1
    assert "No text extracted from empty.txt" in data["errors"][0]
    assert data["extractions"] == []

@pytest.mark.asyncio
@patch("services.ingestion.ingestion_service.process_file", new_callable=AsyncMock)
@patch("services.vector_store.vector_store_service.add_documents", new_callable=MagicMock)
@patch("services.extraction.extraction_service.extract_data", new_callable=MagicMock)
@patch("services.extraction.extraction_service.create_structured_chunk", new_callable=MagicMock)
def test_upload_document_extraction_error(
    create_structured_chunk,
    extract_data,
    add_documents,
    process_file,
):
    # Extraction fails for one file
    process_file.side_effect = lambda content, filename: [make_chunk("chunk1")]
    def extraction_side_effect(text, filename):
        if filename == "fail.txt":
            raise Exception("Extraction failed!")
        return make_extraction_result("ref-ok")
    extract_data.side_effect = extraction_side_effect
    create_structured_chunk.side_effect = lambda extraction_result, filename: make_structured_chunk()
    files = [
        make_upload_file("ok.txt", "good"),
        make_upload_file("fail.txt", "bad"),
    ]
    response = client.post("/upload", files=[
        ("files", files[0]),
        ("files", files[1]),
    ])
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 2 documents."
    assert data["errors"] == []
    assert len(data["extractions"]) == 2
    ok_extraction = next(e for e in data["extractions"] if e["filename"] == "ok.txt")
    fail_extraction = next(e for e in data["extractions"] if e["filename"] == "fail.txt")
    assert ok_extraction["structured_data_extracted"] is True
    assert fail_extraction["structured_data_extracted"] is False
    assert "Extraction failed!" in fail_extraction["error"]

@pytest.mark.asyncio
@patch("services.ingestion.ingestion_service.process_file", new_callable=AsyncMock)
@patch("services.vector_store.vector_store_service.add_documents", new_callable=MagicMock)
@patch("services.extraction.extraction_service.extract_data", new_callable=MagicMock)
@patch("services.extraction.extraction_service.create_structured_chunk", new_callable=MagicMock)
def test_upload_document_file_processing_error(
    create_structured_chunk,
    extract_data,
    add_documents,
    process_file,
):
    # Simulate process_file raises exception
    def process_file_side_effect(content, filename):
        if filename == "bad.txt":
            raise Exception("File corrupted")
        return [make_chunk("chunk1")]
    process_file.side_effect = process_file_side_effect
    extract_data.side_effect = lambda text, filename: make_extraction_result("ref-xyz")
    create_structured_chunk.side_effect = lambda extraction_result, filename: make_structured_chunk()
    files = [
        make_upload_file("good.txt", "ok"),
        make_upload_file("bad.txt", "fail"),
    ]
    response = client.post("/upload", files=[
        ("files", files[0]),
        ("files", files[1]),
    ])
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 1 documents."
    assert len(data["errors"]) == 1
    assert "Error processing bad.txt: File corrupted" in data["errors"][0]
    assert len(data["extractions"]) == 1
    assert data["extractions"][0]["filename"] == "good.txt"

@pytest.mark.asyncio
@patch("services.rag.rag_service.answer_question", new_callable=MagicMock)
def test_ask_question_happy_path(answer_question):
    answer = make_sourced_answer()
    answer_question.return_value = answer
    payload = {
        "question": "What is the answer?",
        "top_k": 3,
        "filters": {},
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "42"
    assert data["confidence_score"] == 0.99
    assert data["sources"] == ["doc1.pdf"]

@pytest.mark.asyncio
@patch("services.rag.rag_service.answer_question", new_callable=MagicMock)
def test_ask_question_error_handling(answer_question):
    answer_question.side_effect = Exception("RAG failed")
    payload = {
        "question": "What is the answer?",
        "top_k": 3,
        "filters": {},
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error processing question."

@pytest.mark.asyncio
@patch("services.ingestion.ingestion_service.process_file", new_callable=AsyncMock)
@patch("services.extraction.extraction_service.extract_data", new_callable=MagicMock)
def test_extract_data_happy_path(extract_data, process_file):
    process_file.side_effect = lambda content, filename: [make_chunk("chunk1"), make_chunk("chunk2")]
    extract_data.side_effect = lambda text, filename: make_extraction_response("doc1.pdf")
    file = make_upload_file("doc1.pdf", "shipment data")
    response = client.post("/extract", files=[("file", file)])
    assert response.status_code == 200
    # ExtractionResponse is a pydantic model, so response is dict-like
    data = response.json()
    assert data["document_id"] == "doc1.pdf"

@pytest.mark.asyncio
@patch("services.ingestion.ingestion_service.process_file", new_callable=AsyncMock)
@patch("services.extraction.extraction_service.extract_data", new_callable=MagicMock)
def test_extract_data_no_text(process_file, extract_data):
    process_file.side_effect = lambda content, filename: []
    file = make_upload_file("empty.pdf", "")
    response = client.post("/extract", files=[("file", file)])
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "empty.pdf"
    # Should return empty data
    assert isinstance(data["data"], dict)

@pytest.mark.asyncio
@patch("services.ingestion.ingestion_service.process_file", new_callable=AsyncMock)
@patch("services.extraction.extraction_service.extract_data", new_callable=MagicMock)
def test_extract_data_error_handling(extract_data, process_file):
    process_file.side_effect = lambda content, filename: [make_chunk("chunk1")]
    extract_data.side_effect = Exception("Extraction failed")
    file = make_upload_file("fail.pdf", "bad data")
    response = client.post("/extract", files=[("file", file)])
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error during extraction."

def test_ping_endpoint():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "pong"}

# Reconciliation tests: compare outputs across equivalent paths

@pytest.mark.asyncio
@patch("services.ingestion.ingestion_service.process_file", new_callable=AsyncMock)
@patch("services.vector_store.vector_store_service.add_documents", new_callable=MagicMock)
@patch("services.extraction.extraction_service.extract_data", new_callable=MagicMock)
@patch("services.extraction.extraction_service.create_structured_chunk", new_callable=MagicMock)
def test_reconciliation_upload_vs_extract(
    create_structured_chunk,
    extract_data,
    add_documents,
    process_file,
):
    # Setup: both endpoints should extract the same reference_id for the same file
    process_file.side_effect = lambda content, filename: [make_chunk("chunk1")]
    extraction_result = make_extraction_result("ref-xyz")
    extract_data.side_effect = lambda text, filename: extraction_result
    create_structured_chunk.side_effect = lambda extraction_result, filename: make_structured_chunk()
    file = make_upload_file("doc1.pdf", "shipment data")
    # /upload
    upload_resp = client.post("/upload", files=[("files", file)])
    assert upload_resp.status_code == 200
    upload_data = upload_resp.json()
    assert len(upload_data["extractions"]) == 1
    upload_ref = upload_data["extractions"][0]["reference_id"]
    # /extract
    # Patch extract_data to return ExtractionResponse with same reference_id
    from models.extraction_schema import ExtractionResponse, ShipmentData
    extraction_response = ExtractionResponse(data=ShipmentData(reference_id="ref-xyz"), document_id="doc1.pdf")
    extract_data.side_effect = lambda text, filename: extraction_response
    extract_resp = client.post("/extract", files=[("file", file)])
    assert extract_resp.status_code == 200
    extract_data_json = extract_resp.json()
    # Reconcile: reference_id from /upload and /extract should match
    assert upload_ref == extract_data_json["data"]["reference_id"]

@pytest.mark.asyncio
@patch("services.rag.rag_service.answer_question", new_callable=MagicMock)
def test_reconciliation_ask_consistency(answer_question):
    # For the same question, answer should be consistent
    answer = make_sourced_answer()
    answer_question.return_value = answer
    payload = {
        "question": "What is the answer?",
        "top_k": 1,
        "filters": {},
    }
    resp1 = client.post("/ask", json=payload)
    resp2 = client.post("/ask", json=payload)
    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp1.json() == resp2.json()
