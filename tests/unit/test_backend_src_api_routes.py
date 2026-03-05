import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from io import BytesIO

import sys
import types

# Patch sys.modules for services and models to allow import of router
mock_ingestion_service = MagicMock()
mock_vector_store_service = MagicMock()
mock_rag_service = MagicMock()
mock_extraction_service = MagicMock()

mock_QAQuery = MagicMock()
mock_SourcedAnswer = MagicMock()
mock_ExtractionResponse = MagicMock()
mock_ShipmentData = MagicMock()

sys.modules["services.ingestion"] = types.SimpleNamespace(ingestion_service=mock_ingestion_service)
sys.modules["services.vector_store"] = types.SimpleNamespace(vector_store_service=mock_vector_store_service)
sys.modules["services.rag"] = types.SimpleNamespace(rag_service=mock_rag_service)
sys.modules["services.extraction"] = types.SimpleNamespace(extraction_service=mock_extraction_service)
sys.modules["models.schemas"] = types.SimpleNamespace(QAQuery=mock_QAQuery, SourcedAnswer=mock_SourcedAnswer)
sys.modules["models.extraction_schema"] = types.SimpleNamespace(
    ExtractionResponse=mock_ExtractionResponse, ShipmentData=mock_ShipmentData
)

from backend.src.api.routes import router

from fastapi import FastAPI

app = FastAPI()
app.include_router(router)

client = TestClient(app)

class DummyChunk:
    def __init__(self, text):
        self.text = text

class DummyExtractionResult:
    def __init__(self, reference_id="ref-123"):
        self.data = MagicMock()
        self.data.reference_id = reference_id

class DummyStructuredChunk:
    def __init__(self, text):
        self.text = text

class DummySourcedAnswer:
    def __init__(self, answer="42", confidence_score=0.99):
        self.answer = answer
        self.confidence_score = confidence_score

class DummyExtractionResponse:
    def __init__(self, data=None, document_id=None):
        self.data = data
        self.document_id = document_id

@pytest.fixture(autouse=True)
def reset_mocks():
    mock_ingestion_service.reset_mock()
    mock_vector_store_service.reset_mock()
    mock_rag_service.reset_mock()
    mock_extraction_service.reset_mock()
    yield

def make_upload_file(filename, content):
    file = MagicMock(spec=UploadFile)
    file.filename = filename
    file.read = AsyncMock(return_value=content)
    return file

@pytest.mark.asyncio
async def test_upload_document_happy_path(monkeypatch):
    # Arrange
    file1 = make_upload_file("doc1.txt", b"file1 content")
    file2 = make_upload_file("doc2.txt", b"file2 content")
    chunks1 = [DummyChunk("chunk1"), DummyChunk("chunk2")]
    chunks2 = [DummyChunk("chunkA")]
    extraction_result1 = DummyExtractionResult("ref-abc")
    extraction_result2 = DummyExtractionResult("ref-def")
    structured_chunk1 = DummyStructuredChunk("structured1")
    structured_chunk2 = DummyStructuredChunk("structured2")

    mock_ingestion_service.process_file.side_effect = [chunks1, chunks2]
    mock_vector_store_service.add_documents.return_value = None
    mock_extraction_service.extract_data.side_effect = [extraction_result1, extraction_result2]
    mock_extraction_service.create_structured_chunk.side_effect = [structured_chunk1, structured_chunk2]

    # Patch logger to avoid noisy output
    with patch("backend.src.api.routes.logger"):
        from backend.src.api.routes import upload_document
        resp = await upload_document([file1, file2])

    # Assert
    assert resp["message"] == "Successfully processed 2 documents."
    assert resp["errors"] == []
    assert len(resp["extractions"]) == 2
    assert resp["extractions"][0]["filename"] == "doc1.txt"
    assert resp["extractions"][0]["text_chunks"] == 2
    assert resp["extractions"][0]["structured_data_extracted"] is True
    assert resp["extractions"][0]["reference_id"] == "ref-abc"
    assert resp["extractions"][1]["filename"] == "doc2.txt"
    assert resp["extractions"][1]["text_chunks"] == 1
    assert resp["extractions"][1]["structured_data_extracted"] is True
    assert resp["extractions"][1]["reference_id"] == "ref-def"

@pytest.mark.asyncio
async def test_upload_document_no_text_extracted(monkeypatch):
    file1 = make_upload_file("empty.txt", b"empty")
    mock_ingestion_service.process_file.return_value = []
    with patch("backend.src.api.routes.logger"):
        from backend.src.api.routes import upload_document
        resp = await upload_document([file1])
    assert resp["message"] == "Successfully processed 0 documents."
    assert "No text extracted from empty.txt" in resp["errors"]
    assert resp["extractions"] == []

@pytest.mark.asyncio
async def test_upload_document_extraction_error(monkeypatch):
    file1 = make_upload_file("doc.txt", b"content")
    chunks = [DummyChunk("chunk1")]
    mock_ingestion_service.process_file.return_value = chunks
    mock_vector_store_service.add_documents.return_value = None
    mock_extraction_service.extract_data.side_effect = Exception("extract fail")
    with patch("backend.src.api.routes.logger"):
        from backend.src.api.routes import upload_document
        resp = await upload_document([file1])
    assert resp["message"] == "Successfully processed 1 documents."
    assert resp["errors"] == []
    assert resp["extractions"][0]["filename"] == "doc.txt"
    assert resp["extractions"][0]["text_chunks"] == 1
    assert resp["extractions"][0]["structured_data_extracted"] is False
    assert "extract fail" in resp["extractions"][0]["error"]

@pytest.mark.asyncio
async def test_upload_document_outer_exception(monkeypatch):
    file1 = make_upload_file("fail.txt", b"fail")
    mock_ingestion_service.process_file.side_effect = Exception("outer fail")
    with patch("backend.src.api.routes.logger"):
        from backend.src.api.routes import upload_document
        resp = await upload_document([file1])
    assert resp["message"] == "Successfully processed 0 documents."
    assert "Error processing fail.txt: outer fail" in resp["errors"]
    assert resp["extractions"] == []

@pytest.mark.asyncio
async def test_upload_document_empty_file_list(monkeypatch):
    with patch("backend.src.api.routes.logger"):
        from backend.src.api.routes import upload_document
        resp = await upload_document([])
    assert resp["message"] == "Successfully processed 0 documents."
    assert resp["errors"] == []
    assert resp["extractions"] == []

def test_ask_question_happy_path(monkeypatch):
    dummy_query = MagicMock()
    dummy_answer = DummySourcedAnswer("The answer", 0.88)
    mock_rag_service.answer_question.return_value = dummy_answer
    with patch("backend.src.api.routes.logger"):
        from backend.src.api.routes import ask_question
        resp = ask_question(dummy_query)
        # ask_question is async, but not awaited in FastAPI, so we call directly
        # But in FastAPI, it is awaited, so we need to run it as async
        import asyncio
        result = asyncio.run(resp)
    assert result.answer == "The answer"
    assert result.confidence_score == 0.88

def test_ask_question_error(monkeypatch):
    dummy_query = MagicMock()
    mock_rag_service.answer_question.side_effect = Exception("fail")
    with patch("backend.src.api.routes.logger"):
        from backend.src.api.routes import ask_question
        import asyncio
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(ask_question(dummy_query))
    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Internal server error processing question."

@pytest.mark.asyncio
async def test_extract_data_happy_path(monkeypatch):
    file = make_upload_file("ship.pdf", b"shipdata")
    chunks = [DummyChunk("shipment text")]
    extraction_result = DummyExtractionResponse(data="shipment", document_id="ship.pdf")
    mock_ingestion_service.process_file.return_value = chunks
    mock_extraction_service.extract_data.return_value = extraction_result
    with patch("backend.src.api.routes.logger"):
        from backend.src.api.routes import extract_data
        resp = await extract_data(file)
    assert resp == extraction_result

@pytest.mark.asyncio
async def test_extract_data_no_text(monkeypatch):
    file = make_upload_file("empty.pdf", b"empty")
    mock_ingestion_service.process_file.return_value = []
    dummy_shipment_data = MagicMock()
    dummy_extraction_response = DummyExtractionResponse(data=dummy_shipment_data, document_id="empty.pdf")
    # Patch ExtractionResponse to return our dummy_extraction_response
    with patch("backend.src.api.routes.ExtractionResponse", return_value=dummy_extraction_response):
        with patch("backend.src.api.routes.logger"):
            from backend.src.api.routes import extract_data
            resp = await extract_data(file)
    assert resp.data == dummy_shipment_data
    assert resp.document_id == "empty.pdf"

@pytest.mark.asyncio
async def test_extract_data_error(monkeypatch):
    file = make_upload_file("fail.pdf", b"fail")
    mock_ingestion_service.process_file.side_effect = Exception("ingest fail")
    with patch("backend.src.api.routes.logger"):
        from backend.src.api.routes import extract_data
        with pytest.raises(HTTPException) as excinfo:
            await extract_data(file)
    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Internal server error during extraction."

def test_ping_endpoint():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "pong"}

def test_upload_document_endpoint_happy_path(monkeypatch):
    # Integration-style test for /upload endpoint
    # Patch all services for deterministic behavior
    chunks = [DummyChunk("chunk1")]
    extraction_result = DummyExtractionResult("ref-xyz")
    structured_chunk = DummyStructuredChunk("structured")
    mock_ingestion_service.process_file.return_value = chunks
    mock_vector_store_service.add_documents.return_value = None
    mock_extraction_service.extract_data.return_value = extraction_result
    mock_extraction_service.create_structured_chunk.return_value = structured_chunk

    files = {'files': ('test.txt', b"hello world", 'text/plain')}
    with patch("backend.src.api.routes.logger"):
        response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Successfully processed 1 documents."
    assert data["errors"] == []
    assert len(data["extractions"]) == 1
    assert data["extractions"][0]["filename"] == "test.txt"
    assert data["extractions"][0]["structured_data_extracted"] is True
    assert data["extractions"][0]["reference_id"] == "ref-xyz"

def test_ask_endpoint_happy_path(monkeypatch):
    dummy_answer = DummySourcedAnswer("42", 0.99)
    mock_rag_service.answer_question.return_value = dummy_answer
    with patch("backend.src.api.routes.logger"):
        response = client.post("/ask", json={"question": "What is the answer?"})
    # Since SourcedAnswer is a MagicMock, FastAPI will serialize as dict
    assert response.status_code == 200
    # The response will be a dict with answer and confidence_score
    assert response.json()["answer"] == "42"
    assert response.json()["confidence_score"] == 0.99

def test_ask_endpoint_error(monkeypatch):
    mock_rag_service.answer_question.side_effect = Exception("fail")
    with patch("backend.src.api.routes.logger"):
        response = client.post("/ask", json={"question": "fail"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error processing question."

def test_extract_endpoint_happy_path(monkeypatch):
    chunks = [DummyChunk("shipment text")]
    extraction_result = DummyExtractionResponse(data="shipment", document_id="ship.pdf")
    mock_ingestion_service.process_file.return_value = chunks
    mock_extraction_service.extract_data.return_value = extraction_result
    with patch("backend.src.api.routes.logger"):
        files = {'file': ('ship.pdf', b"shipdata", 'application/pdf')}
        response = client.post("/extract", files=files)
    assert response.status_code == 200

def test_extract_endpoint_no_text(monkeypatch):
    mock_ingestion_service.process_file.return_value = []
    dummy_shipment_data = MagicMock()
    dummy_extraction_response = DummyExtractionResponse(data=dummy_shipment_data, document_id="empty.pdf")
    with patch("backend.src.api.routes.ExtractionResponse", return_value=dummy_extraction_response):
        with patch("backend.src.api.routes.logger"):
            files = {'file': ('empty.pdf', b"empty", 'application/pdf')}
            response = client.post("/extract", files=files)
    assert response.status_code == 200
    assert response.json()["document_id"] == "empty.pdf"

def test_extract_endpoint_error(monkeypatch):
    mock_ingestion_service.process_file.side_effect = Exception("fail")
    with patch("backend.src.api.routes.logger"):
        files = {'file': ('fail.pdf', b"fail", 'application/pdf')}
        response = client.post("/extract", files=files)
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error during extraction."
