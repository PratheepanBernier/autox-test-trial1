import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from io import BytesIO

import sys
import types

# Patch sys.modules to mock all external services and models
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

from api import routes

import asyncio

@pytest.fixture
def anyio_backend():
    return 'asyncio'

@pytest.fixture
def test_client():
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)

class DummyUploadFile:
    def __init__(self, filename, content):
        self.filename = filename
        self._content = content
        self.read_called = False

    async def read(self):
        self.read_called = True
        return self._content

@pytest.mark.asyncio
async def test_upload_document_happy_path(monkeypatch):
    # Setup
    file_content = b"file content"
    filename = "test.txt"
    dummy_file = DummyUploadFile(filename, file_content)

    # Mock chunk object
    class DummyChunk:
        def __init__(self, text):
            self.text = text

    chunks = [DummyChunk("chunk1"), DummyChunk("chunk2")]

    mock_ingestion_service.process_file.return_value = chunks
    mock_vector_store_service.add_documents.return_value = None

    extraction_result = MagicMock()
    extraction_result.data.reference_id = "ref123"
    mock_extraction_service.extract_data.return_value = extraction_result

    structured_chunk = DummyChunk("structured")
    mock_extraction_service.create_structured_chunk.return_value = structured_chunk

    # Call
    response = await routes.upload_document([dummy_file])

    # Assert
    assert response["message"] == "Successfully processed 1 documents."
    assert response["errors"] == []
    assert response["extractions"][0]["filename"] == filename
    assert response["extractions"][0]["text_chunks"] == 2
    assert response["extractions"][0]["structured_data_extracted"] is True
    assert response["extractions"][0]["reference_id"] == "ref123"
    assert dummy_file.read_called

@pytest.mark.asyncio
async def test_upload_document_no_text_extracted(monkeypatch):
    file_content = b"file content"
    filename = "empty.txt"
    dummy_file = DummyUploadFile(filename, file_content)

    mock_ingestion_service.process_file.return_value = []

    response = await routes.upload_document([dummy_file])

    assert response["message"] == "Successfully processed 0 documents."
    assert len(response["errors"]) == 1
    assert "No text extracted from empty.txt" in response["errors"][0]
    assert response["extractions"] == []

@pytest.mark.asyncio
async def test_upload_document_extraction_failure(monkeypatch):
    file_content = b"file content"
    filename = "fail_extract.txt"
    dummy_file = DummyUploadFile(filename, file_content)

    class DummyChunk:
        def __init__(self, text):
            self.text = text

    chunks = [DummyChunk("chunk1")]
    mock_ingestion_service.process_file.return_value = chunks
    mock_vector_store_service.add_documents.return_value = None

    mock_extraction_service.extract_data.side_effect = Exception("Extraction failed")

    response = await routes.upload_document([dummy_file])

    assert response["message"] == "Successfully processed 1 documents."
    assert response["errors"] == []
    assert response["extractions"][0]["filename"] == filename
    assert response["extractions"][0]["text_chunks"] == 1
    assert response["extractions"][0]["structured_data_extracted"] is False
    assert "Extraction failed" in response["extractions"][0]["error"]

@pytest.mark.asyncio
async def test_upload_document_file_processing_error(monkeypatch):
    file_content = b"file content"
    filename = "badfile.txt"
    dummy_file = DummyUploadFile(filename, file_content)

    mock_ingestion_service.process_file.side_effect = Exception("Processing error")

    response = await routes.upload_document([dummy_file])

    assert response["message"] == "Successfully processed 0 documents."
    assert len(response["errors"]) == 1
    assert "Error processing badfile.txt: Processing error" in response["errors"][0]
    assert response["extractions"] == []

@pytest.mark.asyncio
async def test_upload_document_multiple_files_mixed(monkeypatch):
    # First file: happy path
    file1 = DummyUploadFile("good.txt", b"abc")
    # Second file: no text
    file2 = DummyUploadFile("empty.txt", b"def")
    # Third file: extraction error
    file3 = DummyUploadFile("fail.txt", b"ghi")

    class DummyChunk:
        def __init__(self, text):
            self.text = text

    # Setup mocks for each file
    def process_file_side_effect(content, filename):
        if filename == "good.txt":
            return [DummyChunk("chunk1")]
        elif filename == "empty.txt":
            return []
        elif filename == "fail.txt":
            return [DummyChunk("chunk2")]

    mock_ingestion_service.process_file.side_effect = process_file_side_effect
    mock_vector_store_service.add_documents.return_value = None

    extraction_result = MagicMock()
    extraction_result.data.reference_id = "ref456"
    mock_extraction_service.extract_data.side_effect = [extraction_result, Exception("Extraction failed")]
    mock_extraction_service.create_structured_chunk.return_value = DummyChunk("structured")

    response = await routes.upload_document([file1, file2, file3])

    assert response["message"] == "Successfully processed 2 documents."
    assert len(response["errors"]) == 0 or all("No text extracted" in e for e in response["errors"])
    assert len(response["extractions"]) == 2
    assert response["extractions"][0]["filename"] == "good.txt"
    assert response["extractions"][0]["structured_data_extracted"] is True
    assert response["extractions"][1]["filename"] == "fail.txt"
    assert response["extractions"][1]["structured_data_extracted"] is False

@pytest.mark.asyncio
async def test_ask_question_happy_path(monkeypatch):
    query = MagicMock()
    query.question = "What is the answer?"

    answer = MagicMock()
    answer.confidence_score = 0.95
    mock_rag_service.answer_question.return_value = answer

    result = await routes.ask_question(query)
    assert result == answer
    mock_rag_service.answer_question.assert_called_once_with(query)

@pytest.mark.asyncio
async def test_ask_question_error(monkeypatch):
    query = MagicMock()
    query.question = "What is the answer?"

    mock_rag_service.answer_question.side_effect = Exception("RAG error")

    with pytest.raises(HTTPException) as excinfo:
        await routes.ask_question(query)
    assert excinfo.value.status_code == 500
    assert "Internal server error processing question." in excinfo.value.detail

@pytest.mark.asyncio
async def test_extract_data_happy_path(monkeypatch):
    file = DummyUploadFile("ship.txt", b"abc")

    class DummyChunk:
        def __init__(self, text):
            self.text = text

    chunks = [DummyChunk("chunk1"), DummyChunk("chunk2")]
    mock_ingestion_service.process_file.return_value = chunks

    extraction_response = MagicMock()
    mock_extraction_service.extract_data.return_value = extraction_response

    result = await routes.extract_data(file)
    assert result == extraction_response
    mock_ingestion_service.process_file.assert_called_once()
    mock_extraction_service.extract_data.assert_called_once()

@pytest.mark.asyncio
async def test_extract_data_no_text(monkeypatch):
    file = DummyUploadFile("empty.txt", b"abc")
    mock_ingestion_service.process_file.return_value = []

    empty_shipment_data = MagicMock()
    mock_ShipmentData.return_value = empty_shipment_data
    empty_response = MagicMock()
    mock_ExtractionResponse.return_value = empty_response

    result = await routes.extract_data(file)
    mock_ExtractionResponse.assert_called_once_with(data=empty_shipment_data, document_id="empty.txt")
    assert result == empty_response

@pytest.mark.asyncio
async def test_extract_data_error(monkeypatch):
    file = DummyUploadFile("bad.txt", b"abc")
    mock_ingestion_service.process_file.side_effect = Exception("Extraction error")

    with pytest.raises(HTTPException) as excinfo:
        await routes.extract_data(file)
    assert excinfo.value.status_code == 500
    assert "Internal server error during extraction." in excinfo.value.detail

def test_ping_endpoint(test_client):
    response = test_client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "pong"}
