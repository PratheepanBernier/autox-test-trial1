import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from backend.src.api import routes
from models.schemas import QAQuery, SourcedAnswer
from models.extraction_schema import ExtractionResponse, ShipmentData

import types

@pytest.fixture
def mock_router(monkeypatch):
    # Patch logger to avoid side effects
    monkeypatch.setattr(routes, "logger", MagicMock())
    yield

@pytest.fixture
def dummy_upload_file():
    # Minimal UploadFile mock
    file = MagicMock(spec=UploadFile)
    file.filename = "test.pdf"
    file.read = AsyncMock(return_value=b"filecontent")
    return file

@pytest.fixture
def dummy_chunks():
    chunk = MagicMock()
    chunk.text = "chunk text"
    return [chunk, chunk]

@pytest.fixture
def dummy_structured_chunk():
    chunk = MagicMock()
    chunk.text = "structured data"
    return chunk

@pytest.fixture
def dummy_extraction_result():
    result = MagicMock()
    result.data.reference_id = "ref-123"
    return result

@pytest.mark.asyncio
async def test_upload_document_happy_path(
    mock_router, dummy_upload_file, dummy_chunks, dummy_structured_chunk, dummy_extraction_result
):
    with patch.object(routes.ingestion_service, "process_file", return_value=dummy_chunks) as mock_process, \
         patch.object(routes.vector_store_service, "add_documents") as mock_add_docs, \
         patch.object(routes.extraction_service, "extract_data", return_value=dummy_extraction_result) as mock_extract, \
         patch.object(routes.extraction_service, "create_structured_chunk", return_value=dummy_structured_chunk):

        files = [dummy_upload_file]
        resp = await routes.upload_document(files)

        assert resp["message"] == "Successfully processed 1 documents."
        assert resp["errors"] == []
        assert resp["extractions"][0]["filename"] == "test.pdf"
        assert resp["extractions"][0]["text_chunks"] == 2
        assert resp["extractions"][0]["structured_data_extracted"] is True
        assert resp["extractions"][0]["reference_id"] == "ref-123"
        mock_process.assert_called_once()
        mock_add_docs.assert_any_call(dummy_chunks)
        mock_add_docs.assert_any_call([dummy_structured_chunk])
        mock_extract.assert_called_once()

@pytest.mark.asyncio
async def test_upload_document_no_text_extracted(
    mock_router, dummy_upload_file
):
    with patch.object(routes.ingestion_service, "process_file", return_value=[]) as mock_process, \
         patch.object(routes.vector_store_service, "add_documents") as mock_add_docs:

        files = [dummy_upload_file]
        resp = await routes.upload_document(files)

        assert resp["message"] == "Successfully processed 0 documents."
        assert "No text extracted from test.pdf" in resp["errors"]
        assert resp["extractions"] == []
        mock_process.assert_called_once()
        mock_add_docs.assert_not_called()

@pytest.mark.asyncio
async def test_upload_document_extraction_error(
    mock_router, dummy_upload_file, dummy_chunks
):
    with patch.object(routes.ingestion_service, "process_file", return_value=dummy_chunks), \
         patch.object(routes.vector_store_service, "add_documents"), \
         patch.object(routes.extraction_service, "extract_data", side_effect=Exception("extract fail")), \
         patch.object(routes.extraction_service, "create_structured_chunk"):

        files = [dummy_upload_file]
        resp = await routes.upload_document(files)

        assert resp["message"] == "Successfully processed 1 documents."
        assert resp["errors"] == []
        assert resp["extractions"][0]["filename"] == "test.pdf"
        assert resp["extractions"][0]["text_chunks"] == 2
        assert resp["extractions"][0]["structured_data_extracted"] is False
        assert "extract fail" in resp["extractions"][0]["error"]

@pytest.mark.asyncio
async def test_upload_document_file_processing_error(
    mock_router, dummy_upload_file
):
    with patch.object(routes.ingestion_service, "process_file", side_effect=Exception("process fail")):
        files = [dummy_upload_file]
        resp = await routes.upload_document(files)
        assert resp["message"] == "Successfully processed 0 documents."
        assert "Error processing test.pdf: process fail" in resp["errors"]
        assert resp["extractions"] == []

@pytest.mark.asyncio
async def test_upload_document_multiple_files_mixed_results(
    mock_router, dummy_upload_file, dummy_chunks, dummy_structured_chunk, dummy_extraction_result
):
    file1 = MagicMock(spec=UploadFile)
    file1.filename = "file1.pdf"
    file1.read = AsyncMock(return_value=b"content1")
    file2 = MagicMock(spec=UploadFile)
    file2.filename = "file2.pdf"
    file2.read = AsyncMock(return_value=b"content2")

    def process_file_side_effect(content, filename):
        if filename == "file1.pdf":
            return dummy_chunks
        else:
            return []

    with patch.object(routes.ingestion_service, "process_file", side_effect=process_file_side_effect), \
         patch.object(routes.vector_store_service, "add_documents"), \
         patch.object(routes.extraction_service, "extract_data", return_value=dummy_extraction_result), \
         patch.object(routes.extraction_service, "create_structured_chunk", return_value=dummy_structured_chunk):

        files = [file1, file2]
        resp = await routes.upload_document(files)
        assert resp["message"] == "Successfully processed 1 documents."
        assert "No text extracted from file2.pdf" in resp["errors"]
        assert len(resp["extractions"]) == 1
        assert resp["extractions"][0]["filename"] == "file1.pdf"

@pytest.mark.asyncio
async def test_ask_question_happy_path(mock_router):
    query = QAQuery(question="What is the shipment date?")
    answer = SourcedAnswer(answer="2023-01-01", confidence_score=0.95, sources=["doc1"])
    with patch.object(routes.rag_service, "answer_question", return_value=answer) as mock_answer:
        resp = await routes.ask_question(query)
        assert resp == answer
        mock_answer.assert_called_once_with(query)

@pytest.mark.asyncio
async def test_ask_question_error_handling(mock_router):
    query = QAQuery(question="What is the shipment date?")
    with patch.object(routes.rag_service, "answer_question", side_effect=Exception("fail")):
        with pytest.raises(HTTPException) as excinfo:
            await routes.ask_question(query)
        assert excinfo.value.status_code == 500
        assert excinfo.value.detail == "Internal server error processing question."

@pytest.mark.asyncio
async def test_extract_data_happy_path(mock_router, dummy_upload_file, dummy_chunks):
    dummy_result = ExtractionResponse(data=ShipmentData(reference_id="abc"), document_id="test.pdf")
    with patch.object(routes.ingestion_service, "process_file", return_value=dummy_chunks), \
         patch.object(routes.extraction_service, "extract_data", return_value=dummy_result):

        resp = await routes.extract_data(dummy_upload_file)
        assert isinstance(resp, ExtractionResponse)
        assert resp.data.reference_id == "abc"
        assert resp.document_id == "test.pdf"

@pytest.mark.asyncio
async def test_extract_data_no_text_extracted(mock_router, dummy_upload_file):
    with patch.object(routes.ingestion_service, "process_file", return_value=[]):
        resp = await routes.extract_data(dummy_upload_file)
        assert isinstance(resp, ExtractionResponse)
        assert resp.data == ShipmentData()
        assert resp.document_id == "test.pdf"

@pytest.mark.asyncio
async def test_extract_data_error_handling(mock_router, dummy_upload_file):
    with patch.object(routes.ingestion_service, "process_file", side_effect=Exception("fail")):
        with pytest.raises(HTTPException) as excinfo:
            await routes.extract_data(dummy_upload_file)
        assert excinfo.value.status_code == 500
        assert excinfo.value.detail == "Internal server error during extraction."

@pytest.mark.asyncio
async def test_ping_returns_pong(mock_router):
    resp = await routes.ping()
    assert resp == {"status": "pong"}
