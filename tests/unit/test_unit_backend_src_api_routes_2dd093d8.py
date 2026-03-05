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
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import UploadFile, HTTPException
from backend.src.api import routes
from models.schemas import QAQuery, SourcedAnswer
from models.extraction_schema import ExtractionResponse, ShipmentData

@pytest.mark.asyncio
@patch("backend.src.api.routes.ingestion_service")
@patch("backend.src.api.routes.vector_store_service")
@patch("backend.src.api.routes.extraction_service")
async def test_upload_document_happy_path(mock_extraction_service, mock_vector_store_service, mock_ingestion_service):
    file_mock = AsyncMock(spec=UploadFile)
    file_mock.filename = "test.pdf"
    file_mock.read = AsyncMock(return_value=b"filecontent")
    chunk_mock = MagicMock()
    chunk_mock.text = "chunk text"
    mock_ingestion_service.process_file.return_value = [chunk_mock]
    extraction_result_mock = MagicMock()
    extraction_result_mock.data.reference_id = "ref123"
    mock_extraction_service.extract_data.return_value = extraction_result_mock
    structured_chunk_mock = MagicMock()
    mock_extraction_service.create_structured_chunk.return_value = structured_chunk_mock

    files = [file_mock]
    result = await routes.upload_document(files=files)

    assert result["message"] == "Successfully processed 1 documents."
    assert result["errors"] == []
    assert result["extractions"][0]["filename"] == "test.pdf"
    assert result["extractions"][0]["text_chunks"] == 1
    assert result["extractions"][0]["structured_data_extracted"] is True
    assert result["extractions"][0]["reference_id"] == "ref123"
    mock_vector_store_service.add_documents.assert_any_call([chunk_mock])
    mock_vector_store_service.add_documents.assert_any_call([structured_chunk_mock])

@pytest.mark.asyncio
@patch("backend.src.api.routes.ingestion_service")
@patch("backend.src.api.routes.vector_store_service")
@patch("backend.src.api.routes.extraction_service")
async def test_upload_document_no_chunks(mock_extraction_service, mock_vector_store_service, mock_ingestion_service):
    file_mock = AsyncMock(spec=UploadFile)
    file_mock.filename = "empty.pdf"
    file_mock.read = AsyncMock(return_value=b"filecontent")
    mock_ingestion_service.process_file.return_value = []

    files = [file_mock]
    result = await routes.upload_document(files=files)

    assert result["message"] == "Successfully processed 0 documents."
    assert "No text extracted from empty.pdf" in result["errors"]
    assert result["extractions"] == []

@pytest.mark.asyncio
@patch("backend.src.api.routes.ingestion_service")
@patch("backend.src.api.routes.vector_store_service")
@patch("backend.src.api.routes.extraction_service")
async def test_upload_document_extraction_error(mock_extraction_service, mock_vector_store_service, mock_ingestion_service):
    file_mock = AsyncMock(spec=UploadFile)
    file_mock.filename = "fail_extract.pdf"
    file_mock.read = AsyncMock(return_value=b"filecontent")
    chunk_mock = MagicMock()
    chunk_mock.text = "chunk text"
    mock_ingestion_service.process_file.return_value = [chunk_mock]
    mock_extraction_service.extract_data.side_effect = Exception("Extraction failed")

    files = [file_mock]
    result = await routes.upload_document(files=files)

    assert result["message"] == "Successfully processed 1 documents."
    assert result["errors"] == []
    assert result["extractions"][0]["filename"] == "fail_extract.pdf"
    assert result["extractions"][0]["text_chunks"] == 1
    assert result["extractions"][0]["structured_data_extracted"] is False
    assert "Extraction failed" in result["extractions"][0]["error"]

@pytest.mark.asyncio
@patch("backend.src.api.routes.ingestion_service")
@patch("backend.src.api.routes.vector_store_service")
@patch("backend.src.api.routes.extraction_service")
async def test_upload_document_file_processing_error(mock_extraction_service, mock_vector_store_service, mock_ingestion_service):
    file_mock = AsyncMock(spec=UploadFile)
    file_mock.filename = "badfile.pdf"
    file_mock.read = AsyncMock(return_value=b"filecontent")
    mock_ingestion_service.process_file.side_effect = Exception("Processing error")

    files = [file_mock]
    result = await routes.upload_document(files=files)

    assert result["message"] == "Successfully processed 0 documents."
    assert "Error processing badfile.pdf: Processing error" in result["errors"]
    assert result["extractions"] == []

@pytest.mark.asyncio
@patch("backend.src.api.routes.rag_service")
def test_ask_question_happy_path(mock_rag_service):
    query = QAQuery(question="What is the shipment date?")
    answer = SourcedAnswer(answer="2024-01-01", confidence_score=0.95, sources=["doc1"])
    mock_rag_service.answer_question.return_value = answer

    import asyncio
    result = asyncio.run(routes.ask_question(query=query))
    assert result == answer
    mock_rag_service.answer_question.assert_called_once_with(query)

@pytest.mark.asyncio
@patch("backend.src.api.routes.rag_service")
def test_ask_question_error_handling(mock_rag_service):
    query = QAQuery(question="What is the shipment date?")
    mock_rag_service.answer_question.side_effect = Exception("RAG error")

    import asyncio
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(routes.ask_question(query=query))
    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Internal server error processing question."

@pytest.mark.asyncio
@patch("backend.src.api.routes.ingestion_service")
@patch("backend.src.api.routes.extraction_service")
async def test_extract_data_happy_path(mock_extraction_service, mock_ingestion_service):
    file_mock = AsyncMock(spec=UploadFile)
    file_mock.filename = "extract.pdf"
    file_mock.read = AsyncMock(return_value=b"filecontent")
    chunk_mock = MagicMock()
    chunk_mock.text = "chunk text"
    mock_ingestion_service.process_file.return_value = [chunk_mock]
    extraction_response = ExtractionResponse(data=ShipmentData(), document_id="extract.pdf")
    mock_extraction_service.extract_data.return_value = extraction_response

    result = await routes.extract_data(file=file_mock)
    assert isinstance(result, ExtractionResponse)
    assert result.document_id == "extract.pdf"

@pytest.mark.asyncio
@patch("backend.src.api.routes.ingestion_service")
@patch("backend.src.api.routes.extraction_service")
async def test_extract_data_no_text(mock_extraction_service, mock_ingestion_service):
    file_mock = AsyncMock(spec=UploadFile)
    file_mock.filename = "empty.pdf"
    file_mock.read = AsyncMock(return_value=b"filecontent")
    mock_ingestion_service.process_file.return_value = []

    result = await routes.extract_data(file=file_mock)
    assert isinstance(result, ExtractionResponse)
    assert result.document_id == "empty.pdf"
    assert result.data == ShipmentData()

@pytest.mark.asyncio
@patch("backend.src.api.routes.ingestion_service")
@patch("backend.src.api.routes.extraction_service")
async def test_extract_data_error_handling(mock_extraction_service, mock_ingestion_service):
    file_mock = AsyncMock(spec=UploadFile)
    file_mock.filename = "fail.pdf"
    file_mock.read = AsyncMock(return_value=b"filecontent")
    mock_ingestion_service.process_file.side_effect = Exception("Extraction error")

    with pytest.raises(HTTPException) as excinfo:
        await routes.extract_data(file=file_mock)
    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Internal server error during extraction."

@pytest.mark.asyncio
async def test_ping_returns_pong():
    result = await routes.ping()
    assert result == {"status": "pong"}
