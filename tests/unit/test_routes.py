# source_hash: 7f49f55d5565bc86
import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from backend.src.api.routes import router
from models.schemas import QAQuery, SourcedAnswer
from models.extraction_schema import ExtractionResponse, ShipmentData

import types

client = TestClient(router)

@pytest.fixture
def mock_upload_file():
    file = MagicMock(spec=UploadFile)
    file.filename = "test.pdf"
    file.read = AsyncMock(return_value=b"filecontent")
    return file

@pytest.fixture
def mock_text_chunk():
    chunk = MagicMock()
    chunk.text = "chunk text"
    return chunk

@pytest.fixture
def mock_structured_chunk():
    chunk = MagicMock()
    chunk.text = "structured data"
    return chunk

@pytest.fixture
def mock_extraction_result():
    result = MagicMock()
    result.data.reference_id = "ref-123"
    return result

@pytest.fixture
def mock_extraction_response():
    return ExtractionResponse(data=ShipmentData(reference_id="ref-123"), document_id="test.pdf")

@pytest.mark.asyncio
@patch("backend.src.api.routes.ingestion_service")
@patch("backend.src.api.routes.vector_store_service")
@patch("backend.src.api.routes.extraction_service")
async def test_upload_document_happy_path(
    mock_extraction_service,
    mock_vector_store_service,
    mock_ingestion_service,
    mock_upload_file,
    mock_text_chunk,
    mock_structured_chunk,
    mock_extraction_result,
):
    mock_ingestion_service.process_file.return_value = [mock_text_chunk]
    mock_vector_store_service.add_documents.return_value = None
    mock_extraction_service.extract_data.return_value = mock_extraction_result
    mock_extraction_service.create_structured_chunk.return_value = mock_structured_chunk

    from backend.src.api.routes import upload_document

    files = [mock_upload_file]
    response = await upload_document(files=files)

    assert response["message"] == "Successfully processed 1 documents."
    assert response["errors"] == []
    assert response["extractions"][0]["filename"] == "test.pdf"
    assert response["extractions"][0]["text_chunks"] == 1
    assert response["extractions"][0]["structured_data_extracted"] is True
    assert response["extractions"][0]["reference_id"] == "ref-123"

@pytest.mark.asyncio
@patch("backend.src.api.routes.ingestion_service")
@patch("backend.src.api.routes.vector_store_service")
@patch("backend.src.api.routes.extraction_service")
async def test_upload_document_no_text_extracted(
    mock_extraction_service,
    mock_vector_store_service,
    mock_ingestion_service,
    mock_upload_file,
):
    mock_ingestion_service.process_file.return_value = []

    from backend.src.api.routes import upload_document

    files = [mock_upload_file]
    response = await upload_document(files=files)

    assert response["message"] == "Successfully processed 0 documents."
    assert "No text extracted from test.pdf" in response["errors"]
    assert response["extractions"] == []

@pytest.mark.asyncio
@patch("backend.src.api.routes.ingestion_service")
@patch("backend.src.api.routes.vector_store_service")
@patch("backend.src.api.routes.extraction_service")
async def test_upload_document_extraction_error(
    mock_extraction_service,
    mock_vector_store_service,
    mock_ingestion_service,
    mock_upload_file,
    mock_text_chunk,
):
    mock_ingestion_service.process_file.return_value = [mock_text_chunk]
    mock_vector_store_service.add_documents.return_value = None
    mock_extraction_service.extract_data.side_effect = Exception("Extraction failed")

    from backend.src.api.routes import upload_document

    files = [mock_upload_file]
    response = await upload_document(files=files)

    assert response["message"] == "Successfully processed 1 documents."
    assert response["errors"] == []
    assert response["extractions"][0]["filename"] == "test.pdf"
    assert response["extractions"][0]["text_chunks"] == 1
    assert response["extractions"][0]["structured_data_extracted"] is False
    assert "Extraction failed" in response["extractions"][0]["error"]

@pytest.mark.asyncio
@patch("backend.src.api.routes.ingestion_service")
@patch("backend.src.api.routes.vector_store_service")
@patch("backend.src.api.routes.extraction_service")
async def test_upload_document_file_processing_error(
    mock_extraction_service,
    mock_vector_store_service,
    mock_ingestion_service,
    mock_upload_file,
):
    mock_ingestion_service.process_file.side_effect = Exception("Processing error")

    from backend.src.api.routes import upload_document

    files = [mock_upload_file]
    response = await upload_document(files=files)

    assert response["message"] == "Successfully processed 0 documents."
    assert "Error processing test.pdf: Processing error" in response["errors"]
    assert response["extractions"] == []

@pytest.mark.asyncio
@patch("backend.src.api.routes.rag_service")
async def test_ask_question_happy_path(mock_rag_service):
    mock_answer = SourcedAnswer(
        answer="42",
        sources=["doc1"],
        confidence_score=0.99
    )
    mock_rag_service.answer_question.return_value = mock_answer

    from backend.src.api.routes import ask_question

    query = QAQuery(question="What is the answer?", top_k=1)
    response = await ask_question(query=query)

    assert response.answer == "42"
    assert response.sources == ["doc1"]
    assert response.confidence_score == 0.99

@pytest.mark.asyncio
@patch("backend.src.api.routes.rag_service")
async def test_ask_question_error_handling(mock_rag_service):
    mock_rag_service.answer_question.side_effect = Exception("RAG error")

    from backend.src.api.routes import ask_question

    query = QAQuery(question="What is the answer?", top_k=1)
    with pytest.raises(HTTPException) as excinfo:
        await ask_question(query=query)
    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Internal server error processing question."

@pytest.mark.asyncio
@patch("backend.src.api.routes.ingestion_service")
@patch("backend.src.api.routes.extraction_service")
async def test_extract_data_happy_path(
    mock_extraction_service,
    mock_ingestion_service,
    mock_upload_file,
    mock_text_chunk,
    mock_extraction_response,
):
    mock_ingestion_service.process_file.return_value = [mock_text_chunk]
    mock_text_chunk.text = "some text"
    mock_extraction_service.extract_data.return_value = mock_extraction_response

    from backend.src.api.routes import extract_data

    response = await extract_data(file=mock_upload_file)
    assert isinstance(response, ExtractionResponse)
    assert response.data.reference_id == "ref-123"
    assert response.document_id == "test.pdf"

@pytest.mark.asyncio
@patch("backend.src.api.routes.ingestion_service")
@patch("backend.src.api.routes.extraction_service")
async def test_extract_data_no_text_extracted(
    mock_extraction_service,
    mock_ingestion_service,
    mock_upload_file,
):
    mock_ingestion_service.process_file.return_value = []
    mock_upload_file.filename = "empty.pdf"

    from backend.src.api.routes import extract_data

    response = await extract_data(file=mock_upload_file)
    assert isinstance(response, ExtractionResponse)
    assert response.data == ShipmentData()
    assert response.document_id == "empty.pdf"

@pytest.mark.asyncio
@patch("backend.src.api.routes.ingestion_service")
@patch("backend.src.api.routes.extraction_service")
async def test_extract_data_error_handling(
    mock_extraction_service,
    mock_ingestion_service,
    mock_upload_file,
    mock_text_chunk,
):
    mock_ingestion_service.process_file.return_value = [mock_text_chunk]
    mock_text_chunk.text = "some text"
    mock_extraction_service.extract_data.side_effect = Exception("Extraction error")

    from backend.src.api.routes import extract_data

    with pytest.raises(HTTPException) as excinfo:
        await extract_data(file=mock_upload_file)
    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Internal server error during extraction."

def test_ping_endpoint_returns_pong():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "pong"}
