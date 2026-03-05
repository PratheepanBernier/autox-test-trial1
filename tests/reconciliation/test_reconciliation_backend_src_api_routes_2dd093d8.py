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
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import UploadFile, HTTPException
from fastapi.testclient import TestClient
from backend.src.api.routes import router
from models.schemas import QAQuery, SourcedAnswer
from models.extraction_schema import ExtractionResponse, ShipmentData

import io

import types

from fastapi import FastAPI

@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)

@pytest.fixture
def fake_upload_file():
    def _make(filename, content):
        file = MagicMock(spec=UploadFile)
        file.filename = filename
        file.read = AsyncMock(return_value=content)
        return file
    return _make

@pytest.fixture
def patch_services():
    with patch("backend.src.api.routes.ingestion_service") as ingestion_service, \
         patch("backend.src.api.routes.vector_store_service") as vector_store_service, \
         patch("backend.src.api.routes.extraction_service") as extraction_service, \
         patch("backend.src.api.routes.rag_service") as rag_service:
        yield ingestion_service, vector_store_service, extraction_service, rag_service

@pytest.mark.asyncio
async def test_upload_document_happy_path(patch_services, fake_upload_file):
    ingestion_service, vector_store_service, extraction_service, _ = patch_services

    # Mock chunk object
    chunk = MagicMock()
    chunk.text = "chunk text"
    ingestion_service.process_file.return_value = [chunk]
    extraction_result = MagicMock()
    extraction_result.data.reference_id = "ref-123"
    extraction_service.extract_data.return_value = extraction_result
    structured_chunk = MagicMock()
    extraction_service.create_structured_chunk.return_value = structured_chunk

    file1 = fake_upload_file("doc1.pdf", b"filecontent1")
    file2 = fake_upload_file("doc2.pdf", b"filecontent2")

    from backend.src.api.routes import upload_document

    files = [file1, file2]
    response = await upload_document(files=files)

    assert response["message"] == "Successfully processed 2 documents."
    assert response["errors"] == []
    assert len(response["extractions"]) == 2
    for extraction in response["extractions"]:
        assert extraction["structured_data_extracted"] is True
        assert extraction["reference_id"] == "ref-123"
        assert extraction["text_chunks"] == 1

@pytest.mark.asyncio
async def test_upload_document_no_text_extracted(patch_services, fake_upload_file):
    ingestion_service, vector_store_service, extraction_service, _ = patch_services

    ingestion_service.process_file.return_value = []
    file1 = fake_upload_file("empty.pdf", b"emptycontent")

    from backend.src.api.routes import upload_document

    files = [file1]
    response = await upload_document(files=files)

    assert response["message"] == "Successfully processed 0 documents."
    assert "No text extracted from empty.pdf" in response["errors"][0]
    assert response["extractions"] == []

@pytest.mark.asyncio
async def test_upload_document_extraction_error(patch_services, fake_upload_file):
    ingestion_service, vector_store_service, extraction_service, _ = patch_services

    chunk = MagicMock()
    chunk.text = "chunk text"
    ingestion_service.process_file.return_value = [chunk]
    extraction_service.extract_data.side_effect = Exception("Extraction failed")
    file1 = fake_upload_file("fail.pdf", b"failcontent")
    extraction_service.create_structured_chunk.return_value = MagicMock()

    from backend.src.api.routes import upload_document

    files = [file1]
    response = await upload_document(files=files)

    assert response["message"] == "Successfully processed 1 documents."
    assert response["errors"] == []
    assert len(response["extractions"]) == 1
    extraction = response["extractions"][0]
    assert extraction["filename"] == "fail.pdf"
    assert extraction["structured_data_extracted"] is False
    assert "Extraction failed" in extraction["error"]

@pytest.mark.asyncio
async def test_upload_document_file_processing_error(patch_services, fake_upload_file):
    ingestion_service, vector_store_service, extraction_service, _ = patch_services

    ingestion_service.process_file.side_effect = Exception("File corrupt")
    file1 = fake_upload_file("corrupt.pdf", b"corruptcontent")

    from backend.src.api.routes import upload_document

    files = [file1]
    response = await upload_document(files=files)

    assert response["message"] == "Successfully processed 0 documents."
    assert "Error processing corrupt.pdf: File corrupt" in response["errors"][0]
    assert response["extractions"] == []

@pytest.mark.asyncio
async def test_upload_document_boundary_empty_file_list(patch_services):
    from backend.src.api.routes import upload_document

    files = []
    response = await upload_document(files=files)
    assert response["message"] == "Successfully processed 0 documents."
    assert response["errors"] == []
    assert response["extractions"] == []

def test_ask_question_happy_path(patch_services):
    _, _, _, rag_service = patch_services

    answer = MagicMock(spec=SourcedAnswer)
    answer.confidence_score = 0.95
    rag_service.answer_question.return_value = answer

    from backend.src.api.routes import ask_question

    query = QAQuery(question="What is the shipment date?")
    result = ask_question(query)
    # ask_question is async, but not awaited in FastAPI sync context
    # so we call it directly for test
    assert result == answer

def test_ask_question_error_handling(patch_services):
    _, _, _, rag_service = patch_services

    rag_service.answer_question.side_effect = Exception("RAG error")

    from backend.src.api.routes import ask_question

    query = QAQuery(question="What is the shipment date?")
    with pytest.raises(HTTPException) as excinfo:
        ask_question(query)
    assert excinfo.value.status_code == 500
    assert "Internal server error processing question." in str(excinfo.value.detail)

@pytest.mark.asyncio
async def test_extract_data_happy_path(patch_services, fake_upload_file):
    ingestion_service, _, extraction_service, _ = patch_services

    chunk = MagicMock()
    chunk.text = "chunk text"
    ingestion_service.process_file.return_value = [chunk]
    extraction_response = ExtractionResponse(data=ShipmentData(reference_id="abc"), document_id="doc.pdf")
    extraction_service.extract_data.return_value = extraction_response

    file = fake_upload_file("doc.pdf", b"filecontent")
    from backend.src.api.routes import extract_data

    response = await extract_data(file=file)
    assert isinstance(response, ExtractionResponse)
    assert response.data.reference_id == "abc"
    assert response.document_id == "doc.pdf"

@pytest.mark.asyncio
async def test_extract_data_no_text_extracted(patch_services, fake_upload_file):
    ingestion_service, _, extraction_service, _ = patch_services

    ingestion_service.process_file.return_value = []
    file = fake_upload_file("empty.pdf", b"emptycontent")

    from backend.src.api.routes import extract_data

    response = await extract_data(file=file)
    assert isinstance(response, ExtractionResponse)
    assert response.data == ShipmentData()
    assert response.document_id == "empty.pdf"

@pytest.mark.asyncio
async def test_extract_data_error_handling(patch_services, fake_upload_file):
    ingestion_service, _, extraction_service, _ = patch_services

    ingestion_service.process_file.side_effect = Exception("Extraction error")
    file = fake_upload_file("fail.pdf", b"failcontent")

    from backend.src.api.routes import extract_data

    with pytest.raises(HTTPException) as excinfo:
        await extract_data(file=file)
    assert excinfo.value.status_code == 500
    assert "Internal server error during extraction." in str(excinfo.value.detail)

def test_ping_endpoint(client):
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "pong"}

@pytest.mark.asyncio
async def test_reconciliation_upload_and_extract_equivalent_structured_data(patch_services, fake_upload_file):
    ingestion_service, vector_store_service, extraction_service, _ = patch_services

    # Setup: both endpoints should extract the same structured data for the same file
    chunk = MagicMock()
    chunk.text = "chunk text"
    ingestion_service.process_file.return_value = [chunk]
    extraction_result = MagicMock()
    extraction_result.data.reference_id = "recon-123"
    extraction_service.extract_data.return_value = ExtractionResponse(
        data=ShipmentData(reference_id="recon-123"), document_id="recon.pdf"
    )
    extraction_service.create_structured_chunk.return_value = MagicMock()

    file = fake_upload_file("recon.pdf", b"filecontent")

    from backend.src.api.routes import upload_document, extract_data

    # Call upload_document
    upload_resp = await upload_document(files=[file])
    # Call extract_data
    extract_resp = await extract_data(file=file)

    # Reconciliation: reference_id should match
    upload_extraction = upload_resp["extractions"][0]
    assert upload_extraction["reference_id"] == extract_resp.data.reference_id
    assert upload_extraction["filename"] == extract_resp.document_id
    assert upload_extraction["structured_data_extracted"] is True

@pytest.mark.asyncio
async def test_reconciliation_upload_and_extract_empty_file(patch_services, fake_upload_file):
    ingestion_service, vector_store_service, extraction_service, _ = patch_services

    ingestion_service.process_file.return_value = []
    file = fake_upload_file("empty.pdf", b"")

    from backend.src.api.routes import upload_document, extract_data

    upload_resp = await upload_document(files=[file])
    extract_resp = await extract_data(file=file)

    assert upload_resp["extractions"] == []
    assert extract_resp.data == ShipmentData()
    assert extract_resp.document_id == "empty.pdf"
