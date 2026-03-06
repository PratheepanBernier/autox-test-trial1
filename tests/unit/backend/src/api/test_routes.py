import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from backend.src.api.routes import router
from backend.src.models.extraction_schema import ExtractionResponse, ShipmentData
from backend.src.models.schemas import (
    QAQuery,
    SourcedAnswer,
    UploadResponse,
    UploadExtractionSummary,
)

import io

import sys

import types

import logging

import asyncio

import builtins

import pytest

import typing

from fastapi import FastAPI

app = FastAPI()
app.include_router(router)

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def fake_upload_file():
    file = MagicMock(spec=UploadFile)
    file.filename = "test.pdf"
    file.read = AsyncMock(return_value=b"filecontent")
    return file

@pytest.fixture
def fake_upload_files():
    file1 = MagicMock(spec=UploadFile)
    file1.filename = "file1.pdf"
    file1.read = AsyncMock(return_value=b"content1")
    file2 = MagicMock(spec=UploadFile)
    file2.filename = "file2.pdf"
    file2.read = AsyncMock(return_value=b"content2")
    return [file1, file2]

@pytest.fixture
def fake_container():
    container = MagicMock()
    container.document_pipeline_service.process_uploads = AsyncMock()
    container.rag_service.answer_question = MagicMock()
    container.ingestion_service.process_file = MagicMock()
    container.extraction_service.extract_data = MagicMock()
    return container

@pytest.fixture(autouse=True)
def patch_get_container(fake_container):
    with patch("backend.src.api.routes.get_container", return_value=fake_container):
        yield fake_container

@pytest.mark.asyncio
async def test_upload_document_happy_path(fake_upload_files, fake_container):
    # Arrange
    from backend.src.api.routes import upload_document
    result_mock = MagicMock()
    result_mock.message = "Success"
    result_mock.errors = []
    result_mock.extractions = [
        MagicMock(
            filename="file1.pdf",
            text_chunks=5,
            structured_data_extracted=True,
            reference_id="abc123",
            error=None,
        ),
        MagicMock(
            filename="file2.pdf",
            text_chunks=3,
            structured_data_extracted=False,
            reference_id="def456",
            error="Parse error",
        ),
    ]
    fake_container.document_pipeline_service.process_uploads.return_value = result_mock

    # Act
    resp = await upload_document(fake_upload_files, fake_container)

    # Assert
    assert isinstance(resp, UploadResponse)
    assert resp.message == "Success"
    assert resp.errors == []
    assert len(resp.extractions) == 2
    assert resp.extractions[0].filename == "file1.pdf"
    assert resp.extractions[1].filename == "file2.pdf"
    assert resp.extractions[1].error == "Parse error"

@pytest.mark.asyncio
async def test_upload_document_empty_files(fake_container):
    from backend.src.api.routes import upload_document
    fake_container.document_pipeline_service.process_uploads.return_value = MagicMock(
        message="No files", errors=["No files"], extractions=[]
    )
    resp = await upload_document([], fake_container)
    assert isinstance(resp, UploadResponse)
    assert resp.message == "No files"
    assert resp.errors == ["No files"]
    assert resp.extractions == []

@pytest.mark.asyncio
async def test_upload_document_service_error(fake_upload_files, fake_container):
    from backend.src.api.routes import upload_document
    fake_container.document_pipeline_service.process_uploads.side_effect = Exception("Service down")
    with pytest.raises(Exception):
        await upload_document(fake_upload_files, fake_container)

def test_ask_question_happy_path(fake_container):
    from backend.src.api.routes import ask_question
    query = QAQuery(question="What is the shipment date?")
    answer = SourcedAnswer(answer="2024-01-01", confidence_score=0.99, sources=["doc1"])
    fake_container.rag_service.answer_question.return_value = answer
    resp = ask_question(query, fake_container)
    assert resp == answer

def test_ask_question_error_handling(fake_container):
    from backend.src.api.routes import ask_question
    query = QAQuery(question="What is the shipment date?")
    fake_container.rag_service.answer_question.side_effect = Exception("RAG failure")
    with pytest.raises(HTTPException) as excinfo:
        ask_question(query, fake_container)
    assert excinfo.value.status_code == 500
    assert "Internal server error" in excinfo.value.detail

@pytest.mark.asyncio
async def test_extract_data_happy_path(fake_upload_file, fake_container):
    from backend.src.api.routes import extract_data
    # Simulate ingestion_service returns chunks with text
    chunk1 = MagicMock(text="Hello")
    chunk2 = MagicMock(text="World")
    fake_container.ingestion_service.process_file.return_value = [chunk1, chunk2]
    extraction_result = ExtractionResponse(
        data=ShipmentData(field1="value1"),
        document_id="test.pdf"
    )
    fake_container.extraction_service.extract_data.return_value = extraction_result
    resp = await extract_data(fake_upload_file, fake_container)
    assert isinstance(resp, ExtractionResponse)
    assert resp.data.field1 == "value1"
    assert resp.document_id == "test.pdf"

@pytest.mark.asyncio
async def test_extract_data_no_text_extracted(fake_upload_file, fake_container):
    from backend.src.api.routes import extract_data
    fake_container.ingestion_service.process_file.return_value = []
    resp = await extract_data(fake_upload_file, fake_container)
    assert isinstance(resp, ExtractionResponse)
    assert resp.data == ShipmentData()
    assert resp.document_id == "test.pdf"

@pytest.mark.asyncio
async def test_extract_data_extraction_service_error(fake_upload_file, fake_container):
    from backend.src.api.routes import extract_data
    chunk = MagicMock(text="Some text")
    fake_container.ingestion_service.process_file.return_value = [chunk]
    fake_container.extraction_service.extract_data.side_effect = Exception("Extraction failed")
    with pytest.raises(HTTPException) as excinfo:
        await extract_data(fake_upload_file, fake_container)
    assert excinfo.value.status_code == 500
    assert "Internal server error" in excinfo.value.detail

@pytest.mark.asyncio
async def test_extract_data_ingestion_service_error(fake_upload_file, fake_container):
    from backend.src.api.routes import extract_data
    fake_container.ingestion_service.process_file.side_effect = Exception("Ingestion failed")
    with pytest.raises(HTTPException) as excinfo:
        await extract_data(fake_upload_file, fake_container)
    assert excinfo.value.status_code == 500
    assert "Internal server error" in excinfo.value.detail

def test_ping_endpoint(client):
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert resp.json() == {"status": "pong"}
