import io
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile as StarletteUploadFile
from unittest.mock import AsyncMock, MagicMock, patch

from backend.src.api import routes
from backend.src.models.schemas import (
    QAQuery,
    SourcedAnswer,
    UploadResponse,
    UploadExtractionSummary,
)
from backend.src.models.extraction_schema import (
    ExtractionResponse,
    ShipmentData,
)
from backend.src.dependencies import ServiceContainer

@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(routes.router)
    return app

@pytest.fixture
def client(app):
    return TestClient(app)

@pytest.fixture
def mock_container():
    # Create a mock ServiceContainer with all required services
    container = MagicMock(spec=ServiceContainer)
    container.document_pipeline_service.process_uploads = AsyncMock()
    container.rag_service.answer_question = MagicMock()
    container.ingestion_service.process_file = MagicMock()
    container.extraction_service.extract_data = MagicMock()
    return container

@pytest.fixture(autouse=True)
def patch_get_container(mock_container):
    with patch("backend.src.api.routes.get_container", return_value=mock_container):
        yield

def make_upload_file(filename, content):
    # Simulate FastAPI UploadFile using Starlette's UploadFile
    file = StarletteUploadFile(filename=filename, file=io.BytesIO(content))
    return file

@pytest.mark.asyncio
class TestUploadDocument:
    async def test_upload_document_happy_path(self, app, mock_container):
        # Arrange
        files = [
            make_upload_file("doc1.txt", b"hello world"),
            make_upload_file("doc2.txt", b"another file"),
        ]
        mock_result = MagicMock()
        mock_result.message = "Upload successful"
        mock_result.errors = []
        mock_result.extractions = [
            MagicMock(
                filename="doc1.txt",
                text_chunks=3,
                structured_data_extracted=True,
                reference_id="REF123",
                error=None,
            ),
            MagicMock(
                filename="doc2.txt",
                text_chunks=2,
                structured_data_extracted=False,
                reference_id=None,
                error="Extraction failed",
            ),
        ]
        mock_container.document_pipeline_service.process_uploads.return_value = mock_result

        # Act
        async with TestClient(app) as ac:
            response = await ac.post(
                "/upload",
                files=[
                    ("files", ("doc1.txt", b"hello world", "text/plain")),
                    ("files", ("doc2.txt", b"another file", "text/plain")),
                ],
            )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Upload successful"
        assert data["errors"] == []
        assert len(data["extractions"]) == 2
        assert data["extractions"][0]["filename"] == "doc1.txt"
        assert data["extractions"][0]["text_chunks"] == 3
        assert data["extractions"][0]["structured_data_extracted"] is True
        assert data["extractions"][0]["reference_id"] == "REF123"
        assert data["extractions"][0]["error"] is None
        assert data["extractions"][1]["filename"] == "doc2.txt"
        assert data["extractions"][1]["structured_data_extracted"] is False
        assert data["extractions"][1]["error"] == "Extraction failed"

    async def test_upload_document_with_errors(self, app, mock_container):
        # Arrange
        files = [make_upload_file("badfile.txt", b"bad data")]
        mock_result = MagicMock()
        mock_result.message = "Partial success"
        mock_result.errors = ["badfile.txt: unreadable"]
        mock_result.extractions = [
            MagicMock(
                filename="badfile.txt",
                text_chunks=0,
                structured_data_extracted=False,
                reference_id=None,
                error="Unreadable file",
            )
        ]
        mock_container.document_pipeline_service.process_uploads.return_value = mock_result

        # Act
        async with TestClient(app) as ac:
            response = await ac.post(
                "/upload",
                files=[("files", ("badfile.txt", b"bad data", "text/plain"))],
            )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Partial success"
        assert "badfile.txt: unreadable" in data["errors"]
        assert len(data["extractions"]) == 1
        assert data["extractions"][0]["filename"] == "badfile.txt"
        assert data["extractions"][0]["text_chunks"] == 0
        assert data["extractions"][0]["structured_data_extracted"] is False
        assert data["extractions"][0]["error"] == "Unreadable file"

    async def test_upload_document_empty_files(self, app, mock_container):
        # Arrange
        mock_result = MagicMock()
        mock_result.message = "No files uploaded"
        mock_result.errors = ["No files provided"]
        mock_result.extractions = []
        mock_container.document_pipeline_service.process_uploads.return_value = mock_result

        # Act
        async with TestClient(app) as ac:
            response = await ac.post("/upload", files=[])

        # Assert
        assert response.status_code == 422  # FastAPI validation error for missing required files

@pytest.mark.asyncio
class TestAskQuestion:
    async def test_ask_question_happy_path(self, app, mock_container):
        # Arrange
        query = {"question": "What is the reference ID?", "chat_history": []}
        answer = SourcedAnswer(
            answer="The reference ID is REF123.",
            confidence_score=0.95,
            sources=[],
        )
        mock_container.rag_service.answer_question.return_value = answer

        # Act
        async with TestClient(app) as ac:
            response = await ac.post("/ask", json=query)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "The reference ID is REF123."
        assert data["confidence_score"] == 0.95
        assert data["sources"] == []

    async def test_ask_question_no_sources(self, app, mock_container):
        # Arrange
        query = {"question": "Unknown question?", "chat_history": []}
        answer = SourcedAnswer(
            answer="I cannot find the answer in the provided documents.",
            confidence_score=0.05,
            sources=[],
        )
        mock_container.rag_service.answer_question.return_value = answer

        # Act
        async with TestClient(app) as ac:
            response = await ac.post("/ask", json=query)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "cannot find the answer" in data["answer"].lower()
        assert data["confidence_score"] == 0.05

    async def test_ask_question_internal_error(self, app, mock_container):
        # Arrange
        query = {"question": "Trigger error", "chat_history": []}
        mock_container.rag_service.answer_question.side_effect = Exception("LLM failure")

        # Act
        async with TestClient(app) as ac:
            response = await ac.post("/ask", json=query)

        # Assert
        assert response.status_code == 500
        data = response.json()
        assert data["detail"] == "Internal server error processing question."

@pytest.mark.asyncio
class TestExtractData:
    async def test_extract_data_happy_path(self, app, mock_container):
        # Arrange
        file_content = b"Shipment details: Reference ID REF123"
        filename = "shipment.txt"
        chunks = [
            MagicMock(text="Reference ID REF123", metadata=MagicMock()),
            MagicMock(text="Other details", metadata=MagicMock()),
        ]
        mock_container.ingestion_service.process_file.return_value = chunks
        extraction_response = ExtractionResponse(
            data=ShipmentData(reference_id="REF123"),
            document_id=filename,
        )
        mock_container.extraction_service.extract_data.return_value = extraction_response

        # Act
        async with TestClient(app) as ac:
            response = await ac.post(
                "/extract",
                files=[("file", (filename, file_content, "text/plain"))],
            )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["reference_id"] == "REF123"
        assert data["document_id"] == filename

    async def test_extract_data_no_text(self, app, mock_container):
        # Arrange
        file_content = b""
        filename = "empty.txt"
        mock_container.ingestion_service.process_file.return_value = []
        # ExtractionService.extract_data should not be called if no text
        # So we don't set a return value for it

        # Act
        async with TestClient(app) as ac:
            response = await ac.post(
                "/extract",
                files=[("file", (filename, file_content, "text/plain"))],
            )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["data"] == {}
        assert data["document_id"] == filename

    async def test_extract_data_internal_error(self, app, mock_container):
        # Arrange
        file_content = b"bad content"
        filename = "bad.txt"
        mock_container.ingestion_service.process_file.side_effect = Exception("Parse error")

        # Act
        async with TestClient(app) as ac:
            response = await ac.post(
                "/extract",
                files=[("file", (filename, file_content, "text/plain"))],
            )

        # Assert
        assert response.status_code == 500
        data = response.json()
        assert data["detail"] == "Internal server error during extraction."
