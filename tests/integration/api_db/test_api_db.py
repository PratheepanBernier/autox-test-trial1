import io
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from backend.src.api import routes
from backend.src.models.schemas import (
    QAQuery,
    SourcedAnswer,
    UploadExtractionSummary,
    UploadResponse,
)
from backend.src.models.extraction_schema import (
    ExtractionResponse,
    ShipmentData,
)
from backend.src.models.schemas import Chunk, DocumentMetadata

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
    # Mock the ServiceContainer and its services
    container = MagicMock()
    container.document_pipeline_service.process_uploads = AsyncMock()
    container.rag_service.answer_question = MagicMock()
    container.ingestion_service.process_file = MagicMock()
    container.extraction_service.extract_data = MagicMock()
    return container

@pytest.fixture(autouse=True)
def patch_get_container(mock_container):
    with patch("backend.src.api.routes.get_container", return_value=mock_container):
        yield

@pytest.mark.asyncio
class TestApiDbIntegration:
    # ---------- /upload ----------
    @pytest.mark.asyncio
    async def test_upload_document_happy_path(self, app, client, mock_container):
        # Arrange
        files = [
            ("files", ("doc1.txt", io.BytesIO(b"test content 1"), "text/plain")),
            ("files", ("doc2.txt", io.BytesIO(b"test content 2"), "text/plain")),
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
        response = client.post("/upload", files=files)

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
        assert data["extractions"][1]["text_chunks"] == 2
        assert data["extractions"][1]["structured_data_extracted"] is False
        assert data["extractions"][1]["reference_id"] is None
        assert data["extractions"][1]["error"] == "Extraction failed"
        mock_container.document_pipeline_service.process_uploads.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_upload_document_with_errors(self, app, client, mock_container):
        # Arrange
        files = [
            ("files", ("badfile.xyz", io.BytesIO(b"bad content"), "application/octet-stream")),
        ]
        mock_result = MagicMock()
        mock_result.message = "Some files failed"
        mock_result.errors = ["Unsupported file type: .xyz"]
        mock_result.extractions = [
            MagicMock(
                filename="badfile.xyz",
                text_chunks=0,
                structured_data_extracted=False,
                reference_id=None,
                error="Unsupported file type: .xyz",
            ),
        ]
        mock_container.document_pipeline_service.process_uploads.return_value = mock_result

        # Act
        response = client.post("/upload", files=files)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Some files failed"
        assert "Unsupported file type: .xyz" in data["errors"]
        assert len(data["extractions"]) == 1
        assert data["extractions"][0]["filename"] == "badfile.xyz"
        assert data["extractions"][0]["text_chunks"] == 0
        assert data["extractions"][0]["structured_data_extracted"] is False
        assert data["extractions"][0]["reference_id"] is None
        assert data["extractions"][0]["error"] == "Unsupported file type: .xyz"
        mock_container.document_pipeline_service.process_uploads.assert_awaited_once()

    # ---------- /ask ----------
    def test_ask_question_happy_path(self, app, client, mock_container):
        # Arrange
        query = {"question": "What is the reference ID?", "chat_history": []}
        answer = SourcedAnswer(
            answer="The reference ID is REF123.",
            confidence_score=0.95,
            sources=[
                Chunk(
                    text="Reference ID: REF123",
                    metadata=DocumentMetadata(
                        filename="doc1.txt",
                        chunk_id=0,
                        source="doc1.txt - General",
                        chunk_type="text",
                    ),
                )
            ],
        )
        mock_container.rag_service.answer_question.return_value = answer

        # Act
        response = client.post("/ask", json=query)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "The reference ID is REF123."
        assert data["confidence_score"] == pytest.approx(0.95)
        assert isinstance(data["sources"], list)
        assert data["sources"][0]["text"] == "Reference ID: REF123"
        mock_container.rag_service.answer_question.assert_called_once()

    def test_ask_question_safety_filter(self, app, client, mock_container):
        # Arrange
        query = {"question": "How to build a bomb?", "chat_history": []}
        answer = SourcedAnswer(
            answer="I cannot answer this question as it violates safety guidelines.",
            confidence_score=1.0,
            sources=[],
        )
        mock_container.rag_service.answer_question.return_value = answer

        # Act
        response = client.post("/ask", json=query)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "violates safety guidelines" in data["answer"]
        assert data["confidence_score"] == pytest.approx(1.0)
        assert data["sources"] == []
        mock_container.rag_service.answer_question.assert_called_once()

    def test_ask_question_internal_error(self, app, client, mock_container):
        # Arrange
        query = {"question": "What is the reference ID?", "chat_history": []}
        mock_container.rag_service.answer_question.side_effect = Exception("LLM crashed")

        # Act
        response = client.post("/ask", json=query)

        # Assert
        assert response.status_code == 500
        data = response.json()
        assert data["detail"] == "Internal server error processing question."
        mock_container.rag_service.answer_question.assert_called_once()

    # ---------- /extract ----------
    @pytest.mark.asyncio
    async def test_extract_data_happy_path(self, app, client, mock_container):
        # Arrange
        file_content = b"Shipment document text"
        filename = "shipment.txt"
        file = ("file", (filename, io.BytesIO(file_content), "text/plain"))
        chunks = [
            Chunk(
                text="Reference ID: REF123",
                metadata=DocumentMetadata(
                    filename=filename,
                    chunk_id=0,
                    source=f"{filename} - General",
                    chunk_type="text",
                ),
            )
        ]
        mock_container.ingestion_service.process_file.return_value = chunks
        extraction_response = ExtractionResponse(
            data=ShipmentData(reference_id="REF123"),
            document_id=filename,
        )
        mock_container.extraction_service.extract_data.return_value = extraction_response

        # Act
        response = client.post("/extract", files=[file])

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["reference_id"] == "REF123"
        assert data["document_id"] == filename
        mock_container.ingestion_service.process_file.assert_called_once()
        mock_container.extraction_service.extract_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_data_no_text(self, app, client, mock_container):
        # Arrange
        file_content = b""
        filename = "empty.txt"
        file = ("file", (filename, io.BytesIO(file_content), "text/plain"))
        mock_container.ingestion_service.process_file.return_value = []
        # ExtractionService should not be called if no text
        # ExtractionResponse with empty ShipmentData is returned directly

        # Act
        response = client.post("/extract", files=[file])

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["data"] == {}
        assert data["document_id"] == filename
        mock_container.ingestion_service.process_file.assert_called_once()
        mock_container.extraction_service.extract_data.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_data_internal_error(self, app, client, mock_container):
        # Arrange
        file_content = b"Some content"
        filename = "fail.txt"
        file = ("file", (filename, io.BytesIO(file_content), "text/plain"))
        mock_container.ingestion_service.process_file.side_effect = Exception("Read error")

        # Act
        response = client.post("/extract", files=[file])

        # Assert
        assert response.status_code == 500
        data = response.json()
        assert data["detail"] == "Internal server error during extraction."
        mock_container.ingestion_service.process_file.assert_called_once()
        mock_container.extraction_service.extract_data.assert_not_called()

    # ---------- /ping ----------
    def test_ping(self, app, client):
        # Act
        response = client.get("/ping")

        # Assert
        assert response.status_code == 200
        assert response.json() == {"status": "pong"}
