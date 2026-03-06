import io
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.routes import router as api_router
from models.schemas import QAQuery, SourcedAnswer
from models.extraction_schema import ExtractionResponse, ShipmentData

@pytest.fixture(scope="module")
def test_app():
    app = FastAPI()
    app.include_router(api_router)
    return app

@pytest.fixture
def client(test_app):
    return TestClient(test_app)

def test_upload_document_success(client):
    # Patch all service dependencies for deterministic integration
    with patch("services.ingestion.ingestion_service") as mock_ingestion, \
         patch("services.vector_store.vector_store_service") as mock_vector_store, \
         patch("services.extraction.extraction_service") as mock_extraction:

        # Prepare deterministic chunk and extraction response
        class DummyMetadata:
            def model_dump(self):
                return {
                    "filename": "test.txt",
                    "chunk_id": 0,
                    "source": "test.txt - General",
                    "chunk_type": "text"
                }
        dummy_chunk = MagicMock()
        dummy_chunk.text = "This is a test chunk."
        dummy_chunk.metadata = DummyMetadata()

        mock_ingestion.process_file.return_value = [dummy_chunk]
        mock_vector_store.add_documents.return_value = None

        dummy_extraction = ExtractionResponse(
            data=ShipmentData(reference_id="REF123"),
            document_id="test.txt"
        )
        mock_extraction.extract_data.return_value = dummy_extraction

        dummy_structured_chunk = MagicMock()
        dummy_structured_chunk.text = "Structured data chunk"
        dummy_structured_chunk.metadata = DummyMetadata()
        mock_extraction.create_structured_chunk.return_value = dummy_structured_chunk

        # Prepare file upload
        file_content = b"dummy file content"
        files = {
            "files": ("test.txt", io.BytesIO(file_content), "text/plain")
        }

        response = client.post("/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Successfully processed 1 documents."
        assert data["errors"] == []
        assert len(data["extractions"]) == 1
        extraction = data["extractions"][0]
        assert extraction["filename"] == "test.txt"
        assert extraction["text_chunks"] == 1
        assert extraction["structured_data_extracted"] is True
        assert extraction["reference_id"] == "REF123"

def test_upload_document_no_chunks(client):
    with patch("services.ingestion.ingestion_service") as mock_ingestion, \
         patch("services.vector_store.vector_store_service") as mock_vector_store, \
         patch("services.extraction.extraction_service") as mock_extraction:

        mock_ingestion.process_file.return_value = []
        file_content = b"dummy file content"
        files = {
            "files": ("empty.txt", io.BytesIO(file_content), "text/plain")
        }
        response = client.post("/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Successfully processed 0 documents."
        assert len(data["errors"]) == 1
        assert "No text extracted from empty.txt" in data["errors"][0]
        assert data["extractions"] == []

def test_ask_question_success(client):
    with patch("services.rag.rag_service") as mock_rag:
        mock_answer = SourcedAnswer(
            answer="The answer is 42.",
            confidence_score=0.95,
            sources=[]
        )
        mock_rag.answer_question.return_value = mock_answer

        payload = {
            "question": "What is the answer?",
            "chat_history": []
        }
        response = client.post("/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "The answer is 42."
        assert data["confidence_score"] == 0.95
        assert data["sources"] == []

def test_ask_question_error(client):
    with patch("services.rag.rag_service") as mock_rag:
        mock_rag.answer_question.side_effect = Exception("LLM failure")
        payload = {
            "question": "What is the answer?",
            "chat_history": []
        }
        response = client.post("/ask", json=payload)
        assert response.status_code == 500
        assert response.json()["detail"] == "Internal server error processing question."

def test_extract_data_success(client):
    with patch("services.ingestion.ingestion_service") as mock_ingestion, \
         patch("services.extraction.extraction_service") as mock_extraction:

        class DummyMetadata:
            def model_dump(self):
                return {
                    "filename": "extract.txt",
                    "chunk_id": 0,
                    "source": "extract.txt - General",
                    "chunk_type": "text"
                }
        dummy_chunk = MagicMock()
        dummy_chunk.text = "Extracted text."
        dummy_chunk.metadata = DummyMetadata()
        mock_ingestion.process_file.return_value = [dummy_chunk]

        dummy_extraction = ExtractionResponse(
            data=ShipmentData(reference_id="EXTRACT123"),
            document_id="extract.txt"
        )
        mock_extraction.extract_data.return_value = dummy_extraction

        file_content = b"extract file content"
        files = {
            "file": ("extract.txt", io.BytesIO(file_content), "text/plain")
        }
        response = client.post("/extract", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["reference_id"] == "EXTRACT123"
        assert data["document_id"] == "extract.txt"

def test_extract_data_no_text(client):
    with patch("services.ingestion.ingestion_service") as mock_ingestion, \
         patch("services.extraction.extraction_service") as mock_extraction:

        mock_ingestion.process_file.return_value = []
        file_content = b"empty file"
        files = {
            "file": ("empty.txt", io.BytesIO(file_content), "text/plain")
        }
        response = client.post("/extract", files=files)
        assert response.status_code == 200
        data = response.json()
        # Should return empty ShipmentData
        assert data["data"] == {}
        assert data["document_id"] == "empty.txt"

def test_extract_data_error(client):
    with patch("services.ingestion.ingestion_service") as mock_ingestion, \
         patch("services.extraction.extraction_service") as mock_extraction:

        mock_ingestion.process_file.side_effect = Exception("Extraction failed")
        file_content = b"fail file"
        files = {
            "file": ("fail.txt", io.BytesIO(file_content), "text/plain")
        }
        response = client.post("/extract", files=files)
        assert response.status_code == 500
        assert response.json()["detail"] == "Internal server error during extraction."

def test_ping(client):
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "pong"}
