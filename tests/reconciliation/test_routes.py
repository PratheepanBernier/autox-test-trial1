# source_hash: 7f49f55d5565bc86
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
from backend.src.api.routes import router
from fastapi import FastAPI, UploadFile
from models.schemas import QAQuery, SourcedAnswer
from models.extraction_schema import ExtractionResponse, ShipmentData

app = FastAPI()
app.include_router(router)

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_chunks():
    class Chunk:
        def __init__(self, text):
            self.text = text
    return [Chunk("chunk1 text"), Chunk("chunk2 text")]

@pytest.fixture
def mock_structured_chunk():
    class Chunk:
        def __init__(self, text):
            self.text = text
    return Chunk("structured data chunk")

@pytest.fixture
def mock_extraction_result():
    class Data:
        reference_id = "ref-123"
    class ExtractionResult:
        data = Data()
    return ExtractionResult()

@pytest.fixture
def mock_sourced_answer():
    class SourcedAnswerMock:
        answer = "42"
        confidence_score = 0.99
        sources = ["doc1"]
    return SourcedAnswerMock()

@pytest.fixture
def mock_extraction_response():
    return ExtractionResponse(data=ShipmentData(reference_id="abc"), document_id="file1.pdf")

@pytest.fixture
def upload_file(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("test content")
    file = MagicMock(spec=UploadFile)
    file.filename = "test.txt"
    file.read = AsyncMock(return_value=b"test content")
    return file

@pytest.fixture
def upload_file_empty(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("")
    file = MagicMock(spec=UploadFile)
    file.filename = "empty.txt"
    file.read = AsyncMock(return_value=b"")
    return file

@pytest.mark.asyncio
async def test_upload_document_happy_path(monkeypatch, mock_chunks, mock_structured_chunk, mock_extraction_result, upload_file):
    # Patch services
    monkeypatch.setattr("services.ingestion.ingestion_service.process_file", MagicMock(return_value=mock_chunks))
    monkeypatch.setattr("services.vector_store.vector_store_service.add_documents", MagicMock())
    monkeypatch.setattr("services.extraction.extraction_service.extract_data", MagicMock(return_value=mock_extraction_result))
    monkeypatch.setattr("services.extraction.extraction_service.create_structured_chunk", MagicMock(return_value=mock_structured_chunk))

    # Simulate FastAPI dependency injection for UploadFile
    from fastapi.encoders import jsonable_encoder
    from starlette.datastructures import UploadFile as StarletteUploadFile
    from io import BytesIO

    file = StarletteUploadFile(filename="test.txt", file=BytesIO(b"test content"))
    files = [("files", (file.filename, file.file, "text/plain"))]

    with TestClient(app) as client:
        response = client.post("/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Successfully processed 1 documents."
        assert data["errors"] == []
        assert data["extractions"][0]["filename"] == "test.txt"
        assert data["extractions"][0]["text_chunks"] == 2
        assert data["extractions"][0]["structured_data_extracted"] is True
        assert data["extractions"][0]["reference_id"] == "ref-123"

@pytest.mark.asyncio
async def test_upload_document_no_text_extracted(monkeypatch, upload_file):
    monkeypatch.setattr("services.ingestion.ingestion_service.process_file", MagicMock(return_value=[]))
    monkeypatch.setattr("services.vector_store.vector_store_service.add_documents", MagicMock())
    monkeypatch.setattr("services.extraction.extraction_service.extract_data", MagicMock())
    monkeypatch.setattr("services.extraction.extraction_service.create_structured_chunk", MagicMock())

    from fastapi.encoders import jsonable_encoder
    from starlette.datastructures import UploadFile as StarletteUploadFile
    from io import BytesIO

    file = StarletteUploadFile(filename="empty.txt", file=BytesIO(b""))
    files = [("files", (file.filename, file.file, "text/plain"))]

    with TestClient(app) as client:
        response = client.post("/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Successfully processed 0 documents."
        assert "No text extracted from empty.txt" in data["errors"][0]
        assert data["extractions"] == []

@pytest.mark.asyncio
async def test_upload_document_extraction_error(monkeypatch, mock_chunks, upload_file):
    monkeypatch.setattr("services.ingestion.ingestion_service.process_file", MagicMock(return_value=mock_chunks))
    monkeypatch.setattr("services.vector_store.vector_store_service.add_documents", MagicMock())
    monkeypatch.setattr("services.extraction.extraction_service.extract_data", MagicMock(side_effect=Exception("Extraction failed")))
    monkeypatch.setattr("services.extraction.extraction_service.create_structured_chunk", MagicMock())

    from fastapi.encoders import jsonable_encoder
    from starlette.datastructures import UploadFile as StarletteUploadFile
    from io import BytesIO

    file = StarletteUploadFile(filename="fail.txt", file=BytesIO(b"test content"))
    files = [("files", (file.filename, file.file, "text/plain"))]

    with TestClient(app) as client:
        response = client.post("/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Successfully processed 1 documents."
        assert data["errors"] == []
        assert data["extractions"][0]["filename"] == "fail.txt"
        assert data["extractions"][0]["text_chunks"] == 2
        assert data["extractions"][0]["structured_data_extracted"] is False
        assert "Extraction failed" in data["extractions"][0]["error"]

@pytest.mark.asyncio
async def test_upload_document_file_processing_error(monkeypatch, upload_file):
    monkeypatch.setattr("services.ingestion.ingestion_service.process_file", MagicMock(side_effect=Exception("File error")))
    monkeypatch.setattr("services.vector_store.vector_store_service.add_documents", MagicMock())
    monkeypatch.setattr("services.extraction.extraction_service.extract_data", MagicMock())
    monkeypatch.setattr("services.extraction.extraction_service.create_structured_chunk", MagicMock())

    from fastapi.encoders import jsonable_encoder
    from starlette.datastructures import UploadFile as StarletteUploadFile
    from io import BytesIO

    file = StarletteUploadFile(filename="bad.txt", file=BytesIO(b"bad content"))
    files = [("files", (file.filename, file.file, "text/plain"))]

    with TestClient(app) as client:
        response = client.post("/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Successfully processed 0 documents."
        assert "Error processing bad.txt: File error" in data["errors"][0]
        assert data["extractions"] == []

def test_ask_question_happy_path(monkeypatch, mock_sourced_answer):
    monkeypatch.setattr("services.rag.rag_service.answer_question", MagicMock(return_value=mock_sourced_answer))

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "What is the answer?", "top_k": 1})
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "42"
        assert data["confidence_score"] == 0.99
        assert data["sources"] == ["doc1"]

def test_ask_question_error(monkeypatch):
    monkeypatch.setattr("services.rag.rag_service.answer_question", MagicMock(side_effect=Exception("RAG error")))

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "fail?", "top_k": 1})
        assert response.status_code == 500
        assert response.json()["detail"] == "Internal server error processing question."

@pytest.mark.asyncio
async def test_extract_data_happy_path(monkeypatch, mock_chunks, mock_extraction_response):
    monkeypatch.setattr("services.ingestion.ingestion_service.process_file", MagicMock(return_value=mock_chunks))
    monkeypatch.setattr("services.extraction.extraction_service.extract_data", MagicMock(return_value=mock_extraction_response))

    from fastapi.encoders import jsonable_encoder
    from starlette.datastructures import UploadFile as StarletteUploadFile
    from io import BytesIO

    file = StarletteUploadFile(filename="file1.pdf", file=BytesIO(b"pdf content"))
    files = {"file": (file.filename, file.file, "application/pdf")}

    with TestClient(app) as client:
        response = client.post("/extract", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["reference_id"] == "abc"
        assert data["document_id"] == "file1.pdf"

@pytest.mark.asyncio
async def test_extract_data_no_text(monkeypatch):
    monkeypatch.setattr("services.ingestion.ingestion_service.process_file", MagicMock(return_value=[]))
    monkeypatch.setattr("services.extraction.extraction_service.extract_data", MagicMock())

    from fastapi.encoders import jsonable_encoder
    from starlette.datastructures import UploadFile as StarletteUploadFile
    from io import BytesIO

    file = StarletteUploadFile(filename="empty.pdf", file=BytesIO(b""))
    files = {"file": (file.filename, file.file, "application/pdf")}

    with TestClient(app) as client:
        response = client.post("/extract", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["data"] == {}
        assert data["document_id"] == "empty.pdf"

@pytest.mark.asyncio
async def test_extract_data_error(monkeypatch, mock_chunks):
    monkeypatch.setattr("services.ingestion.ingestion_service.process_file", MagicMock(return_value=mock_chunks))
    monkeypatch.setattr("services.extraction.extraction_service.extract_data", MagicMock(side_effect=Exception("Extraction error")))

    from fastapi.encoders import jsonable_encoder
    from starlette.datastructures import UploadFile as StarletteUploadFile
    from io import BytesIO

    file = StarletteUploadFile(filename="fail.pdf", file=BytesIO(b"fail content"))
    files = {"file": (file.filename, file.file, "application/pdf")}

    with TestClient(app) as client:
        response = client.post("/extract", files=files)
        assert response.status_code == 500
        assert response.json()["detail"] == "Internal server error during extraction."

def test_ping_and_reconciliation(client):
    # /ping should always return {"status": "pong"}
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert resp.json() == {"status": "pong"}

    # Reconciliation: /ping and /ping called twice should be identical
    resp2 = client.get("/ping")
    assert resp.json() == resp2.json()

def test_upload_and_extract_reconciliation(monkeypatch, mock_chunks, mock_structured_chunk, mock_extraction_result, mock_extraction_response):
    # Patch for upload
    monkeypatch.setattr("services.ingestion.ingestion_service.process_file", MagicMock(return_value=mock_chunks))
    monkeypatch.setattr("services.vector_store.vector_store_service.add_documents", MagicMock())
    monkeypatch.setattr("services.extraction.extraction_service.extract_data", MagicMock(return_value=mock_extraction_result))
    monkeypatch.setattr("services.extraction.extraction_service.create_structured_chunk", MagicMock(return_value=mock_structured_chunk))

    from fastapi.encoders import jsonable_encoder
    from starlette.datastructures import UploadFile as StarletteUploadFile
    from io import BytesIO

    file = StarletteUploadFile(filename="rec.pdf", file=BytesIO(b"rec content"))
    files = [("files", (file.filename, file.file, "application/pdf"))]

    with TestClient(app) as client:
        upload_resp = client.post("/upload", files=files)
        assert upload_resp.status_code == 200
        upload_data = upload_resp.json()
        # Now patch extract_data to return a compatible ExtractionResponse
        monkeypatch.setattr("services.ingestion.ingestion_service.process_file", MagicMock(return_value=mock_chunks))
        monkeypatch.setattr("services.extraction.extraction_service.extract_data", MagicMock(return_value=mock_extraction_response))
        extract_files = {"file": (file.filename, BytesIO(b"rec content"), "application/pdf")}
        extract_resp = client.post("/extract", files=extract_files)
        assert extract_resp.status_code == 200
        extract_data = extract_resp.json()
        # Reconciliation: reference_id in upload extractions and extract response should match if extraction logic is consistent
        upload_ref_id = upload_data["extractions"][0]["reference_id"]
        extract_ref_id = extract_data["data"]["reference_id"]
        assert upload_ref_id == "ref-123"
        assert extract_ref_id == "abc" or upload_ref_id == extract_ref_id  # If mock_extraction_result and mock_extraction_response are aligned

def test_ask_and_upload_reconciliation(monkeypatch, mock_sourced_answer, mock_chunks, mock_structured_chunk, mock_extraction_result):
    # Patch upload
    monkeypatch.setattr("services.ingestion.ingestion_service.process_file", MagicMock(return_value=mock_chunks))
    monkeypatch.setattr("services.vector_store.vector_store_service.add_documents", MagicMock())
    monkeypatch.setattr("services.extraction.extraction_service.extract_data", MagicMock(return_value=mock_extraction_result))
    monkeypatch.setattr("services.extraction.extraction_service.create_structured_chunk", MagicMock(return_value=mock_structured_chunk))
    # Patch ask
    monkeypatch.setattr("services.rag.rag_service.answer_question", MagicMock(return_value=mock_sourced_answer))

    from fastapi.encoders import jsonable_encoder
    from starlette.datastructures import UploadFile as StarletteUploadFile
    from io import BytesIO

    file = StarletteUploadFile(filename="qa.pdf", file=BytesIO(b"qa content"))
    files = [("files", (file.filename, file.file, "application/pdf"))]

    with TestClient(app) as client:
        upload_resp = client.post("/upload", files=files)
        assert upload_resp.status_code == 200
        ask_resp = client.post("/ask", json={"question": "What is the answer?", "top_k": 1})
        assert ask_resp.status_code == 200
        # Reconciliation: upload and ask are independent, but after upload, ask should succeed and return the mocked answer
        assert ask_resp.json()["answer"] == "42"
        assert ask_resp.json()["confidence_score"] == 0.99
        assert ask_resp.json()["sources"] == ["doc1"]
