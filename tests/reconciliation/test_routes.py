import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
from backend.src.api.routes import router
from fastapi import FastAPI, UploadFile
from models.schemas import SourcedAnswer, QAQuery
from models.extraction_schema import ExtractionResponse, ShipmentData

app = FastAPI()
app.include_router(router)

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def fake_chunks():
    class Chunk:
        def __init__(self, text):
            self.text = text
    return [Chunk("chunk1 text"), Chunk("chunk2 text")]

@pytest.fixture
def fake_structured_chunk():
    class Chunk:
        def __init__(self, text):
            self.text = text
    return Chunk("structured data chunk")

@pytest.fixture
def fake_extraction_result():
    class Data:
        reference_id = "ref-123"
    class ExtractionResult:
        data = Data()
    return ExtractionResult()

@pytest.fixture
def fake_extraction_response():
    return ExtractionResponse(data=ShipmentData(reference_id="ref-123"), document_id="file1.pdf")

@pytest.fixture
def fake_sourced_answer():
    return SourcedAnswer(answer="42", sources=["doc1"], confidence_score=0.99)

def make_upload_file(filename, content):
    file = MagicMock(spec=UploadFile)
    file.filename = filename
    file.read = AsyncMock(return_value=content)
    return file

@pytest.mark.asyncio
@patch("services.ingestion.ingestion_service.process_file")
@patch("services.vector_store.vector_store_service.add_documents")
@patch("services.extraction.extraction_service.extract_data")
@patch("services.extraction.extraction_service.create_structured_chunk")
async def test_upload_document_happy_path(
    mock_create_structured_chunk,
    mock_extract_data,
    mock_add_documents,
    mock_process_file,
    fake_chunks,
    fake_structured_chunk,
    fake_extraction_result,
    client
):
    mock_process_file.return_value = fake_chunks
    mock_add_documents.return_value = None
    mock_extract_data.return_value = fake_extraction_result
    mock_create_structured_chunk.return_value = fake_structured_chunk

    file1 = make_upload_file("file1.pdf", b"file1 content")
    file2 = make_upload_file("file2.pdf", b"file2 content")

    # Use TestClient to call the endpoint
    with patch("fastapi.UploadFile", autospec=True):
        response = await app.router.routes[0].endpoint([file1, file2])

    assert response["message"] == "Successfully processed 2 documents."
    assert response["errors"] == []
    assert len(response["extractions"]) == 2
    for extraction in response["extractions"]:
        assert extraction["structured_data_extracted"] is True
        assert extraction["reference_id"] == "ref-123"
        assert extraction["text_chunks"] == 2

@pytest.mark.asyncio
@patch("services.ingestion.ingestion_service.process_file")
@patch("services.vector_store.vector_store_service.add_documents")
@patch("services.extraction.extraction_service.extract_data")
@patch("services.extraction.extraction_service.create_structured_chunk")
async def test_upload_document_no_text_extracted(
    mock_create_structured_chunk,
    mock_extract_data,
    mock_add_documents,
    mock_process_file,
    client
):
    mock_process_file.return_value = []
    file1 = make_upload_file("file1.pdf", b"file1 content")

    with patch("fastapi.UploadFile", autospec=True):
        response = await app.router.routes[0].endpoint([file1])

    assert response["message"] == "Successfully processed 0 documents."
    assert "No text extracted from file1.pdf" in response["errors"][0]
    assert response["extractions"] == []

@pytest.mark.asyncio
@patch("services.ingestion.ingestion_service.process_file")
@patch("services.vector_store.vector_store_service.add_documents")
@patch("services.extraction.extraction_service.extract_data")
@patch("services.extraction.extraction_service.create_structured_chunk")
async def test_upload_document_extraction_error(
    mock_create_structured_chunk,
    mock_extract_data,
    mock_add_documents,
    mock_process_file,
    fake_chunks,
    client
):
    mock_process_file.return_value = fake_chunks
    mock_add_documents.return_value = None
    mock_extract_data.side_effect = Exception("Extraction failed")

    file1 = make_upload_file("file1.pdf", b"file1 content")

    with patch("fastapi.UploadFile", autospec=True):
        response = await app.router.routes[0].endpoint([file1])

    assert response["message"] == "Successfully processed 1 documents."
    assert response["errors"] == []
    assert len(response["extractions"]) == 1
    extraction = response["extractions"][0]
    assert extraction["structured_data_extracted"] is False
    assert "Extraction failed" in extraction["error"]
    assert extraction["filename"] == "file1.pdf"
    assert extraction["text_chunks"] == 2

@pytest.mark.asyncio
@patch("services.ingestion.ingestion_service.process_file")
@patch("services.vector_store.vector_store_service.add_documents")
@patch("services.extraction.extraction_service.extract_data")
@patch("services.extraction.extraction_service.create_structured_chunk")
async def test_upload_document_file_processing_error(
    mock_create_structured_chunk,
    mock_extract_data,
    mock_add_documents,
    mock_process_file,
    client
):
    mock_process_file.side_effect = Exception("File corrupted")
    file1 = make_upload_file("file1.pdf", b"file1 content")

    with patch("fastapi.UploadFile", autospec=True):
        response = await app.router.routes[0].endpoint([file1])

    assert response["message"] == "Successfully processed 0 documents."
    assert "Error processing file1.pdf: File corrupted" in response["errors"][0]
    assert response["extractions"] == []

@pytest.mark.asyncio
@patch("services.rag.rag_service.answer_question")
async def test_ask_question_happy_path(mock_answer_question, fake_sourced_answer, client):
    mock_answer_question.return_value = fake_sourced_answer
    query = QAQuery(question="What is the answer?", top_k=1)
    response = await app.router.routes[1].endpoint(query)
    assert response.answer == "42"
    assert response.sources == ["doc1"]
    assert response.confidence_score == 0.99

@pytest.mark.asyncio
@patch("services.rag.rag_service.answer_question")
async def test_ask_question_error_handling(mock_answer_question, client):
    mock_answer_question.side_effect = Exception("RAG failure")
    query = QAQuery(question="What is the answer?", top_k=1)
    with pytest.raises(Exception) as excinfo:
        await app.router.routes[1].endpoint(query)
    assert "Internal server error processing question." in str(excinfo.value)

@pytest.mark.asyncio
@patch("services.ingestion.ingestion_service.process_file")
@patch("services.extraction.extraction_service.extract_data")
async def test_extract_data_happy_path(
    mock_extract_data,
    mock_process_file,
    fake_chunks,
    fake_extraction_response,
    client
):
    mock_process_file.return_value = fake_chunks
    mock_extract_data.return_value = fake_extraction_response
    file = make_upload_file("file1.pdf", b"file1 content")
    response = await app.router.routes[2].endpoint(file)
    assert isinstance(response, ExtractionResponse)
    assert response.data.reference_id == "ref-123"
    assert response.document_id == "file1.pdf"

@pytest.mark.asyncio
@patch("services.ingestion.ingestion_service.process_file")
@patch("services.extraction.extraction_service.extract_data")
async def test_extract_data_no_text_extracted(
    mock_extract_data,
    mock_process_file,
    client
):
    mock_process_file.return_value = []
    file = make_upload_file("file1.pdf", b"file1 content")
    response = await app.router.routes[2].endpoint(file)
    assert isinstance(response, ExtractionResponse)
    assert response.data.reference_id is None
    assert response.document_id == "file1.pdf"

@pytest.mark.asyncio
@patch("services.ingestion.ingestion_service.process_file")
@patch("services.extraction.extraction_service.extract_data")
async def test_extract_data_error_handling(
    mock_extract_data,
    mock_process_file,
    fake_chunks,
    client
):
    mock_process_file.return_value = fake_chunks
    mock_extract_data.side_effect = Exception("Extraction failed")
    file = make_upload_file("file1.pdf", b"file1 content")
    with pytest.raises(Exception) as excinfo:
        await app.router.routes[2].endpoint(file)
    assert "Internal server error during extraction." in str(excinfo.value)

def test_ping_endpoint(client):
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "pong"}

# Reconciliation tests

@pytest.mark.asyncio
@patch("services.ingestion.ingestion_service.process_file")
@patch("services.vector_store.vector_store_service.add_documents")
@patch("services.extraction.extraction_service.extract_data")
@patch("services.extraction.extraction_service.create_structured_chunk")
async def test_upload_and_extract_consistency(
    mock_create_structured_chunk,
    mock_extract_data,
    mock_add_documents,
    mock_process_file,
    fake_chunks,
    fake_structured_chunk,
    fake_extraction_result,
    fake_extraction_response,
    client
):
    # Setup mocks for upload
    mock_process_file.return_value = fake_chunks
    mock_add_documents.return_value = None
    mock_extract_data.return_value = fake_extraction_result
    mock_create_structured_chunk.return_value = fake_structured_chunk

    file = make_upload_file("file1.pdf", b"file1 content")

    # Call upload endpoint
    upload_response = await app.router.routes[0].endpoint([file])

    # Now, setup extract endpoint to return ExtractionResponse with same reference_id
    mock_extract_data.return_value = fake_extraction_response
    mock_process_file.return_value = fake_chunks

    extract_response = await app.router.routes[2].endpoint(file)

    # Reconciliation: reference_id in upload extraction matches extract endpoint
    upload_ref_id = upload_response["extractions"][0]["reference_id"]
    extract_ref_id = extract_response.data.reference_id
    assert upload_ref_id == extract_ref_id

@pytest.mark.asyncio
@patch("services.rag.rag_service.answer_question")
async def test_ask_question_consistency(mock_answer_question, fake_sourced_answer, client):
    # Simulate two equivalent queries
    mock_answer_question.return_value = fake_sourced_answer
    query1 = QAQuery(question="What is the answer?", top_k=1)
    query2 = QAQuery(question="What is the answer?", top_k=1)
    response1 = await app.router.routes[1].endpoint(query1)
    response2 = await app.router.routes[1].endpoint(query2)
    assert response1.answer == response2.answer
    assert response1.sources == response2.sources
    assert response1.confidence_score == response2.confidence_score

@pytest.mark.asyncio
@patch("services.ingestion.ingestion_service.process_file")
@patch("services.extraction.extraction_service.extract_data")
async def test_extract_and_upload_extraction_error_consistency(
    mock_extract_data,
    mock_process_file,
    fake_chunks,
    client
):
    # Both endpoints should report extraction error in a consistent way
    mock_process_file.return_value = fake_chunks
    mock_extract_data.side_effect = Exception("Extraction failed")

    file = make_upload_file("file1.pdf", b"file1 content")

    # Upload endpoint
    with patch("services.vector_store.vector_store_service.add_documents"):
        upload_response = await app.router.routes[0].endpoint([file])
    upload_extraction = upload_response["extractions"][0]
    assert upload_extraction["structured_data_extracted"] is False
    assert "Extraction failed" in upload_extraction["error"]

    # Extract endpoint
    with pytest.raises(Exception) as excinfo:
        await app.router.routes[2].endpoint(file)
    assert "Internal server error during extraction." in str(excinfo.value)
