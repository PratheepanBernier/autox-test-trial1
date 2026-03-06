import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

from backend.src.use_cases.document_pipeline import (
    DocumentPipelineService,
    UploadExtractionSummary,
    UploadResult,
)
from backend.src.models.schemas import Chunk
from backend.src.models.extraction_schema import ExtractionResponse

class DummyReferenceData:
    reference_id = "ref-123"

@pytest.fixture
def mock_ingestion_service():
    return Mock()

@pytest.fixture
def mock_vector_store_service():
    return Mock()

@pytest.fixture
def mock_extraction_service():
    return Mock()

@pytest.fixture
def document_pipeline_service(
    mock_ingestion_service, mock_vector_store_service, mock_extraction_service
):
    return DocumentPipelineService(
        ingestion_service=mock_ingestion_service,
        vector_store_service=mock_vector_store_service,
        extraction_service=mock_extraction_service,
    )

def make_upload_file(filename, content: bytes):
    file = AsyncMock()
    file.filename = filename
    file.read = AsyncMock(return_value=content)
    return file

def make_chunk(text, meta=None):
    # Simulate a Chunk object
    return Chunk(text=text, meta=meta or {})

def make_extraction_response(reference_id="ref-123"):
    # Simulate an ExtractionResponse object
    resp = Mock(spec=ExtractionResponse)
    resp.data = DummyReferenceData()
    resp.data.reference_id = reference_id
    return resp

@pytest.mark.asyncio
async def test_process_uploads_happy_path(document_pipeline_service, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    # Arrange
    file1 = make_upload_file("doc1.txt", b"hello world")
    chunks = [make_chunk("chunk1"), make_chunk("chunk2")]
    extraction_response = make_extraction_response("abc-001")
    structured_chunk = make_chunk("structured")

    mock_ingestion_service.process_file.return_value = chunks
    mock_extraction_service.extract_data.return_value = extraction_response
    mock_extraction_service.create_structured_chunk.return_value = structured_chunk

    # Act
    result = await document_pipeline_service.process_uploads([file1])

    # Assert
    assert isinstance(result, UploadResult)
    assert result.processed_count == 1
    assert result.errors == []
    assert len(result.extractions) == 1
    summary = result.extractions[0]
    assert summary.filename == "doc1.txt"
    assert summary.text_chunks == 2
    assert summary.structured_data_extracted is True
    assert summary.reference_id == "abc-001"
    assert summary.error is None

    mock_ingestion_service.process_file.assert_called_once_with(b"hello world", "doc1.txt")
    mock_vector_store_service.add_documents.assert_has_calls([call(chunks), call([structured_chunk])])
    mock_extraction_service.extract_data.assert_called_once_with("chunk1\nchunk2", "doc1.txt")
    mock_extraction_service.create_structured_chunk.assert_called_once_with(extraction_response, "doc1.txt")

@pytest.mark.asyncio
async def test_process_uploads_no_chunks(document_pipeline_service, mock_ingestion_service, mock_vector_store_service):
    # Arrange
    file1 = make_upload_file("empty.txt", b"")
    mock_ingestion_service.process_file.return_value = []

    # Act
    result = await document_pipeline_service.process_uploads([file1])

    # Assert
    assert result.processed_count == 0
    assert result.errors == ["No text extracted from empty.txt"]
    assert result.extractions == []
    mock_ingestion_service.process_file.assert_called_once_with(b"", "empty.txt")
    mock_vector_store_service.add_documents.assert_not_called()

@pytest.mark.asyncio
async def test_process_uploads_ingestion_exception(document_pipeline_service, mock_ingestion_service):
    # Arrange
    file1 = make_upload_file("fail.txt", b"abc")
    mock_ingestion_service.process_file.side_effect = Exception("ingestion failed")

    # Act
    result = await document_pipeline_service.process_uploads([file1])

    # Assert
    assert result.processed_count == 0
    assert result.errors[0].startswith("Error processing fail.txt: ingestion failed")
    assert result.extractions == []

@pytest.mark.asyncio
async def test_process_uploads_extraction_exception(document_pipeline_service, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    # Arrange
    file1 = make_upload_file("doc2.txt", b"data")
    chunks = [make_chunk("chunkA")]
    mock_ingestion_service.process_file.return_value = chunks
    mock_extraction_service.extract_data.side_effect = Exception("extraction error")

    # Act
    result = await document_pipeline_service.process_uploads([file1])

    # Assert
    assert result.processed_count == 1
    assert result.errors == []
    assert len(result.extractions) == 1
    summary = result.extractions[0]
    assert summary.filename == "doc2.txt"
    assert summary.text_chunks == 1
    assert summary.structured_data_extracted is False
    assert summary.reference_id is None
    assert "extraction error" in summary.error

    mock_vector_store_service.add_documents.assert_called_once_with(chunks)
    mock_extraction_service.create_structured_chunk.assert_not_called()

@pytest.mark.asyncio
async def test_process_uploads_multiple_files_mixed_results(document_pipeline_service, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    # Arrange
    file1 = make_upload_file("good.txt", b"good")
    file2 = make_upload_file("bad.txt", b"bad")
    file3 = make_upload_file("empty.txt", b"")
    chunks1 = [make_chunk("c1")]
    chunks2 = [make_chunk("c2")]
    extraction_response1 = make_extraction_response("good-ref")
    structured_chunk1 = make_chunk("structured1")

    # file1: happy path
    # file2: extraction fails
    # file3: no chunks
    mock_ingestion_service.process_file.side_effect = [
        chunks1, chunks2, []
    ]
    mock_extraction_service.extract_data.side_effect = [
        extraction_response1, Exception("extract fail")
    ]
    mock_extraction_service.create_structured_chunk.return_value = structured_chunk1

    # Act
    result = await document_pipeline_service.process_uploads([file1, file2, file3])

    # Assert
    assert result.processed_count == 2
    assert result.errors == ["No text extracted from empty.txt"]
    assert len(result.extractions) == 2

    # file1 summary
    summary1 = result.extractions[0]
    assert summary1.filename == "good.txt"
    assert summary1.text_chunks == 1
    assert summary1.structured_data_extracted is True
    assert summary1.reference_id == "good-ref"
    assert summary1.error is None

    # file2 summary
    summary2 = result.extractions[1]
    assert summary2.filename == "bad.txt"
    assert summary2.text_chunks == 1
    assert summary2.structured_data_extracted is False
    assert summary2.reference_id is None
    assert "extract fail" in summary2.error

@pytest.mark.asyncio
async def test_process_uploads_boundary_empty_file_list(document_pipeline_service):
    # Arrange
    files = []

    # Act
    result = await document_pipeline_service.process_uploads(files)

    # Assert
    assert result.processed_count == 0
    assert result.errors == []
    assert result.extractions == []

@pytest.mark.asyncio
async def test_process_uploads_file_read_exception(document_pipeline_service):
    # Arrange
    file1 = make_upload_file("unreadable.txt", b"irrelevant")
    file1.read.side_effect = Exception("read error")

    # Act
    result = await document_pipeline_service.process_uploads([file1])

    # Assert
    assert result.processed_count == 0
    assert result.errors[0].startswith("Error processing unreadable.txt: read error")
    assert result.extractions == []

def test_upload_result_message_property():
    # Arrange
    result = UploadResult(processed_count=3)
    # Act & Assert
    assert result.message == "Successfully processed 3 documents."
