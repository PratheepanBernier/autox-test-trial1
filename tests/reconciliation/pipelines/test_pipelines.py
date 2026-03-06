import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, call

from backend.src.use_cases.document_pipeline import (
    DocumentPipelineService,
    UploadExtractionSummary,
    UploadResult,
)
from backend.src.models.schemas import Chunk
from backend.src.models.extraction_schema import ExtractionResponse

class DummyUploadFile:
    def __init__(self, filename, content):
        self.filename = filename
        self._content = content
        self._read_called = False

    async def read(self):
        if self._read_called:
            return b""
        self._read_called = True
        return self._content

@pytest.fixture
def dummy_chunks():
    return [
        Chunk(text="chunk1", metadata={"page": 1}),
        Chunk(text="chunk2", metadata={"page": 2}),
    ]

@pytest.fixture
def dummy_structured_chunk():
    return Chunk(text="structured", metadata={"structured": True})

@pytest.fixture
def dummy_extraction_response():
    class DummyData:
        reference_id = "ref-123"
    return ExtractionResponse(data=DummyData())

@pytest.fixture
def ingestion_service(dummy_chunks):
    svc = MagicMock()
    svc.process_file = MagicMock(return_value=dummy_chunks)
    return svc

@pytest.fixture
def vector_store_service():
    svc = MagicMock()
    svc.add_documents = MagicMock()
    return svc

@pytest.fixture
def extraction_service(dummy_extraction_response, dummy_structured_chunk):
    svc = MagicMock()
    svc.extract_data = MagicMock(return_value=dummy_extraction_response)
    svc.create_structured_chunk = MagicMock(return_value=dummy_structured_chunk)
    return svc

@pytest.mark.asyncio
async def test_process_uploads_happy_path(
    ingestion_service,
    vector_store_service,
    extraction_service,
    dummy_chunks,
    dummy_structured_chunk,
    dummy_extraction_response,
):
    # Arrange
    files = [
        DummyUploadFile("file1.txt", b"file1-content"),
        DummyUploadFile("file2.txt", b"file2-content"),
    ]
    pipeline = DocumentPipelineService(
        ingestion_service, vector_store_service, extraction_service
    )

    # Act
    result = await pipeline.process_uploads(files)

    # Assert
    # Record counts
    assert result.processed_count == 2
    assert len(result.errors) == 0
    assert len(result.extractions) == 2

    # Aggregate consistency: each file's extraction summary matches chunk count and extraction
    for i, summary in enumerate(result.extractions):
        assert summary.filename == f"file{i+1}.txt"
        assert summary.text_chunks == len(dummy_chunks)
        assert summary.structured_data_extracted is True
        assert summary.reference_id == dummy_extraction_response.data.reference_id
        assert summary.error is None

    # Vector store should receive correct calls: 2 files * 2 chunks + 2 structured chunks
    expected_calls = [
        call(dummy_chunks),
        call([dummy_structured_chunk]),
        call(dummy_chunks),
        call([dummy_structured_chunk]),
    ]
    assert vector_store_service.add_documents.call_args_list == expected_calls

    # No duplicate or missing records: each file processed, each chunk and structured chunk added
    assert ingestion_service.process_file.call_count == 2
    assert extraction_service.extract_data.call_count == 2
    assert extraction_service.create_structured_chunk.call_count == 2

@pytest.mark.asyncio
async def test_process_uploads_no_text_extracted(
    ingestion_service,
    vector_store_service,
    extraction_service,
):
    # Arrange
    ingestion_service.process_file.return_value = []
    files = [DummyUploadFile("emptyfile.txt", b"")]
    pipeline = DocumentPipelineService(
        ingestion_service, vector_store_service, extraction_service
    )

    # Act
    result = await pipeline.process_uploads(files)

    # Assert
    assert result.processed_count == 0
    assert len(result.errors) == 1
    assert "No text extracted from emptyfile.txt" in result.errors[0]
    assert len(result.extractions) == 0
    vector_store_service.add_documents.assert_not_called()
    extraction_service.extract_data.assert_not_called()
    extraction_service.create_structured_chunk.assert_not_called()

@pytest.mark.asyncio
async def test_process_uploads_extraction_failure(
    ingestion_service,
    vector_store_service,
    extraction_service,
    dummy_chunks,
):
    # Arrange
    extraction_service.extract_data.side_effect = Exception("Extraction failed")
    files = [DummyUploadFile("failfile.txt", b"fail-content")]
    pipeline = DocumentPipelineService(
        ingestion_service, vector_store_service, extraction_service
    )

    # Act
    result = await pipeline.process_uploads(files)

    # Assert
    assert result.processed_count == 1
    assert len(result.errors) == 0
    assert len(result.extractions) == 1
    summary = result.extractions[0]
    assert summary.filename == "failfile.txt"
    assert summary.text_chunks == len(dummy_chunks)
    assert summary.structured_data_extracted is False
    assert summary.reference_id is None
    assert "Extraction failed" in summary.error

    # Only text chunks added, not structured chunk
    vector_store_service.add_documents.assert_called_once_with(dummy_chunks)
    extraction_service.create_structured_chunk.assert_not_called()

@pytest.mark.asyncio
async def test_process_uploads_ingestion_failure(
    ingestion_service,
    vector_store_service,
    extraction_service,
):
    # Arrange
    ingestion_service.process_file.side_effect = Exception("Ingestion error")
    files = [DummyUploadFile("badfile.txt", b"bad-content")]
    pipeline = DocumentPipelineService(
        ingestion_service, vector_store_service, extraction_service
    )

    # Act
    result = await pipeline.process_uploads(files)

    # Assert
    assert result.processed_count == 0
    assert len(result.errors) == 1
    assert "Error processing badfile.txt: Ingestion error" in result.errors[0]
    assert len(result.extractions) == 0
    vector_store_service.add_documents.assert_not_called()
    extraction_service.extract_data.assert_not_called()
    extraction_service.create_structured_chunk.assert_not_called()

@pytest.mark.asyncio
async def test_process_uploads_duplicate_filenames(
    ingestion_service,
    vector_store_service,
    extraction_service,
    dummy_chunks,
    dummy_structured_chunk,
    dummy_extraction_response,
):
    # Arrange
    files = [
        DummyUploadFile("dup.txt", b"content1"),
        DummyUploadFile("dup.txt", b"content2"),
    ]
    pipeline = DocumentPipelineService(
        ingestion_service, vector_store_service, extraction_service
    )

    # Act
    result = await pipeline.process_uploads(files)

    # Assert
    # Both files processed independently, even with same filename
    assert result.processed_count == 2
    assert len(result.errors) == 0
    assert len(result.extractions) == 2
    filenames = [summary.filename for summary in result.extractions]
    assert filenames == ["dup.txt", "dup.txt"]

@pytest.mark.asyncio
async def test_process_uploads_boundary_empty_file_list(
    ingestion_service,
    vector_store_service,
    extraction_service,
):
    # Arrange
    files = []
    pipeline = DocumentPipelineService(
        ingestion_service, vector_store_service, extraction_service
    )

    # Act
    result = await pipeline.process_uploads(files)

    # Assert
    assert result.processed_count == 0
    assert len(result.errors) == 0
    assert len(result.extractions) == 0
    ingestion_service.process_file.assert_not_called()
    vector_store_service.add_documents.assert_not_called()
    extraction_service.extract_data.assert_not_called()
    extraction_service.create_structured_chunk.assert_not_called()

@pytest.mark.asyncio
async def test_process_uploads_transformation_correctness(
    ingestion_service,
    vector_store_service,
    extraction_service,
    dummy_chunks,
    dummy_structured_chunk,
    dummy_extraction_response,
):
    # Arrange
    # Simulate transformation: structured chunk must reference extraction data
    def create_structured_chunk(extraction, filename):
        assert extraction == dummy_extraction_response
        assert filename == "file.txt"
        return dummy_structured_chunk

    extraction_service.create_structured_chunk.side_effect = create_structured_chunk
    files = [DummyUploadFile("file.txt", b"abc")]
    pipeline = DocumentPipelineService(
        ingestion_service, vector_store_service, extraction_service
    )

    # Act
    result = await pipeline.process_uploads(files)

    # Assert
    assert result.processed_count == 1
    assert len(result.errors) == 0
    assert len(result.extractions) == 1
    summary = result.extractions[0]
    assert summary.structured_data_extracted is True
    assert summary.reference_id == dummy_extraction_response.data.reference_id
    # Structured chunk was created with correct transformation
    extraction_service.create_structured_chunk.assert_called_once_with(
        dummy_extraction_response, "file.txt"
    )
    # Structured chunk added to vector store
    assert call([dummy_structured_chunk]) in vector_store_service.add_documents.call_args_list
