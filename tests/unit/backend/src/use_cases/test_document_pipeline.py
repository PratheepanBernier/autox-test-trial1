import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, call

from backend.src.use_cases.document_pipeline import (
    DocumentPipelineService,
    UploadExtractionSummary,
    UploadResult,
)
from backend.src.models.schemas import Chunk
from backend.src.models.extraction_schema import ExtractionResponse

class DummyUploadFile:
    def __init__(self, filename, content: bytes):
        self.filename = filename
        self._content = content
        self._read_called = False

    async def read(self):
        if self._read_called:
            # Simulate FastAPI UploadFile behavior: .read() returns b'' after first call
            return b''
        self._read_called = True
        return self._content

@pytest.fixture
def mock_ingestion_service():
    return Mock()

@pytest.fixture
def mock_vector_store_service():
    return Mock()

@pytest.fixture
def mock_extraction_service():
    return Mock()

@pytest.mark.asyncio
async def test_process_uploads_happy_path(
    mock_ingestion_service, mock_vector_store_service, mock_extraction_service
):
    # Arrange
    file_content = b"file content"
    filename = "test.txt"
    upload_file = DummyUploadFile(filename, file_content)
    chunk1 = Chunk(text="chunk1", metadata={"page": 1})
    chunk2 = Chunk(text="chunk2", metadata={"page": 2})
    chunks = [chunk1, chunk2]

    extraction_response = ExtractionResponse(
        data=type("Data", (), {"reference_id": "ref-123"})()
    )
    structured_chunk = Chunk(text="structured", metadata={"structured": True})

    mock_ingestion_service.process_file.return_value = chunks
    mock_vector_store_service.add_documents.return_value = None
    mock_extraction_service.extract_data.return_value = extraction_response
    mock_extraction_service.create_structured_chunk.return_value = structured_chunk

    service = DocumentPipelineService(
        ingestion_service=mock_ingestion_service,
        vector_store_service=mock_vector_store_service,
        extraction_service=mock_extraction_service,
    )

    # Act
    result = await service.process_uploads([upload_file])

    # Assert
    assert isinstance(result, UploadResult)
    assert result.processed_count == 1
    assert result.errors == []
    assert len(result.extractions) == 1
    summary = result.extractions[0]
    assert summary.filename == filename
    assert summary.text_chunks == 2
    assert summary.structured_data_extracted is True
    assert summary.reference_id == "ref-123"
    assert summary.error is None

    mock_ingestion_service.process_file.assert_called_once_with(file_content, filename)
    mock_vector_store_service.add_documents.assert_has_calls([
        call(chunks),
        call([structured_chunk])
    ])
    mock_extraction_service.extract_data.assert_called_once_with("chunk1\nchunk2", filename)
    mock_extraction_service.create_structured_chunk.assert_called_once_with(extraction_response, filename)

@pytest.mark.asyncio
async def test_process_uploads_no_chunks_extracted(
    mock_ingestion_service, mock_vector_store_service, mock_extraction_service
):
    # Arrange
    file_content = b"irrelevant"
    filename = "empty.pdf"
    upload_file = DummyUploadFile(filename, file_content)
    mock_ingestion_service.process_file.return_value = []

    service = DocumentPipelineService(
        ingestion_service=mock_ingestion_service,
        vector_store_service=mock_vector_store_service,
        extraction_service=mock_extraction_service,
    )

    # Act
    result = await service.process_uploads([upload_file])

    # Assert
    assert result.processed_count == 0
    assert result.errors == [f"No text extracted from {filename}"]
    assert result.extractions == []
    mock_ingestion_service.process_file.assert_called_once_with(file_content, filename)
    mock_vector_store_service.add_documents.assert_not_called()
    mock_extraction_service.extract_data.assert_not_called()
    mock_extraction_service.create_structured_chunk.assert_not_called()

@pytest.mark.asyncio
async def test_process_uploads_ingestion_service_raises(
    mock_ingestion_service, mock_vector_store_service, mock_extraction_service
):
    # Arrange
    file_content = b"bad"
    filename = "fail.docx"
    upload_file = DummyUploadFile(filename, file_content)
    mock_ingestion_service.process_file.side_effect = Exception("ingestion failed")

    service = DocumentPipelineService(
        ingestion_service=mock_ingestion_service,
        vector_store_service=mock_vector_store_service,
        extraction_service=mock_extraction_service,
    )

    # Act
    result = await service.process_uploads([upload_file])

    # Assert
    assert result.processed_count == 0
    assert result.errors == [f"Error processing {filename}: ingestion failed"]
    assert result.extractions == []
    mock_ingestion_service.process_file.assert_called_once_with(file_content, filename)
    mock_vector_store_service.add_documents.assert_not_called()
    mock_extraction_service.extract_data.assert_not_called()
    mock_extraction_service.create_structured_chunk.assert_not_called()

@pytest.mark.asyncio
async def test_process_uploads_extraction_service_raises(
    mock_ingestion_service, mock_vector_store_service, mock_extraction_service
):
    # Arrange
    file_content = b"ok"
    filename = "extract_fail.txt"
    upload_file = DummyUploadFile(filename, file_content)
    chunk = Chunk(text="chunk", metadata={})
    mock_ingestion_service.process_file.return_value = [chunk]
    mock_vector_store_service.add_documents.return_value = None
    mock_extraction_service.extract_data.side_effect = Exception("extraction error")

    service = DocumentPipelineService(
        ingestion_service=mock_ingestion_service,
        vector_store_service=mock_vector_store_service,
        extraction_service=mock_extraction_service,
    )

    # Act
    result = await service.process_uploads([upload_file])

    # Assert
    assert result.processed_count == 1
    assert result.errors == []
    assert len(result.extractions) == 1
    summary = result.extractions[0]
    assert summary.filename == filename
    assert summary.text_chunks == 1
    assert summary.structured_data_extracted is False
    assert summary.reference_id is None
    assert "extraction error" in summary.error

    mock_ingestion_service.process_file.assert_called_once_with(file_content, filename)
    mock_vector_store_service.add_documents.assert_called_once_with([chunk])
    mock_extraction_service.extract_data.assert_called_once_with("chunk", filename)
    mock_extraction_service.create_structured_chunk.assert_not_called()

@pytest.mark.asyncio
async def test_process_uploads_multiple_files_mixed_results(
    mock_ingestion_service, mock_vector_store_service, mock_extraction_service
):
    # Arrange
    file1 = DummyUploadFile("good1.txt", b"abc")
    file2 = DummyUploadFile("empty.txt", b"def")
    file3 = DummyUploadFile("bad.pdf", b"ghi")
    file4 = DummyUploadFile("extract_fail.docx", b"jkl")

    chunk1 = Chunk(text="t1", metadata={})
    chunk2 = Chunk(text="t2", metadata={})
    extraction_response = ExtractionResponse(
        data=type("Data", (), {"reference_id": "RID"})()
    )
    structured_chunk = Chunk(text="structured", metadata={})

    # Setup mocks for each file
    def process_file_side_effect(content, filename):
        if filename == "good1.txt":
            return [chunk1, chunk2]
        elif filename == "empty.txt":
            return []
        elif filename == "bad.pdf":
            raise Exception("ingest fail")
        elif filename == "extract_fail.docx":
            return [chunk1]
        else:
            return []

    def extract_data_side_effect(text, filename):
        if filename == "good1.txt":
            return extraction_response
        elif filename == "extract_fail.docx":
            raise Exception("extract fail")
        else:
            return extraction_response

    mock_ingestion_service.process_file.side_effect = process_file_side_effect
    mock_vector_store_service.add_documents.return_value = None
    mock_extraction_service.extract_data.side_effect = extract_data_side_effect
    mock_extraction_service.create_structured_chunk.return_value = structured_chunk

    service = DocumentPipelineService(
        ingestion_service=mock_ingestion_service,
        vector_store_service=mock_vector_store_service,
        extraction_service=mock_extraction_service,
    )

    # Act
    result = await service.process_uploads([file1, file2, file3, file4])

    # Assert
    assert result.processed_count == 2  # good1.txt and extract_fail.docx
    assert len(result.errors) == 2
    assert "No text extracted from empty.txt" in result.errors
    assert "Error processing bad.pdf: ingest fail" in result.errors
    assert len(result.extractions) == 2

    # Check good1.txt summary
    summary1 = next(e for e in result.extractions if e.filename == "good1.txt")
    assert summary1.text_chunks == 2
    assert summary1.structured_data_extracted is True
    assert summary1.reference_id == "RID"
    assert summary1.error is None

    # Check extract_fail.docx summary
    summary2 = next(e for e in result.extractions if e.filename == "extract_fail.docx")
    assert summary2.text_chunks == 1
    assert summary2.structured_data_extracted is False
    assert summary2.reference_id is None
    assert "extract fail" in summary2.error

@pytest.mark.asyncio
async def test_process_uploads_boundary_empty_file_list(
    mock_ingestion_service, mock_vector_store_service, mock_extraction_service
):
    # Arrange
    service = DocumentPipelineService(
        ingestion_service=mock_ingestion_service,
        vector_store_service=mock_vector_store_service,
        extraction_service=mock_extraction_service,
    )

    # Act
    result = await service.process_uploads([])

    # Assert
    assert result.processed_count == 0
    assert result.errors == []
    assert result.extractions == []

@pytest.mark.asyncio
async def test_process_uploads_file_read_raises(
    mock_ingestion_service, mock_vector_store_service, mock_extraction_service
):
    # Arrange
    class FailingUploadFile:
        filename = "fail.bin"
        async def read(self):
            raise Exception("read error")

    upload_file = FailingUploadFile()
    service = DocumentPipelineService(
        ingestion_service=mock_ingestion_service,
        vector_store_service=mock_vector_store_service,
        extraction_service=mock_extraction_service,
    )

    # Act
    result = await service.process_uploads([upload_file])

    # Assert
    assert result.processed_count == 0
    assert result.errors == ["Error processing fail.bin: read error"]
    assert result.extractions == []
    mock_ingestion_service.process_file.assert_not_called()
    mock_vector_store_service.add_documents.assert_not_called()
    mock_extraction_service.extract_data.assert_not_called()
    mock_extraction_service.create_structured_chunk.assert_not_called()
