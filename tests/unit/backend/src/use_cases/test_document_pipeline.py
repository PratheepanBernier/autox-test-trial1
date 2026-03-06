import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, create_autospec, call

from backend.src.use_cases.document_pipeline import (
    DocumentPipelineService,
    UploadExtractionSummary,
    UploadResult,
)
from backend.src.models.schemas import Chunk
from backend.src.models.extraction_schema import ExtractionResponse

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
def make_upload_file():
    # Simulate FastAPI's UploadFile with async read
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

    return DummyUploadFile

@pytest.mark.asyncio
async def test_process_uploads_happy_path(
    mock_ingestion_service,
    mock_vector_store_service,
    mock_extraction_service,
    make_upload_file,
):
    # Arrange
    file_content = b"file-content"
    filename = "test.pdf"
    chunk1 = Chunk(text="chunk1", metadata={"page": 1})
    chunk2 = Chunk(text="chunk2", metadata={"page": 2})
    chunks = [chunk1, chunk2]

    extraction_response = ExtractionResponse(
        data=type("Data", (), {"reference_id": "ref-123"})(),
        raw_text="chunk1\nchunk2",
        schema_version="1.0",
    )
    structured_chunk = Chunk(text="structured", metadata={"type": "structured"})

    mock_ingestion_service.process_file.return_value = chunks
    mock_vector_store_service.add_documents.return_value = None
    mock_extraction_service.extract_data.return_value = extraction_response
    mock_extraction_service.create_structured_chunk.return_value = structured_chunk

    upload_file = make_upload_file(filename, file_content)
    service = DocumentPipelineService(
        mock_ingestion_service,
        mock_vector_store_service,
        mock_extraction_service,
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
    mock_ingestion_service,
    mock_vector_store_service,
    mock_extraction_service,
    make_upload_file,
):
    file_content = b"empty"
    filename = "empty.pdf"
    mock_ingestion_service.process_file.return_value = []

    upload_file = make_upload_file(filename, file_content)
    service = DocumentPipelineService(
        mock_ingestion_service,
        mock_vector_store_service,
        mock_extraction_service,
    )

    result = await service.process_uploads([upload_file])

    assert result.processed_count == 0
    assert result.errors == [f"No text extracted from {filename}"]
    assert result.extractions == []
    mock_ingestion_service.process_file.assert_called_once_with(file_content, filename)
    mock_vector_store_service.add_documents.assert_not_called()
    mock_extraction_service.extract_data.assert_not_called()
    mock_extraction_service.create_structured_chunk.assert_not_called()

@pytest.mark.asyncio
async def test_process_uploads_extraction_error(
    mock_ingestion_service,
    mock_vector_store_service,
    mock_extraction_service,
    make_upload_file,
):
    file_content = b"file-content"
    filename = "fail_extract.pdf"
    chunk = Chunk(text="chunk", metadata={})
    chunks = [chunk]

    mock_ingestion_service.process_file.return_value = chunks
    mock_vector_store_service.add_documents.return_value = None
    mock_extraction_service.extract_data.side_effect = RuntimeError("Extraction failed")

    upload_file = make_upload_file(filename, file_content)
    service = DocumentPipelineService(
        mock_ingestion_service,
        mock_vector_store_service,
        mock_extraction_service,
    )

    result = await service.process_uploads([upload_file])

    assert result.processed_count == 1
    assert result.errors == []
    assert len(result.extractions) == 1
    summary = result.extractions[0]
    assert summary.filename == filename
    assert summary.text_chunks == 1
    assert summary.structured_data_extracted is False
    assert summary.reference_id is None
    assert "Extraction failed" in summary.error

    mock_ingestion_service.process_file.assert_called_once_with(file_content, filename)
    mock_vector_store_service.add_documents.assert_called_once_with(chunks)
    mock_extraction_service.extract_data.assert_called_once_with("chunk", filename)
    mock_extraction_service.create_structured_chunk.assert_not_called()

@pytest.mark.asyncio
async def test_process_uploads_ingestion_error(
    mock_ingestion_service,
    mock_vector_store_service,
    mock_extraction_service,
    make_upload_file,
):
    file_content = b"file-content"
    filename = "bad.pdf"
    mock_ingestion_service.process_file.side_effect = ValueError("Bad file")

    upload_file = make_upload_file(filename, file_content)
    service = DocumentPipelineService(
        mock_ingestion_service,
        mock_vector_store_service,
        mock_extraction_service,
    )

    result = await service.process_uploads([upload_file])

    assert result.processed_count == 0
    assert len(result.errors) == 1
    assert result.errors[0].startswith(f"Error processing {filename}: Bad file")
    assert result.extractions == []
    mock_ingestion_service.process_file.assert_called_once_with(file_content, filename)
    mock_vector_store_service.add_documents.assert_not_called()
    mock_extraction_service.extract_data.assert_not_called()
    mock_extraction_service.create_structured_chunk.assert_not_called()

@pytest.mark.asyncio
async def test_process_uploads_multiple_files_mixed_results(
    mock_ingestion_service,
    mock_vector_store_service,
    mock_extraction_service,
    make_upload_file,
):
    # File 1: happy path
    file1_content = b"content1"
    file1_name = "file1.pdf"
    chunk1 = Chunk(text="chunk1", metadata={})
    chunks1 = [chunk1]
    extraction1 = ExtractionResponse(
        data=type("Data", (), {"reference_id": "ref1"})(),
        raw_text="chunk1",
        schema_version="1.0",
    )
    structured_chunk1 = Chunk(text="structured1", metadata={})

    # File 2: no chunks
    file2_content = b"content2"
    file2_name = "file2.pdf"

    # File 3: extraction error
    file3_content = b"content3"
    file3_name = "file3.pdf"
    chunk3 = Chunk(text="chunk3", metadata={})
    chunks3 = [chunk3]

    def process_file_side_effect(content, filename):
        if filename == file1_name:
            return chunks1
        elif filename == file2_name:
            return []
        elif filename == file3_name:
            return chunks3
        else:
            return []

    mock_ingestion_service.process_file.side_effect = process_file_side_effect
    mock_vector_store_service.add_documents.return_value = None

    def extract_data_side_effect(text, filename):
        if filename == file1_name:
            return extraction1
        elif filename == file3_name:
            raise RuntimeError("Extraction failed for file3")
        else:
            raise AssertionError("Unexpected call")

    mock_extraction_service.extract_data.side_effect = extract_data_side_effect
    mock_extraction_service.create_structured_chunk.return_value = structured_chunk1

    files = [
        make_upload_file(file1_name, file1_content),
        make_upload_file(file2_name, file2_content),
        make_upload_file(file3_name, file3_content),
    ]
    service = DocumentPipelineService(
        mock_ingestion_service,
        mock_vector_store_service,
        mock_extraction_service,
    )

    result = await service.process_uploads(files)

    assert result.processed_count == 2
    assert len(result.errors) == 1
    assert result.errors[0] == f"No text extracted from {file2_name}"
    assert len(result.extractions) == 2

    # File 1 summary
    summary1 = next(e for e in result.extractions if e.filename == file1_name)
    assert summary1.text_chunks == 1
    assert summary1.structured_data_extracted is True
    assert summary1.reference_id == "ref1"
    assert summary1.error is None

    # File 3 summary
    summary3 = next(e for e in result.extractions if e.filename == file3_name)
    assert summary3.text_chunks == 1
    assert summary3.structured_data_extracted is False
    assert summary3.reference_id is None
    assert "Extraction failed for file3" in summary3.error

@pytest.mark.asyncio
async def test_process_uploads_empty_file_list(
    mock_ingestion_service,
    mock_vector_store_service,
    mock_extraction_service,
):
    service = DocumentPipelineService(
        mock_ingestion_service,
        mock_vector_store_service,
        mock_extraction_service,
    )
    result = await service.process_uploads([])
    assert isinstance(result, UploadResult)
    assert result.processed_count == 0
    assert result.errors == []
    assert result.extractions == []
    assert result.message == "Successfully processed 0 documents."
    mock_ingestion_service.process_file.assert_not_called()
    mock_vector_store_service.add_documents.assert_not_called()
    mock_extraction_service.extract_data.assert_not_called()
    mock_extraction_service.create_structured_chunk.assert_not_called()
