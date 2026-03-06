import pytest
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
def mock_ingestion_service():
    return MagicMock()

@pytest.fixture
def mock_vector_store_service():
    return MagicMock()

@pytest.fixture
def mock_extraction_service():
    return MagicMock()

@pytest.fixture
def pipeline(mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    return DocumentPipelineService(
        ingestion_service=mock_ingestion_service,
        vector_store_service=mock_vector_store_service,
        extraction_service=mock_extraction_service,
    )

def make_chunk(text, chunk_id):
    return Chunk(text=text, chunk_id=chunk_id)

def make_extraction_response(reference_id="ref-123"):
    class DummyData:
        def __init__(self, reference_id):
            self.reference_id = reference_id
    return ExtractionResponse(data=DummyData(reference_id=reference_id))

@pytest.mark.asyncio
async def test_happy_path_single_file(pipeline, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    # Arrange
    file_content = b"file-content"
    filename = "doc1.txt"
    upload_file = DummyUploadFile(filename, file_content)
    chunks = [make_chunk("chunk1", "c1"), make_chunk("chunk2", "c2")]
    extraction_response = make_extraction_response("ref-abc")
    structured_chunk = make_chunk("structured", "sc1")

    mock_ingestion_service.process_file.return_value = chunks
    mock_extraction_service.extract_data.return_value = extraction_response
    mock_extraction_service.create_structured_chunk.return_value = structured_chunk

    # Act
    result = await pipeline.process_uploads([upload_file])

    # Assert
    # Record count: processed_count should be 1
    assert result.processed_count == 1
    # No errors
    assert result.errors == []
    # Extraction summary
    assert len(result.extractions) == 1
    summary = result.extractions[0]
    assert summary.filename == filename
    assert summary.text_chunks == 2
    assert summary.structured_data_extracted is True
    assert summary.reference_id == "ref-abc"
    assert summary.error is None

    # Vector store should receive both text chunks and structured chunk
    expected_calls = [
        call.add_documents(chunks),
        call.add_documents([structured_chunk]),
    ]
    mock_vector_store_service.assert_has_calls(expected_calls, any_order=False)

    # Transformation correctness: structured_chunk is added after extraction
    mock_extraction_service.extract_data.assert_called_once_with("chunk1\nchunk2", filename)
    mock_extraction_service.create_structured_chunk.assert_called_once_with(extraction_response, filename)

@pytest.mark.asyncio
async def test_no_text_extracted(pipeline, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    # Arrange
    file_content = b"irrelevant"
    filename = "empty.txt"
    upload_file = DummyUploadFile(filename, file_content)
    mock_ingestion_service.process_file.return_value = []

    # Act
    result = await pipeline.process_uploads([upload_file])

    # Assert
    assert result.processed_count == 0
    assert result.errors == [f"No text extracted from {filename}"]
    assert result.extractions == []
    mock_vector_store_service.add_documents.assert_not_called()
    mock_extraction_service.extract_data.assert_not_called()
    mock_extraction_service.create_structured_chunk.assert_not_called()

@pytest.mark.asyncio
async def test_extraction_failure(pipeline, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    # Arrange
    file_content = b"file-content"
    filename = "fail_extract.txt"
    upload_file = DummyUploadFile(filename, file_content)
    chunks = [make_chunk("chunk1", "c1")]
    mock_ingestion_service.process_file.return_value = chunks
    mock_extraction_service.extract_data.side_effect = Exception("Extraction failed")

    # Act
    result = await pipeline.process_uploads([upload_file])

    # Assert
    assert result.processed_count == 1
    assert result.errors == []
    assert len(result.extractions) == 1
    summary = result.extractions[0]
    assert summary.filename == filename
    assert summary.text_chunks == 1
    assert summary.structured_data_extracted is False
    assert summary.reference_id is None
    assert "Extraction failed" in summary.error
    # Only text chunks added to vector store
    mock_vector_store_service.add_documents.assert_called_once_with(chunks)
    mock_extraction_service.create_structured_chunk.assert_not_called()

@pytest.mark.asyncio
async def test_ingestion_failure(pipeline, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    # Arrange
    file_content = b"file-content"
    filename = "bad.txt"
    upload_file = DummyUploadFile(filename, file_content)
    mock_ingestion_service.process_file.side_effect = Exception("Ingestion error")

    # Act
    result = await pipeline.process_uploads([upload_file])

    # Assert
    assert result.processed_count == 0
    assert result.errors == [f"Error processing {filename}: Ingestion error"]
    assert result.extractions == []
    mock_vector_store_service.add_documents.assert_not_called()
    mock_extraction_service.extract_data.assert_not_called()
    mock_extraction_service.create_structured_chunk.assert_not_called()

@pytest.mark.asyncio
async def test_multiple_files_with_duplicates_and_missing(pipeline, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    # Arrange
    files = [
        DummyUploadFile("doc1.txt", b"abc"),
        DummyUploadFile("doc2.txt", b"def"),
        DummyUploadFile("doc1.txt", b"abc"),  # duplicate filename/content
        DummyUploadFile("empty.txt", b""),
    ]
    # doc1.txt and doc2.txt yield chunks, empty.txt yields none
    chunks_doc1 = [make_chunk("chunkA", "cA")]
    chunks_doc2 = [make_chunk("chunkB", "cB"), make_chunk("chunkC", "cC")]
    extraction_doc1 = make_extraction_response("ref-1")
    extraction_doc2 = make_extraction_response("ref-2")
    structured_chunk1 = make_chunk("structured1", "sc1")
    structured_chunk2 = make_chunk("structured2", "sc2")

    def process_file_side_effect(content, filename):
        if filename == "doc1.txt":
            return chunks_doc1
        if filename == "doc2.txt":
            return chunks_doc2
        return []

    def extract_data_side_effect(text, filename):
        if filename == "doc1.txt":
            return extraction_doc1
        if filename == "doc2.txt":
            return extraction_doc2
        raise Exception("Should not extract")

    def create_structured_chunk_side_effect(extraction, filename):
        if filename == "doc1.txt":
            return structured_chunk1
        if filename == "doc2.txt":
            return structured_chunk2
        raise Exception("Should not create")

    mock_ingestion_service.process_file.side_effect = process_file_side_effect
    mock_extraction_service.extract_data.side_effect = extract_data_side_effect
    mock_extraction_service.create_structured_chunk.side_effect = create_structured_chunk_side_effect

    # Act
    result = await pipeline.process_uploads(files)

    # Assert
    # Record count: 3 processed (empty.txt yields no chunks)
    assert result.processed_count == 3
    # One error for empty.txt
    assert f"No text extracted from empty.txt" in result.errors
    # Extraction summaries: 3 (including duplicate doc1.txt)
    assert len(result.extractions) == 3
    # Check for duplicates by filename
    filenames = [e.filename for e in result.extractions]
    assert filenames.count("doc1.txt") == 2
    assert filenames.count("doc2.txt") == 1
    assert filenames.count("empty.txt") == 0

    # Check for missing: empty.txt not in extractions
    # Check aggregate: total chunks processed
    total_chunks = sum(e.text_chunks for e in result.extractions)
    assert total_chunks == 1 + 2 + 1  # doc1.txt (1), doc2.txt (2), doc1.txt duplicate (1)

    # Check that structured data was extracted for all processed files
    for e in result.extractions:
        assert e.structured_data_extracted is True
        assert e.reference_id in {"ref-1", "ref-2"}

    # Vector store: should have been called for each chunk and structured chunk
    expected_add_calls = [
        call.add_documents(chunks_doc1),
        call.add_documents([structured_chunk1]),
        call.add_documents(chunks_doc2),
        call.add_documents([structured_chunk2]),
        call.add_documents(chunks_doc1),
        call.add_documents([structured_chunk1]),
    ]
    # Flatten actual calls to add_documents
    actual_add_calls = [c for c in mock_vector_store_service.mock_calls if c[0] == "add_documents"]
    assert actual_add_calls == expected_add_calls

@pytest.mark.asyncio
async def test_boundary_conditions_empty_file_list(pipeline, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    # Arrange
    files = []

    # Act
    result = await pipeline.process_uploads(files)

    # Assert
    assert result.processed_count == 0
    assert result.errors == []
    assert result.extractions == []
    mock_ingestion_service.process_file.assert_not_called()
    mock_vector_store_service.add_documents.assert_not_called()
    mock_extraction_service.extract_data.assert_not_called()
    mock_extraction_service.create_structured_chunk.assert_not_called()
