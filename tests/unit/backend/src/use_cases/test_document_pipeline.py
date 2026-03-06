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

class DummyUploadFile:
    def __init__(self, filename, content):
        self.filename = filename
        self._content = content
        self._read_called = False

    async def read(self):
        if self._read_called:
            # Simulate FastAPI UploadFile behavior: .read() returns b"" after first call
            return b""
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

@pytest.fixture
def pipeline(mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    return DocumentPipelineService(
        ingestion_service=mock_ingestion_service,
        vector_store_service=mock_vector_store_service,
        extraction_service=mock_extraction_service,
    )

def make_chunk(text, meta=None):
    return Chunk(text=text, meta=meta or {})

def make_extraction_response(reference_id="ref-123"):
    class DummyData:
        def __init__(self, reference_id):
            self.reference_id = reference_id
    return ExtractionResponse(data=DummyData(reference_id=reference_id))

@pytest.mark.asyncio
async def test_process_uploads_happy_path(pipeline, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    file1 = DummyUploadFile("doc1.txt", b"file1 content")
    file2 = DummyUploadFile("doc2.txt", b"file2 content")

    # Arrange
    chunks1 = [make_chunk("chunk1"), make_chunk("chunk2")]
    chunks2 = [make_chunk("chunkA")]
    mock_ingestion_service.process_file.side_effect = [chunks1, chunks2]

    extraction1 = make_extraction_response("ref-abc")
    extraction2 = make_extraction_response("ref-def")
    mock_extraction_service.extract_data.side_effect = [extraction1, extraction2]
    structured_chunk1 = make_chunk("structured1")
    structured_chunk2 = make_chunk("structured2")
    mock_extraction_service.create_structured_chunk.side_effect = [structured_chunk1, structured_chunk2]

    # Act
    result = await pipeline.process_uploads([file1, file2])

    # Assert
    assert isinstance(result, UploadResult)
    assert result.processed_count == 2
    assert result.errors == []
    assert len(result.extractions) == 2

    # Check extraction summaries
    summary1 = result.extractions[0]
    summary2 = result.extractions[1]
    assert summary1.filename == "doc1.txt"
    assert summary1.text_chunks == 2
    assert summary1.structured_data_extracted is True
    assert summary1.reference_id == "ref-abc"
    assert summary1.error is None

    assert summary2.filename == "doc2.txt"
    assert summary2.text_chunks == 1
    assert summary2.structured_data_extracted is True
    assert summary2.reference_id == "ref-def"
    assert summary2.error is None

    # Check calls
    assert mock_ingestion_service.process_file.call_args_list == [
        call(b"file1 content", "doc1.txt"),
        call(b"file2 content", "doc2.txt"),
    ]
    # Each chunk list is added, then structured chunk is added
    assert mock_vector_store_service.add_documents.call_args_list == [
        call(chunks1),
        call([structured_chunk1]),
        call(chunks2),
        call([structured_chunk2]),
    ]
    assert mock_extraction_service.extract_data.call_count == 2
    assert mock_extraction_service.create_structured_chunk.call_count == 2

@pytest.mark.asyncio
async def test_process_uploads_no_chunks_extracted(pipeline, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    file1 = DummyUploadFile("empty.pdf", b"irrelevant")
    mock_ingestion_service.process_file.return_value = []

    # Act
    result = await pipeline.process_uploads([file1])

    # Assert
    assert result.processed_count == 0
    assert result.errors == ["No text extracted from empty.pdf"]
    assert result.extractions == []
    mock_vector_store_service.add_documents.assert_not_called()
    mock_extraction_service.extract_data.assert_not_called()
    mock_extraction_service.create_structured_chunk.assert_not_called()

@pytest.mark.asyncio
async def test_process_uploads_ingestion_service_raises(pipeline, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    file1 = DummyUploadFile("bad.docx", b"bad content")
    mock_ingestion_service.process_file.side_effect = Exception("ingestion failed")

    # Act
    result = await pipeline.process_uploads([file1])

    # Assert
    assert result.processed_count == 0
    assert len(result.errors) == 1
    assert result.errors[0].startswith("Error processing bad.docx: ingestion failed")
    assert result.extractions == []
    mock_vector_store_service.add_documents.assert_not_called()
    mock_extraction_service.extract_data.assert_not_called()

@pytest.mark.asyncio
async def test_process_uploads_extraction_service_raises(pipeline, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    file1 = DummyUploadFile("doc3.txt", b"file3 content")
    chunks = [make_chunk("chunkX")]
    mock_ingestion_service.process_file.return_value = chunks
    mock_extraction_service.extract_data.side_effect = Exception("extraction error")

    # Act
    result = await pipeline.process_uploads([file1])

    # Assert
    assert result.processed_count == 1
    assert result.errors == []
    assert len(result.extractions) == 1
    summary = result.extractions[0]
    assert summary.filename == "doc3.txt"
    assert summary.text_chunks == 1
    assert summary.structured_data_extracted is False
    assert summary.reference_id is None
    assert "extraction error" in summary.error
    # Should still add the original chunks to vector store, but not the structured chunk
    mock_vector_store_service.add_documents.assert_called_once_with(chunks)
    mock_extraction_service.create_structured_chunk.assert_not_called()

@pytest.mark.asyncio
async def test_process_uploads_structured_chunk_add_raises(pipeline, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    file1 = DummyUploadFile("doc4.txt", b"file4 content")
    chunks = [make_chunk("chunkY")]
    mock_ingestion_service.process_file.return_value = chunks
    extraction = make_extraction_response("ref-xyz")
    mock_extraction_service.extract_data.return_value = extraction
    structured_chunk = make_chunk("structuredY")
    mock_extraction_service.create_structured_chunk.return_value = structured_chunk

    # Simulate add_documents raising on structured chunk
    def add_documents_side_effect(arg):
        if arg == [structured_chunk]:
            raise Exception("vector store add failed")
    mock_vector_store_service.add_documents.side_effect = add_documents_side_effect

    # Act
    result = await pipeline.process_uploads([file1])

    # Assert
    # The error should be caught at the outer try/except, so it's an error for the file
    assert result.processed_count == 0
    assert len(result.errors) == 1
    assert result.errors[0].startswith("Error processing doc4.txt: vector store add failed")
    assert result.extractions == []

@pytest.mark.asyncio
async def test_process_uploads_multiple_files_mixed_results(pipeline, mock_ingestion_service, mock_vector_store_service, mock_extraction_service):
    file1 = DummyUploadFile("good.txt", b"good content")
    file2 = DummyUploadFile("empty.txt", b"empty content")
    file3 = DummyUploadFile("fail.doc", b"fail content")

    # file1: happy path
    chunks1 = [make_chunk("good chunk")]
    extraction1 = make_extraction_response("good-ref")
    structured_chunk1 = make_chunk("good structured")
    # file2: no chunks
    # file3: ingestion fails
    mock_ingestion_service.process_file.side_effect = [
        chunks1,
        [],
        Exception("ingestion fail"),
    ]
    mock_extraction_service.extract_data.return_value = extraction1
    mock_extraction_service.create_structured_chunk.return_value = structured_chunk1

    # Act
    result = await pipeline.process_uploads([file1, file2, file3])

    # Assert
    assert result.processed_count == 1
    assert len(result.errors) == 2
    assert "No text extracted from empty.txt" in result.errors
    assert result.errors[1].startswith("Error processing fail.doc: ingestion fail")
    assert len(result.extractions) == 1
    summary = result.extractions[0]
    assert summary.filename == "good.txt"
    assert summary.text_chunks == 1
    assert summary.structured_data_extracted is True
    assert summary.reference_id == "good-ref"
    assert summary.error is None

@pytest.mark.asyncio
async def test_process_uploads_empty_file_list(pipeline):
    # Act
    result = await pipeline.process_uploads([])

    # Assert
    assert isinstance(result, UploadResult)
    assert result.processed_count == 0
    assert result.errors == []
    assert result.extractions == []
    assert result.message == "Successfully processed 0 documents."

@pytest.mark.asyncio
async def test_process_uploads_file_read_raises(pipeline, mock_ingestion_service):
    class FailingUploadFile(DummyUploadFile):
        async def read(self):
            raise IOError("read failed")
    file1 = FailingUploadFile("badfile.txt", b"irrelevant")

    # Act
    result = await pipeline.process_uploads([file1])

    # Assert
    assert result.processed_count == 0
    assert len(result.errors) == 1
    assert result.errors[0].startswith("Error processing badfile.txt: read failed")
