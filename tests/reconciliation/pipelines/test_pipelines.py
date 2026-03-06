import pytest
from unittest.mock import AsyncMock, MagicMock, call
from types import SimpleNamespace

from backend.src.use_cases.document_pipeline import (
    DocumentPipelineService,
    UploadExtractionSummary,
    UploadResult,
)

@pytest.fixture
def fake_chunk():
    def _make(text, chunk_id):
        return SimpleNamespace(text=text, chunk_id=chunk_id)
    return _make

@pytest.fixture
def fake_extraction_response():
    def _make(reference_id):
        return SimpleNamespace(data=SimpleNamespace(reference_id=reference_id))
    return _make

@pytest.fixture
def fake_upload_file():
    class FakeUploadFile:
        def __init__(self, filename, content):
            self.filename = filename
            self._content = content
            self._read_called = False
        async def read(self):
            if self._read_called:
                return b""
            self._read_called = True
            return self._content
    return FakeUploadFile

@pytest.mark.asyncio
async def test_pipeline_record_counts_and_aggregate_consistency(
    fake_chunk, fake_extraction_response, fake_upload_file
):
    # Prepare test data
    files = [
        fake_upload_file("doc1.txt", b"file1 content"),
        fake_upload_file("doc2.txt", b"file2 content"),
    ]
    # Each file will be split into 2 chunks
    chunks_per_file = {
        "doc1.txt": [fake_chunk("chunk1a", "c1a"), fake_chunk("chunk1b", "c1b")],
        "doc2.txt": [fake_chunk("chunk2a", "c2a"), fake_chunk("chunk2b", "c2b")],
    }
    # Extraction responses
    extraction_per_file = {
        "doc1.txt": fake_extraction_response("ref1"),
        "doc2.txt": fake_extraction_response("ref2"),
    }
    # Structured chunk per file
    structured_chunk_per_file = {
        "doc1.txt": fake_chunk("structured1", "sc1"),
        "doc2.txt": fake_chunk("structured2", "sc2"),
    }

    # Mocks
    ingestion_service = MagicMock()
    ingestion_service.process_file.side_effect = lambda content, filename: chunks_per_file[filename]

    vector_store_service = MagicMock()
    vector_store_service.add_documents = MagicMock()

    extraction_service = MagicMock()
    extraction_service.extract_data.side_effect = lambda text, filename="unknown": extraction_per_file[filename]
    extraction_service.create_structured_chunk.side_effect = lambda extraction, filename: structured_chunk_per_file[filename]

    pipeline = DocumentPipelineService(
        ingestion_service=ingestion_service,
        vector_store_service=vector_store_service,
        extraction_service=extraction_service,
    )

    result: UploadResult = await pipeline.process_uploads(files)

    # 1. Record counts: processed_count matches input files
    assert result.processed_count == 2
    # 2. Aggregate: sum of text_chunks matches total chunks processed
    total_chunks = sum(len(chunks) for chunks in chunks_per_file.values())
    assert sum(e.text_chunks for e in result.extractions) == total_chunks
    # 3. Structured data extracted for all files
    assert all(e.structured_data_extracted for e in result.extractions)
    # 4. Reference IDs are correct and unique
    reference_ids = [e.reference_id for e in result.extractions]
    assert set(reference_ids) == {"ref1", "ref2"}
    # 5. No errors
    assert result.errors == []
    # 6. Vector store add_documents called for each chunk batch and each structured chunk
    expected_calls = [
        call(chunks_per_file["doc1.txt"]),
        call([structured_chunk_per_file["doc1.txt"]]),
        call(chunks_per_file["doc2.txt"]),
        call([structured_chunk_per_file["doc2.txt"]]),
    ]
    vector_store_service.add_documents.assert_has_calls(expected_calls, any_order=False)
    # 7. No duplicate or missing records in extractions
    filenames = [e.filename for e in result.extractions]
    assert sorted(filenames) == ["doc1.txt", "doc2.txt"]
    assert len(set(filenames)) == len(filenames)

@pytest.mark.asyncio
async def test_pipeline_duplicate_and_missing_detection(
    fake_chunk, fake_extraction_response, fake_upload_file
):
    # Prepare test data: one file yields no chunks, one yields chunks
    files = [
        fake_upload_file("empty.txt", b""),
        fake_upload_file("good.txt", b"good content"),
    ]
    chunks_per_file = {
        "empty.txt": [],
        "good.txt": [fake_chunk("good1", "g1"), fake_chunk("good2", "g2")],
    }
    extraction_per_file = {
        "good.txt": fake_extraction_response("goodref"),
    }
    structured_chunk_per_file = {
        "good.txt": fake_chunk("structured_good", "sgood"),
    }

    ingestion_service = MagicMock()
    ingestion_service.process_file.side_effect = lambda content, filename: chunks_per_file[filename]

    vector_store_service = MagicMock()
    vector_store_service.add_documents = MagicMock()

    extraction_service = MagicMock()
    extraction_service.extract_data.side_effect = lambda text, filename="unknown": extraction_per_file[filename]
    extraction_service.create_structured_chunk.side_effect = lambda extraction, filename: structured_chunk_per_file[filename]

    pipeline = DocumentPipelineService(
        ingestion_service=ingestion_service,
        vector_store_service=vector_store_service,
        extraction_service=extraction_service,
    )

    result: UploadResult = await pipeline.process_uploads(files)

    # 1. Only one file processed successfully
    assert result.processed_count == 1
    # 2. Error for empty.txt
    assert any("No text extracted from empty.txt" in err for err in result.errors)
    # 3. Only one extraction summary, for good.txt
    assert len(result.extractions) == 1
    assert result.extractions[0].filename == "good.txt"
    # 4. No duplicate extraction summaries
    filenames = [e.filename for e in result.extractions]
    assert len(set(filenames)) == len(filenames)
    # 5. No missing extraction for good.txt
    assert result.extractions[0].structured_data_extracted is True
    # 6. Vector store add_documents called only for good.txt
    expected_calls = [
        call(chunks_per_file["good.txt"]),
        call([structured_chunk_per_file["good.txt"]]),
    ]
    vector_store_service.add_documents.assert_has_calls(expected_calls, any_order=False)

@pytest.mark.asyncio
async def test_pipeline_transformation_correctness(
    fake_chunk, fake_extraction_response, fake_upload_file
):
    # Prepare test data: verify that transformation (extraction + structured chunk) is correct
    files = [
        fake_upload_file("trans.txt", b"transform me"),
    ]
    chunks_per_file = {
        "trans.txt": [fake_chunk("t1", "t1id"), fake_chunk("t2", "t2id")],
    }
    extraction_per_file = {
        "trans.txt": fake_extraction_response("transref"),
    }
    structured_chunk_per_file = {
        "trans.txt": fake_chunk("structured_trans", "strans"),
    }

    ingestion_service = MagicMock()
    ingestion_service.process_file.side_effect = lambda content, filename: chunks_per_file[filename]

    vector_store_service = MagicMock()
    vector_store_service.add_documents = MagicMock()

    extraction_service = MagicMock()
    extraction_service.extract_data.side_effect = lambda text, filename="unknown": extraction_per_file[filename]
    extraction_service.create_structured_chunk.side_effect = lambda extraction, filename: structured_chunk_per_file[filename]

    pipeline = DocumentPipelineService(
        ingestion_service=ingestion_service,
        vector_store_service=vector_store_service,
        extraction_service=extraction_service,
    )

    result: UploadResult = await pipeline.process_uploads(files)

    # 1. Extraction called with correct full_text
    expected_full_text = "\n".join(chunk.text for chunk in chunks_per_file["trans.txt"])
    extraction_service.extract_data.assert_called_once_with(expected_full_text, "trans.txt")
    # 2. Structured chunk created with correct extraction and filename
    extraction_service.create_structured_chunk.assert_called_once_with(
        extraction_per_file["trans.txt"], "trans.txt"
    )
    # 3. Structured chunk added to vector store
    vector_store_service.add_documents.assert_any_call([structured_chunk_per_file["trans.txt"]])
    # 4. Extraction summary reflects transformation
    summary = result.extractions[0]
    assert summary.filename == "trans.txt"
    assert summary.text_chunks == 2
    assert summary.structured_data_extracted is True
    assert summary.reference_id == "transref"
    # 5. No errors
    assert result.errors == []
