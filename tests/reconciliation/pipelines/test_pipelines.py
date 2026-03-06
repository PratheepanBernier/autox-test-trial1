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

@pytest.mark.asyncio
class TestDocumentPipelineReconciliation:
    @pytest.fixture
    def chunks(self):
        return [
            Chunk(text="chunk1", metadata={"page": 1}),
            Chunk(text="chunk2", metadata={"page": 2}),
        ]

    @pytest.fixture
    def structured_chunk(self):
        return Chunk(text="structured", metadata={"structured": True})

    @pytest.fixture
    def extraction_response(self):
        class DummyData:
            reference_id = "ref-123"
        return ExtractionResponse(data=DummyData())

    @pytest.fixture
    def ingestion_service(self, chunks):
        svc = MagicMock()
        svc.process_file = MagicMock(return_value=chunks)
        return svc

    @pytest.fixture
    def vector_store_service(self):
        svc = MagicMock()
        svc.add_documents = MagicMock()
        return svc

    @pytest.fixture
    def extraction_service(self, extraction_response, structured_chunk):
        svc = MagicMock()
        svc.extract_data = MagicMock(return_value=extraction_response)
        svc.create_structured_chunk = MagicMock(return_value=structured_chunk)
        return svc

    @pytest.fixture
    def pipeline(self, ingestion_service, vector_store_service, extraction_service):
        return DocumentPipelineService(
            ingestion_service=ingestion_service,
            vector_store_service=vector_store_service,
            extraction_service=extraction_service,
        )

    async def test_happy_path_counts_and_aggregates(
        self, pipeline, ingestion_service, vector_store_service, extraction_service, chunks, structured_chunk
    ):
        files = [
            DummyUploadFile("doc1.pdf", b"filecontent1"),
            DummyUploadFile("doc2.pdf", b"filecontent2"),
        ]

        result = await pipeline.process_uploads(files)

        # Assert processed count matches input files
        assert result.processed_count == 2
        assert len(result.errors) == 0
        assert len(result.extractions) == 2

        # Assert record counts: each file's chunks + 1 structured chunk per file
        expected_add_calls = [
            call(chunks),
            call([structured_chunk]),
            call(chunks),
            call([structured_chunk]),
        ]
        assert vector_store_service.add_documents.call_args_list == expected_add_calls

        # Assert extraction summaries match
        for summary in result.extractions:
            assert summary.text_chunks == len(chunks)
            assert summary.structured_data_extracted is True
            assert summary.reference_id == "ref-123"
            assert summary.error is None

    async def test_duplicate_detection_and_missing_records(
        self, pipeline, ingestion_service, vector_store_service, extraction_service, chunks, structured_chunk
    ):
        # Simulate ingestion service returning duplicate chunks for one file
        chunks_with_duplicate = chunks + [chunks[0]]
        ingestion_service.process_file.side_effect = [chunks_with_duplicate, []]

        files = [
            DummyUploadFile("dup.pdf", b"dupcontent"),
            DummyUploadFile("empty.pdf", b"emptycontent"),
        ]

        result = await pipeline.process_uploads(files)

        # First file: duplicate chunk, second file: no chunks
        assert result.processed_count == 1  # Only one file processed successfully
        assert len(result.errors) == 1
        assert "No text extracted from empty.pdf" in result.errors[0]

        # Check for duplicate chunk in extraction summary
        summary = result.extractions[0]
        assert summary.filename == "dup.pdf"
        assert summary.text_chunks == len(chunks_with_duplicate)
        assert summary.structured_data_extracted is True

        # Vector store should be called for all chunks (including duplicate) and structured chunk
        expected_add_calls = [
            call(chunks_with_duplicate),
            call([structured_chunk]),
        ]
        assert vector_store_service.add_documents.call_args_list == expected_add_calls

    async def test_transformation_correctness(
        self, pipeline, ingestion_service, extraction_service, structured_chunk, extraction_response
    ):
        # Simulate transformation: structured_chunk must contain reference_id from extraction_response
        def create_structured_chunk(extraction, filename):
            return Chunk(text=f"structured-{extraction.data.reference_id}", metadata={"filename": filename})

        extraction_service.create_structured_chunk.side_effect = create_structured_chunk

        files = [DummyUploadFile("doc.pdf", b"abc")]

        result = await pipeline.process_uploads(files)

        # Check that structured_chunk text contains reference_id
        structured_call = extraction_service.create_structured_chunk.call_args[0]
        extraction_arg, filename_arg = structured_call
        assert extraction_arg.data.reference_id == "ref-123"
        assert filename_arg == "doc.pdf"

        # Extraction summary should reflect correct reference_id
        summary = result.extractions[0]
        assert summary.reference_id == "ref-123"
        assert summary.structured_data_extracted is True

    async def test_extraction_failure_and_error_propagation(
        self, pipeline, ingestion_service, extraction_service, chunks
    ):
        # Extraction service raises error
        extraction_service.extract_data.side_effect = Exception("Extraction failed")

        files = [DummyUploadFile("fail.pdf", b"failcontent")]

        result = await pipeline.process_uploads(files)

        assert result.processed_count == 1
        assert len(result.errors) == 0
        assert len(result.extractions) == 1
        summary = result.extractions[0]
        assert summary.structured_data_extracted is False
        assert "Extraction failed" in summary.error

    async def test_ingestion_failure(
        self, pipeline, ingestion_service
    ):
        # Ingestion service raises error
        ingestion_service.process_file.side_effect = Exception("Ingestion error")

        files = [DummyUploadFile("bad.pdf", b"badcontent")]

        result = await pipeline.process_uploads(files)

        assert result.processed_count == 0
        assert len(result.errors) == 1
        assert "Error processing bad.pdf: Ingestion error" in result.errors[0]
        assert len(result.extractions) == 0

    async def test_boundary_conditions_empty_file_list(self, pipeline):
        files = []
        result = await pipeline.process_uploads(files)
        assert result.processed_count == 0
        assert len(result.errors) == 0
        assert len(result.extractions) == 0

    async def test_boundary_conditions_large_number_of_files(
        self, pipeline, ingestion_service, vector_store_service, extraction_service, chunks, structured_chunk
    ):
        files = [DummyUploadFile(f"file_{i}.pdf", f"content_{i}".encode()) for i in range(20)]
        result = await pipeline.process_uploads(files)
        assert result.processed_count == 20
        assert len(result.errors) == 0
        assert len(result.extractions) == 20
        # Each file: add_documents called twice (chunks, structured_chunk)
        assert vector_store_service.add_documents.call_count == 40
        for summary in result.extractions:
            assert summary.text_chunks == len(chunks)
            assert summary.structured_data_extracted is True
            assert summary.reference_id == "ref-123"
            assert summary.error is None
