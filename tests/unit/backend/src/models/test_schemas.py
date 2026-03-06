import pytest
from pydantic import ValidationError
from backend.src.models.schemas import (
    DocumentMetadata,
    Chunk,
    QAQuery,
    SourcedAnswer,
    ExtractionRequest,
    UploadExtractionSummary,
    UploadResponse,
)

def test_document_metadata_happy_path():
    metadata = DocumentMetadata(
        filename="doc.pdf",
        page_number=5,
        chunk_id=1,
        source="upload",
        chunk_type="structured_data"
    )
    assert metadata.filename == "doc.pdf"
    assert metadata.page_number == 5
    assert metadata.chunk_id == 1
    assert metadata.source == "upload"
    assert metadata.chunk_type == "structured_data"

def test_document_metadata_defaults_and_optional_fields():
    metadata = DocumentMetadata(
        filename="doc.txt",
        chunk_id=2,
        source="api"
    )
    assert metadata.page_number is None
    assert metadata.chunk_type == "text"

def test_document_metadata_missing_required_fields_raises():
    with pytest.raises(ValidationError):
        DocumentMetadata(filename="doc.txt", source="api")  # missing chunk_id

def test_chunk_happy_path():
    metadata = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=3,
        source="upload"
    )
    chunk = Chunk(
        text="This is a chunk of text.",
        metadata=metadata
    )
    assert chunk.text == "This is a chunk of text."
    assert chunk.metadata == metadata

def test_chunk_missing_metadata_raises():
    with pytest.raises(ValidationError):
        Chunk(text="abc")  # missing metadata

def test_qaquery_happy_path_and_default_history():
    query = QAQuery(question="What is AI?")
    assert query.question == "What is AI?"
    assert query.chat_history == []

def test_qaquery_with_chat_history():
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    query = QAQuery(question="Next?", chat_history=history)
    assert query.chat_history == history

def test_qaquery_invalid_chat_history_type_raises():
    with pytest.raises(ValidationError):
        QAQuery(question="?", chat_history="notalist")

def test_sourced_answer_happy_path():
    metadata = DocumentMetadata(filename="f", chunk_id=1, source="s")
    chunk = Chunk(text="t", metadata=metadata)
    answer = SourcedAnswer(
        answer="42",
        confidence_score=0.99,
        sources=[chunk]
    )
    assert answer.answer == "42"
    assert answer.confidence_score == 0.99
    assert answer.sources == [chunk]

def test_sourced_answer_confidence_score_boundaries():
    metadata = DocumentMetadata(filename="f", chunk_id=1, source="s")
    chunk = Chunk(text="t", metadata=metadata)
    answer = SourcedAnswer(answer="a", confidence_score=0.0, sources=[chunk])
    assert answer.confidence_score == 0.0
    answer = SourcedAnswer(answer="a", confidence_score=1.0, sources=[chunk])
    assert answer.confidence_score == 1.0

def test_sourced_answer_invalid_sources_type_raises():
    with pytest.raises(ValidationError):
        SourcedAnswer(answer="a", confidence_score=0.5, sources="notalist")

def test_extraction_request_happy_path():
    req = ExtractionRequest(document_text="Some text")
    assert req.document_text == "Some text"

def test_extraction_request_missing_document_text_raises():
    with pytest.raises(ValidationError):
        ExtractionRequest()

def test_upload_extraction_summary_happy_path():
    summary = UploadExtractionSummary(
        filename="file.csv",
        text_chunks=10,
        structured_data_extracted=True,
        reference_id="abc123",
        error=None
    )
    assert summary.filename == "file.csv"
    assert summary.text_chunks == 10
    assert summary.structured_data_extracted is True
    assert summary.reference_id == "abc123"
    assert summary.error is None

def test_upload_extraction_summary_optional_fields():
    summary = UploadExtractionSummary(
        filename="file.csv",
        text_chunks=0,
        structured_data_extracted=False
    )
    assert summary.reference_id is None
    assert summary.error is None

def test_upload_extraction_summary_missing_required_fields_raises():
    with pytest.raises(ValidationError):
        UploadExtractionSummary(filename="f", structured_data_extracted=True)  # missing text_chunks

def test_upload_response_happy_path():
    summary1 = UploadExtractionSummary(
        filename="f1", text_chunks=1, structured_data_extracted=True
    )
    summary2 = UploadExtractionSummary(
        filename="f2", text_chunks=2, structured_data_extracted=False, error="Parse error"
    )
    resp = UploadResponse(
        message="Upload complete",
        errors=["err1"],
        extractions=[summary1, summary2]
    )
    assert resp.message == "Upload complete"
    assert resp.errors == ["err1"]
    assert resp.extractions == [summary1, summary2]

def test_upload_response_empty_errors_and_extractions():
    resp = UploadResponse(
        message="Done",
        errors=[],
        extractions=[]
    )
    assert resp.errors == []
    assert resp.extractions == []

def test_upload_response_invalid_extractions_type_raises():
    with pytest.raises(ValidationError):
        UploadResponse(message="m", errors=[], extractions="notalist")
