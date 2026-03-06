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
    meta = DocumentMetadata(
        filename="doc.pdf",
        page_number=2,
        chunk_id=1,
        source="upload",
        chunk_type="structured_data"
    )
    assert meta.filename == "doc.pdf"
    assert meta.page_number == 2
    assert meta.chunk_id == 1
    assert meta.source == "upload"
    assert meta.chunk_type == "structured_data"

def test_document_metadata_defaults_and_optional_fields():
    meta = DocumentMetadata(
        filename="doc.txt",
        chunk_id=5,
        source="api"
    )
    assert meta.page_number is None
    assert meta.chunk_type == "text"

def test_document_metadata_missing_required_fields_raises():
    with pytest.raises(ValidationError):
        DocumentMetadata(filename="doc.txt", source="api")  # missing chunk_id

def test_chunk_happy_path():
    meta = DocumentMetadata(
        filename="file.docx",
        chunk_id=10,
        source="system"
    )
    chunk = Chunk(text="Hello world", metadata=meta)
    assert chunk.text == "Hello world"
    assert chunk.metadata == meta

def test_chunk_invalid_metadata_type_raises():
    with pytest.raises(ValidationError):
        Chunk(text="abc", metadata={"filename": "f", "chunk_id": 1, "source": "s"})

def test_qaquery_happy_path_and_default_history():
    query = QAQuery(question="What is AI?")
    assert query.question == "What is AI?"
    assert query.chat_history == []

def test_qaquery_with_chat_history():
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    query = QAQuery(question="Next?", chat_history=history)
    assert query.chat_history == history

def test_qaquery_invalid_chat_history_raises():
    with pytest.raises(ValidationError):
        QAQuery(question="?", chat_history="not a list")

def test_sourced_answer_happy_path():
    meta = DocumentMetadata(filename="f", chunk_id=1, source="s")
    chunk = Chunk(text="t", metadata=meta)
    answer = SourcedAnswer(answer="42", confidence_score=0.99, sources=[chunk])
    assert answer.answer == "42"
    assert answer.confidence_score == 0.99
    assert answer.sources == [chunk]

def test_sourced_answer_confidence_score_boundary():
    meta = DocumentMetadata(filename="f", chunk_id=1, source="s")
    chunk = Chunk(text="t", metadata=meta)
    answer = SourcedAnswer(answer="ans", confidence_score=0.0, sources=[chunk])
    assert answer.confidence_score == 0.0
    answer = SourcedAnswer(answer="ans", confidence_score=1.0, sources=[chunk])
    assert answer.confidence_score == 1.0

def test_sourced_answer_empty_sources():
    answer = SourcedAnswer(answer="none", confidence_score=0.5, sources=[])
    assert answer.sources == []

def test_extraction_request_happy_path():
    req = ExtractionRequest(document_text="Some text")
    assert req.document_text == "Some text"

def test_extraction_request_missing_document_text_raises():
    with pytest.raises(ValidationError):
        ExtractionRequest()

def test_upload_extraction_summary_happy_path():
    summary = UploadExtractionSummary(
        filename="file.csv",
        text_chunks=3,
        structured_data_extracted=True,
        reference_id="abc123",
        error=None
    )
    assert summary.filename == "file.csv"
    assert summary.text_chunks == 3
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
        UploadExtractionSummary(filename="f", structured_data_extracted=True)

def test_upload_response_happy_path():
    summary1 = UploadExtractionSummary(
        filename="f1", text_chunks=1, structured_data_extracted=True
    )
    summary2 = UploadExtractionSummary(
        filename="f2", text_chunks=2, structured_data_extracted=False, error="err"
    )
    resp = UploadResponse(
        message="Done",
        errors=["err1", "err2"],
        extractions=[summary1, summary2]
    )
    assert resp.message == "Done"
    assert resp.errors == ["err1", "err2"]
    assert resp.extractions == [summary1, summary2]

def test_upload_response_empty_errors_and_extractions():
    resp = UploadResponse(message="OK", errors=[], extractions=[])
    assert resp.errors == []
    assert resp.extractions == []

def test_upload_response_missing_required_fields_raises():
    with pytest.raises(ValidationError):
        UploadResponse(message="msg", errors=[])
    with pytest.raises(ValidationError):
        UploadResponse(errors=[], extractions=[])
    with pytest.raises(ValidationError):
        UploadResponse(message="msg", extractions=[])
