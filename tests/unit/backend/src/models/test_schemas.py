import pytest
from pydantic import ValidationError
from backend.src.models import schemas

def test_document_metadata_happy_path():
    metadata = schemas.DocumentMetadata(
        filename="doc.pdf",
        page_number=2,
        chunk_id=1,
        source="upload",
        chunk_type="structured_data"
    )
    assert metadata.filename == "doc.pdf"
    assert metadata.page_number == 2
    assert metadata.chunk_id == 1
    assert metadata.source == "upload"
    assert metadata.chunk_type == "structured_data"

def test_document_metadata_default_chunk_type():
    metadata = schemas.DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    assert metadata.chunk_type == "text"

def test_document_metadata_optional_page_number_none():
    metadata = schemas.DocumentMetadata(
        filename="doc.pdf",
        page_number=None,
        chunk_id=1,
        source="upload"
    )
    assert metadata.page_number is None

def test_document_metadata_missing_required_fields_raises():
    with pytest.raises(ValidationError) as excinfo:
        schemas.DocumentMetadata(
            filename="doc.pdf",
            source="upload"
        )
    assert "chunk_id" in str(excinfo.value)

def test_chunk_happy_path():
    metadata = schemas.DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    chunk = schemas.Chunk(
        text="This is a chunk.",
        metadata=metadata
    )
    assert chunk.text == "This is a chunk."
    assert chunk.metadata == metadata

def test_chunk_invalid_metadata_type_raises():
    with pytest.raises(ValidationError):
        schemas.Chunk(
            text="Some text",
            metadata={"filename": "doc.pdf", "chunk_id": 1, "source": "upload"}
        )

def test_qaquery_happy_path():
    chat_history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    query = schemas.QAQuery(
        question="What is AI?",
        chat_history=chat_history
    )
    assert query.question == "What is AI?"
    assert query.chat_history == chat_history

def test_qaquery_default_chat_history():
    query = schemas.QAQuery(question="What is AI?")
    assert query.chat_history == []

def test_qaquery_invalid_chat_history_type_raises():
    with pytest.raises(ValidationError):
        schemas.QAQuery(
            question="What is AI?",
            chat_history="not a list"
        )

def test_sourced_answer_happy_path():
    metadata = schemas.DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    chunk = schemas.Chunk(
        text="Chunk text",
        metadata=metadata
    )
    answer = schemas.SourcedAnswer(
        answer="42",
        confidence_score=0.99,
        sources=[chunk]
    )
    assert answer.answer == "42"
    assert answer.confidence_score == 0.99
    assert answer.sources == [chunk]

def test_sourced_answer_empty_sources():
    answer = schemas.SourcedAnswer(
        answer="No sources",
        confidence_score=0.0,
        sources=[]
    )
    assert answer.sources == []

def test_sourced_answer_confidence_score_bounds():
    answer = schemas.SourcedAnswer(
        answer="test",
        confidence_score=1.0,
        sources=[]
    )
    assert answer.confidence_score == 1.0
    answer = schemas.SourcedAnswer(
        answer="test",
        confidence_score=0.0,
        sources=[]
    )
    assert answer.confidence_score == 0.0

def test_sourced_answer_invalid_sources_type_raises():
    with pytest.raises(ValidationError):
        schemas.SourcedAnswer(
            answer="test",
            confidence_score=0.5,
            sources="not a list"
        )

def test_extraction_request_happy_path():
    req = schemas.ExtractionRequest(document_text="Some text")
    assert req.document_text == "Some text"

def test_extraction_request_missing_document_text_raises():
    with pytest.raises(ValidationError):
        schemas.ExtractionRequest()

def test_upload_extraction_summary_happy_path():
    summary = schemas.UploadExtractionSummary(
        filename="doc.pdf",
        text_chunks=5,
        structured_data_extracted=True,
        reference_id="abc123",
        error=None
    )
    assert summary.filename == "doc.pdf"
    assert summary.text_chunks == 5
    assert summary.structured_data_extracted is True
    assert summary.reference_id == "abc123"
    assert summary.error is None

def test_upload_extraction_summary_optional_fields_none():
    summary = schemas.UploadExtractionSummary(
        filename="doc.pdf",
        text_chunks=0,
        structured_data_extracted=False
    )
    assert summary.reference_id is None
    assert summary.error is None

def test_upload_extraction_summary_missing_required_fields_raises():
    with pytest.raises(ValidationError):
        schemas.UploadExtractionSummary(
            text_chunks=1,
            structured_data_extracted=True
        )

def test_upload_response_happy_path():
    summary1 = schemas.UploadExtractionSummary(
        filename="doc1.pdf",
        text_chunks=2,
        structured_data_extracted=True
    )
    summary2 = schemas.UploadExtractionSummary(
        filename="doc2.pdf",
        text_chunks=3,
        structured_data_extracted=False,
        error="Parse error"
    )
    resp = schemas.UploadResponse(
        message="Upload complete",
        errors=["error1", "error2"],
        extractions=[summary1, summary2]
    )
    assert resp.message == "Upload complete"
    assert resp.errors == ["error1", "error2"]
    assert resp.extractions == [summary1, summary2]

def test_upload_response_empty_errors_and_extractions():
    resp = schemas.UploadResponse(
        message="No errors",
        errors=[],
        extractions=[]
    )
    assert resp.errors == []
    assert resp.extractions == []

def test_upload_response_missing_required_fields_raises():
    with pytest.raises(ValidationError):
        schemas.UploadResponse(
            message="Missing fields"
        )

def test_upload_response_invalid_extractions_type_raises():
    with pytest.raises(ValidationError):
        schemas.UploadResponse(
            message="Invalid",
            errors=[],
            extractions="not a list"
        )
