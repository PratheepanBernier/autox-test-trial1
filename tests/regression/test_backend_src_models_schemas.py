import pytest
from pydantic import ValidationError
from datetime import datetime
from backend.src.models.schemas import (
    DocumentMetadata,
    Chunk,
    QAQuery,
    SourcedAnswer,
    ExtractionRequest,
)

def test_document_metadata_happy_path():
    meta = DocumentMetadata(
        filename="doc.pdf",
        page_number=5,
        chunk_id=1,
        source="upload",
        chunk_type="structured_data"
    )
    assert meta.filename == "doc.pdf"
    assert meta.page_number == 5
    assert meta.chunk_id == 1
    assert meta.source == "upload"
    assert meta.chunk_type == "structured_data"

def test_document_metadata_defaults_and_optional_page_number():
    meta = DocumentMetadata(
        filename="doc2.pdf",
        chunk_id=2,
        source="api"
    )
    assert meta.filename == "doc2.pdf"
    assert meta.page_number is None
    assert meta.chunk_id == 2
    assert meta.source == "api"
    assert meta.chunk_type == "text"

def test_document_metadata_missing_required_fields_raises():
    with pytest.raises(ValidationError) as excinfo:
        DocumentMetadata(
            page_number=1,
            chunk_id=3,
            source="missing_filename"
        )
    assert "filename" in str(excinfo.value)

def test_document_metadata_invalid_chunk_type():
    # chunk_type is not validated, so any string is accepted
    meta = DocumentMetadata(
        filename="doc3.pdf",
        chunk_id=4,
        source="test",
        chunk_type="invalid_type"
    )
    assert meta.chunk_type == "invalid_type"

def test_chunk_happy_path():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    chunk = Chunk(
        text="This is a chunk of text.",
        metadata=meta
    )
    assert chunk.text == "This is a chunk of text."
    assert chunk.metadata == meta

def test_chunk_missing_metadata_raises():
    with pytest.raises(ValidationError) as excinfo:
        Chunk(
            text="Missing metadata"
        )
    assert "metadata" in str(excinfo.value)

def test_chunk_empty_text():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    chunk = Chunk(
        text="",
        metadata=meta
    )
    assert chunk.text == ""

def test_qaquery_happy_path():
    q = QAQuery(
        question="What is the capital of France?",
        chat_history=[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    )
    assert q.question == "What is the capital of France?"
    assert isinstance(q.chat_history, list)
    assert q.chat_history[0]["role"] == "user"

def test_qaquery_default_chat_history():
    q = QAQuery(
        question="What is the capital of Germany?"
    )
    assert q.chat_history == []

def test_qaquery_invalid_chat_history_type():
    with pytest.raises(ValidationError) as excinfo:
        QAQuery(
            question="Test",
            chat_history="not a list"
        )
    assert "value is not a valid list" in str(excinfo.value)

def test_sourced_answer_happy_path():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    chunk = Chunk(
        text="Paris is the capital of France.",
        metadata=meta
    )
    answer = SourcedAnswer(
        answer="Paris",
        confidence_score=0.95,
        sources=[chunk]
    )
    assert answer.answer == "Paris"
    assert answer.confidence_score == 0.95
    assert answer.sources[0] == chunk

def test_sourced_answer_empty_sources():
    answer = SourcedAnswer(
        answer="No sources found.",
        confidence_score=0.0,
        sources=[]
    )
    assert answer.sources == []

def test_sourced_answer_confidence_score_boundaries():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    chunk = Chunk(
        text="Test",
        metadata=meta
    )
    answer_low = SourcedAnswer(
        answer="Low",
        confidence_score=0.0,
        sources=[chunk]
    )
    answer_high = SourcedAnswer(
        answer="High",
        confidence_score=1.0,
        sources=[chunk]
    )
    assert answer_low.confidence_score == 0.0
    assert answer_high.confidence_score == 1.0

def test_sourced_answer_invalid_confidence_score_type():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    chunk = Chunk(
        text="Test",
        metadata=meta
    )
    with pytest.raises(ValidationError) as excinfo:
        SourcedAnswer(
            answer="Test",
            confidence_score="not a float",
            sources=[chunk]
        )
    assert "value is not a valid float" in str(excinfo.value)

def test_extraction_request_happy_path():
    req = ExtractionRequest(
        document_text="This is the document text."
    )
    assert req.document_text == "This is the document text."

def test_extraction_request_empty_document_text():
    req = ExtractionRequest(
        document_text=""
    )
    assert req.document_text == ""

def test_extraction_request_missing_document_text_raises():
    with pytest.raises(ValidationError) as excinfo:
        ExtractionRequest()
    assert "document_text" in str(excinfo.value)
