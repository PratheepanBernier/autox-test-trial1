import pytest
from pydantic import ValidationError
from backend.src.models.schemas import (
    DocumentMetadata,
    Chunk,
    QAQuery,
    SourcedAnswer,
    ExtractionRequest,
)

def test_document_metadata_equivalent_paths():
    # Happy path: all fields provided
    meta1 = DocumentMetadata(
        filename="doc.pdf",
        page_number=2,
        chunk_id=1,
        source="upload",
        chunk_type="text"
    )
    # Omit chunk_type (should default)
    meta2 = DocumentMetadata(
        filename="doc.pdf",
        page_number=2,
        chunk_id=1,
        source="upload"
    )
    assert meta1.filename == meta2.filename
    assert meta1.page_number == meta2.page_number
    assert meta1.chunk_id == meta2.chunk_id
    assert meta1.source == meta2.source
    assert meta1.chunk_type == meta2.chunk_type == "text"

def test_document_metadata_boundary_and_error_handling():
    # Edge: page_number is None
    meta = DocumentMetadata(
        filename="doc.pdf",
        page_number=None,
        chunk_id=1,
        source="upload"
    )
    assert meta.page_number is None

    # Error: missing required fields
    with pytest.raises(ValidationError):
        DocumentMetadata(filename="doc.pdf", chunk_id=1)  # missing source

    # Error: wrong type for chunk_id
    with pytest.raises(ValidationError):
        DocumentMetadata(filename="doc.pdf", chunk_id="not-an-int", source="upload")

def test_chunk_equivalent_paths():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    # Happy path
    chunk1 = Chunk(text="Hello", metadata=meta)
    # Construct metadata as dict
    chunk2 = Chunk(text="Hello", metadata=meta.dict())
    assert chunk1.text == chunk2.text
    assert chunk1.metadata == chunk2.metadata

def test_chunk_error_handling():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    # Error: missing text
    with pytest.raises(ValidationError):
        Chunk(metadata=meta)
    # Error: missing metadata
    with pytest.raises(ValidationError):
        Chunk(text="Hello")

def test_qaquery_equivalent_paths():
    # Happy path: with chat_history
    q1 = QAQuery(question="What is AI?", chat_history=[{"role": "user", "content": "Hi"}])
    # Omit chat_history (should default to empty list)
    q2 = QAQuery(question="What is AI?")
    assert q1.question == q2.question
    assert isinstance(q2.chat_history, list)
    assert q2.chat_history == []

def test_qaquery_error_handling():
    # Error: missing question
    with pytest.raises(ValidationError):
        QAQuery(chat_history=[])
    # Error: wrong type for chat_history
    with pytest.raises(ValidationError):
        QAQuery(question="What?", chat_history="not-a-list")

def test_sourced_answer_equivalent_paths():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    chunk = Chunk(text="Hello", metadata=meta)
    # Happy path
    ans1 = SourcedAnswer(
        answer="42",
        confidence_score=0.99,
        sources=[chunk]
    )
    # Construct sources as list of dicts
    ans2 = SourcedAnswer(
        answer="42",
        confidence_score=0.99,
        sources=[chunk.dict()]
    )
    assert ans1.answer == ans2.answer
    assert ans1.confidence_score == ans2.confidence_score
    assert ans1.sources == ans2.sources

def test_sourced_answer_boundary_and_error_handling():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    chunk = Chunk(text="Hello", metadata=meta)
    # Edge: confidence_score at boundary
    ans = SourcedAnswer(
        answer="yes",
        confidence_score=0.0,
        sources=[chunk]
    )
    assert ans.confidence_score == 0.0

    # Error: missing required fields
    with pytest.raises(ValidationError):
        SourcedAnswer(answer="yes", confidence_score=0.5)
    with pytest.raises(ValidationError):
        SourcedAnswer(confidence_score=0.5, sources=[chunk])

    # Error: wrong type for sources
    with pytest.raises(ValidationError):
        SourcedAnswer(answer="yes", confidence_score=0.5, sources="not-a-list")

def test_extraction_request_equivalent_paths():
    # Happy path
    req1 = ExtractionRequest(document_text="Some text")
    # Construct from dict
    req2 = ExtractionRequest(**{"document_text": "Some text"})
    assert req1.document_text == req2.document_text

def test_extraction_request_error_handling():
    # Error: missing document_text
    with pytest.raises(ValidationError):
        ExtractionRequest()
    # Error: wrong type
    with pytest.raises(ValidationError):
        ExtractionRequest(document_text=123)
