import pytest
from pydantic import ValidationError
from backend.src.models.schemas import (
    DocumentMetadata,
    Chunk,
    QAQuery,
    SourcedAnswer,
    ExtractionRequest,
)
from datetime import datetime

def test_document_metadata_happy_path():
    metadata = DocumentMetadata(
        filename="file.pdf",
        page_number=2,
        chunk_id=1,
        source="upload",
        chunk_type="structured_data"
    )
    assert metadata.filename == "file.pdf"
    assert metadata.page_number == 2
    assert metadata.chunk_id == 1
    assert metadata.source == "upload"
    assert metadata.chunk_type == "structured_data"

def test_document_metadata_default_chunk_type():
    metadata = DocumentMetadata(
        filename="file.pdf",
        chunk_id=1,
        source="upload"
    )
    assert metadata.chunk_type == "text"

def test_document_metadata_optional_page_number_none():
    metadata = DocumentMetadata(
        filename="file.pdf",
        chunk_id=1,
        source="upload"
    )
    assert metadata.page_number is None

def test_document_metadata_missing_required_fields_raises():
    with pytest.raises(ValidationError) as excinfo:
        DocumentMetadata(
            filename="file.pdf",
            source="upload"
        )
    assert "chunk_id" in str(excinfo.value)

def test_document_metadata_invalid_chunk_id_type():
    with pytest.raises(ValidationError):
        DocumentMetadata(
            filename="file.pdf",
            chunk_id="not_an_int",
            source="upload"
        )

def test_chunk_happy_path():
    metadata = DocumentMetadata(
        filename="doc.txt",
        chunk_id=10,
        source="api"
    )
    chunk = Chunk(
        text="This is a chunk of text.",
        metadata=metadata
    )
    assert chunk.text == "This is a chunk of text."
    assert chunk.metadata == metadata

def test_chunk_invalid_metadata_type():
    with pytest.raises(ValidationError):
        Chunk(
            text="Some text",
            metadata={"filename": "doc.txt", "chunk_id": 1, "source": "api"}
        )

def test_qaquery_happy_path():
    query = QAQuery(
        question="What is the capital of France?",
        chat_history=[{"role": "user", "content": "Hello"}]
    )
    assert query.question == "What is the capital of France?"
    assert query.chat_history == [{"role": "user", "content": "Hello"}]

def test_qaquery_default_chat_history():
    query = QAQuery(question="What is AI?")
    assert query.chat_history == []

def test_qaquery_invalid_chat_history_type():
    with pytest.raises(ValidationError):
        QAQuery(
            question="Test?",
            chat_history="not_a_list"
        )

def test_sourced_answer_happy_path():
    metadata = DocumentMetadata(
        filename="source.pdf",
        chunk_id=5,
        source="external"
    )
    chunk = Chunk(
        text="Relevant text.",
        metadata=metadata
    )
    answer = SourcedAnswer(
        answer="42",
        confidence_score=0.99,
        sources=[chunk]
    )
    assert answer.answer == "42"
    assert answer.confidence_score == 0.99
    assert answer.sources == [chunk]

def test_sourced_answer_empty_sources():
    answer = SourcedAnswer(
        answer="No sources",
        confidence_score=0.5,
        sources=[]
    )
    assert answer.sources == []

def test_sourced_answer_confidence_score_bounds():
    # No explicit bounds in schema, but test float edge cases
    answer = SourcedAnswer(
        answer="Low confidence",
        confidence_score=0.0,
        sources=[]
    )
    assert answer.confidence_score == 0.0
    answer = SourcedAnswer(
        answer="High confidence",
        confidence_score=1.0,
        sources=[]
    )
    assert answer.confidence_score == 1.0

def test_sourced_answer_invalid_sources_type():
    with pytest.raises(ValidationError):
        SourcedAnswer(
            answer="Bad sources",
            confidence_score=0.5,
            sources="not_a_list"
        )

def test_extraction_request_happy_path():
    req = ExtractionRequest(document_text="Extract this text.")
    assert req.document_text == "Extract this text."

def test_extraction_request_missing_document_text():
    with pytest.raises(ValidationError):
        ExtractionRequest()

def test_extraction_request_empty_document_text():
    req = ExtractionRequest(document_text="")
    assert req.document_text == ""

def test_document_metadata_repr_and_equality():
    meta1 = DocumentMetadata(
        filename="a.txt",
        chunk_id=1,
        source="src"
    )
    meta2 = DocumentMetadata(
        filename="a.txt",
        chunk_id=1,
        source="src"
    )
    assert meta1 == meta2
    assert repr(meta1).startswith("DocumentMetadata(")
