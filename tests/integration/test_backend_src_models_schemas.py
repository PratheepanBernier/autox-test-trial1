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
        filename="file.pdf",
        page_number=2,
        chunk_id=1,
        source="upload",
        chunk_type="structured_data"
    )
    assert meta.filename == "file.pdf"
    assert meta.page_number == 2
    assert meta.chunk_id == 1
    assert meta.source == "upload"
    assert meta.chunk_type == "structured_data"

def test_document_metadata_default_chunk_type():
    meta = DocumentMetadata(
        filename="file.pdf",
        chunk_id=2,
        source="upload"
    )
    assert meta.chunk_type == "text"

def test_document_metadata_optional_page_number():
    meta = DocumentMetadata(
        filename="file.pdf",
        chunk_id=3,
        source="upload"
    )
    assert meta.page_number is None

def test_document_metadata_missing_required_fields_raises():
    with pytest.raises(ValidationError) as excinfo:
        DocumentMetadata(
            filename="file.pdf",
            source="upload"
        )
    assert "chunk_id" in str(excinfo.value)

def test_chunk_happy_path():
    meta = DocumentMetadata(
        filename="doc.txt",
        chunk_id=5,
        source="api"
    )
    chunk = Chunk(
        text="This is a chunk of text.",
        metadata=meta
    )
    assert chunk.text == "This is a chunk of text."
    assert chunk.metadata.filename == "doc.txt"

def test_chunk_invalid_metadata_type_raises():
    with pytest.raises(ValidationError):
        Chunk(
            text="Some text",
            metadata={"filename": "doc.txt", "chunk_id": 1, "source": "api"}
        )

def test_qaquery_happy_path():
    query = QAQuery(
        question="What is the capital of France?",
        chat_history=[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    )
    assert query.question == "What is the capital of France?"
    assert len(query.chat_history) == 2
    assert query.chat_history[0]["role"] == "user"

def test_qaquery_default_chat_history():
    query = QAQuery(question="What is AI?")
    assert query.chat_history == []

def test_qaquery_invalid_chat_history_type_raises():
    with pytest.raises(ValidationError):
        QAQuery(question="Q?", chat_history="not a list")

def test_sourced_answer_happy_path():
    meta = DocumentMetadata(
        filename="source.pdf",
        chunk_id=10,
        source="reference"
    )
    chunk = Chunk(
        text="Relevant text.",
        metadata=meta
    )
    answer = SourcedAnswer(
        answer="42",
        confidence_score=0.99,
        sources=[chunk]
    )
    assert answer.answer == "42"
    assert answer.confidence_score == 0.99
    assert len(answer.sources) == 1
    assert answer.sources[0].text == "Relevant text."

def test_sourced_answer_confidence_score_boundary():
    meta = DocumentMetadata(
        filename="source.pdf",
        chunk_id=11,
        source="reference"
    )
    chunk = Chunk(
        text="Boundary test.",
        metadata=meta
    )
    answer = SourcedAnswer(
        answer="Boundary",
        confidence_score=0.0,
        sources=[chunk]
    )
    assert answer.confidence_score == 0.0
    answer = SourcedAnswer(
        answer="Boundary",
        confidence_score=1.0,
        sources=[chunk]
    )
    assert answer.confidence_score == 1.0

def test_sourced_answer_empty_sources():
    answer = SourcedAnswer(
        answer="No sources",
        confidence_score=0.5,
        sources=[]
    )
    assert answer.sources == []

def test_sourced_answer_invalid_sources_type_raises():
    with pytest.raises(ValidationError):
        SourcedAnswer(
            answer="Invalid",
            confidence_score=0.5,
            sources="not a list"
        )

def test_extraction_request_happy_path():
    req = ExtractionRequest(document_text="Extract this text.")
    assert req.document_text == "Extract this text."

def test_extraction_request_missing_document_text_raises():
    with pytest.raises(ValidationError):
        ExtractionRequest()

def test_document_metadata_repr_and_equality():
    meta1 = DocumentMetadata(
        filename="a.pdf",
        chunk_id=1,
        source="src"
    )
    meta2 = DocumentMetadata(
        filename="a.pdf",
        chunk_id=1,
        source="src"
    )
    assert meta1 == meta2
    assert repr(meta1).startswith("DocumentMetadata(")

def test_chunk_repr_and_equality():
    meta = DocumentMetadata(
        filename="b.pdf",
        chunk_id=2,
        source="src"
    )
    chunk1 = Chunk(text="abc", metadata=meta)
    chunk2 = Chunk(text="abc", metadata=meta)
    assert chunk1 == chunk2
    assert repr(chunk1).startswith("Chunk(")

def test_qaquery_repr_and_equality():
    q1 = QAQuery(question="Q?", chat_history=[])
    q2 = QAQuery(question="Q?", chat_history=[])
    assert q1 == q2
    assert repr(q1).startswith("QAQuery(")

def test_sourced_answer_repr_and_equality():
    meta = DocumentMetadata(
        filename="c.pdf",
        chunk_id=3,
        source="src"
    )
    chunk = Chunk(text="xyz", metadata=meta)
    a1 = SourcedAnswer(answer="ans", confidence_score=0.7, sources=[chunk])
    a2 = SourcedAnswer(answer="ans", confidence_score=0.7, sources=[chunk])
    assert a1 == a2
    assert repr(a1).startswith("SourcedAnswer(")

def test_extraction_request_repr_and_equality():
    r1 = ExtractionRequest(document_text="abc")
    r2 = ExtractionRequest(document_text="abc")
    assert r1 == r2
    assert repr(r1).startswith("ExtractionRequest(")
