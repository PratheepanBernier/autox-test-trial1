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

def test_document_metadata_defaults_and_optional():
    meta = DocumentMetadata(
        filename="file.txt",
        chunk_id=42,
        source="api"
    )
    assert meta.filename == "file.txt"
    assert meta.page_number is None
    assert meta.chunk_id == 42
    assert meta.source == "api"
    assert meta.chunk_type == "text"

def test_document_metadata_missing_required_fields_raises():
    with pytest.raises(ValidationError) as excinfo:
        DocumentMetadata()
    errors = excinfo.value.errors()
    required_fields = {e['loc'][0] for e in errors}
    assert "filename" in required_fields
    assert "chunk_id" in required_fields
    assert "source" in required_fields

def test_document_metadata_invalid_types():
    with pytest.raises(ValidationError):
        DocumentMetadata(
            filename=123,  # should be str
            chunk_id="not-an-int",  # should be int
            source=456,  # should be str
        )

def test_chunk_happy_path():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=5,
        source="system"
    )
    chunk = Chunk(
        text="This is a chunk of text.",
        metadata=meta
    )
    assert chunk.text == "This is a chunk of text."
    assert isinstance(chunk.metadata, DocumentMetadata)
    assert chunk.metadata.filename == "doc.pdf"

def test_chunk_invalid_metadata_type():
    with pytest.raises(ValidationError):
        Chunk(
            text="Some text",
            metadata={"filename": "a", "chunk_id": 1, "source": "b"}  # not a DocumentMetadata instance
        )

def test_qaquery_happy_path():
    query = QAQuery(
        question="What is the capital of France?",
        chat_history=[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    )
    assert query.question == "What is the capital of France?"
    assert isinstance(query.chat_history, list)
    assert query.chat_history[0]["role"] == "user"

def test_qaquery_default_chat_history():
    query = QAQuery(question="Test?")
    assert query.chat_history == []

def test_qaquery_invalid_chat_history_type():
    with pytest.raises(ValidationError):
        QAQuery(question="Test?", chat_history="not-a-list")

def test_sourced_answer_happy_path():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=7,
        source="system"
    )
    chunk = Chunk(
        text="Relevant text.",
        metadata=meta
    )
    answer = SourcedAnswer(
        answer="Paris",
        confidence_score=0.95,
        sources=[chunk]
    )
    assert answer.answer == "Paris"
    assert answer.confidence_score == 0.95
    assert len(answer.sources) == 1
    assert isinstance(answer.sources[0], Chunk)

def test_sourced_answer_confidence_score_boundaries():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=8,
        source="system"
    )
    chunk = Chunk(
        text="Boundary test.",
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
        chunk_id=9,
        source="system"
    )
    chunk = Chunk(
        text="Invalid score.",
        metadata=meta
    )
    with pytest.raises(ValidationError):
        SourcedAnswer(
            answer="Oops",
            confidence_score="not-a-float",
            sources=[chunk]
        )

def test_sourced_answer_sources_empty_list():
    answer = SourcedAnswer(
        answer="No sources",
        confidence_score=0.5,
        sources=[]
    )
    assert answer.sources == []

def test_extraction_request_happy_path():
    req = ExtractionRequest(document_text="Some document content.")
    assert req.document_text == "Some document content."

def test_extraction_request_missing_document_text():
    with pytest.raises(ValidationError):
        ExtractionRequest()

def test_extraction_request_invalid_document_text_type():
    with pytest.raises(ValidationError):
        ExtractionRequest(document_text=12345)

def test_document_metadata_repr_and_equality():
    meta1 = DocumentMetadata(
        filename="file1.txt",
        chunk_id=1,
        source="src"
    )
    meta2 = DocumentMetadata(
        filename="file1.txt",
        chunk_id=1,
        source="src"
    )
    assert repr(meta1).startswith("DocumentMetadata(")
    assert meta1 == meta2

def test_chunk_repr_and_equality():
    meta = DocumentMetadata(
        filename="file2.txt",
        chunk_id=2,
        source="src"
    )
    chunk1 = Chunk(text="abc", metadata=meta)
    chunk2 = Chunk(text="abc", metadata=meta)
    assert repr(chunk1).startswith("Chunk(")
    assert chunk1 == chunk2

def test_qaquery_repr_and_equality():
    q1 = QAQuery(question="Q?", chat_history=[])
    q2 = QAQuery(question="Q?", chat_history=[])
    assert repr(q1).startswith("QAQuery(")
    assert q1 == q2

def test_sourced_answer_repr_and_equality():
    meta = DocumentMetadata(
        filename="file3.txt",
        chunk_id=3,
        source="src"
    )
    chunk = Chunk(text="xyz", metadata=meta)
    a1 = SourcedAnswer(answer="A", confidence_score=0.8, sources=[chunk])
    a2 = SourcedAnswer(answer="A", confidence_score=0.8, sources=[chunk])
    assert repr(a1).startswith("SourcedAnswer(")
    assert a1 == a2

def test_extraction_request_repr_and_equality():
    r1 = ExtractionRequest(document_text="abc")
    r2 = ExtractionRequest(document_text="abc")
    assert repr(r1).startswith("ExtractionRequest(")
    assert r1 == r2
