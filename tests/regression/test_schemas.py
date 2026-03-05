import pytest
from pydantic import ValidationError
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
        source="api"
    )
    assert meta.chunk_type == "text"

def test_document_metadata_optional_page_number_none():
    meta = DocumentMetadata(
        filename="file.pdf",
        chunk_id=3,
        source="api"
    )
    assert meta.page_number is None

def test_document_metadata_missing_required_fields_raises():
    with pytest.raises(ValidationError) as excinfo:
        DocumentMetadata(
            filename="file.pdf",
            source="api"
        )
    assert "chunk_id" in str(excinfo.value)

def test_document_metadata_invalid_chunk_type():
    meta = DocumentMetadata(
        filename="file.pdf",
        chunk_id=4,
        source="api",
        chunk_type="invalid_type"
    )
    # Pydantic does not restrict chunk_type, so it should accept any string
    assert meta.chunk_type == "invalid_type"

def test_chunk_happy_path():
    meta = DocumentMetadata(
        filename="doc.txt",
        chunk_id=5,
        source="system"
    )
    chunk = Chunk(
        text="This is a chunk of text.",
        metadata=meta
    )
    assert chunk.text == "This is a chunk of text."
    assert chunk.metadata == meta

def test_chunk_empty_text():
    meta = DocumentMetadata(
        filename="doc.txt",
        chunk_id=6,
        source="system"
    )
    chunk = Chunk(
        text="",
        metadata=meta
    )
    assert chunk.text == ""

def test_chunk_invalid_metadata_type_raises():
    with pytest.raises(ValidationError):
        Chunk(
            text="Some text",
            metadata={"filename": "a", "chunk_id": 1, "source": "b"}
        )

def test_qaquery_happy_path():
    query = QAQuery(
        question="What is the capital of France?",
        chat_history=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    )
    assert query.question == "What is the capital of France?"
    assert len(query.chat_history) == 2

def test_qaquery_default_chat_history():
    query = QAQuery(
        question="What is the capital of Germany?"
    )
    assert query.chat_history == []

def test_qaquery_empty_question_accepted():
    query = QAQuery(
        question=""
    )
    assert query.question == ""

def test_qaquery_invalid_chat_history_type_raises():
    with pytest.raises(ValidationError):
        QAQuery(
            question="Test",
            chat_history="not a list"
        )

def test_sourced_answer_happy_path():
    meta = DocumentMetadata(
        filename="doc.txt",
        chunk_id=7,
        source="system"
    )
    chunk = Chunk(
        text="Some text",
        metadata=meta
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
        confidence_score=0.0,
        sources=[]
    )
    assert answer.sources == []

def test_sourced_answer_confidence_score_boundaries():
    meta = DocumentMetadata(
        filename="doc.txt",
        chunk_id=8,
        source="system"
    )
    chunk = Chunk(
        text="Some text",
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

def test_sourced_answer_invalid_sources_type_raises():
    with pytest.raises(ValidationError):
        SourcedAnswer(
            answer="Invalid",
            confidence_score=0.5,
            sources="not a list"
        )

def test_extraction_request_happy_path():
    req = ExtractionRequest(
        document_text="Extract this text."
    )
    assert req.document_text == "Extract this text."

def test_extraction_request_empty_document_text():
    req = ExtractionRequest(
        document_text=""
    )
    assert req.document_text == ""

def test_extraction_request_missing_document_text_raises():
    with pytest.raises(ValidationError):
        ExtractionRequest()

def test_document_metadata_equivalent_paths_reconciliation():
    # chunk_type default vs explicit
    meta_default = DocumentMetadata(
        filename="file.pdf",
        chunk_id=9,
        source="api"
    )
    meta_explicit = DocumentMetadata(
        filename="file.pdf",
        chunk_id=9,
        source="api",
        chunk_type="text"
    )
    assert meta_default.dict() == meta_explicit.dict()
