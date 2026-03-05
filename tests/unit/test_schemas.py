# source_hash: 66e6706c1aa7e23f
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

def test_document_metadata_defaults_and_optional():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=5,
        source="api"
    )
    assert meta.filename == "doc.pdf"
    assert meta.page_number is None
    assert meta.chunk_id == 5
    assert meta.source == "api"
    assert meta.chunk_type == "text"  # default

def test_document_metadata_missing_required_fields_raises():
    with pytest.raises(ValidationError) as excinfo:
        DocumentMetadata(
            filename="doc.pdf",
            source="api"
        )
    assert "chunk_id" in str(excinfo.value)

def test_document_metadata_invalid_page_number_type():
    with pytest.raises(ValidationError):
        DocumentMetadata(
            filename="doc.pdf",
            page_number="not_an_int",
            chunk_id=1,
            source="api"
        )

def test_chunk_happy_path():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=2,
        source="upload"
    )
    chunk = Chunk(
        text="This is a chunk of text.",
        metadata=meta
    )
    assert chunk.text == "This is a chunk of text."
    assert chunk.metadata == meta

def test_chunk_invalid_metadata_type():
    with pytest.raises(ValidationError):
        Chunk(
            text="Some text",
            metadata={"filename": "doc.pdf", "chunk_id": 1, "source": "api"}
        )

def test_qaquery_happy_path_with_history():
    query = QAQuery(
        question="What is the capital of France?",
        chat_history=[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    )
    assert query.question == "What is the capital of France?"
    assert len(query.chat_history) == 2
    assert query.chat_history[0]["role"] == "user"

def test_qaquery_empty_history_default():
    query = QAQuery(
        question="What is the capital of France?"
    )
    assert query.chat_history == []

def test_qaquery_invalid_chat_history_type():
    with pytest.raises(ValidationError):
        QAQuery(
            question="Test?",
            chat_history="not_a_list"
        )

def test_sourced_answer_happy_path():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=3,
        source="upload"
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
    assert answer.sources[0] == chunk

def test_sourced_answer_empty_sources():
    answer = SourcedAnswer(
        answer="No sources found.",
        confidence_score=0.0,
        sources=[]
    )
    assert answer.sources == []

def test_sourced_answer_confidence_score_bounds():
    # Lower bound
    answer = SourcedAnswer(
        answer="Low confidence",
        confidence_score=0.0,
        sources=[]
    )
    assert answer.confidence_score == 0.0
    # Upper bound
    answer = SourcedAnswer(
        answer="High confidence",
        confidence_score=1.0,
        sources=[]
    )
    assert answer.confidence_score == 1.0

def test_sourced_answer_invalid_confidence_score_type():
    with pytest.raises(ValidationError):
        SourcedAnswer(
            answer="Test",
            confidence_score="not_a_float",
            sources=[]
        )

def test_sourced_answer_invalid_sources_type():
    with pytest.raises(ValidationError):
        SourcedAnswer(
            answer="Test",
            confidence_score=0.5,
            sources="not_a_list"
        )

def test_extraction_request_happy_path():
    req = ExtractionRequest(
        document_text="Some document text."
    )
    assert req.document_text == "Some document text."

def test_extraction_request_missing_document_text():
    with pytest.raises(ValidationError):
        ExtractionRequest()

def test_extraction_request_empty_document_text():
    req = ExtractionRequest(
        document_text=""
    )
    assert req.document_text == ""

def test_document_metadata_chunk_type_invalid_value():
    # chunk_type is not validated for allowed values, so any string is accepted
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="api",
        chunk_type="invalid_type"
    )
    assert meta.chunk_type == "invalid_type"

def test_document_metadata_boundary_chunk_id():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=0,
        source="api"
    )
    assert meta.chunk_id == 0
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=-1,
        source="api"
    )
    assert meta.chunk_id == -1

def test_document_metadata_filename_empty_string():
    meta = DocumentMetadata(
        filename="",
        chunk_id=1,
        source="api"
    )
    assert meta.filename == ""
