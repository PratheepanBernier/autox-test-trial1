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

def test_document_metadata_default_chunk_type():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    assert meta.chunk_type == "text"

def test_document_metadata_page_number_optional():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    assert meta.page_number is None

def test_document_metadata_missing_required_fields_raises():
    with pytest.raises(ValidationError) as excinfo:
        DocumentMetadata(filename="doc.pdf", source="upload")
    assert "chunk_id" in str(excinfo.value)

def test_document_metadata_invalid_chunk_type():
    # chunk_type is not validated, so any string is accepted
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload",
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
    with pytest.raises(ValidationError):
        Chunk(text="abc")

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
    query = QAQuery(
        question="What is the capital of France?",
        chat_history=[{"role": "user", "content": "Hello"}]
    )
    assert query.question == "What is the capital of France?"
    assert query.chat_history == [{"role": "user", "content": "Hello"}]

def test_qaquery_default_chat_history():
    query = QAQuery(question="What is the capital of France?")
    assert query.chat_history == []

def test_qaquery_empty_question_accepted():
    query = QAQuery(question="")
    assert query.question == ""

def test_qaquery_invalid_chat_history_type():
    with pytest.raises(ValidationError):
        QAQuery(question="Q", chat_history="notalist")

def test_sourced_answer_happy_path():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    chunk = Chunk(
        text="Some text.",
        metadata=meta
    )
    answer = SourcedAnswer(
        answer="Paris",
        confidence_score=0.95,
        sources=[chunk]
    )
    assert answer.answer == "Paris"
    assert answer.confidence_score == 0.95
    assert answer.sources == [chunk]

def test_sourced_answer_empty_sources():
    answer = SourcedAnswer(
        answer="Paris",
        confidence_score=0.95,
        sources=[]
    )
    assert answer.sources == []

def test_sourced_answer_confidence_score_bounds():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    chunk = Chunk(
        text="Some text.",
        metadata=meta
    )
    # Lower bound
    answer = SourcedAnswer(
        answer="Paris",
        confidence_score=0.0,
        sources=[chunk]
    )
    assert answer.confidence_score == 0.0
    # Upper bound
    answer = SourcedAnswer(
        answer="Paris",
        confidence_score=1.0,
        sources=[chunk]
    )
    assert answer.confidence_score == 1.0

def test_sourced_answer_invalid_confidence_score_type():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    chunk = Chunk(
        text="Some text.",
        metadata=meta
    )
    with pytest.raises(ValidationError):
        SourcedAnswer(
            answer="Paris",
            confidence_score="high",
            sources=[chunk]
        )

def test_sourced_answer_missing_sources_raises():
    with pytest.raises(ValidationError):
        SourcedAnswer(
            answer="Paris",
            confidence_score=0.9
        )

def test_extraction_request_happy_path():
    req = ExtractionRequest(document_text="Some document text.")
    assert req.document_text == "Some document text."

def test_extraction_request_empty_text():
    req = ExtractionRequest(document_text="")
    assert req.document_text == ""

def test_extraction_request_missing_document_text_raises():
    with pytest.raises(ValidationError):
        ExtractionRequest()

def test_document_metadata_boundary_chunk_id():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=0,
        source="upload"
    )
    assert meta.chunk_id == 0
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=-1,
        source="upload"
    )
    assert meta.chunk_id == -1

def test_document_metadata_long_filename():
    long_name = "a" * 1024
    meta = DocumentMetadata(
        filename=long_name,
        chunk_id=1,
        source="upload"
    )
    assert meta.filename == long_name

def test_chunk_metadata_equivalence():
    meta1 = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    meta2 = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    chunk1 = Chunk(text="abc", metadata=meta1)
    chunk2 = Chunk(text="abc", metadata=meta2)
    assert chunk1 == chunk2

def test_sourced_answer_equivalent_paths():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    chunk = Chunk(text="abc", metadata=meta)
    answer1 = SourcedAnswer(
        answer="Paris",
        confidence_score=0.9,
        sources=[chunk]
    )
    answer2 = SourcedAnswer.parse_obj(answer1.dict())
    assert answer1 == answer2
