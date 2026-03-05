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

def test_document_metadata_defaults_and_optional_fields():
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
        chunk_id=1,
        source="api"
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
    qa = QAQuery(
        question="What is AI?",
        chat_history=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    )
    assert qa.question == "What is AI?"
    assert len(qa.chat_history) == 2
    assert qa.chat_history[0]["role"] == "user"

def test_qaquery_empty_history_default():
    qa = QAQuery(question="What is AI?")
    assert qa.chat_history == []

def test_qaquery_invalid_chat_history_type():
    with pytest.raises(ValidationError):
        QAQuery(question="Q?", chat_history="not_a_list")

def test_sourced_answer_happy_path():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="api"
    )
    chunk = Chunk(text="Relevant text", metadata=meta)
    answer = SourcedAnswer(
        answer="42",
        confidence_score=0.99,
        sources=[chunk]
    )
    assert answer.answer == "42"
    assert answer.confidence_score == 0.99
    assert answer.sources[0] == chunk

def test_sourced_answer_empty_sources():
    answer = SourcedAnswer(
        answer="No sources",
        confidence_score=0.0,
        sources=[]
    )
    assert answer.sources == []

def test_sourced_answer_invalid_confidence_score_type():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="api"
    )
    chunk = Chunk(text="Relevant text", metadata=meta)
    with pytest.raises(ValidationError):
        SourcedAnswer(
            answer="bad",
            confidence_score="not_a_float",
            sources=[chunk]
        )

def test_extraction_request_happy_path():
    req = ExtractionRequest(document_text="Some document content.")
    assert req.document_text == "Some document content."

def test_extraction_request_missing_document_text_raises():
    with pytest.raises(ValidationError):
        ExtractionRequest()

def test_extraction_request_empty_string():
    req = ExtractionRequest(document_text="")
    assert req.document_text == ""

def test_document_metadata_boundary_chunk_id_zero():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=0,
        source="api"
    )
    assert meta.chunk_id == 0

def test_document_metadata_boundary_chunk_id_negative():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=-1,
        source="api"
    )
    assert meta.chunk_id == -1

def test_document_metadata_long_filename():
    long_name = "a" * 1024 + ".pdf"
    meta = DocumentMetadata(
        filename=long_name,
        chunk_id=1,
        source="api"
    )
    assert meta.filename == long_name

def test_qaquery_chat_history_dicts_with_extra_keys():
    qa = QAQuery(
        question="Test?",
        chat_history=[{"role": "user", "content": "Hi", "extra": "ignored"}]
    )
    assert qa.chat_history[0]["extra"] == "ignored"

def test_sourced_answer_sources_with_multiple_chunks():
    meta1 = DocumentMetadata(filename="a.pdf", chunk_id=1, source="api")
    meta2 = DocumentMetadata(filename="b.pdf", chunk_id=2, source="api")
    chunk1 = Chunk(text="Text1", metadata=meta1)
    chunk2 = Chunk(text="Text2", metadata=meta2)
    answer = SourcedAnswer(
        answer="Combined",
        confidence_score=0.5,
        sources=[chunk1, chunk2]
    )
    assert len(answer.sources) == 2
    assert answer.sources[1].metadata.filename == "b.pdf"

def test_chunk_text_empty_string():
    meta = DocumentMetadata(filename="doc.pdf", chunk_id=1, source="api")
    chunk = Chunk(text="", metadata=meta)
    assert chunk.text == ""
