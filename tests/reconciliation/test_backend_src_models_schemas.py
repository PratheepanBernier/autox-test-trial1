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

def test_document_metadata_happy_path_and_defaults():
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

    # Test default chunk_type
    meta2 = DocumentMetadata(
        filename="file2.pdf",
        chunk_id=2,
        source="api"
    )
    assert meta2.chunk_type == "text"
    assert meta2.page_number is None

def test_document_metadata_missing_required_fields_raises():
    with pytest.raises(ValidationError):
        DocumentMetadata(
            filename="file.pdf",
            # chunk_id missing
            source="upload"
        )
    with pytest.raises(ValidationError):
        DocumentMetadata(
            # filename missing
            chunk_id=1,
            source="upload"
        )
    with pytest.raises(ValidationError):
        DocumentMetadata(
            filename="file.pdf",
            chunk_id=1,
            # source missing
        )

def test_document_metadata_boundary_conditions():
    # page_number at boundary (0, negative, large)
    meta_zero = DocumentMetadata(
        filename="file.pdf",
        page_number=0,
        chunk_id=1,
        source="upload"
    )
    assert meta_zero.page_number == 0

    meta_negative = DocumentMetadata(
        filename="file.pdf",
        page_number=-1,
        chunk_id=1,
        source="upload"
    )
    assert meta_negative.page_number == -1

    meta_large = DocumentMetadata(
        filename="file.pdf",
        page_number=10**6,
        chunk_id=1,
        source="upload"
    )
    assert meta_large.page_number == 10**6

def test_chunk_happy_path_and_metadata_reconciliation():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=5,
        source="api"
    )
    chunk = Chunk(
        text="This is a chunk of text.",
        metadata=meta
    )
    assert chunk.text == "This is a chunk of text."
    assert chunk.metadata == meta

    # Reconciliation: Construct via dict and via model, should be equivalent
    chunk_dict = {
        "text": "This is a chunk of text.",
        "metadata": {
            "filename": "doc.pdf",
            "chunk_id": 5,
            "source": "api"
        }
    }
    chunk2 = Chunk(**chunk_dict)
    assert chunk2.text == chunk.text
    assert chunk2.metadata == chunk.metadata

def test_chunk_invalid_metadata_raises():
    with pytest.raises(ValidationError):
        Chunk(
            text="abc",
            metadata={
                "filename": "doc.pdf",
                # chunk_id missing
                "source": "api"
            }
        )

def test_qaquery_happy_path_and_defaults():
    q = QAQuery(question="What is the answer?")
    assert q.question == "What is the answer?"
    assert q.chat_history == []

    q2 = QAQuery(
        question="Next?",
        chat_history=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    )
    assert q2.chat_history[0]["role"] == "user"
    assert q2.chat_history[1]["content"] == "Hello"

def test_qaquery_invalid_chat_history_raises():
    with pytest.raises(ValidationError):
        QAQuery(
            question="Test",
            chat_history="not a list"
        )
    with pytest.raises(ValidationError):
        QAQuery(
            question="Test",
            chat_history=[{"role": "user"}]  # missing 'content'
        )

def test_sourced_answer_happy_path_and_reconciliation():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
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
    assert answer.sources[0] == chunk

    # Reconciliation: Construct via dict and via model, should be equivalent
    answer_dict = {
        "answer": "42",
        "confidence_score": 0.99,
        "sources": [
            {
                "text": "Relevant text.",
                "metadata": {
                    "filename": "doc.pdf",
                    "chunk_id": 1,
                    "source": "upload"
                }
            }
        ]
    }
    answer2 = SourcedAnswer(**answer_dict)
    assert answer2.answer == answer.answer
    assert answer2.confidence_score == answer.confidence_score
    assert answer2.sources[0].text == chunk.text
    assert answer2.sources[0].metadata == chunk.metadata

def test_sourced_answer_invalid_confidence_score_raises():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    chunk = Chunk(
        text="Relevant text.",
        metadata=meta
    )
    with pytest.raises(ValidationError):
        SourcedAnswer(
            answer="bad",
            confidence_score="not a float",
            sources=[chunk]
        )
    with pytest.raises(ValidationError):
        SourcedAnswer(
            answer="bad",
            confidence_score=0.5,
            sources="not a list"
        )

def test_extraction_request_happy_path_and_reconciliation():
    req = ExtractionRequest(document_text="Some document text.")
    assert req.document_text == "Some document text."

    # Reconciliation: Construct via dict and via model, should be equivalent
    req_dict = {"document_text": "Some document text."}
    req2 = ExtractionRequest(**req_dict)
    assert req2.document_text == req.document_text

def test_extraction_request_missing_document_text_raises():
    with pytest.raises(ValidationError):
        ExtractionRequest()
    with pytest.raises(ValidationError):
        ExtractionRequest(document_text=None)

def test_document_metadata_strict_types_and_extra_fields():
    # Extra fields should be ignored or raise error depending on config (default: ignored)
    meta = DocumentMetadata(
        filename="file.pdf",
        chunk_id=1,
        source="upload",
        extra_field="should be ignored"
    )
    assert not hasattr(meta, "extra_field")

def test_chunk_type_accepts_only_valid_strings():
    # Accepts any string, but let's check for regression
    meta = DocumentMetadata(
        filename="file.pdf",
        chunk_id=1,
        source="upload",
        chunk_type="structured_data"
    )
    assert meta.chunk_type == "structured_data"
    meta2 = DocumentMetadata(
        filename="file.pdf",
        chunk_id=1,
        source="upload",
        chunk_type="random_type"
    )
    assert meta2.chunk_type == "random_type"
