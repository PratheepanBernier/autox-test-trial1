# source_hash: 66e6706c1aa7e23f
# import_target: backend.src.models.schemas
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import pytest
from pydantic import ValidationError
from datetime import datetime
from backend.src.models import schemas


def test_document_metadata_happy_path():
    meta = schemas.DocumentMetadata(
        filename="doc.pdf",
        page_number=2,
        chunk_id=10,
        source="upload",
        chunk_type="structured_data"
    )
    assert meta.filename == "doc.pdf"
    assert meta.page_number == 2
    assert meta.chunk_id == 10
    assert meta.source == "upload"
    assert meta.chunk_type == "structured_data"

def test_document_metadata_defaults_and_optional():
    meta = schemas.DocumentMetadata(
        filename="doc.txt",
        chunk_id=1,
        source="api"
    )
    assert meta.filename == "doc.txt"
    assert meta.page_number is None
    assert meta.chunk_id == 1
    assert meta.source == "api"
    assert meta.chunk_type == "text"

def test_document_metadata_missing_required_fields_raises():
    with pytest.raises(ValidationError) as excinfo:
        schemas.DocumentMetadata(
            page_number=1,
            chunk_id=2,
            source="src"
        )
    assert "field required" in str(excinfo.value)
    with pytest.raises(ValidationError) as excinfo2:
        schemas.DocumentMetadata(
            filename="f",
            page_number=1,
            source="src"
        )
    assert "field required" in str(excinfo2.value)

def test_document_metadata_invalid_chunk_type():
    with pytest.raises(ValidationError) as excinfo:
        schemas.DocumentMetadata(
            filename="f",
            chunk_id=1,
            source="src",
            chunk_type=123
        )
    assert "str type expected" in str(excinfo.value)

def test_chunk_happy_path():
    meta = schemas.DocumentMetadata(
        filename="file.pdf",
        chunk_id=5,
        source="upload"
    )
    chunk = schemas.Chunk(
        text="This is a chunk.",
        metadata=meta
    )
    assert chunk.text == "This is a chunk."
    assert chunk.metadata == meta

def test_chunk_invalid_metadata_type():
    with pytest.raises(ValidationError) as excinfo:
        schemas.Chunk(
            text="abc",
            metadata={"filename": "f", "chunk_id": 1, "source": "src"}
        )
    assert "value is not a valid dict" not in str(excinfo.value)  # Pydantic will coerce dict to model
    # But missing required fields will raise
    with pytest.raises(ValidationError):
        schemas.Chunk(
            text="abc",
            metadata={"chunk_id": 1, "source": "src"}
        )

def test_qaquery_happy_path():
    q = schemas.QAQuery(
        question="What is AI?",
        chat_history=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    )
    assert q.question == "What is AI?"
    assert isinstance(q.chat_history, list)
    assert q.chat_history[0]["role"] == "user"

def test_qaquery_default_chat_history():
    q = schemas.QAQuery(question="Q?")
    assert q.chat_history == []

def test_qaquery_invalid_chat_history_type():
    with pytest.raises(ValidationError) as excinfo:
        schemas.QAQuery(
            question="Q?",
            chat_history="notalist"
        )
    assert "value is not a valid list" in str(excinfo.value)

def test_sourced_answer_happy_path():
    meta = schemas.DocumentMetadata(
        filename="f",
        chunk_id=1,
        source="src"
    )
    chunk = schemas.Chunk(
        text="chunk text",
        metadata=meta
    )
    ans = schemas.SourcedAnswer(
        answer="42",
        confidence_score=0.99,
        sources=[chunk]
    )
    assert ans.answer == "42"
    assert ans.confidence_score == 0.99
    assert ans.sources[0] == chunk

def test_sourced_answer_confidence_score_bounds():
    meta = schemas.DocumentMetadata(
        filename="f",
        chunk_id=1,
        source="src"
    )
    chunk = schemas.Chunk(
        text="chunk text",
        metadata=meta
    )
    ans = schemas.SourcedAnswer(
        answer="ans",
        confidence_score=0.0,
        sources=[chunk]
    )
    assert ans.confidence_score == 0.0
    ans2 = schemas.SourcedAnswer(
        answer="ans",
        confidence_score=1.0,
        sources=[chunk]
    )
    assert ans2.confidence_score == 1.0

def test_sourced_answer_invalid_confidence_score_type():
    meta = schemas.DocumentMetadata(
        filename="f",
        chunk_id=1,
        source="src"
    )
    chunk = schemas.Chunk(
        text="chunk text",
        metadata=meta
    )
    with pytest.raises(ValidationError) as excinfo:
        schemas.SourcedAnswer(
            answer="ans",
            confidence_score="high",
            sources=[chunk]
        )
    assert "value is not a valid float" in str(excinfo.value)

def test_sourced_answer_sources_empty_list():
    ans = schemas.SourcedAnswer(
        answer="none",
        confidence_score=0.5,
        sources=[]
    )
    assert ans.sources == []

def test_extraction_request_happy_path():
    req = schemas.ExtractionRequest(document_text="Some document text.")
    assert req.document_text == "Some document text."

def test_extraction_request_missing_document_text():
    with pytest.raises(ValidationError) as excinfo:
        schemas.ExtractionRequest()
    assert "field required" in str(excinfo.value)

def test_extraction_request_empty_document_text():
    req = schemas.ExtractionRequest(document_text="")
    assert req.document_text == ""
