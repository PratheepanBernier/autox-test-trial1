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
from backend.src.models import schemas
from datetime import datetime

def test_document_metadata_happy_path():
    meta = schemas.DocumentMetadata(
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
    meta = schemas.DocumentMetadata(
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
        schemas.DocumentMetadata(
            page_number=1,
            chunk_id=2,
            source="src"
        )
    assert "filename" in str(excinfo.value)

    with pytest.raises(ValidationError) as excinfo:
        schemas.DocumentMetadata(
            filename="f",
            page_number=1,
            source="src"
        )
    assert "chunk_id" in str(excinfo.value)

    with pytest.raises(ValidationError) as excinfo:
        schemas.DocumentMetadata(
            filename="f",
            chunk_id=1,
            page_number=1
        )
    assert "source" in str(excinfo.value)

def test_document_metadata_invalid_types():
    with pytest.raises(ValidationError):
        schemas.DocumentMetadata(
            filename=123,
            chunk_id="not-an-int",
            source=456
        )

def test_chunk_happy_path():
    meta = schemas.DocumentMetadata(
        filename="doc.pdf",
        chunk_id=3,
        source="test"
    )
    chunk = schemas.Chunk(
        text="This is a chunk.",
        metadata=meta
    )
    assert chunk.text == "This is a chunk."
    assert chunk.metadata == meta

def test_chunk_invalid_metadata_type():
    with pytest.raises(ValidationError):
        schemas.Chunk(
            text="abc",
            metadata={"filename": "f", "chunk_id": 1, "source": "s"}
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
    q = schemas.QAQuery(question="Test?")
    assert q.chat_history == []

def test_qaquery_invalid_chat_history_type():
    with pytest.raises(ValidationError):
        schemas.QAQuery(question="Q", chat_history="not-a-list")

def test_sourced_answer_happy_path():
    meta = schemas.DocumentMetadata(
        filename="doc.pdf",
        chunk_id=5,
        source="src"
    )
    chunk = schemas.Chunk(
        text="Relevant text.",
        metadata=meta
    )
    answer = schemas.SourcedAnswer(
        answer="42",
        confidence_score=0.99,
        sources=[chunk]
    )
    assert answer.answer == "42"
    assert answer.confidence_score == 0.99
    assert answer.sources[0] == chunk

def test_sourced_answer_empty_sources():
    answer = schemas.SourcedAnswer(
        answer="No sources",
        confidence_score=0.0,
        sources=[]
    )
    assert answer.sources == []

def test_sourced_answer_invalid_confidence_score_type():
    meta = schemas.DocumentMetadata(
        filename="doc.pdf",
        chunk_id=5,
        source="src"
    )
    chunk = schemas.Chunk(
        text="Relevant text.",
        metadata=meta
    )
    with pytest.raises(ValidationError):
        schemas.SourcedAnswer(
            answer="bad",
            confidence_score="high",
            sources=[chunk]
        )

def test_extraction_request_happy_path():
    req = schemas.ExtractionRequest(document_text="Extract this text.")
    assert req.document_text == "Extract this text."

def test_extraction_request_missing_document_text():
    with pytest.raises(ValidationError):
        schemas.ExtractionRequest()

def test_extraction_request_invalid_type():
    with pytest.raises(ValidationError):
        schemas.ExtractionRequest(document_text=12345)
