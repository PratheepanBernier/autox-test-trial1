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
        chunk_id=1,
        source="upload",
        chunk_type="structured_data"
    )
    assert meta.filename == "doc.pdf"
    assert meta.page_number == 2
    assert meta.chunk_id == 1
    assert meta.source == "upload"
    assert meta.chunk_type == "structured_data"

def test_document_metadata_defaults_and_optional_page_number():
    meta = schemas.DocumentMetadata(
        filename="doc.pdf",
        chunk_id=5,
        source="api"
    )
    assert meta.filename == "doc.pdf"
    assert meta.page_number is None
    assert meta.chunk_id == 5
    assert meta.source == "api"
    assert meta.chunk_type == "text"

def test_document_metadata_missing_required_fields_raises():
    with pytest.raises(ValidationError) as excinfo:
        schemas.DocumentMetadata(
            filename="doc.pdf",
            source="api"
        )
    assert "chunk_id" in str(excinfo.value)

def test_document_metadata_invalid_page_number_type():
    with pytest.raises(ValidationError):
        schemas.DocumentMetadata(
            filename="doc.pdf",
            page_number="not_an_int",
            chunk_id=1,
            source="api"
        )

def test_chunk_happy_path():
    meta = schemas.DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    chunk = schemas.Chunk(
        text="This is a chunk of text.",
        metadata=meta
    )
    assert chunk.text == "This is a chunk of text."
    assert chunk.metadata == meta

def test_chunk_invalid_metadata_type():
    with pytest.raises(ValidationError):
        schemas.Chunk(
            text="Some text",
            metadata={"filename": "doc.pdf", "chunk_id": 1, "source": "api"}
        )

def test_qaquery_happy_path():
    q = schemas.QAQuery(
        question="What is the capital of France?",
        chat_history=[{"role": "user", "content": "Hello"}]
    )
    assert q.question == "What is the capital of France?"
    assert q.chat_history == [{"role": "user", "content": "Hello"}]

def test_qaquery_default_chat_history():
    q = schemas.QAQuery(
        question="What is the capital of France?"
    )
    assert q.chat_history == []

def test_qaquery_invalid_chat_history_type():
    with pytest.raises(ValidationError):
        schemas.QAQuery(
            question="Test?",
            chat_history="not_a_list"
        )

def test_sourced_answer_happy_path():
    meta = schemas.DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    chunk = schemas.Chunk(
        text="Chunk text",
        metadata=meta
    )
    answer = schemas.SourcedAnswer(
        answer="Paris",
        confidence_score=0.95,
        sources=[chunk]
    )
    assert answer.answer == "Paris"
    assert answer.confidence_score == 0.95
    assert answer.sources == [chunk]

def test_sourced_answer_empty_sources():
    answer = schemas.SourcedAnswer(
        answer="Paris",
        confidence_score=0.95,
        sources=[]
    )
    assert answer.sources == []

def test_sourced_answer_confidence_score_bounds():
    meta = schemas.DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    chunk = schemas.Chunk(
        text="Chunk text",
        metadata=meta
    )
    answer = schemas.SourcedAnswer(
        answer="Paris",
        confidence_score=0.0,
        sources=[chunk]
    )
    assert answer.confidence_score == 0.0
    answer = schemas.SourcedAnswer(
        answer="Paris",
        confidence_score=1.0,
        sources=[chunk]
    )
    assert answer.confidence_score == 1.0

def test_sourced_answer_invalid_confidence_score_type():
    meta = schemas.DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    chunk = schemas.Chunk(
        text="Chunk text",
        metadata=meta
    )
    with pytest.raises(ValidationError):
        schemas.SourcedAnswer(
            answer="Paris",
            confidence_score="high",
            sources=[chunk]
        )

def test_extraction_request_happy_path():
    req = schemas.ExtractionRequest(
        document_text="This is the document text."
    )
    assert req.document_text == "This is the document text."

def test_extraction_request_missing_document_text():
    with pytest.raises(ValidationError):
        schemas.ExtractionRequest()

def test_document_metadata_chunk_type_boundary():
    meta = schemas.DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload",
        chunk_type=""
    )
    assert meta.chunk_type == ""

def test_document_metadata_long_filename():
    long_filename = "a" * 1024 + ".pdf"
    meta = schemas.DocumentMetadata(
        filename=long_filename,
        chunk_id=1,
        source="upload"
    )
    assert meta.filename == long_filename

def test_document_metadata_negative_page_number():
    meta = schemas.DocumentMetadata(
        filename="doc.pdf",
        page_number=-1,
        chunk_id=1,
        source="upload"
    )
    assert meta.page_number == -1

def test_chunk_empty_text():
    meta = schemas.DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    chunk = schemas.Chunk(
        text="",
        metadata=meta
    )
    assert chunk.text == ""

def test_qaquery_empty_question():
    q = schemas.QAQuery(
        question="",
        chat_history=[]
    )
    assert q.question == ""

def test_sourced_answer_empty_answer():
    meta = schemas.DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    chunk = schemas.Chunk(
        text="Chunk text",
        metadata=meta
    )
    answer = schemas.SourcedAnswer(
        answer="",
        confidence_score=0.5,
        sources=[chunk]
    )
    assert answer.answer == ""
