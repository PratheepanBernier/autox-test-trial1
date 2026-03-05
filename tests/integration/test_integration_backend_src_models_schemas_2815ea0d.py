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

def test_document_metadata_default_chunk_type():
    meta = schemas.DocumentMetadata(
        filename="doc.txt",
        chunk_id=5,
        source="api"
    )
    assert meta.chunk_type == "text"

def test_document_metadata_page_number_optional():
    meta = schemas.DocumentMetadata(
        filename="doc.txt",
        chunk_id=5,
        source="api"
    )
    assert meta.page_number is None

def test_document_metadata_missing_required_fields_raises():
    with pytest.raises(ValidationError) as excinfo:
        schemas.DocumentMetadata(
            filename="doc.txt",
            source="api"
        )
    assert "chunk_id" in str(excinfo.value)

def test_document_metadata_invalid_chunk_id_type():
    with pytest.raises(ValidationError):
        schemas.DocumentMetadata(
            filename="doc.txt",
            chunk_id="not_an_int",
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
            metadata={"filename": "doc.pdf", "chunk_id": 1, "source": "upload"}
        )

def test_qaquery_happy_path():
    query = schemas.QAQuery(
        question="What is the capital of France?",
        chat_history=[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    )
    assert query.question == "What is the capital of France?"
    assert isinstance(query.chat_history, list)
    assert query.chat_history[0]["role"] == "user"

def test_qaquery_default_chat_history():
    query = schemas.QAQuery(
        question="What is the capital of France?"
    )
    assert query.chat_history == []

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
        text="Paris is the capital of France.",
        metadata=meta
    )
    answer = schemas.SourcedAnswer(
        answer="Paris",
        confidence_score=0.98,
        sources=[chunk]
    )
    assert answer.answer == "Paris"
    assert answer.confidence_score == 0.98
    assert answer.sources[0].text == "Paris is the capital of France."

def test_sourced_answer_empty_sources():
    answer = schemas.SourcedAnswer(
        answer="Paris",
        confidence_score=0.98,
        sources=[]
    )
    assert answer.sources == []

def test_sourced_answer_invalid_confidence_score_type():
    meta = schemas.DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    chunk = schemas.Chunk(
        text="Paris is the capital of France.",
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
        document_text="Some document content."
    )
    assert req.document_text == "Some document content."

def test_extraction_request_missing_document_text():
    with pytest.raises(ValidationError):
        schemas.ExtractionRequest()

def test_document_metadata_boundary_chunk_id_zero():
    meta = schemas.DocumentMetadata(
        filename="doc.pdf",
        chunk_id=0,
        source="upload"
    )
    assert meta.chunk_id == 0

def test_document_metadata_boundary_chunk_id_negative():
    meta = schemas.DocumentMetadata(
        filename="doc.pdf",
        chunk_id=-1,
        source="upload"
    )
    assert meta.chunk_id == -1

def test_document_metadata_empty_filename():
    meta = schemas.DocumentMetadata(
        filename="",
        chunk_id=1,
        source="upload"
    )
    assert meta.filename == ""

def test_qaquery_empty_question():
    query = schemas.QAQuery(
        question=""
    )
    assert query.question == ""

def test_sourced_answer_confidence_score_bounds():
    meta = schemas.DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    chunk = schemas.Chunk(
        text="Paris is the capital of France.",
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

def test_chunk_type_accepts_only_valid_values():
    meta = schemas.DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload",
        chunk_type="text"
    )
    assert meta.chunk_type == "text"
    meta2 = schemas.DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload",
        chunk_type="structured_data"
    )
    assert meta2.chunk_type == "structured_data"
    # Accepts any string, as no validation is enforced

def test_document_metadata_repr_and_str():
    meta = schemas.DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    s = str(meta)
    r = repr(meta)
    assert "filename='doc.pdf'" in s
    assert "chunk_id=1" in r
