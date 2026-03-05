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

def test_document_metadata_happy_path_and_defaults():
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

    # Default chunk_type
    meta2 = schemas.DocumentMetadata(
        filename="doc2.pdf",
        chunk_id=2,
        source="api"
    )
    assert meta2.chunk_type == "text"
    assert meta2.page_number is None

def test_document_metadata_missing_required_fields_raises():
    with pytest.raises(ValidationError):
        schemas.DocumentMetadata(
            filename="doc.pdf",
            source="upload"
            # missing chunk_id
        )
    with pytest.raises(ValidationError):
        schemas.DocumentMetadata(
            chunk_id=1,
            source="upload"
            # missing filename
        )
    with pytest.raises(ValidationError):
        schemas.DocumentMetadata(
            filename="doc.pdf",
            chunk_id=1
            # missing source
        )

def test_document_metadata_boundary_conditions():
    # page_number at 0 and negative
    meta = schemas.DocumentMetadata(
        filename="doc.pdf",
        page_number=0,
        chunk_id=1,
        source="upload"
    )
    assert meta.page_number == 0

    meta_neg = schemas.DocumentMetadata(
        filename="doc.pdf",
        page_number=-1,
        chunk_id=1,
        source="upload"
    )
    assert meta_neg.page_number == -1

def test_chunk_happy_path_and_metadata_equivalence():
    meta = schemas.DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    chunk = schemas.Chunk(
        text="This is a chunk.",
        metadata=meta
    )
    assert chunk.text == "This is a chunk."
    assert chunk.metadata == meta

    # Reconciliation: Construct via dict
    chunk2 = schemas.Chunk.parse_obj({
        "text": "This is a chunk.",
        "metadata": meta.dict()
    })
    assert chunk2.text == chunk.text
    assert chunk2.metadata == chunk.metadata

def test_chunk_missing_fields_raises():
    meta = schemas.DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="upload"
    )
    with pytest.raises(ValidationError):
        schemas.Chunk(
            metadata=meta
        )
    with pytest.raises(ValidationError):
        schemas.Chunk(
            text="Some text"
        )

def test_qaquery_happy_path_and_defaults():
    q = schemas.QAQuery(
        question="What is the answer?",
        chat_history=[{"role": "user", "content": "Hi"}]
    )
    assert q.question == "What is the answer?"
    assert q.chat_history == [{"role": "user", "content": "Hi"}]

    # Default chat_history
    q2 = schemas.QAQuery(
        question="Another question"
    )
    assert q2.chat_history == []

def test_qaquery_chat_history_edge_cases():
    # Empty chat_history
    q = schemas.QAQuery(
        question="Q?",
        chat_history=[]
    )
    assert q.chat_history == []

    # Chat history with empty dict
    q2 = schemas.QAQuery(
        question="Q?",
        chat_history=[{}]
    )
    assert q2.chat_history == [{}]

def test_qaquery_missing_question_raises():
    with pytest.raises(ValidationError):
        schemas.QAQuery(
            chat_history=[{"role": "user", "content": "Hi"}]
        )

def test_sourced_answer_happy_path_and_equivalence():
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
        answer="42",
        confidence_score=0.99,
        sources=[chunk]
    )
    assert answer.answer == "42"
    assert answer.confidence_score == 0.99
    assert answer.sources == [chunk]

    # Reconciliation: Construct via dict
    answer2 = schemas.SourcedAnswer.parse_obj({
        "answer": "42",
        "confidence_score": 0.99,
        "sources": [chunk.dict()]
    })
    assert answer2.answer == answer.answer
    assert answer2.confidence_score == answer.confidence_score
    assert answer2.sources[0].text == chunk.text
    assert answer2.sources[0].metadata == chunk.metadata

def test_sourced_answer_missing_fields_raises():
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
            confidence_score=0.5,
            sources=[chunk]
        )
    with pytest.raises(ValidationError):
        schemas.SourcedAnswer(
            answer="A",
            sources=[chunk]
        )
    with pytest.raises(ValidationError):
        schemas.SourcedAnswer(
            answer="A",
            confidence_score=0.5
        )

def test_extraction_request_happy_path_and_equivalence():
    req = schemas.ExtractionRequest(
        document_text="Some document text"
    )
    assert req.document_text == "Some document text"

    # Reconciliation: Construct via dict
    req2 = schemas.ExtractionRequest.parse_obj({
        "document_text": "Some document text"
    })
    assert req2.document_text == req.document_text

def test_extraction_request_missing_document_text_raises():
    with pytest.raises(ValidationError):
        schemas.ExtractionRequest()

def test_document_metadata_equivalent_paths():
    # Reconciliation: direct and parse_obj
    data = {
        "filename": "doc.pdf",
        "page_number": 5,
        "chunk_id": 10,
        "source": "api",
        "chunk_type": "structured_data"
    }
    meta1 = schemas.DocumentMetadata(**data)
    meta2 = schemas.DocumentMetadata.parse_obj(data)
    assert meta1 == meta2

def test_chunk_equivalent_paths():
    meta_data = {
        "filename": "doc.pdf",
        "chunk_id": 1,
        "source": "upload"
    }
    chunk_data = {
        "text": "Chunk text",
        "metadata": meta_data
    }
    chunk1 = schemas.Chunk(**chunk_data)
    chunk2 = schemas.Chunk.parse_obj(chunk_data)
    assert chunk1 == chunk2

def test_qaquery_equivalent_paths():
    data = {
        "question": "Q?",
        "chat_history": [{"role": "user", "content": "Hi"}]
    }
    q1 = schemas.QAQuery(**data)
    q2 = schemas.QAQuery.parse_obj(data)
    assert q1 == q2

def test_sourced_answer_equivalent_paths():
    meta_data = {
        "filename": "doc.pdf",
        "chunk_id": 1,
        "source": "upload"
    }
    chunk_data = {
        "text": "Chunk text",
        "metadata": meta_data
    }
    answer_data = {
        "answer": "A",
        "confidence_score": 0.5,
        "sources": [chunk_data]
    }
    ans1 = schemas.SourcedAnswer(**answer_data)
    ans2 = schemas.SourcedAnswer.parse_obj(answer_data)
    assert ans1 == ans2

def test_extraction_request_equivalent_paths():
    data = {"document_text": "Text"}
    req1 = schemas.ExtractionRequest(**data)
    req2 = schemas.ExtractionRequest.parse_obj(data)
    assert req1 == req2
