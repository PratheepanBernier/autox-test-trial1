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

def test_document_metadata_and_chunk_equivalence():
    # Happy path: create DocumentMetadata and Chunk, compare round-trip serialization
    meta = DocumentMetadata(
        filename="doc.pdf",
        page_number=1,
        chunk_id=42,
        source="upload",
        chunk_type="text"
    )
    chunk = Chunk(
        text="Sample text",
        metadata=meta
    )
    # Serialize and deserialize, should be equivalent
    chunk_dict = chunk.dict()
    chunk2 = Chunk(**chunk_dict)
    assert chunk == chunk2
    assert chunk.metadata == chunk2.metadata

def test_document_metadata_chunk_type_default_and_override():
    # chunk_type default
    meta_default = DocumentMetadata(
        filename="doc.pdf",
        page_number=2,
        chunk_id=1,
        source="api"
    )
    assert meta_default.chunk_type == "text"
    # chunk_type override
    meta_structured = DocumentMetadata(
        filename="doc2.pdf",
        page_number=3,
        chunk_id=2,
        source="api",
        chunk_type="structured_data"
    )
    assert meta_structured.chunk_type == "structured_data"

def test_document_metadata_missing_required_fields_raises():
    # Missing filename
    with pytest.raises(ValidationError):
        DocumentMetadata(
            page_number=1,
            chunk_id=1,
            source="api"
        )
    # Missing chunk_id
    with pytest.raises(ValidationError):
        DocumentMetadata(
            filename="doc.pdf",
            page_number=1,
            source="api"
        )
    # Missing source
    with pytest.raises(ValidationError):
        DocumentMetadata(
            filename="doc.pdf",
            page_number=1,
            chunk_id=1
        )

def test_chunk_requires_text_and_metadata():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="api"
    )
    # Missing text
    with pytest.raises(ValidationError):
        Chunk(metadata=meta)
    # Missing metadata
    with pytest.raises(ValidationError):
        Chunk(text="abc")

def test_qaquery_equivalent_paths():
    # Happy path: empty chat_history
    q1 = QAQuery(question="What is AI?")
    q2 = QAQuery(question="What is AI?", chat_history=[])
    assert q1 == q2
    # Non-empty chat_history
    chat = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    q3 = QAQuery(question="What is AI?", chat_history=chat)
    q4 = QAQuery.parse_obj(q3.dict())
    assert q3 == q4

def test_qaquery_chat_history_edge_cases():
    # chat_history with empty dicts
    q = QAQuery(question="Q", chat_history=[{}, {}])
    assert q.chat_history == [{}, {}]
    # chat_history with missing keys
    q2 = QAQuery(question="Q", chat_history=[{"role": "user"}])
    assert q2.chat_history == [{"role": "user"}]

def test_sourced_answer_equivalence_and_sources():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="api"
    )
    chunk = Chunk(text="abc", metadata=meta)
    # Happy path
    sa1 = SourcedAnswer(
        answer="42",
        confidence_score=0.99,
        sources=[chunk]
    )
    sa2 = SourcedAnswer.parse_obj(sa1.dict())
    assert sa1 == sa2
    # Edge: empty sources
    sa3 = SourcedAnswer(
        answer="none",
        confidence_score=0.0,
        sources=[]
    )
    assert sa3.sources == []

def test_sourced_answer_confidence_score_boundaries():
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="api"
    )
    chunk = Chunk(text="abc", metadata=meta)
    # Lower boundary
    sa_low = SourcedAnswer(answer="low", confidence_score=0.0, sources=[chunk])
    assert sa_low.confidence_score == 0.0
    # Upper boundary
    sa_high = SourcedAnswer(answer="high", confidence_score=1.0, sources=[chunk])
    assert sa_high.confidence_score == 1.0
    # Out of bounds (should not raise, as no validation)
    sa_neg = SourcedAnswer(answer="neg", confidence_score=-1.0, sources=[chunk])
    assert sa_neg.confidence_score == -1.0
    sa_over = SourcedAnswer(answer="over", confidence_score=2.0, sources=[chunk])
    assert sa_over.confidence_score == 2.0

def test_extraction_request_equivalence():
    # Happy path
    er1 = ExtractionRequest(document_text="Some text")
    er2 = ExtractionRequest.parse_obj(er1.dict())
    assert er1 == er2

def test_extraction_request_missing_document_text_raises():
    with pytest.raises(ValidationError):
        ExtractionRequest()

def test_document_metadata_page_number_optional_and_none_equivalence():
    # page_number omitted vs. explicit None
    meta1 = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="api"
    )
    meta2 = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="api",
        page_number=None
    )
    assert meta1 == meta2

def test_document_metadata_page_number_edge_cases():
    # page_number = 0 (boundary)
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="api",
        page_number=0
    )
    assert meta.page_number == 0
    # page_number negative
    meta_neg = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="api",
        page_number=-1
    )
    assert meta_neg.page_number == -1
    # page_number large
    meta_large = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="api",
        page_number=10**6
    )
    assert meta_large.page_number == 10**6

def test_chunk_and_sourced_answer_round_trip_equivalence():
    # Reconcile: Chunk in SourcedAnswer sources round-trip
    meta = DocumentMetadata(
        filename="doc.pdf",
        chunk_id=1,
        source="api"
    )
    chunk = Chunk(text="abc", metadata=meta)
    sa = SourcedAnswer(answer="ok", confidence_score=0.5, sources=[chunk])
    # Serialize SourcedAnswer, then reconstruct
    sa_dict = sa.dict()
    sa2 = SourcedAnswer.parse_obj(sa_dict)
    # Compare all fields
    assert sa == sa2
    assert sa.sources[0] == sa2.sources[0]
    assert sa.sources[0].metadata == sa2.sources[0].metadata
