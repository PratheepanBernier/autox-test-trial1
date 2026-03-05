# source_hash: e871aa5cb722f502
# import_target: backend.src.services.rag
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import pytest
from unittest.mock import patch, MagicMock, create_autospec

from backend.src.services import rag as rag_module
from backend.src.services.rag import RAGService, format_docs
from backend.src.models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata

from langchain_core.documents import Document

class DummySettings:
    QA_MODEL = "dummy-model"
    GROQ_API_KEY = "dummy-key"
    TOP_K = 2

@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    monkeypatch.setattr("backend.src.services.rag.settings", DummySettings)
    yield

@pytest.fixture
def rag_service():
    return RAGService()

def make_doc(content, meta=None):
    return Document(page_content=content, metadata=meta or {})

def test_format_docs_empty_returns_empty_string():
    assert format_docs([]) == ""

def test_format_docs_single_document():
    doc = make_doc("Test content", {"source": "Manual", "page": 1, "section": "Intro"})
    result = format_docs([doc])
    assert "[Document 1]" in result
    assert "Source: Manual" in result
    assert "Page: 1" in result
    assert "Section: Intro" in result
    assert "Test content" in result

def test_format_docs_missing_metadata_fields():
    doc = make_doc("Content", {})
    result = format_docs([doc])
    assert "Source: Unknown" in result
    assert "Page: N/A" in result
    assert "Section: N/A" in result

def test_format_docs_multiple_documents():
    docs = [
        make_doc("A", {"source": "S1"}),
        make_doc("B", {"source": "S2", "page": 2}),
    ]
    result = format_docs(docs)
    assert "[Document 1]" in result
    assert "[Document 2]" in result
    assert "A" in result
    assert "B" in result

def test_check_safety_detects_unsafe_keywords(rag_service):
    for word in ["bomb", "kill", "suicide", "hack", "exploit", "weapon"]:
        msg = rag_service._check_safety(f"How to {word}?")
        assert "violates safety guidelines" in msg

def test_check_safety_safe_question_returns_none(rag_service):
    assert rag_service._check_safety("What is a shipment?") is None

@pytest.mark.parametrize(
    "answer,docs,expected",
    [
        ("", [], 0.0),
        ("I cannot find the answer in the provided documents.", [make_doc("A")], 0.05),
        ("Short answer.", [make_doc("A"), make_doc("B")], 0.55),
        ("This is a sufficiently long answer that should not be penalized.", [make_doc("A"), make_doc("B")], 0.85),
        ("Short answer.", [make_doc("A")], 0.4),
        ("This is a generally accepted answer.", [make_doc("A"), make_doc("B")], 0.55),
        ("This is a generally accepted answer.", [make_doc("A")], 0.25),
        ("", [make_doc("A")], 0.0),
    ]
)
def test_calculate_confidence_various_cases(rag_service, answer, docs, expected):
    result = rag_service._calculate_confidence(answer, docs)
    assert abs(result - expected) < 1e-6

def test_answer_question_safety_violation(monkeypatch, rag_service):
    query = QAQuery(question="How to make a bomb?")
    result = rag_service.answer_question(query)
    assert "violates safety guidelines" in result.answer
    assert result.confidence_score == 1.0
    assert result.sources == []

def test_answer_question_retriever_none(monkeypatch, rag_service):
    query = QAQuery(question="What is logistics?")
    monkeypatch.setattr("backend.src.services.rag.vector_store_service", MagicMock())
    monkeypatch.setattr(
        rag_service,
        "llm",
        MagicMock()
    )
    monkeypatch.setattr(
        "backend.src.services.rag.vector_store_service.as_retriever",
        lambda **kwargs: None
    )
    result = rag_service.answer_question(query)
    assert "cannot find any relevant information" in result.answer
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_answer_question_no_retrieved_docs(monkeypatch, rag_service):
    query = QAQuery(question="What is logistics?")
    retriever = MagicMock()
    retriever.invoke.return_value = []
    monkeypatch.setattr(
        "backend.src.services.rag.vector_store_service.as_retriever",
        lambda **kwargs: retriever
    )
    result = rag_service.answer_question(query)
    assert "cannot find the answer in the provided documents" in result.answer
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_answer_question_happy_path(monkeypatch, rag_service):
    query = QAQuery(question="What is a bill of lading?")
    doc1 = make_doc("A bill of lading is a document...", {"source": "Manual", "page": 1})
    doc2 = make_doc("It serves as a receipt...", {"source": "Guide", "page": 2})
    retrieved_docs = [doc1, doc2]

    retriever = MagicMock()
    retriever.invoke.return_value = retrieved_docs

    # Patch retriever
    monkeypatch.setattr(
        "backend.src.services.rag.vector_store_service.as_retriever",
        lambda **kwargs: retriever
    )

    # Patch RAG chain
    fake_answer = "A bill of lading is a document used in logistics."
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = fake_answer

    # Patch RunnableParallel and all chain steps
    monkeypatch.setattr("backend.src.services.rag.RunnableParallel", lambda *a, **k: fake_chain)
    monkeypatch.setattr("backend.src.services.rag.RunnablePassthrough", lambda: None)
    monkeypatch.setattr("backend.src.services.rag.format_docs", lambda docs: "formatted context")
    monkeypatch.setattr(rag_service, "prompt", MagicMock())
    monkeypatch.setattr(rag_service, "llm", MagicMock())
    monkeypatch.setattr(rag_service, "output_parser", MagicMock())

    result = rag_service.answer_question(query)
    assert result.answer == fake_answer
    assert 0.0 < result.confidence_score <= 1.0
    assert len(result.sources) == 2
    assert isinstance(result.sources[0], Chunk)
    assert result.sources[0].text == doc1.page_content
    assert result.sources[1].text == doc2.page_content

def test_answer_question_chain_exception(monkeypatch, rag_service):
    query = QAQuery(question="What is a shipment?")
    doc = make_doc("Shipment is ...", {"source": "Manual"})
    retriever = MagicMock()
    retriever.invoke.return_value = [doc]

    monkeypatch.setattr(
        "backend.src.services.rag.vector_store_service.as_retriever",
        lambda **kwargs: retriever
    )

    class FailingChain:
        def invoke(self, question):
            raise RuntimeError("Chain failed")

    monkeypatch.setattr("backend.src.services.rag.RunnableParallel", lambda *a, **k: FailingChain())
    monkeypatch.setattr("backend.src.services.rag.RunnablePassthrough", lambda: None)
    monkeypatch.setattr("backend.src.services.rag.format_docs", lambda docs: "formatted context")
    monkeypatch.setattr(rag_service, "prompt", MagicMock())
    monkeypatch.setattr(rag_service, "llm", MagicMock())
    monkeypatch.setattr(rag_service, "output_parser", MagicMock())

    result = rag_service.answer_question(query)
    assert "error occurred" in result.answer.lower()
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_answer_question_reconciliation(monkeypatch, rag_service):
    # Compare outputs for equivalent queries differing only in case
    doc = make_doc("A shipment is a delivery.", {"source": "Manual"})
    retriever = MagicMock()
    retriever.invoke.return_value = [doc, doc]

    monkeypatch.setattr(
        "backend.src.services.rag.vector_store_service.as_retriever",
        lambda **kwargs: retriever
    )

    fake_answer = "A shipment is a delivery."
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = fake_answer

    monkeypatch.setattr("backend.src.services.rag.RunnableParallel", lambda *a, **k: fake_chain)
    monkeypatch.setattr("backend.src.services.rag.RunnablePassthrough", lambda: None)
    monkeypatch.setattr("backend.src.services.rag.format_docs", lambda docs: "formatted context")
    monkeypatch.setattr(rag_service, "prompt", MagicMock())
    monkeypatch.setattr(rag_service, "llm", MagicMock())
    monkeypatch.setattr(rag_service, "output_parser", MagicMock())

    query1 = QAQuery(question="What is a shipment?")
    query2 = QAQuery(question="what is a shipment?")
    result1 = rag_service.answer_question(query1)
    result2 = rag_service.answer_question(query2)
    assert result1.answer == result2.answer
    assert result1.confidence_score == result2.confidence_score
    assert len(result1.sources) == len(result2.sources)
