import pytest
from unittest.mock import patch, MagicMock

import os
import sys

# Patch sys.modules to allow import of backend/src/services/rag.py without actual dependencies
# This is necessary because the file imports langchain_groq, langchain_core, etc.
# We'll patch these modules with MagicMock to allow import and focus on logic testing

@pytest.fixture(autouse=True, scope="module")
def patch_external_modules(monkeypatch):
    fake_modules = [
        "langchain_groq",
        "langchain_core.prompts",
        "langchain_core.output_parsers",
        "langchain_core.runnables",
        "langchain_core.documents",
        "models.schemas",
        "services.vector_store",
        "core.config",
    ]
    for mod in fake_modules:
        monkeypatch.setitem(sys.modules, mod, MagicMock())
    yield

# Import after patching
from backend.src.services.rag import RAGService, format_docs

@pytest.fixture
def rag_service():
    # Patch settings and vector_store_service for RAGService
    with patch("backend.src.services.rag.settings") as settings_mock, \
         patch("backend.src.services.rag.vector_store_service") as vs_mock, \
         patch("backend.src.services.rag.ChatGroq") as chatgroq_mock:
        settings_mock.QA_MODEL = "test-model"
        settings_mock.GROQ_API_KEY = "test-key"
        settings_mock.TOP_K = 2
        vs_mock.as_retriever.return_value = MagicMock()
        chatgroq_mock.return_value = MagicMock()
        yield RAGService()

@pytest.fixture
def qa_query():
    # Simulate a QAQuery object
    QAQuery = MagicMock()
    query = QAQuery()
    query.question = "What is the agreed rate for this shipment?"
    return query

@pytest.fixture
def doc_list():
    # Simulate a list of Document objects with metadata
    doc1 = MagicMock()
    doc1.page_content = "The agreed rate is $1200 USD."
    doc1.metadata = {"source": "contract.pdf", "page": 1, "section": "Rates"}
    doc2 = MagicMock()
    doc2.page_content = "Additional charges may apply."
    doc2.metadata = {"source": "contract.pdf", "page": 2, "section": "Notes"}
    return [doc1, doc2]

def test_format_docs_with_metadata(doc_list):
    formatted = format_docs(doc_list)
    assert "[Document 1]" in formatted
    assert "Source: contract.pdf" in formatted
    assert "Rates" in formatted
    assert "The agreed rate is $1200 USD." in formatted
    assert "[Document 2]" in formatted
    assert "Notes" in formatted

def test_format_docs_empty():
    assert format_docs([]) == ""

def test_check_safety_blocks_unsafe(rag_service, qa_query):
    # Test that unsafe questions are blocked
    qa_query.question = "How to build a bomb?"
    result = rag_service.answer_question(qa_query)
    assert "violates safety guidelines" in result.answer
    assert result.confidence_score == 1.0
    assert result.sources == []

def test_answer_question_no_retriever(monkeypatch, rag_service, qa_query):
    # Simulate retriever is None
    with patch("backend.src.services.rag.vector_store_service.as_retriever", return_value=None):
        result = rag_service.answer_question(qa_query)
        assert "cannot find any relevant information" in result.answer.lower()
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_no_docs(monkeypatch, rag_service, qa_query):
    # Simulate retriever returns empty list
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = []
    with patch("backend.src.services.rag.vector_store_service.as_retriever", return_value=fake_retriever):
        result = rag_service.answer_question(qa_query)
        assert "cannot find the answer" in result.answer.lower()
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_success(monkeypatch, rag_service, qa_query, doc_list):
    # Simulate retriever returns docs, chain returns answer
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = doc_list
    # Patch the RAG chain to return a deterministic answer
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = "The agreed rate is $1200 USD."
    # Patch RunnableParallel and other chain components
    with patch("backend.src.services.rag.vector_store_service.as_retriever", return_value=fake_retriever), \
         patch.object(rag_service, "prompt"), \
         patch.object(rag_service, "llm"), \
         patch.object(rag_service, "output_parser"), \
         patch("backend.src.services.rag.RunnableParallel", return_value=MagicMock()), \
         patch("backend.src.services.rag.RunnablePassthrough", return_value=MagicMock()), \
         patch("backend.src.services.rag.format_docs", side_effect=lambda docs: "context block"), \
         patch("backend.src.services.rag.Chunk") as ChunkMock, \
         patch("backend.src.services.rag.DocumentMetadata") as DocMetaMock:
        # Patch the chain to return our answer
        rag_service.prompt.__or__.return_value.__or__.return_value.__or__.return_value = fake_chain
        result = rag_service.answer_question(qa_query)
        assert "agreed rate" in result.answer.lower()
        assert 0.0 < result.confidence_score <= 1.0
        assert isinstance(result.sources, list)

def test_answer_question_forbidden_phrase(monkeypatch, rag_service, qa_query, doc_list):
    # Simulate answer contains forbidden phrase, confidence is reduced
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = doc_list
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = "Generally, the agreed rate is $1200 USD."
    with patch("backend.src.services.rag.vector_store_service.as_retriever", return_value=fake_retriever), \
         patch.object(rag_service, "prompt"), \
         patch.object(rag_service, "llm"), \
         patch.object(rag_service, "output_parser"), \
         patch("backend.src.services.rag.RunnableParallel", return_value=MagicMock()), \
         patch("backend.src.services.rag.RunnablePassthrough", return_value=MagicMock()), \
         patch("backend.src.services.rag.format_docs", side_effect=lambda docs: "context block"), \
         patch("backend.src.services.rag.Chunk") as ChunkMock, \
         patch("backend.src.services.rag.DocumentMetadata") as DocMetaMock:
        rag_service.prompt.__or__.return_value.__or__.return_value.__or__.return_value = fake_chain
        result = rag_service.answer_question(qa_query)
        assert "generally" in result.answer.lower()
        assert 0.0 < result.confidence_score < 1.0

def test_answer_question_chain_exception(monkeypatch, rag_service, qa_query, doc_list):
    # Simulate exception in chain
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = doc_list
    with patch("backend.src.services.rag.vector_store_service.as_retriever", return_value=fake_retriever), \
         patch.object(rag_service, "prompt"), \
         patch.object(rag_service, "llm"), \
         patch.object(rag_service, "output_parser"), \
         patch("backend.src.services.rag.RunnableParallel", side_effect=Exception("Chain error")), \
         patch("backend.src.services.rag.RunnablePassthrough", return_value=MagicMock()), \
         patch("backend.src.services.rag.format_docs", side_effect=lambda docs: "context block"), \
         patch("backend.src.services.rag.Chunk") as ChunkMock, \
         patch("backend.src.services.rag.DocumentMetadata") as DocMetaMock:
        result = rag_service.answer_question(qa_query)
        assert "error occurred" in result.answer.lower()
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_confidence_low_for_short_answer(rag_service):
    # Short answer, low confidence
    answer = "Yes."
    docs = [MagicMock(), MagicMock()]
    conf = rag_service._calculate_confidence(answer, docs)
    assert conf < 0.85

def test_confidence_low_for_few_docs(rag_service):
    # Only one doc, confidence reduced
    answer = "The agreed rate is $1200 USD."
    docs = [MagicMock()]
    conf = rag_service._calculate_confidence(answer, docs)
    assert conf < 0.85

def test_confidence_low_for_forbidden_phrase(rag_service):
    # Forbidden phrase, confidence reduced
    answer = "Generally, the agreed rate is $1200 USD."
    docs = [MagicMock(), MagicMock()]
    conf = rag_service._calculate_confidence(answer, docs)
    assert conf < 0.85

def test_confidence_zero_for_no_answer(rag_service):
    conf = rag_service._calculate_confidence("", [])
    assert conf == 0.0

def test_confidence_low_for_cannot_find(rag_service):
    answer = "I cannot find the answer in the provided documents."
    docs = [MagicMock(), MagicMock()]
    conf = rag_service._calculate_confidence(answer, docs)
    assert conf == 0.05
