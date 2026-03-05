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
from langchain_core.documents import Document
from models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata

@pytest.fixture
def sample_docs():
    return [
        Document(
            page_content="Content A",
            metadata={"source": "Manual.pdf", "page": 1, "section": "Intro"}
        ),
        Document(
            page_content="Content B",
            metadata={"source": "Guide.docx", "page": 2, "section": "Usage"}
        ),
    ]

def test_format_docs_with_multiple_documents(sample_docs):
    formatted = format_docs(sample_docs)
    assert "[Document 1]" in formatted
    assert "Manual.pdf" in formatted
    assert "Content A" in formatted
    assert "[Document 2]" in formatted
    assert "Guide.docx" in formatted
    assert "Content B" in formatted
    assert formatted.count("[Document") == 2

def test_format_docs_with_empty_list():
    assert format_docs([]) == ""

def test_format_docs_with_missing_metadata():
    docs = [Document(page_content="No meta", metadata=None)]
    formatted = format_docs(docs)
    assert "Unknown" in formatted
    assert "N/A" in formatted
    assert "No meta" in formatted

def test_check_safety_blocks_unsafe_questions():
    service = RAGService()
    for word in ["bomb", "kill", "suicide", "hack", "exploit", "weapon"]:
        result = service._check_safety(f"How to {word}?")
        assert result == "I cannot answer this question as it violates safety guidelines."

def test_check_safety_allows_safe_questions():
    service = RAGService()
    assert service._check_safety("How to ship a package?") is None

@pytest.mark.parametrize(
    "answer,retrieved_docs,expected",
    [
        ("", [], 0.0),
        ("I cannot find the answer in the provided documents.", [Document(page_content="", metadata={})], 0.05),
        ("Short answer.", [Document(page_content="", metadata={}), Document(page_content="", metadata={})], 0.55),
        ("A sufficiently long answer that is more than thirty characters.", [Document(page_content="", metadata={}), Document(page_content="", metadata={})], 0.85),
        ("A sufficiently long answer that is more than thirty characters.", [Document(page_content="", metadata={})], 0.7),
        ("This is generally the case.", [Document(page_content="", metadata={}), Document(page_content="", metadata={})], 0.55),
        ("This is generally the case.", [Document(page_content="", metadata={})], 0.4),
        ("", [Document(page_content="", metadata={})], 0.0),
    ]
)
def test_calculate_confidence_various_cases(answer, retrieved_docs, expected):
    service = RAGService()
    result = service._calculate_confidence(answer, retrieved_docs)
    assert abs(result - expected) < 1e-6

def make_mock_retriever(return_docs):
    mock = MagicMock()
    mock.invoke.return_value = return_docs
    return mock

def test_answer_question_safety_violation(monkeypatch):
    service = RAGService()
    query = QAQuery(question="How to make a bomb?")
    with patch.object(service, "_check_safety", return_value="BLOCKED"):
        result = service.answer_question(query)
    assert isinstance(result, SourcedAnswer)
    assert result.answer == "BLOCKED"
    assert result.confidence_score == 1.0
    assert result.sources == []

def test_answer_question_retriever_none(monkeypatch):
    service = RAGService()
    query = QAQuery(question="What is TMS?")
    monkeypatch.setattr(rag_module.vector_store_service, "as_retriever", lambda **kwargs: None)
    with patch.object(service, "_check_safety", return_value=None):
        result = service.answer_question(query)
    assert result.answer == "I cannot find any relevant information in the uploaded documents."
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_answer_question_no_documents(monkeypatch):
    service = RAGService()
    query = QAQuery(question="What is TMS?")
    mock_retriever = make_mock_retriever([])
    monkeypatch.setattr(rag_module.vector_store_service, "as_retriever", lambda **kwargs: mock_retriever)
    with patch.object(service, "_check_safety", return_value=None):
        result = service.answer_question(query)
    assert result.answer == "I cannot find the answer in the provided documents."
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_answer_question_happy_path(monkeypatch, sample_docs):
    service = RAGService()
    query = QAQuery(question="What is TMS?")
    mock_retriever = make_mock_retriever(sample_docs)
    monkeypatch.setattr(rag_module.vector_store_service, "as_retriever", lambda **kwargs: mock_retriever)
    with patch.object(service, "_check_safety", return_value=None), \
         patch.object(service, "_calculate_confidence", return_value=0.77) as conf_patch, \
         patch.object(service, "prompt") as mock_prompt, \
         patch.object(service, "llm") as mock_llm, \
         patch.object(service, "output_parser") as mock_parser:
        # Simulate LCEL chain
        class DummyChain:
            def invoke(self, question):
                return "This is the answer from the chain."
        dummy_chain = DummyChain()
        with patch("backend.src.services.rag.RunnableParallel", return_value=dummy_chain), \
             patch("backend.src.services.rag.RunnablePassthrough"):
            result = service.answer_question(query)
    assert isinstance(result, SourcedAnswer)
    assert result.answer == "This is the answer from the chain."
    assert result.confidence_score == 0.77
    assert len(result.sources) == len(sample_docs)
    for chunk, doc in zip(result.sources, sample_docs):
        assert isinstance(chunk, Chunk)
        assert chunk.text == doc.page_content
        assert isinstance(chunk.metadata, DocumentMetadata)

def test_answer_question_chain_raises_exception(monkeypatch, sample_docs):
    service = RAGService()
    query = QAQuery(question="What is TMS?")
    mock_retriever = make_mock_retriever(sample_docs)
    monkeypatch.setattr(rag_module.vector_store_service, "as_retriever", lambda **kwargs: mock_retriever)
    with patch.object(service, "_check_safety", return_value=None), \
         patch.object(service, "prompt") as mock_prompt, \
         patch.object(service, "llm") as mock_llm, \
         patch.object(service, "output_parser") as mock_parser:
        class DummyChain:
            def invoke(self, question):
                raise RuntimeError("Chain failed")
        dummy_chain = DummyChain()
        with patch("backend.src.services.rag.RunnableParallel", return_value=dummy_chain), \
             patch("backend.src.services.rag.RunnablePassthrough"):
            result = service.answer_question(query)
    assert isinstance(result, SourcedAnswer)
    assert result.answer == "An error occurred while generating the answer."
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_rag_service_singleton_instance():
    from backend.src.services.rag import rag_service
    assert isinstance(rag_service, RAGService)
