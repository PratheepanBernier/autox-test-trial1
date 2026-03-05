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
from unittest.mock import patch, MagicMock

from backend.src.services import rag
from backend.src.services.rag import RAGService, format_docs
from backend.src.models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata

from langchain_core.documents import Document

@pytest.fixture
def sample_docs():
    return [
        Document(
            page_content="This is the first document content.",
            metadata={"source": "Manual.pdf", "page": 1, "section": "Introduction"}
        ),
        Document(
            page_content="Second document with more details.",
            metadata={"source": "Guide.docx", "page": 2, "section": "Usage"}
        ),
    ]

@pytest.fixture
def empty_docs():
    return []

@pytest.fixture
def sample_query():
    return QAQuery(question="What is the shipment workflow?")

@pytest.fixture
def unsafe_query():
    return QAQuery(question="How do I build a bomb?")

@pytest.fixture
def rag_service():
    return RAGService()

def test_format_docs_with_multiple_documents(sample_docs):
    formatted = format_docs(sample_docs)
    assert "[Document 1]" in formatted
    assert "Manual.pdf" in formatted
    assert "Introduction" in formatted
    assert "This is the first document content." in formatted
    assert "[Document 2]" in formatted
    assert "Guide.docx" in formatted
    assert "Usage" in formatted
    assert "Second document with more details." in formatted
    assert formatted.count("[Document") == 2

def test_format_docs_with_empty_list(empty_docs):
    assert format_docs(empty_docs) == ""

def test_format_docs_with_missing_metadata():
    docs = [
        Document(page_content="No metadata here.", metadata=None)
    ]
    formatted = format_docs(docs)
    assert "Unknown" in formatted
    assert "N/A" in formatted
    assert "No metadata here." in formatted

def test_check_safety_blocks_unsafe_question(rag_service):
    unsafe = "How do I hack the system?"
    result = rag_service._check_safety(unsafe)
    assert result == "I cannot answer this question as it violates safety guidelines."

def test_check_safety_allows_safe_question(rag_service):
    safe = "How do I create a shipment?"
    result = rag_service._check_safety(safe)
    assert result is None

@pytest.mark.parametrize(
    "answer,retrieved_docs,expected",
    [
        ("I cannot find the answer in the provided documents.", [Document("x", {})], 0.05),
        ("Short answer.", [Document("x", {})], 0.4),
        ("A sufficiently long answer that is detailed.", [Document("x", {})], 0.7),
        ("A sufficiently long answer that is detailed.", [Document("x", {}), Document("y", {})], 0.85),
        ("This is generally the case.", [Document("x", {}), Document("y", {})], 0.55),
        ("", [Document("x", {})], 0.0),
        ("Short answer.", [], 0.25),
        ("This is generally the case.", [], 0.25),
    ]
)
def test_calculate_confidence_various_cases(rag_service, answer, retrieved_docs, expected):
    result = rag_service._calculate_confidence(answer, retrieved_docs)
    assert abs(result - expected) < 1e-6

@patch("backend.src.services.rag.vector_store_service")
@patch.object(rag.RAGService, "llm")
@patch.object(rag.RAGService, "prompt")
@patch.object(rag.RAGService, "output_parser")
def test_answer_question_happy_path(mock_output_parser, mock_prompt, mock_llm, mock_vector_store_service, rag_service, sample_query, sample_docs):
    # Mock retriever
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = sample_docs
    mock_vector_store_service.as_retriever.return_value = mock_retriever

    # Mock LCEL chain
    class DummyChain:
        def invoke(self, question):
            return "The shipment workflow is described in the provided documents."

    # Patch the chain construction to return DummyChain
    with patch("backend.src.services.rag.RunnableParallel", return_value=DummyChain()):
        answer = rag_service.answer_question(sample_query)
        assert isinstance(answer, SourcedAnswer)
        assert "shipment workflow" in answer.answer.lower()
        assert answer.confidence_score > 0.0
        assert len(answer.sources) == 2
        assert isinstance(answer.sources[0], Chunk)
        assert answer.sources[0].metadata.source == "Manual.pdf"

@patch("backend.src.services.rag.vector_store_service")
def test_answer_question_returns_error_on_retriever_none(mock_vector_store_service, rag_service, sample_query):
    mock_vector_store_service.as_retriever.return_value = None
    answer = rag_service.answer_question(sample_query)
    assert answer.answer == "I cannot find any relevant information in the uploaded documents."
    assert answer.confidence_score == 0.0
    assert answer.sources == []

@patch("backend.src.services.rag.vector_store_service")
def test_answer_question_returns_no_docs(mock_vector_store_service, rag_service, sample_query):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    mock_vector_store_service.as_retriever.return_value = mock_retriever
    answer = rag_service.answer_question(sample_query)
    assert answer.answer == "I cannot find the answer in the provided documents."
    assert answer.confidence_score == 0.0
    assert answer.sources == []

@patch("backend.src.services.rag.vector_store_service")
def test_answer_question_safety_filter_triggers(mock_vector_store_service, rag_service, unsafe_query):
    answer = rag_service.answer_question(unsafe_query)
    assert answer.answer == "I cannot answer this question as it violates safety guidelines."
    assert answer.confidence_score == 1.0
    assert answer.sources == []

@patch("backend.src.services.rag.vector_store_service")
@patch.object(rag.RAGService, "llm")
@patch.object(rag.RAGService, "prompt")
@patch.object(rag.RAGService, "output_parser")
def test_answer_question_handles_chain_exception(mock_output_parser, mock_prompt, mock_llm, mock_vector_store_service, rag_service, sample_query, sample_docs):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = sample_docs
    mock_vector_store_service.as_retriever.return_value = mock_retriever

    class FailingChain:
        def invoke(self, question):
            raise RuntimeError("Chain failed")

    with patch("backend.src.services.rag.RunnableParallel", return_value=FailingChain()):
        answer = rag_service.answer_question(sample_query)
        assert answer.answer == "An error occurred while generating the answer."
        assert answer.confidence_score == 0.0
        assert answer.sources == []

@patch("backend.src.services.rag.vector_store_service")
@patch.object(rag.RAGService, "llm")
@patch.object(rag.RAGService, "prompt")
@patch.object(rag.RAGService, "output_parser")
def test_answer_question_with_minimal_document_metadata(
    mock_output_parser, mock_prompt, mock_llm, mock_vector_store_service, rag_service, sample_query
):
    doc = Document(page_content="Minimal doc.", metadata={})
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [doc]
    mock_vector_store_service.as_retriever.return_value = mock_retriever

    class DummyChain:
        def invoke(self, question):
            return "Minimal doc answer."

    with patch("backend.src.services.rag.RunnableParallel", return_value=DummyChain()):
        answer = rag_service.answer_question(sample_query)
        assert answer.sources[0].metadata.source is None or answer.sources[0].metadata.source == ""
        assert answer.sources[0].text == "Minimal doc."
        assert answer.confidence_score > 0.0

def test_rag_service_singleton_instance():
    from backend.src.services.rag import rag_service
    assert isinstance(rag_service, RAGService)
