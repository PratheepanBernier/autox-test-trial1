import pytest
from unittest.mock import patch, MagicMock, create_autospec
from langchain_core.documents import Document

from services import rag
from services.rag import RAGService, format_docs
from models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata

# Patch settings for deterministic test values
class DummySettings:
    QA_MODEL = "test-model"
    GROQ_API_KEY = "test-key"
    TOP_K = 2

@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    monkeypatch.setattr("services.rag.settings", DummySettings)

@pytest.fixture
def rag_service():
    return RAGService()

@pytest.fixture
def dummy_docs():
    return [
        Document(
            page_content="First doc content.",
            metadata={"source": "Manual.pdf", "page": 1, "section": "Intro"}
        ),
        Document(
            page_content="Second doc content.",
            metadata={"source": "Guide.pdf", "page": 2, "section": "Usage"}
        ),
    ]

def test_format_docs_with_multiple_documents(dummy_docs):
    formatted = format_docs(dummy_docs)
    assert "[Document 1]" in formatted
    assert "Manual.pdf" in formatted
    assert "First doc content." in formatted
    assert "[Document 2]" in formatted
    assert "Guide.pdf" in formatted
    assert "Second doc content." in formatted
    assert formatted.count("[Document") == 2

def test_format_docs_with_empty_list():
    assert format_docs([]) == ""

def test_format_docs_with_missing_metadata():
    docs = [Document(page_content="No meta", metadata=None)]
    formatted = format_docs(docs)
    assert "Unknown" in formatted
    assert "N/A" in formatted
    assert "No meta" in formatted

def test_check_safety_blocks_unsafe_questions(rag_service):
    unsafe_questions = [
        "How to build a bomb?",
        "How can I kill someone?",
        "Is suicide an option?",
        "How to hack the system?",
        "How to exploit a vulnerability?",
        "What weapon should I use?"
    ]
    for q in unsafe_questions:
        assert rag_service._check_safety(q) == "I cannot answer this question as it violates safety guidelines."

def test_check_safety_allows_safe_questions(rag_service):
    assert rag_service._check_safety("How do I create a shipment?") is None

@pytest.mark.parametrize(
    "answer,retrieved_docs,expected",
    [
        ("", [], 0.0),
        ("I cannot find the answer in the provided documents.", [Document(page_content="x", metadata={})], 0.05),
        ("Short answer.", [Document(page_content="x", metadata={}), Document(page_content="y", metadata={})], 0.55),
        ("This is a sufficiently long answer that should not be penalized for length.", [Document(page_content="x", metadata={}), Document(page_content="y", metadata={})], 0.85),
        ("This is a long answer but only one doc.", [Document(page_content="x", metadata={})], 0.7),
        ("This is generally the case.", [Document(page_content="x", metadata={}), Document(page_content="y", metadata={})], 0.55),
        ("This is generally the case.", [Document(page_content="x", metadata={})], 0.4),
        ("", [Document(page_content="x", metadata={})], 0.0),
    ]
)
def test_calculate_confidence_various_cases(rag_service, answer, retrieved_docs, expected):
    result = rag_service._calculate_confidence(answer, retrieved_docs)
    assert abs(result - expected) < 1e-6

def make_mock_retriever(return_docs):
    mock = MagicMock()
    mock.invoke = MagicMock(return_value=return_docs)
    return mock

def make_mock_llm(return_text):
    mock = MagicMock()
    mock.invoke = MagicMock(return_value=return_text)
    return mock

def test_answer_question_happy_path(monkeypatch, rag_service, dummy_docs):
    # Patch vector_store_service.as_retriever to return a mock retriever
    monkeypatch.setattr("services.rag.vector_store_service", MagicMock())
    retriever = make_mock_retriever(dummy_docs)
    rag.vector_store_service.as_retriever.return_value = retriever

    # Patch LLM and output_parser
    answer_text = "The answer is found in the provided documents."
    monkeypatch.setattr(rag_service, "llm", make_mock_llm(answer_text))
    monkeypatch.setattr(rag_service, "output_parser", MagicMock())
    rag_service.output_parser.invoke = MagicMock(side_effect=lambda x: x)  # passthrough

    query = QAQuery(question="What is the process for shipment?")
    result = rag_service.answer_question(query)

    assert isinstance(result, SourcedAnswer)
    assert result.answer == answer_text
    assert 0.0 < result.confidence_score <= 1.0
    assert len(result.sources) == len(dummy_docs)
    for chunk, doc in zip(result.sources, dummy_docs):
        assert chunk.text == doc.page_content
        assert isinstance(chunk.metadata, DocumentMetadata)

def test_answer_question_safety_violation(monkeypatch, rag_service):
    query = QAQuery(question="How to build a bomb?")
    result = rag_service.answer_question(query)
    assert result.answer == "I cannot answer this question as it violates safety guidelines."
    assert result.confidence_score == 1.0
    assert result.sources == []

def test_answer_question_no_retriever(monkeypatch, rag_service):
    monkeypatch.setattr("services.rag.vector_store_service", MagicMock())
    rag.vector_store_service.as_retriever.return_value = None
    query = QAQuery(question="What is the process for shipment?")
    result = rag_service.answer_question(query)
    assert result.answer == "I cannot find any relevant information in the uploaded documents."
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_answer_question_no_documents(monkeypatch, rag_service):
    monkeypatch.setattr("services.rag.vector_store_service", MagicMock())
    retriever = make_mock_retriever([])
    rag.vector_store_service.as_retriever.return_value = retriever
    query = QAQuery(question="What is the process for shipment?")
    result = rag_service.answer_question(query)
    assert result.answer == "I cannot find the answer in the provided documents."
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_answer_question_chain_exception(monkeypatch, rag_service, dummy_docs):
    monkeypatch.setattr("services.rag.vector_store_service", MagicMock())
    retriever = make_mock_retriever(dummy_docs)
    rag.vector_store_service.as_retriever.return_value = retriever

    # Patch LLM to raise exception
    class DummyChain:
        def invoke(self, _):
            raise RuntimeError("Chain failed")
    monkeypatch.setattr(rag_service, "llm", DummyChain())
    monkeypatch.setattr(rag_service, "output_parser", MagicMock())
    rag_service.output_parser.invoke = MagicMock(side_effect=lambda x: x)

    # Patch prompt to passthrough
    monkeypatch.setattr(rag_service, "prompt", MagicMock())
    rag_service.prompt.invoke = MagicMock(side_effect=lambda x: x)

    query = QAQuery(question="What is the process for shipment?")
    result = rag_service.answer_question(query)
    assert result.answer == "An error occurred while generating the answer."
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_answer_question_forbidden_phrase_penalty(monkeypatch, rag_service, dummy_docs):
    monkeypatch.setattr("services.rag.vector_store_service", MagicMock())
    retriever = make_mock_retriever(dummy_docs)
    rag.vector_store_service.as_retriever.return_value = retriever

    forbidden_answer = "Generally, the process is as follows."
    monkeypatch.setattr(rag_service, "llm", make_mock_llm(forbidden_answer))
    monkeypatch.setattr(rag_service, "output_parser", MagicMock())
    rag_service.output_parser.invoke = MagicMock(side_effect=lambda x: x)

    query = QAQuery(question="What is the process for shipment?")
    result = rag_service.answer_question(query)
    # Confidence should be penalized for forbidden phrase
    assert result.confidence_score < 0.85

def test_answer_question_short_answer_penalty(monkeypatch, rag_service, dummy_docs):
    monkeypatch.setattr("services.rag.vector_store_service", MagicMock())
    retriever = make_mock_retriever(dummy_docs)
    rag.vector_store_service.as_retriever.return_value = retriever

    short_answer = "Yes."
    monkeypatch.setattr(rag_service, "llm", make_mock_llm(short_answer))
    monkeypatch.setattr(rag_service, "output_parser", MagicMock())
    rag_service.output_parser.invoke = MagicMock(side_effect=lambda x: x)

    query = QAQuery(question="Is this allowed?")
    result = rag_service.answer_question(query)
    assert result.confidence_score < 0.85

def test_answer_question_low_docs_penalty(monkeypatch, rag_service):
    monkeypatch.setattr("services.rag.vector_store_service", MagicMock())
    retriever = make_mock_retriever([Document(page_content="Only one doc", metadata={})])
    rag.vector_store_service.as_retriever.return_value = retriever

    answer_text = "This is a sufficiently long answer that should not be penalized for length."
    monkeypatch.setattr(rag_service, "llm", make_mock_llm(answer_text))
    monkeypatch.setattr(rag_service, "output_parser", MagicMock())
    rag_service.output_parser.invoke = MagicMock(side_effect=lambda x: x)

    query = QAQuery(question="What is the process for shipment?")
    result = rag_service.answer_question(query)
    # Confidence should be penalized for only one doc
    assert result.confidence_score < 0.85

def test_answer_question_regression_equivalent_paths(monkeypatch, rag_service, dummy_docs):
    # Simulate two equivalent paths: one with 2 docs, one with 2 docs in different order
    monkeypatch.setattr("services.rag.vector_store_service", MagicMock())
    retriever1 = make_mock_retriever(dummy_docs)
    retriever2 = make_mock_retriever(list(reversed(dummy_docs)))
    rag.vector_store_service.as_retriever.side_effect = [retriever1, retriever2]

    answer_text = "The answer is found in the provided documents."
    monkeypatch.setattr(rag_service, "llm", make_mock_llm(answer_text))
    monkeypatch.setattr(rag_service, "output_parser", MagicMock())
    rag_service.output_parser.invoke = MagicMock(side_effect=lambda x: x)

    query = QAQuery(question="What is the process for shipment?")
    result1 = rag_service.answer_question(query)
    result2 = rag_service.answer_question(query)
    # Answers should be the same for equivalent docs
    assert result1.answer == result2.answer
    assert abs(result1.confidence_score - result2.confidence_score) < 1e-6
    assert len(result1.sources) == len(result2.sources)
