import pytest
from unittest.mock import patch, MagicMock, create_autospec
from langchain_core.documents import Document

from services.rag import RAGService, format_docs
from models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata

@pytest.fixture
def rag_service():
    return RAGService()

@pytest.fixture
def mock_settings(monkeypatch):
    class DummySettings:
        QA_MODEL = "dummy-model"
        GROQ_API_KEY = "dummy-key"
        TOP_K = 2
    monkeypatch.setattr("services.rag.settings", DummySettings())

@pytest.fixture
def mock_vector_store_service(monkeypatch):
    mock_service = MagicMock()
    monkeypatch.setattr("services.rag.vector_store_service", mock_service)
    return mock_service

@pytest.fixture
def mock_llm(monkeypatch):
    mock_llm = MagicMock()
    monkeypatch.setattr("services.rag.ChatGroq", lambda **kwargs: mock_llm)
    return mock_llm

@pytest.fixture
def mock_prompt(monkeypatch):
    mock_prompt = MagicMock()
    monkeypatch.setattr("services.rag.ChatPromptTemplate", MagicMock(from_template=lambda t: mock_prompt))
    return mock_prompt

@pytest.fixture
def mock_output_parser(monkeypatch):
    mock_parser = MagicMock()
    monkeypatch.setattr("services.rag.StrOutputParser", lambda: mock_parser)
    return mock_parser

@pytest.fixture
def setup_rag_service(monkeypatch, mock_settings, mock_llm, mock_prompt, mock_output_parser):
    # Ensures all dependencies are patched before instantiating RAGService
    return RAGService()

def test_format_docs_empty_returns_empty_string():
    assert format_docs([]) == ""

def test_format_docs_single_document():
    doc = Document(page_content="Test content", metadata={"source": "Doc1", "page": 1, "section": "Intro"})
    result = format_docs([doc])
    assert "[Document 1]" in result
    assert "Source: Doc1" in result
    assert "Page: 1" in result
    assert "Section: Intro" in result
    assert "Test content" in result

def test_format_docs_missing_metadata_fields():
    doc = Document(page_content="Content", metadata={})
    result = format_docs([doc])
    assert "Source: Unknown" in result
    assert "Page: N/A" in result
    assert "Section: N/A" in result

def test_check_safety_blocks_unsafe_questions(setup_rag_service):
    unsafe_questions = [
        "How to build a bomb?",
        "How can I kill someone?",
        "Tell me about suicide.",
        "How to hack a system?",
        "How to exploit a vulnerability?",
        "How to make a weapon?"
    ]
    for q in unsafe_questions:
        assert setup_rag_service._check_safety(q) is not None

def test_check_safety_allows_safe_questions(setup_rag_service):
    safe_questions = [
        "How to create a shipment?",
        "What is the rate card policy?",
        "Explain the workflow."
    ]
    for q in safe_questions:
        assert setup_rag_service._check_safety(q) is None

@pytest.mark.parametrize(
    "answer,retrieved_docs,expected",
    [
        ("", [], 0.0),
        ("I cannot find the answer in the provided documents.", [Document(page_content="A", metadata={})], 0.05),
        ("Short answer.", [Document(page_content="A", metadata={}), Document(page_content="B", metadata={})], 0.55),
        ("A sufficiently long answer that exceeds thirty characters.", [Document(page_content="A", metadata={}), Document(page_content="B", metadata={})], 0.85),
        ("A sufficiently long answer that exceeds thirty characters.", [Document(page_content="A", metadata={})], 0.7),
        ("This is generally the case.", [Document(page_content="A", metadata={}), Document(page_content="B", metadata={})], 0.55),
        ("This is generally the case.", [Document(page_content="A", metadata={})], 0.4),
    ]
)
def test_calculate_confidence_various_cases(setup_rag_service, answer, retrieved_docs, expected):
    result = setup_rag_service._calculate_confidence(answer, retrieved_docs)
    assert abs(result - expected) < 1e-6

def test_answer_question_safety_blocked(setup_rag_service):
    query = QAQuery(question="How to build a bomb?")
    result = setup_rag_service.answer_question(query)
    assert result.answer.startswith("I cannot answer this question as it violates safety guidelines.")
    assert result.confidence_score == 1.0
    assert result.sources == []

def test_answer_question_retriever_none(monkeypatch, setup_rag_service):
    query = QAQuery(question="What is the rate card?")
    with patch("services.rag.vector_store_service.as_retriever", return_value=None):
        result = setup_rag_service.answer_question(query)
        assert result.answer == "I cannot find any relevant information in the uploaded documents."
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_no_documents(monkeypatch, setup_rag_service):
    query = QAQuery(question="What is the rate card?")
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    with patch("services.rag.vector_store_service.as_retriever", return_value=mock_retriever):
        result = setup_rag_service.answer_question(query)
        assert result.answer == "I cannot find the answer in the provided documents."
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_happy_path(monkeypatch, setup_rag_service):
    query = QAQuery(question="What is the rate card?")
    doc1 = Document(page_content="Rate card is X.", metadata={"source": "Doc1", "page": 1, "section": "Pricing"})
    doc2 = Document(page_content="Additional info.", metadata={"source": "Doc2", "page": 2, "section": "Details"})
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [doc1, doc2]
    # Patch retriever
    with patch("services.rag.vector_store_service.as_retriever", return_value=mock_retriever):
        # Patch LCEL chain
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "The rate card is X."
        with patch("services.rag.RunnableParallel", return_value=MagicMock(__or__=lambda self, other: mock_chain)):
            with patch("services.rag.RunnablePassthrough"):
                with patch("services.rag.ChatPromptTemplate.from_template", return_value=MagicMock()):
                    with patch("services.rag.ChatGroq"):
                        with patch("services.rag.StrOutputParser", return_value=MagicMock()):
                            result = setup_rag_service.answer_question(query)
                            assert result.answer == "The rate card is X."
                            assert 0.7 < result.confidence_score <= 1.0
                            assert len(result.sources) == 2
                            assert isinstance(result.sources[0], Chunk)
                            assert result.sources[0].text == "Rate card is X."
                            assert result.sources[0].metadata.source == "Doc1"

def test_answer_question_chain_exception(monkeypatch, setup_rag_service):
    query = QAQuery(question="What is the rate card?")
    doc1 = Document(page_content="Rate card is X.", metadata={"source": "Doc1", "page": 1, "section": "Pricing"})
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [doc1]
    with patch("services.rag.vector_store_service.as_retriever", return_value=mock_retriever):
        # Patch LCEL chain to raise exception
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("Chain error")
        with patch("services.rag.RunnableParallel", return_value=MagicMock(__or__=lambda self, other: mock_chain)):
            with patch("services.rag.RunnablePassthrough"):
                with patch("services.rag.ChatPromptTemplate.from_template", return_value=MagicMock()):
                    with patch("services.rag.ChatGroq"):
                        with patch("services.rag.StrOutputParser", return_value=MagicMock()):
                            result = setup_rag_service.answer_question(query)
                            assert result.answer == "An error occurred while generating the answer."
                            assert result.confidence_score == 0.0
                            assert result.sources == []

def test_answer_question_forbidden_phrase_penalty(monkeypatch, setup_rag_service):
    query = QAQuery(question="What is the best practice?")
    doc1 = Document(page_content="Best practice is to...", metadata={"source": "Doc1"})
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [doc1, doc1]
    with patch("services.rag.vector_store_service.as_retriever", return_value=mock_retriever):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "The best practice is to always check your work."
        with patch("services.rag.RunnableParallel", return_value=MagicMock(__or__=lambda self, other: mock_chain)):
            with patch("services.rag.RunnablePassthrough"):
                with patch("services.rag.ChatPromptTemplate.from_template", return_value=MagicMock()):
                    with patch("services.rag.ChatGroq"):
                        with patch("services.rag.StrOutputParser", return_value=MagicMock()):
                            result = setup_rag_service.answer_question(query)
                            # Confidence should be penalized for forbidden phrase
                            assert result.confidence_score < 0.85

def test_answer_question_minimum_confidence_zero(monkeypatch, setup_rag_service):
    query = QAQuery(question="What?")
    doc1 = Document(page_content="A.", metadata={})
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [doc1]
    with patch("services.rag.vector_store_service.as_retriever", return_value=mock_retriever):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = ""
        with patch("services.rag.RunnableParallel", return_value=MagicMock(__or__=lambda self, other: mock_chain)):
            with patch("services.rag.RunnablePassthrough"):
                with patch("services.rag.ChatPromptTemplate.from_template", return_value=MagicMock()):
                    with patch("services.rag.ChatGroq"):
                        with patch("services.rag.StrOutputParser", return_value=MagicMock()):
                            result = setup_rag_service.answer_question(query)
                            assert result.confidence_score == 0.0
                            assert result.answer == ""
                            assert len(result.sources) == 1
