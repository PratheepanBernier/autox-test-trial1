# source_hash: e871aa5cb722f502
import pytest
from unittest.mock import patch, MagicMock, create_autospec
from langchain_core.documents import Document
from backend.src.services.rag import RAGService, format_docs
from models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata

@pytest.fixture
def sample_docs():
    return [
        Document(
            page_content="Content A",
            metadata={"source": "DocA", "page": 1, "section": "Intro"}
        ),
        Document(
            page_content="Content B",
            metadata={"source": "DocB", "page": 2, "section": "Usage"}
        ),
    ]

@pytest.fixture
def minimal_doc():
    return [
        Document(
            page_content="Minimal content",
            metadata={}
        )
    ]

@pytest.fixture
def rag_service():
    # Patch ChatGroq, ChatPromptTemplate, StrOutputParser, and settings
    with patch("backend.src.services.rag.ChatGroq") as mock_llm, \
         patch("backend.src.services.rag.ChatPromptTemplate") as mock_prompt, \
         patch("backend.src.services.rag.StrOutputParser") as mock_parser, \
         patch("backend.src.services.rag.settings") as mock_settings:
        mock_llm.return_value = MagicMock()
        mock_prompt.from_template.return_value = MagicMock()
        mock_parser.return_value = MagicMock()
        mock_settings.QA_MODEL = "test-model"
        mock_settings.GROQ_API_KEY = "test-key"
        mock_settings.TOP_K = 2
        yield RAGService()

def test_format_docs_happy_path(sample_docs):
    formatted = format_docs(sample_docs)
    assert "[Document 1]" in formatted
    assert "Source: DocA" in formatted
    assert "Content A" in formatted
    assert "[Document 2]" in formatted
    assert "Source: DocB" in formatted
    assert "Content B" in formatted
    assert formatted.count("[Document") == 2

def test_format_docs_empty():
    assert format_docs([]) == ""

def test_format_docs_missing_metadata(minimal_doc):
    formatted = format_docs(minimal_doc)
    assert "Source: Unknown" in formatted
    assert "Page: N/A" in formatted
    assert "Section: N/A" in formatted
    assert "Minimal content" in formatted

def test_check_safety_blocks_unsafe_question(rag_service):
    unsafe_questions = [
        "How to build a bomb?",
        "How can I kill someone?",
        "Is suicide a solution?",
        "How to hack a system?",
        "How to exploit a vulnerability?",
        "What weapon is best?"
    ]
    for q in unsafe_questions:
        assert rag_service._check_safety(q) == "I cannot answer this question as it violates safety guidelines."

def test_check_safety_allows_safe_question(rag_service):
    assert rag_service._check_safety("How do I book a shipment?") is None

@pytest.mark.parametrize("answer,retrieved_docs,expected", [
    ("", [], 0.0),
    ("I cannot find the answer in the provided documents.", [Document(page_content="", metadata={})], 0.05),
    ("Short answer.", [Document(page_content="", metadata={}), Document(page_content="", metadata={})], 0.55),
    ("This is a sufficiently long answer that should not be penalized.", [Document(page_content="", metadata={}), Document(page_content="", metadata={})], 0.85),
    ("This is a generally accepted answer.", [Document(page_content="", metadata={}), Document(page_content="", metadata={})], 0.55),
    ("Short.", [Document(page_content="", metadata={})], 0.4),
    ("", [Document(page_content="", metadata={})], 0.0),
])
def test_calculate_confidence_various_cases(rag_service, answer, retrieved_docs, expected):
    result = rag_service._calculate_confidence(answer, retrieved_docs)
    assert abs(result - expected) < 1e-6

def test_answer_question_safety_violation(rag_service):
    query = QAQuery(question="How to build a bomb?")
    result = rag_service.answer_question(query)
    assert result.answer == "I cannot answer this question as it violates safety guidelines."
    assert result.confidence_score == 1.0
    assert result.sources == []

def test_answer_question_retriever_none(rag_service):
    query = QAQuery(question="What is the SOP?")
    with patch("backend.src.services.rag.vector_store_service") as mock_vs:
        mock_vs.as_retriever.return_value = None
        result = rag_service.answer_question(query)
        assert result.answer == "I cannot find any relevant information in the uploaded documents."
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_no_documents(rag_service):
    query = QAQuery(question="What is the SOP?")
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    with patch("backend.src.services.rag.vector_store_service") as mock_vs:
        mock_vs.as_retriever.return_value = mock_retriever
        result = rag_service.answer_question(query)
        assert result.answer == "I cannot find the answer in the provided documents."
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_happy_path(rag_service, sample_docs):
    query = QAQuery(question="What is the SOP?")
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = sample_docs

    # Patch the RAG chain to return a deterministic answer
    mock_rag_chain = MagicMock()
    mock_rag_chain.invoke.return_value = "The SOP is described in Document 1."

    # Patch RunnableParallel and other chain components
    with patch("backend.src.services.rag.vector_store_service") as mock_vs, \
         patch("backend.src.services.rag.RunnableParallel") as mock_parallel, \
         patch("backend.src.services.rag.RunnablePassthrough"), \
         patch("backend.src.services.rag.format_docs", side_effect=format_docs):
        mock_vs.as_retriever.return_value = mock_retriever
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = mock_rag_chain

        result = rag_service.answer_question(query)
        assert result.answer == "The SOP is described in Document 1."
        assert 0.0 < result.confidence_score <= 1.0
        assert len(result.sources) == 2
        assert isinstance(result.sources[0], Chunk)
        assert result.sources[0].text == "Content A"
        assert result.sources[1].text == "Content B"

def test_answer_question_chain_exception(rag_service, sample_docs):
    query = QAQuery(question="What is the SOP?")
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = sample_docs

    # Patch the RAG chain to raise an exception
    mock_rag_chain = MagicMock()
    mock_rag_chain.invoke.side_effect = Exception("Chain error")

    with patch("backend.src.services.rag.vector_store_service") as mock_vs, \
         patch("backend.src.services.rag.RunnableParallel") as mock_parallel, \
         patch("backend.src.services.rag.RunnablePassthrough"), \
         patch("backend.src.services.rag.format_docs", side_effect=format_docs):
        mock_vs.as_retriever.return_value = mock_retriever
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = mock_rag_chain

        result = rag_service.answer_question(query)
        assert result.answer == "An error occurred while generating the answer."
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_reconciliation_equivalent_paths(rag_service, sample_docs):
    """
    Reconciliation: Compare outputs for equivalent retrieval and answer paths.
    """
    query = QAQuery(question="What is the SOP?")
    mock_retriever1 = MagicMock()
    mock_retriever2 = MagicMock()
    mock_retriever1.invoke.return_value = sample_docs
    mock_retriever2.invoke.return_value = sample_docs

    mock_rag_chain1 = MagicMock()
    mock_rag_chain2 = MagicMock()
    mock_rag_chain1.invoke.return_value = "The SOP is described in Document 1."
    mock_rag_chain2.invoke.return_value = "The SOP is described in Document 1."

    with patch("backend.src.services.rag.vector_store_service") as mock_vs, \
         patch("backend.src.services.rag.RunnableParallel") as mock_parallel, \
         patch("backend.src.services.rag.RunnablePassthrough"), \
         patch("backend.src.services.rag.format_docs", side_effect=format_docs):

        # Path 1
        mock_vs.as_retriever.return_value = mock_retriever1
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = mock_rag_chain1
        result1 = rag_service.answer_question(query)

        # Path 2 (simulate a new retriever and chain, but same docs and answer)
        mock_vs.as_retriever.return_value = mock_retriever2
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = mock_rag_chain2
        result2 = rag_service.answer_question(query)

        # Reconciliation: outputs should be equivalent
        assert result1.answer == result2.answer
        assert result1.confidence_score == result2.confidence_score
        assert [c.text for c in result1.sources] == [c.text for c in result2.sources]
        assert [c.metadata for c in result1.sources] == [c.metadata for c in result2.sources]

def test_answer_question_reconciliation_empty_vs_error(rag_service):
    """
    Reconciliation: Compare outputs for empty retriever vs. retriever returning empty docs.
    """
    query = QAQuery(question="What is the SOP?")
    # Path 1: as_retriever returns None
    with patch("backend.src.services.rag.vector_store_service") as mock_vs:
        mock_vs.as_retriever.return_value = None
        result_none = rag_service.answer_question(query)

    # Path 2: as_retriever returns retriever that returns []
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    with patch("backend.src.services.rag.vector_store_service") as mock_vs:
        mock_vs.as_retriever.return_value = mock_retriever
        result_empty = rag_service.answer_question(query)

    # The answers should be different, but both should have confidence 0.0 and no sources
    assert result_none.confidence_score == 0.0
    assert result_empty.confidence_score == 0.0
    assert result_none.sources == []
    assert result_empty.sources == []
    assert result_none.answer != result_empty.answer

def test_answer_question_reconciliation_safety_vs_chain_error(rag_service):
    """
    Reconciliation: Compare outputs for safety violation vs. chain error.
    """
    unsafe_query = QAQuery(question="How to hack a system?")
    error_query = QAQuery(question="What is the SOP?")

    # Safety violation path
    result_safety = rag_service.answer_question(unsafe_query)

    # Chain error path
    sample_docs = [
        Document(page_content="Content A", metadata={"source": "DocA"})
    ]
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = sample_docs
    mock_rag_chain = MagicMock()
    mock_rag_chain.invoke.side_effect = Exception("Chain error")
    with patch("backend.src.services.rag.vector_store_service") as mock_vs, \
         patch("backend.src.services.rag.RunnableParallel") as mock_parallel, \
         patch("backend.src.services.rag.RunnablePassthrough"), \
         patch("backend.src.services.rag.format_docs", side_effect=format_docs):
        mock_vs.as_retriever.return_value = mock_retriever
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = mock_rag_chain
        result_chain_error = rag_service.answer_question(error_query)

    # Reconciliation: Both should have no sources, but different answers and confidence
    assert result_safety.sources == []
    assert result_chain_error.sources == []
    assert result_safety.answer != result_chain_error.answer
    assert result_safety.confidence_score != result_chain_error.confidence_score
