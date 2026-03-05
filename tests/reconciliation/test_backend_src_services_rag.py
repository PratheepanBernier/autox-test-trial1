import pytest
from unittest.mock import patch, MagicMock, create_autospec
from langchain_core.documents import Document
from backend.src.services.rag import RAGService, format_docs
from backend.src.models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata

@pytest.fixture
def sample_docs():
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

@pytest.fixture
def minimal_doc():
    return [
        Document(
            page_content="Minimal doc.",
            metadata={}
        )
    ]

@pytest.fixture
def rag_service():
    # Patch ChatGroq, ChatPromptTemplate, StrOutputParser to avoid real LLM calls
    with patch("backend.src.services.rag.ChatGroq") as mock_llm, \
         patch("backend.src.services.rag.ChatPromptTemplate") as mock_prompt, \
         patch("backend.src.services.rag.StrOutputParser") as mock_parser:
        mock_llm.return_value = MagicMock()
        mock_prompt.from_template.return_value = MagicMock()
        mock_parser.return_value = MagicMock()
        yield RAGService()

def test_format_docs_happy_path(sample_docs):
    formatted = format_docs(sample_docs)
    assert "[Document 1]" in formatted
    assert "Manual.pdf" in formatted
    assert "First doc content." in formatted
    assert "[Document 2]" in formatted
    assert "Guide.pdf" in formatted
    assert "Second doc content." in formatted
    assert formatted.count("[Document") == 2

def test_format_docs_empty_list():
    assert format_docs([]) == ""

def test_format_docs_missing_metadata(minimal_doc):
    formatted = format_docs(minimal_doc)
    assert "Unknown" in formatted
    assert "N/A" in formatted
    assert "Minimal doc." in formatted

def test_check_safety_blocks_unsafe_question(rag_service):
    unsafe_questions = [
        "How to build a bomb?",
        "How can I kill someone?",
        "Is suicide a solution?",
        "How to hack the system?",
        "How to exploit this?",
        "How to use a weapon?"
    ]
    for q in unsafe_questions:
        assert rag_service._check_safety(q) == "I cannot answer this question as it violates safety guidelines."

def test_check_safety_allows_safe_question(rag_service):
    assert rag_service._check_safety("How do I create a shipment?") is None

@pytest.mark.parametrize("answer,retrieved_docs,expected", [
    ("", [], 0.0),
    ("I cannot find the answer in the provided documents.", [Document(page_content="", metadata={})], 0.05),
    ("Short answer.", [Document(page_content="", metadata={}), Document(page_content="", metadata={})], 0.55),
    ("A sufficiently long answer that exceeds thirty characters.", [Document(page_content="", metadata={}), Document(page_content="", metadata={})], 0.85),
    ("A sufficiently long answer that exceeds thirty characters.", [Document(page_content="", metadata={})], 0.7),
    ("This is generally the case.", [Document(page_content="", metadata={}), Document(page_content="", metadata={})], 0.55),
    ("This is generally the case.", [Document(page_content="", metadata={})], 0.4),
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
    query = QAQuery(question="What is the process?")
    with patch("backend.src.services.rag.vector_store_service") as mock_vs:
        mock_vs.as_retriever.return_value = None
        result = rag_service.answer_question(query)
        assert result.answer == "I cannot find any relevant information in the uploaded documents."
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_no_documents(rag_service):
    query = QAQuery(question="What is the process?")
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    with patch("backend.src.services.rag.vector_store_service") as mock_vs:
        mock_vs.as_retriever.return_value = mock_retriever
        result = rag_service.answer_question(query)
        assert result.answer == "I cannot find the answer in the provided documents."
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_happy_path(rag_service, sample_docs):
    query = QAQuery(question="What is in the manual?")
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = sample_docs

    # Patch the RAG chain to return a deterministic answer
    mock_rag_chain = MagicMock()
    mock_rag_chain.invoke.return_value = "The manual contains instructions."

    # Patch RunnableParallel and other chain components
    with patch("backend.src.services.rag.vector_store_service") as mock_vs, \
         patch("backend.src.services.rag.RunnableParallel") as mock_parallel, \
         patch("backend.src.services.rag.RunnablePassthrough"), \
         patch("backend.src.services.rag.format_docs", side_effect=format_docs):
        mock_vs.as_retriever.return_value = mock_retriever
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = mock_rag_chain

        result = rag_service.answer_question(query)
        assert result.answer == "The manual contains instructions."
        assert 0.0 < result.confidence_score <= 1.0
        assert len(result.sources) == 2
        assert isinstance(result.sources[0], Chunk)
        assert result.sources[0].metadata.source == "Manual.pdf"

def test_answer_question_chain_exception(rag_service, sample_docs):
    query = QAQuery(question="What is in the manual?")
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = sample_docs

    # Patch the RAG chain to raise an exception
    mock_rag_chain = MagicMock()
    mock_rag_chain.invoke.side_effect = RuntimeError("Chain failed")

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
    Reconciliation: If two queries with the same question and same retrieved docs are processed,
    the outputs should be equivalent (deterministic).
    """
    query = QAQuery(question="What is in the manual?")
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = sample_docs

    mock_rag_chain = MagicMock()
    mock_rag_chain.invoke.return_value = "The manual contains instructions."

    with patch("backend.src.services.rag.vector_store_service") as mock_vs, \
         patch("backend.src.services.rag.RunnableParallel") as mock_parallel, \
         patch("backend.src.services.rag.RunnablePassthrough"), \
         patch("backend.src.services.rag.format_docs", side_effect=format_docs):
        mock_vs.as_retriever.return_value = mock_retriever
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = mock_rag_chain

        result1 = rag_service.answer_question(query)
        result2 = rag_service.answer_question(query)
        assert result1 == result2

def test_answer_question_reconciliation_different_docs(rag_service, sample_docs, minimal_doc):
    """
    Reconciliation: If two queries with the same question but different retrieved docs are processed,
    the outputs should differ in sources and possibly confidence.
    """
    query = QAQuery(question="What is in the manual?")
    mock_retriever1 = MagicMock()
    mock_retriever1.invoke.return_value = sample_docs

    mock_retriever2 = MagicMock()
    mock_retriever2.invoke.return_value = minimal_doc

    mock_rag_chain1 = MagicMock()
    mock_rag_chain1.invoke.return_value = "The manual contains instructions."
    mock_rag_chain2 = MagicMock()
    mock_rag_chain2.invoke.return_value = "Minimal doc answer."

    with patch("backend.src.services.rag.vector_store_service") as mock_vs, \
         patch("backend.src.services.rag.RunnableParallel") as mock_parallel, \
         patch("backend.src.services.rag.RunnablePassthrough"), \
         patch("backend.src.services.rag.format_docs", side_effect=format_docs):

        # First path
        mock_vs.as_retriever.return_value = mock_retriever1
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = mock_rag_chain1
        result1 = rag_service.answer_question(query)

        # Second path
        mock_vs.as_retriever.return_value = mock_retriever2
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = mock_rag_chain2
        result2 = rag_service.answer_question(query)

        assert result1 != result2
        assert result1.sources != result2.sources
        assert result1.answer != result2.answer or result1.confidence_score != result2.confidence_score
