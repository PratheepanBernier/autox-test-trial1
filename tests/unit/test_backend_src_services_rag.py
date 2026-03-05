import pytest
from unittest.mock import patch, MagicMock, create_autospec
from langchain_core.documents import Document

from services.rag import RAGService, format_docs
from models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata

# -----------------------------
# format_docs tests
# -----------------------------

def test_format_docs_empty_list_returns_empty_string():
    assert format_docs([]) == ""

def test_format_docs_single_document_with_metadata():
    doc = Document(
        page_content="This is a test document.",
        metadata={"source": "Manual.pdf", "page": 3, "section": "2.1"}
    )
    expected = (
        "[Document 1]\n"
        "Source: Manual.pdf | Page: 3 | Section: 2.1\n"
        "This is a test document."
    )
    assert format_docs([doc]) == expected

def test_format_docs_multiple_documents_with_partial_metadata():
    doc1 = Document(
        page_content="Doc1 content.",
        metadata={"source": "A.pdf", "page": 1}
    )
    doc2 = Document(
        page_content="Doc2 content.",
        metadata=None
    )
    expected = (
        "[Document 1]\n"
        "Source: A.pdf | Page: 1 | Section: N/A\n"
        "Doc1 content.\n\n"
        "[Document 2]\n"
        "Source: Unknown | Page: N/A | Section: N/A\n"
        "Doc2 content."
    )
    assert format_docs([doc1, doc2]) == expected

# -----------------------------
# RAGService._check_safety tests
# -----------------------------

@pytest.mark.parametrize("question,expected", [
    ("How to hack the system?", "I cannot answer this question as it violates safety guidelines."),
    ("Tell me about bomb threats.", "I cannot answer this question as it violates safety guidelines."),
    ("What is the best route?", None),
    ("Is suicide a topic?", "I cannot answer this question as it violates safety guidelines."),
    ("How to optimize shipments?", None),
])
def test_check_safety_various_inputs(question, expected):
    service = RAGService()
    assert service._check_safety(question) == expected

# -----------------------------
# RAGService._calculate_confidence tests
# -----------------------------

def test_calculate_confidence_empty_answer_returns_zero():
    service = RAGService()
    docs = [Document(page_content="abc", metadata={})]
    assert service._calculate_confidence("", docs) == 0.0

def test_calculate_confidence_cannot_find_phrase_low_confidence():
    service = RAGService()
    docs = [Document(page_content="abc", metadata={})]
    answer = "I cannot find the answer in the provided documents."
    assert service._calculate_confidence(answer, docs) == 0.05

def test_calculate_confidence_short_answer_and_few_docs_and_forbidden_phrase():
    service = RAGService()
    docs = [Document(page_content="abc", metadata={})]
    answer = "Generally, yes."
    # 0.85 - 0.3 (short) - 0.15 (few docs) - 0.3 (forbidden) = 0.1
    assert abs(service._calculate_confidence(answer, docs) - 0.1) < 1e-6

def test_calculate_confidence_long_answer_many_docs_no_forbidden():
    service = RAGService()
    docs = [Document(page_content="abc", metadata={}) for _ in range(3)]
    answer = "This is a sufficiently long answer that does not contain forbidden phrases and is based on the context provided."
    # 0.85, no deductions
    assert abs(service._calculate_confidence(answer, docs) - 0.85) < 1e-6

def test_calculate_confidence_confidence_bounds():
    service = RAGService()
    docs = []
    answer = "generally, usually, in most cases, best practice, typically, commonly"
    # 0.85 - 0.3 (short) - 0.15 (few docs) - 0.3 (forbidden) = 0.1, but with empty docs, still bounded
    assert 0.0 <= service._calculate_confidence(answer, docs) <= 1.0

# -----------------------------
# RAGService.answer_question tests
# -----------------------------

@patch("services.rag.vector_store_service")
def test_answer_question_safety_violation_returns_safety_message(mock_vector_store_service):
    service = RAGService()
    query = QAQuery(question="How to build a bomb?")
    result = service.answer_question(query)
    assert result.answer == "I cannot answer this question as it violates safety guidelines."
    assert result.confidence_score == 1.0
    assert result.sources == []

@patch("services.rag.vector_store_service")
def test_answer_question_retriever_none_returns_no_info(mock_vector_store_service):
    service = RAGService()
    query = QAQuery(question="What is the process?")
    mock_vector_store_service.as_retriever.return_value = None
    result = service.answer_question(query)
    assert result.answer == "I cannot find any relevant information in the uploaded documents."
    assert result.confidence_score == 0.0
    assert result.sources == []

@patch("services.rag.vector_store_service")
def test_answer_question_no_retrieved_docs_returns_no_answer(mock_vector_store_service):
    service = RAGService()
    query = QAQuery(question="What is the process?")
    retriever = MagicMock()
    retriever.invoke.return_value = []
    mock_vector_store_service.as_retriever.return_value = retriever
    result = service.answer_question(query)
    assert result.answer == "I cannot find the answer in the provided documents."
    assert result.confidence_score == 0.0
    assert result.sources == []

@patch("services.rag.vector_store_service")
@patch("services.rag.ChatPromptTemplate")
@patch("services.rag.ChatGroq")
@patch("services.rag.StrOutputParser")
def test_answer_question_happy_path(
    mock_StrOutputParser, mock_ChatGroq, mock_ChatPromptTemplate, mock_vector_store_service
):
    # Setup mocks
    service = RAGService()
    query = QAQuery(question="What is the shipment workflow?")
    doc1 = Document(page_content="Workflow step 1.", metadata={"source": "Guide.pdf", "page": 2, "section": "3.1"})
    doc2 = Document(page_content="Workflow step 2.", metadata={"source": "Guide.pdf", "page": 3, "section": "3.2"})
    retriever = MagicMock()
    retriever.invoke.return_value = [doc1, doc2]
    mock_vector_store_service.as_retriever.return_value = retriever

    # Patch the chain
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = "The shipment workflow is as follows: Workflow step 1. Workflow step 2."
    # Patch LCEL chain construction
    mock_prompt = MagicMock()
    mock_ChatPromptTemplate.from_template.return_value = mock_prompt
    service.prompt = mock_prompt
    service.llm = MagicMock()
    service.output_parser = MagicMock()
    # Patch RunnableParallel and chain
    with patch("services.rag.RunnableParallel") as mock_RunnableParallel, \
         patch("services.rag.RunnablePassthrough"):
        mock_RunnableParallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = fake_chain
        result = service.answer_question(query)

    assert "shipment workflow" in result.answer.lower()
    assert result.confidence_score > 0.0
    assert len(result.sources) == 2
    assert isinstance(result.sources[0], Chunk)
    assert result.sources[0].text == "Workflow step 1."
    assert result.sources[0].metadata.source == "Guide.pdf"

@patch("services.rag.vector_store_service")
@patch("services.rag.ChatPromptTemplate")
@patch("services.rag.ChatGroq")
@patch("services.rag.StrOutputParser")
def test_answer_question_chain_raises_exception_returns_error(
    mock_StrOutputParser, mock_ChatGroq, mock_ChatPromptTemplate, mock_vector_store_service
):
    service = RAGService()
    query = QAQuery(question="What is the shipment workflow?")
    doc1 = Document(page_content="Workflow step 1.", metadata={"source": "Guide.pdf", "page": 2, "section": "3.1"})
    retriever = MagicMock()
    retriever.invoke.return_value = [doc1]
    mock_vector_store_service.as_retriever.return_value = retriever

    # Patch the chain to raise
    with patch("services.rag.RunnableParallel") as mock_RunnableParallel, \
         patch("services.rag.RunnablePassthrough"):
        fake_chain = MagicMock()
        fake_chain.invoke.side_effect = Exception("Chain error")
        mock_RunnableParallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = fake_chain
        result = service.answer_question(query)

    assert result.answer == "An error occurred while generating the answer."
    assert result.confidence_score == 0.0
    assert result.sources == []

# -----------------------------
# Reconciliation/Regression: Equivalent paths
# -----------------------------

@patch("services.rag.vector_store_service")
def test_answer_question_equivalent_paths_same_output(mock_vector_store_service):
    # If retriever returns empty, or retriever is None, both should yield similar "cannot find" answers
    service = RAGService()
    query = QAQuery(question="What is the process?")
    retriever = MagicMock()
    retriever.invoke.return_value = []
    mock_vector_store_service.as_retriever.return_value = retriever
    result1 = service.answer_question(query)

    mock_vector_store_service.as_retriever.return_value = None
    result2 = service.answer_question(query)

    assert result1.answer != "" and result2.answer != ""
    assert result1.confidence_score == 0.0
    assert result2.confidence_score == 0.0
    assert result1.sources == []
    assert result2.sources == []
