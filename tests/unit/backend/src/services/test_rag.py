import pytest
from unittest.mock import patch, MagicMock, create_autospec
from langchain_core.documents import Document

from services.rag import RAGService, format_docs
from models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata

# --------------------------
# format_docs tests
# --------------------------

def test_format_docs_empty_list_returns_empty_string():
    assert format_docs([]) == ""

def test_format_docs_single_document_with_full_metadata():
    doc = Document(
        page_content="This is the content.",
        metadata={"source": "Manual.pdf", "page": 5, "section": "2.1"}
    )
    expected = (
        "[Document 1]\n"
        "Source: Manual.pdf | Page: 5 | Section: 2.1\n"
        "This is the content."
    )
    assert format_docs([doc]) == expected

def test_format_docs_multiple_documents_with_partial_metadata():
    doc1 = Document(
        page_content="Doc1 content.",
        metadata={"source": "A.pdf"}
    )
    doc2 = Document(
        page_content="Doc2 content.",
        metadata={}
    )
    expected = (
        "[Document 1]\n"
        "Source: A.pdf | Page: N/A | Section: N/A\n"
        "Doc1 content.\n\n"
        "[Document 2]\n"
        "Source: Unknown | Page: N/A | Section: N/A\n"
        "Doc2 content."
    )
    assert format_docs([doc1, doc2]) == expected

def test_format_docs_document_with_none_metadata():
    doc = Document(
        page_content="No meta.",
        metadata=None
    )
    expected = (
        "[Document 1]\n"
        "Source: Unknown | Page: N/A | Section: N/A\n"
        "No meta."
    )
    assert format_docs([doc]) == expected

# --------------------------
# RAGService._check_safety tests
# --------------------------

@pytest.mark.parametrize("question,expected", [
    ("How to bomb a system?", "I cannot answer this question as it violates safety guidelines."),
    ("Tell me how to hack into a server.", "I cannot answer this question as it violates safety guidelines."),
    ("What is the shipment workflow?", None),
    ("Is suicide a topic in the docs?", "I cannot answer this question as it violates safety guidelines."),
    ("How to configure the rate card?", None),
])
def test_check_safety_various_inputs(question, expected):
    service = RAGService()
    assert service._check_safety(question) == expected

# --------------------------
# RAGService._calculate_confidence tests
# --------------------------

def make_doc(content="abc", meta=None):
    return Document(page_content=content, metadata=meta or {})

def test_calculate_confidence_empty_answer():
    service = RAGService()
    docs = [make_doc()]
    assert service._calculate_confidence("", docs) == 0.0

def test_calculate_confidence_cannot_find_phrase():
    service = RAGService()
    docs = [make_doc()]
    ans = "I cannot find the answer in the provided documents."
    assert service._calculate_confidence(ans, docs) == 0.05

def test_calculate_confidence_short_answer_penalty():
    service = RAGService()
    docs = [make_doc(), make_doc()]
    ans = "Short answer."
    # 0.85 - 0.3 (short) = 0.55
    assert service._calculate_confidence(ans, docs) == 0.55

def test_calculate_confidence_few_docs_penalty():
    service = RAGService()
    docs = [make_doc()]
    ans = "This is a sufficiently long answer that should not be penalized for length."
    # 0.85 - 0.15 (few docs)
    assert service._calculate_confidence(ans, docs) == 0.7

def test_calculate_confidence_forbidden_phrase_penalty():
    service = RAGService()
    docs = [make_doc(), make_doc()]
    ans = "Generally, the process is as follows and usually works."
    # 0.85 - 0.3 (forbidden)
    assert service._calculate_confidence(ans, docs) == 0.55

def test_calculate_confidence_multiple_penalties_and_bounds():
    service = RAGService()
    docs = []
    ans = "Generally, short."
    # 0.85 - 0.3 (short) - 0.15 (few docs) - 0.3 (forbidden) = 0.1
    assert service._calculate_confidence(ans, docs) == 0.1

def test_calculate_confidence_upper_and_lower_bounds():
    service = RAGService()
    docs = [make_doc(), make_doc()]
    ans = "A" * 2000  # long answer, no forbidden, enough docs
    assert service._calculate_confidence(ans, docs) == 0.85

    ans = "generally"
    docs = []
    # 0.85 - 0.3 (short) - 0.15 (few docs) - 0.3 (forbidden) = 0.1
    assert service._calculate_confidence(ans, docs) == 0.1

    ans = ""
    assert service._calculate_confidence(ans, docs) == 0.0

# --------------------------
# RAGService.answer_question tests
# --------------------------

@pytest.fixture
def qa_query():
    return QAQuery(question="What is the shipment workflow?")

@pytest.fixture
def rag_service():
    # Patch ChatGroq, ChatPromptTemplate, StrOutputParser, etc. to avoid real calls
    with patch("services.rag.ChatGroq"), \
         patch("services.rag.ChatPromptTemplate"), \
         patch("services.rag.StrOutputParser"):
        yield RAGService()

def test_answer_question_safety_violation(rag_service):
    query = QAQuery(question="How to build a bomb?")
    result = rag_service.answer_question(query)
    assert result.answer == "I cannot answer this question as it violates safety guidelines."
    assert result.confidence_score == 1.0
    assert result.sources == []

def test_answer_question_retriever_none(rag_service, qa_query):
    with patch("services.rag.vector_store_service") as mock_vs:
        mock_vs.as_retriever.return_value = None
        result = rag_service.answer_question(qa_query)
        assert result.answer == "I cannot find any relevant information in the uploaded documents."
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_no_retrieved_docs(rag_service, qa_query):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    with patch("services.rag.vector_store_service") as mock_vs:
        mock_vs.as_retriever.return_value = mock_retriever
        result = rag_service.answer_question(qa_query)
        assert result.answer == "I cannot find the answer in the provided documents."
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_happy_path(rag_service, qa_query):
    doc1 = Document(page_content="Shipment workflow is step A, B, C.", metadata={"source": "Manual.pdf", "page": 1, "section": "Intro"})
    doc2 = Document(page_content="Additional info.", metadata={"source": "Guide.pdf", "page": 2, "section": "Details"})
    retrieved_docs = [doc1, doc2]

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = retrieved_docs

    # Patch vector_store_service.as_retriever to return our mock retriever
    with patch("services.rag.vector_store_service") as mock_vs, \
         patch.object(rag_service, "prompt"), \
         patch.object(rag_service, "llm"), \
         patch.object(rag_service, "output_parser"):

        mock_vs.as_retriever.return_value = mock_retriever

        # Patch the RAG chain to return a deterministic answer
        fake_chain = MagicMock()
        fake_chain.invoke.return_value = "Shipment workflow is step A, B, C."
        # Patch the chain construction to return our fake_chain
        with patch("services.rag.RunnableParallel", return_value=fake_chain), \
             patch("services.rag.RunnablePassthrough"):
            result = rag_service.answer_question(qa_query)

        assert result.answer == "Shipment workflow is step A, B, C."
        assert 0.0 < result.confidence_score <= 1.0
        assert len(result.sources) == 2
        assert result.sources[0].text == "Shipment workflow is step A, B, C."
        assert isinstance(result.sources[0].metadata, DocumentMetadata)

def test_answer_question_chain_exception_returns_error(rag_service, qa_query):
    doc = Document(page_content="Some content.", metadata={})
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [doc]

    with patch("services.rag.vector_store_service") as mock_vs, \
         patch.object(rag_service, "prompt"), \
         patch.object(rag_service, "llm"), \
         patch.object(rag_service, "output_parser"):

        mock_vs.as_retriever.return_value = mock_retriever

        # Patch the RAG chain to raise an exception
        fake_chain = MagicMock()
        fake_chain.invoke.side_effect = Exception("Chain error")
        with patch("services.rag.RunnableParallel", return_value=fake_chain), \
             patch("services.rag.RunnablePassthrough"):
            result = rag_service.answer_question(qa_query)

        assert result.answer == "An error occurred while generating the answer."
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_reconciliation_equivalent_paths(rag_service, qa_query):
    # This test checks that two equivalent queries with different doc order yield the same answer and confidence
    doc1 = Document(page_content="Shipment workflow is step A, B, C.", metadata={"source": "Manual.pdf"})
    doc2 = Document(page_content="Additional info.", metadata={"source": "Guide.pdf"})
    docs_a = [doc1, doc2]
    docs_b = [doc2, doc1]

    mock_retriever_a = MagicMock()
    mock_retriever_a.invoke.return_value = docs_a
    mock_retriever_b = MagicMock()
    mock_retriever_b.invoke.return_value = docs_b

    with patch("services.rag.vector_store_service") as mock_vs, \
         patch.object(rag_service, "prompt"), \
         patch.object(rag_service, "llm"), \
         patch.object(rag_service, "output_parser"):

        # Patch chain to return same answer for both
        fake_chain = MagicMock()
        fake_chain.invoke.return_value = "Shipment workflow is step A, B, C."
        with patch("services.rag.RunnableParallel", return_value=fake_chain), \
             patch("services.rag.RunnablePassthrough"):

            mock_vs.as_retriever.return_value = mock_retriever_a
            result_a = rag_service.answer_question(qa_query)

            mock_vs.as_retriever.return_value = mock_retriever_b
            result_b = rag_service.answer_question(qa_query)

    assert result_a.answer == result_b.answer
    assert result_a.confidence_score == result_b.confidence_score
    assert {s.text for s in result_a.sources} == {s.text for s in result_b.sources}
