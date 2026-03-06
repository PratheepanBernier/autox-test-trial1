import pytest
from unittest.mock import MagicMock, patch, create_autospec

from backend.src.services.rag import RAGService, format_docs
from backend.src.models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata

@pytest.fixture
def mock_vector_store_service():
    return MagicMock()

@pytest.fixture
def rag_service(mock_vector_store_service):
    # Patch langchain and groq imports to avoid side effects
    with patch("backend.src.services.rag.ChatGroq") as mock_llm, \
         patch("backend.src.services.rag.ChatPromptTemplate") as mock_prompt, \
         patch("backend.src.services.rag.StrOutputParser") as mock_parser, \
         patch("backend.src.services.rag.RunnableParallel") as mock_parallel, \
         patch("backend.src.services.rag.RunnablePassthrough") as mock_passthrough, \
         patch("backend.src.services.rag.settings") as mock_settings:
        mock_settings.QA_MODEL = "test-model"
        mock_settings.GROQ_API_KEY = "test-key"
        mock_settings.TOP_K = 2
        # Set up prompt and parser to be callable
        mock_prompt.from_template.return_value = MagicMock()
        mock_parser.return_value = MagicMock()
        yield RAGService(mock_vector_store_service)

# -----------------------------
# format_docs tests
# -----------------------------

class DummyDoc:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata

def test_format_docs_empty_returns_empty_string():
    assert format_docs([]) == ""

def test_format_docs_single_doc_with_full_metadata():
    doc = DummyDoc(
        page_content="This is the content.",
        metadata={"source": "Manual.pdf", "page": 5, "section": "Introduction"}
    )
    result = format_docs([doc])
    assert "[Document 1]" in result
    assert "Source: Manual.pdf" in result
    assert "Page: 5" in result
    assert "Section: Introduction" in result
    assert "This is the content." in result

def test_format_docs_multiple_docs_with_partial_metadata():
    doc1 = DummyDoc("Doc1 content", {"source": "Doc1.pdf"})
    doc2 = DummyDoc("Doc2 content", None)
    result = format_docs([doc1, doc2])
    assert "[Document 1]" in result
    assert "[Document 2]" in result
    assert "Source: Doc1.pdf" in result
    assert "Source: Unknown" in result
    assert "Page: N/A" in result
    assert "Section: N/A" in result
    assert "Doc1 content" in result
    assert "Doc2 content" in result

# -----------------------------
# RAGService._check_safety tests
# -----------------------------

@pytest.mark.parametrize("question,expected", [
    ("How to hack the system?", "I cannot answer this question as it violates safety guidelines."),
    ("Tell me about bomb disposal.", "I cannot answer this question as it violates safety guidelines."),
    ("What is the weather today?", None),
    ("Is suicide prevention covered?", "I cannot answer this question as it violates safety guidelines."),
    ("How to optimize shipment?", None),
])
def test_check_safety_detects_unsafe(rag_service, question, expected):
    assert rag_service._check_safety(question) == expected

# -----------------------------
# RAGService._calculate_confidence tests
# -----------------------------

def test_calculate_confidence_answer_empty(rag_service):
    assert rag_service._calculate_confidence("", [object()]) == 0.0

def test_calculate_confidence_cannot_find_phrase(rag_service):
    answer = "I cannot find the answer in the provided documents."
    assert rag_service._calculate_confidence(answer, [object(), object()]) == 0.05

def test_calculate_confidence_short_answer_penalty(rag_service):
    answer = "Short answer."
    docs = [object(), object()]
    conf = rag_service._calculate_confidence(answer, docs)
    assert 0.5 < conf < 0.7  # 0.85 - 0.3 = 0.55

def test_calculate_confidence_few_docs_penalty(rag_service):
    answer = "This is a sufficiently long answer that should not trigger the short answer penalty."
    docs = [object()]
    conf = rag_service._calculate_confidence(answer, docs)
    assert 0.6 < conf < 0.8  # 0.85 - 0.15 = 0.7

def test_calculate_confidence_forbidden_phrase_penalty(rag_service):
    answer = "Generally, the process is as follows."
    docs = [object(), object()]
    conf = rag_service._calculate_confidence(answer, docs)
    assert 0.5 < conf < 0.7  # 0.85 - 0.3 = 0.55

def test_calculate_confidence_multiple_penalties(rag_service):
    answer = "Usually, short."
    docs = [object()]
    conf = rag_service._calculate_confidence(answer, docs)
    # 0.85 - 0.3 (short) - 0.15 (few docs) - 0.3 (forbidden) = 0.1
    assert 0.09 < conf < 0.11

def test_calculate_confidence_bounds(rag_service):
    answer = "generally, short."
    docs = []
    conf = rag_service._calculate_confidence(answer, docs)
    assert conf == 0.0  # Should not go below 0.0

    answer = "A" * 1000
    docs = [object()] * 10
    conf = rag_service._calculate_confidence(answer, docs)
    assert conf == 0.85  # Should not exceed 0.85

# -----------------------------
# RAGService.answer_question tests
# -----------------------------

def make_mock_doc(content, meta=None):
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = meta
    return doc

def test_answer_question_safety_violation(rag_service):
    query = QAQuery(question="How to make a bomb?")
    result = rag_service.answer_question(query)
    assert result.answer == "I cannot answer this question as it violates safety guidelines."
    assert result.confidence_score == 1.0
    assert result.sources == []

def test_answer_question_retriever_none(rag_service, mock_vector_store_service):
    query = QAQuery(question="What is the process?")
    mock_vector_store_service.as_retriever.return_value = None
    result = rag_service.answer_question(query)
    assert result.answer == "I cannot find any relevant information in the uploaded documents."
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_answer_question_no_docs_found(rag_service, mock_vector_store_service):
    query = QAQuery(question="What is the process?")
    retriever = MagicMock()
    retriever.invoke.return_value = []
    mock_vector_store_service.as_retriever.return_value = retriever
    result = rag_service.answer_question(query)
    assert result.answer == "I cannot find the answer in the provided documents."
    assert result.confidence_score == 0.0
    assert result.sources == []

@patch("backend.src.services.rag.format_docs")
def test_answer_question_happy_path(mock_format_docs, rag_service, mock_vector_store_service):
    query = QAQuery(question="What is the shipment workflow?")
    doc1 = make_mock_doc("Content 1", {"source": "Doc1.pdf", "page": 1})
    doc2 = make_mock_doc("Content 2", {"source": "Doc2.pdf", "page": 2})
    retrieved_docs = [doc1, doc2]
    retriever = MagicMock()
    retriever.invoke.return_value = retrieved_docs
    # The retriever is also used as a pipe in the chain, so mock __or__ to return format_docs
    retriever.__or__.return_value = mock_format_docs
    mock_vector_store_service.as_retriever.return_value = retriever

    # Patch the RAG chain
    rag_chain = MagicMock()
    rag_chain.invoke.return_value = "The shipment workflow is described in the documents."
    with patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
         patch.object(rag_service, "_runnable_passthrough") as mock_passthrough, \
         patch.object(rag_service, "prompt"), \
         patch.object(rag_service, "llm"), \
         patch.object(rag_service, "output_parser"):
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = rag_chain
        result = rag_service.answer_question(query)

    assert result.answer == "The shipment workflow is described in the documents."
    assert 0.7 < result.confidence_score <= 1.0
    assert len(result.sources) == 2
    assert isinstance(result.sources[0], Chunk)
    assert result.sources[0].text == "Content 1"
    assert result.sources[1].text == "Content 2"

def test_answer_question_chain_exception(rag_service, mock_vector_store_service):
    query = QAQuery(question="What is the process?")
    retriever = MagicMock()
    retriever.invoke.return_value = [make_mock_doc("Content", {"source": "Doc.pdf"})]
    mock_vector_store_service.as_retriever.return_value = retriever

    # Patch the RAG chain to raise
    with patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
         patch.object(rag_service, "_runnable_passthrough") as mock_passthrough, \
         patch.object(rag_service, "prompt"), \
         patch.object(rag_service, "llm"), \
         patch.object(rag_service, "output_parser"):
        chain = MagicMock()
        chain.invoke.side_effect = Exception("Chain error")
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = chain
        result = rag_service.answer_question(query)

    assert result.answer == "An error occurred while generating the answer."
    assert result.confidence_score == 0.0
    assert result.sources == []

# -----------------------------
# Reconciliation/Regression: Equivalent paths
# -----------------------------

def test_answer_question_equivalent_paths_same_output(rag_service, mock_vector_store_service):
    """
    If two queries with different casing or whitespace yield the same retriever and chain output,
    the SourcedAnswer should be equivalent (regression/reconciliation intent).
    """
    doc = make_mock_doc("Content", {"source": "Doc.pdf"})
    retriever = MagicMock()
    retriever.invoke.return_value = [doc]
    retriever.__or__.return_value = format_docs
    mock_vector_store_service.as_retriever.return_value = retriever

    rag_chain = MagicMock()
    rag_chain.invoke.return_value = "The answer is in the document."
    with patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
         patch.object(rag_service, "_runnable_passthrough") as mock_passthrough, \
         patch.object(rag_service, "prompt"), \
         patch.object(rag_service, "llm"), \
         patch.object(rag_service, "output_parser"):
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = rag_chain
        query1 = QAQuery(question="What is the answer?")
        query2 = QAQuery(question="  what is the answer?  ")
        result1 = rag_service.answer_question(query1)
        result2 = rag_service.answer_question(query2)

    assert result1.answer == result2.answer
    assert result1.confidence_score == result2.confidence_score
    assert [c.text for c in result1.sources] == [c.text for c in result2.sources]
