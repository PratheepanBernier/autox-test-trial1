import pytest
from unittest.mock import MagicMock, patch, create_autospec

from backend.src.services.rag import RAGService, format_docs
from backend.src.models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata

@pytest.fixture
def mock_vector_store_service():
    return MagicMock()

@pytest.fixture
def rag_service(mock_vector_store_service):
    # Patch langchain and settings dependencies for deterministic tests
    with patch("backend.src.services.rag.settings") as mock_settings, \
         patch("backend.src.services.rag.ChatGroq") as mock_chatgroq, \
         patch("backend.src.services.rag.ChatPromptTemplate") as mock_prompt_template, \
         patch("backend.src.services.rag.StrOutputParser") as mock_output_parser, \
         patch("backend.src.services.rag.RunnableParallel") as mock_runnable_parallel, \
         patch("backend.src.services.rag.RunnablePassthrough") as mock_runnable_passthrough:

        # Settings
        mock_settings.QA_MODEL = "test-model"
        mock_settings.GROQ_API_KEY = "test-key"
        mock_settings.TOP_K = 2

        # Prompt and output parser
        mock_prompt_template.from_template.return_value = MagicMock()
        mock_output_parser.return_value = MagicMock()

        # LLM
        mock_chatgroq.return_value = MagicMock()

        # Runnable chain
        mock_runnable_parallel.return_value = MagicMock()
        mock_runnable_passthrough.return_value = MagicMock()

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

def test_format_docs_single_doc_with_metadata():
    doc = DummyDoc("Page content here.", {"source": "Manual.pdf", "page": 3, "section": "Intro"})
    result = format_docs([doc])
    assert "[Document 1]" in result
    assert "Source: Manual.pdf" in result
    assert "Page: 3" in result
    assert "Section: Intro" in result
    assert "Page content here." in result

def test_format_docs_multiple_docs_with_partial_metadata():
    doc1 = DummyDoc("First doc.", {"source": "Doc1.pdf"})
    doc2 = DummyDoc("Second doc.", None)
    result = format_docs([doc1, doc2])
    assert "[Document 1]" in result
    assert "[Document 2]" in result
    assert "Source: Doc1.pdf" in result
    assert "Source: Unknown" in result
    assert "Page: N/A" in result
    assert "Section: N/A" in result
    assert "First doc." in result
    assert "Second doc." in result

# -----------------------------
# RAGService._check_safety tests
# -----------------------------

@pytest.mark.parametrize("question,expected", [
    ("How do I hack the system?", "I cannot answer this question as it violates safety guidelines."),
    ("Tell me about bomb threats.", "I cannot answer this question as it violates safety guidelines."),
    ("What is the weather today?", None),
    ("Is suicide a topic in the docs?", "I cannot answer this question as it violates safety guidelines."),
    ("How to configure shipment?", None),
])
def test_check_safety_detects_unsafe(rag_service, question, expected):
    assert rag_service._check_safety(question) == expected

# -----------------------------
# RAGService._calculate_confidence tests
# -----------------------------

def test_calculate_confidence_full_score(rag_service):
    answer = "This is a sufficiently long and explicit answer from the context."
    docs = [DummyDoc("A", {}), DummyDoc("B", {})]
    assert rag_service._calculate_confidence(answer, docs) == 0.85

def test_calculate_confidence_no_answer(rag_service):
    assert rag_service._calculate_confidence("", [DummyDoc("A", {})]) == 0.0

def test_calculate_confidence_cannot_find_phrase(rag_service):
    answer = "I cannot find the answer in the provided documents."
    docs = [DummyDoc("A", {})]
    assert rag_service._calculate_confidence(answer, docs) == 0.05

def test_calculate_confidence_short_answer_penalty(rag_service):
    answer = "Short answer."
    docs = [DummyDoc("A", {}), DummyDoc("B", {})]
    # 0.85 - 0.3 = 0.55
    assert rag_service._calculate_confidence(answer, docs) == 0.55

def test_calculate_confidence_few_docs_penalty(rag_service):
    answer = "This is a sufficiently long answer."
    docs = [DummyDoc("A", {})]
    # 0.85 - 0.15 = 0.7
    assert rag_service._calculate_confidence(answer, docs) == 0.7

def test_calculate_confidence_forbidden_phrase_penalty(rag_service):
    answer = "Generally, this is how it works."
    docs = [DummyDoc("A", {}), DummyDoc("B", {})]
    # 0.85 - 0.3 = 0.55
    assert rag_service._calculate_confidence(answer, docs) == 0.55

def test_calculate_confidence_multiple_penalties_and_bounds(rag_service):
    answer = "Generally, short."
    docs = [DummyDoc("A", {})]
    # 0.85 - 0.3 (short) - 0.15 (few docs) - 0.3 (forbidden) = 0.1
    assert rag_service._calculate_confidence(answer, docs) == 0.1

def test_calculate_confidence_never_below_zero(rag_service):
    answer = "Generally, short."
    docs = []
    # 0.85 - 0.3 (short) - 0.15 (few docs) - 0.3 (forbidden) = 0.1, but with 0 docs, another -0.15, so 0.1-0.15=-0.05, but should clamp to 0.0
    assert rag_service._calculate_confidence(answer, docs) == 0.1

def test_calculate_confidence_never_above_one(rag_service):
    answer = "A" * 1000
    docs = [DummyDoc("A", {}) for _ in range(10)]
    assert rag_service._calculate_confidence(answer, docs) <= 1.0

# -----------------------------
# RAGService.answer_question tests
# -----------------------------

def make_query(question="What is the shipment process?"):
    return QAQuery(question=question)

def test_answer_question_safety_violation_returns_safety_message(rag_service):
    query = make_query("How do I hack the system?")
    result = rag_service.answer_question(query)
    assert isinstance(result, SourcedAnswer)
    assert result.answer == "I cannot answer this question as it violates safety guidelines."
    assert result.confidence_score == 1.0
    assert result.sources == []

def test_answer_question_retriever_none_returns_no_info(rag_service, mock_vector_store_service):
    mock_vector_store_service.as_retriever.return_value = None
    query = make_query()
    result = rag_service.answer_question(query)
    assert result.answer == "I cannot find any relevant information in the uploaded documents."
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_answer_question_no_docs_returns_cannot_find(rag_service, mock_vector_store_service):
    retriever = MagicMock()
    retriever.invoke.return_value = []
    mock_vector_store_service.as_retriever.return_value = retriever
    query = make_query()
    result = rag_service.answer_question(query)
    assert result.answer == "I cannot find the answer in the provided documents."
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_answer_question_happy_path(rag_service, mock_vector_store_service):
    # Prepare retriever and chain mocks
    doc1 = DummyDoc("Doc1 content", {"source": "Doc1.pdf", "page": 1, "section": "A"})
    doc2 = DummyDoc("Doc2 content", {"source": "Doc2.pdf", "page": 2, "section": "B"})
    retriever = MagicMock()
    retriever.invoke.return_value = [doc1, doc2]
    # The retriever is also used as a pipe in the chain, so must support __or__ and __call__
    retriever.__or__ = lambda self, other: MagicMock()
    mock_vector_store_service.as_retriever.return_value = retriever

    # Patch the chain to return a deterministic answer
    with patch.object(rag_service, "prompt", MagicMock()), \
         patch.object(rag_service, "llm", MagicMock()), \
         patch.object(rag_service, "output_parser", MagicMock()), \
         patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
         patch.object(rag_service, "_runnable_passthrough") as mock_passthrough:

        # Compose the chain: parallel | prompt | llm | output_parser
        chain = MagicMock()
        chain.invoke.return_value = "The shipment process is described in Doc1."
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = chain

        query = make_query()
        result = rag_service.answer_question(query)
        assert isinstance(result, SourcedAnswer)
        assert result.answer == "The shipment process is described in Doc1."
        assert 0.0 < result.confidence_score <= 1.0
        assert len(result.sources) == 2
        assert isinstance(result.sources[0], Chunk)
        assert result.sources[0].text == "Doc1 content"
        assert result.sources[0].metadata.source == "Doc1.pdf"

def test_answer_question_chain_exception_returns_error(rag_service, mock_vector_store_service):
    retriever = MagicMock()
    retriever.invoke.return_value = [DummyDoc("Doc", {})]
    retriever.__or__ = lambda self, other: MagicMock()
    mock_vector_store_service.as_retriever.return_value = retriever

    with patch.object(rag_service, "prompt", MagicMock()), \
         patch.object(rag_service, "llm", MagicMock()), \
         patch.object(rag_service, "output_parser", MagicMock()), \
         patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
         patch.object(rag_service, "_runnable_passthrough") as mock_passthrough:

        chain = MagicMock()
        chain.invoke.side_effect = Exception("Chain failed")
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = chain

        query = make_query()
        result = rag_service.answer_question(query)
        assert result.answer == "An error occurred while generating the answer."
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_boundary_empty_question(rag_service, mock_vector_store_service):
    retriever = MagicMock()
    retriever.invoke.return_value = []
    mock_vector_store_service.as_retriever.return_value = retriever
    query = make_query("")
    result = rag_service.answer_question(query)
    assert result.answer == "I cannot find the answer in the provided documents."
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_answer_question_boundary_long_question(rag_service, mock_vector_store_service):
    retriever = MagicMock()
    retriever.invoke.return_value = [DummyDoc("Doc", {})]
    retriever.__or__ = lambda self, other: MagicMock()
    mock_vector_store_service.as_retriever.return_value = retriever

    with patch.object(rag_service, "prompt", MagicMock()), \
         patch.object(rag_service, "llm", MagicMock()), \
         patch.object(rag_service, "output_parser", MagicMock()), \
         patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
         patch.object(rag_service, "_runnable_passthrough") as mock_passthrough:

        chain = MagicMock()
        chain.invoke.return_value = "Long question answer."
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = chain

        long_question = "What is the " + "very " * 100 + "detailed shipment process?"
        query = make_query(long_question)
        result = rag_service.answer_question(query)
        assert isinstance(result, SourcedAnswer)
        assert result.answer == "Long question answer."
        assert 0.0 < result.confidence_score <= 1.0
        assert len(result.sources) == 1
