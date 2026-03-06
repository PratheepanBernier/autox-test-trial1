import pytest
from unittest.mock import MagicMock, patch, create_autospec

from backend.src.services import rag
from backend.src.services.rag import RAGService, format_docs
from backend.src.models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata

# -----------------------------
# Fixtures and Mocks
# -----------------------------

class DummyDoc:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

@pytest.fixture
def dummy_vector_store_service():
    return MagicMock(spec=["as_retriever"])

@pytest.fixture
def dummy_settings(monkeypatch):
    class DummySettings:
        QA_MODEL = "test-model"
        GROQ_API_KEY = "test-key"
        TOP_K = 2
    monkeypatch.setattr(rag, "settings", DummySettings())

@pytest.fixture
def rag_service(dummy_vector_store_service, dummy_settings):
    # Patch langchain and groq imports to avoid runtime dependencies
    with patch("backend.src.services.rag.ChatGroq") as mock_llm, \
         patch("backend.src.services.rag.ChatPromptTemplate") as mock_prompt, \
         patch("backend.src.services.rag.StrOutputParser") as mock_parser, \
         patch("backend.src.services.rag.RunnableParallel") as mock_parallel, \
         patch("backend.src.services.rag.RunnablePassthrough") as mock_passthrough:
        mock_llm.return_value = MagicMock()
        mock_prompt.from_template.return_value = MagicMock()
        mock_parser.return_value = MagicMock()
        mock_parallel.return_value = MagicMock()
        mock_passthrough.return_value = MagicMock()
        yield RAGService(dummy_vector_store_service)

# -----------------------------
# format_docs tests
# -----------------------------

def test_format_docs_empty_returns_empty_string():
    assert format_docs([]) == ""

def test_format_docs_single_doc_with_metadata():
    doc = DummyDoc("Page content here.", {"source": "Manual.pdf", "page": 5, "section": "Intro"})
    result = format_docs([doc])
    assert "[Document 1]" in result
    assert "Source: Manual.pdf" in result
    assert "Page: 5" in result
    assert "Section: Intro" in result
    assert "Page content here." in result

def test_format_docs_multiple_docs_with_partial_metadata():
    doc1 = DummyDoc("First doc.", {"source": "Doc1.pdf"})
    doc2 = DummyDoc("Second doc.", {})
    result = format_docs([doc1, doc2])
    assert "[Document 1]" in result
    assert "[Document 2]" in result
    assert "Source: Doc1.pdf" in result
    assert "Source: Unknown" in result
    assert "Page: N/A" in result
    assert "Section: N/A" in result

# -----------------------------
# _check_safety tests
# -----------------------------

@pytest.mark.parametrize("question,expected", [
    ("How to hack the system?", "I cannot answer this question as it violates safety guidelines."),
    ("Tell me about bomb disposal.", "I cannot answer this question as it violates safety guidelines."),
    ("What is the weather today?", None),
    ("Is suicide a topic in the docs?", "I cannot answer this question as it violates safety guidelines."),
    ("How to optimize shipment?", None),
])
def test_check_safety_detects_unsafe(rag_service, question, expected):
    assert rag_service._check_safety(question) == expected

# -----------------------------
# _calculate_confidence tests
# -----------------------------

def test_calculate_confidence_happy_path(rag_service):
    answer = "This is a sufficiently long answer present in the documents."
    docs = [DummyDoc("A", {}), DummyDoc("B", {})]
    conf = rag_service._calculate_confidence(answer, docs)
    assert 0.8 < conf <= 1.0

def test_calculate_confidence_no_answer(rag_service):
    conf = rag_service._calculate_confidence("", [DummyDoc("A", {})])
    assert conf == 0.0

def test_calculate_confidence_cannot_find_phrase(rag_service):
    conf = rag_service._calculate_confidence("I cannot find the answer in the provided documents.", [DummyDoc("A", {})])
    assert conf == 0.05

def test_calculate_confidence_short_answer_penalty(rag_service):
    conf = rag_service._calculate_confidence("Short answer.", [DummyDoc("A", {}), DummyDoc("B", {})])
    # 0.85 - 0.3 = 0.55
    assert 0.5 < conf < 0.6

def test_calculate_confidence_few_docs_penalty(rag_service):
    conf = rag_service._calculate_confidence("This is a sufficiently long answer.", [DummyDoc("A", {})])
    # 0.85 - 0.15 = 0.7
    assert 0.6 < conf < 0.8

def test_calculate_confidence_forbidden_phrase_penalty(rag_service):
    answer = "Generally, the process is as follows."
    conf = rag_service._calculate_confidence(answer, [DummyDoc("A", {}), DummyDoc("B", {})])
    # 0.85 - 0.3 = 0.55
    assert 0.5 < conf < 0.6

def test_calculate_confidence_multiple_penalties(rag_service):
    answer = "Usually short."
    conf = rag_service._calculate_confidence(answer, [DummyDoc("A", {})])
    # 0.85 - 0.3 (short) - 0.15 (few docs) - 0.3 (forbidden) = 0.1
    assert 0.09 < conf < 0.11

def test_calculate_confidence_never_below_zero(rag_service):
    answer = "Generally short."
    conf = rag_service._calculate_confidence(answer, [])
    assert conf == 0.0

def test_calculate_confidence_never_above_one(rag_service):
    answer = "This is a very long answer." * 100
    conf = rag_service._calculate_confidence(answer, [DummyDoc("A", {}), DummyDoc("B", {})])
    assert 0.0 <= conf <= 1.0

# -----------------------------
# answer_question tests
# -----------------------------

def make_query(question="What is the shipment process?"):
    return QAQuery(question=question)

def test_answer_question_safety_violation(rag_service):
    query = make_query("How to build a bomb?")
    result = rag_service.answer_question(query)
    assert result.answer == "I cannot answer this question as it violates safety guidelines."
    assert result.confidence_score == 1.0
    assert result.sources == []

def test_answer_question_retriever_none(rag_service):
    rag_service._vector_store_service.as_retriever.return_value = None
    query = make_query()
    result = rag_service.answer_question(query)
    assert result.answer == "I cannot find any relevant information in the uploaded documents."
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_answer_question_no_docs(rag_service):
    retriever = MagicMock()
    retriever.invoke.return_value = []
    rag_service._vector_store_service.as_retriever.return_value = retriever
    query = make_query()
    result = rag_service.answer_question(query)
    assert result.answer == "I cannot find the answer in the provided documents."
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_answer_question_happy_path(rag_service):
    # Setup retriever to return docs
    doc1 = DummyDoc("Shipment process is as follows.", {"source": "SOP.pdf", "page": 1, "section": "Process"})
    doc2 = DummyDoc("Additional info.", {"source": "Manual.pdf", "page": 2, "section": "Details"})
    retriever = MagicMock()
    retriever.invoke.return_value = [doc1, doc2]
    # Patch the chain to return a deterministic answer
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = "Shipment process is as follows."
    # Patch the RAG chain construction
    with patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
         patch.object(rag_service, "_runnable_passthrough") as mock_passthrough, \
         patch.object(rag_service, "prompt"), \
         patch.object(rag_service, "llm"), \
         patch.object(rag_service, "output_parser"):
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = fake_chain
        rag_service._vector_store_service.as_retriever.return_value = retriever
        query = make_query()
        result = rag_service.answer_question(query)
        assert result.answer == "Shipment process is as follows."
        assert 0.8 < result.confidence_score <= 1.0
        assert len(result.sources) == 2
        assert result.sources[0].text == "Shipment process is as follows."
        assert result.sources[0].metadata.source == "SOP.pdf"

def test_answer_question_chain_exception(rag_service):
    doc = DummyDoc("Some content.", {"source": "Doc.pdf"})
    retriever = MagicMock()
    retriever.invoke.return_value = [doc]
    # Patch the chain to raise
    fake_chain = MagicMock()
    fake_chain.invoke.side_effect = Exception("Chain error")
    with patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
         patch.object(rag_service, "_runnable_passthrough") as mock_passthrough, \
         patch.object(rag_service, "prompt"), \
         patch.object(rag_service, "llm"), \
         patch.object(rag_service, "output_parser"):
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = fake_chain
        rag_service._vector_store_service.as_retriever.return_value = retriever
        query = make_query()
        result = rag_service.answer_question(query)
        assert result.answer == "An error occurred while generating the answer."
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_boundary_top_k(monkeypatch, dummy_vector_store_service):
    # Test with TOP_K = 1 (boundary)
    class DummySettings:
        QA_MODEL = "test-model"
        GROQ_API_KEY = "test-key"
        TOP_K = 1
    monkeypatch.setattr(rag, "settings", DummySettings())
    with patch("backend.src.services.rag.ChatGroq"), \
         patch("backend.src.services.rag.ChatPromptTemplate"), \
         patch("backend.src.services.rag.StrOutputParser"), \
         patch("backend.src.services.rag.RunnableParallel"), \
         patch("backend.src.services.rag.RunnablePassthrough"):
        service = RAGService(dummy_vector_store_service)
    retriever = MagicMock()
    retriever.invoke.return_value = [DummyDoc("Boundary doc.", {"source": "Doc.pdf"})]
    dummy_vector_store_service.as_retriever.return_value = retriever
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = "Boundary doc."
    with patch.object(service, "_runnable_parallel") as mock_parallel, \
         patch.object(service, "_runnable_passthrough"), \
         patch.object(service, "prompt"), \
         patch.object(service, "llm"), \
         patch.object(service, "output_parser"):
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = fake_chain
        query = QAQuery(question="Boundary test?")
        result = service.answer_question(query)
        assert result.answer == "Boundary doc."
        assert result.sources[0].text == "Boundary doc."
        assert result.sources[0].metadata.source == "Doc.pdf"
        assert 0.6 < result.confidence_score < 0.8
