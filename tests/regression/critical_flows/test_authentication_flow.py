import pytest
from unittest.mock import MagicMock, patch, create_autospec
from backend.src.services.rag import RAGService, format_docs
from backend.src.models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata

@pytest.fixture
def mock_settings(monkeypatch):
    class DummySettings:
        QA_MODEL = "test-model"
        GROQ_API_KEY = "test-key"
        TOP_K = 2
    monkeypatch.setattr("backend.src.services.rag.settings", DummySettings())

@pytest.fixture
def mock_vector_store_service():
    svc = MagicMock()
    return svc

@pytest.fixture
def rag_service(mock_vector_store_service, mock_settings):
    # Patch langchain imports to avoid real initialization
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
        return RAGService(mock_vector_store_service)

class DummyDoc:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

def make_query(question):
    return QAQuery(question=question)

def make_docs(n):
    return [DummyDoc(f"Content {i}", {"source": f"src{i}", "page": i, "section": f"sec{i}"}) for i in range(1, n+1)]

def test_format_docs_empty():
    assert format_docs([]) == ""

def test_format_docs_basic():
    docs = [DummyDoc("Hello", {"source": "A", "page": 1, "section": "Intro"})]
    result = format_docs(docs)
    assert "[Document 1]" in result
    assert "Source: A" in result
    assert "Page: 1" in result
    assert "Section: Intro" in result
    assert "Hello" in result

def test_format_docs_missing_metadata():
    docs = [DummyDoc("Hi", None)]
    result = format_docs(docs)
    assert "Unknown" in result
    assert "N/A" in result

def test_check_safety_blocks_unsafe(rag_service):
    unsafe = ["bomb", "kill", "suicide", "hack", "exploit", "weapon"]
    for word in unsafe:
        assert rag_service._check_safety(f"How to {word}?") is not None

def test_check_safety_allows_safe(rag_service):
    assert rag_service._check_safety("How to ship a package?") is None

def test_calculate_confidence_answer_not_found(rag_service):
    docs = make_docs(2)
    conf = rag_service._calculate_confidence("I cannot find the answer in the provided documents.", docs)
    assert conf == 0.05

def test_calculate_confidence_short_answer(rag_service):
    docs = make_docs(2)
    conf = rag_service._calculate_confidence("Short answer.", docs)
    assert 0.5 < conf < 0.85

def test_calculate_confidence_low_docs(rag_service):
    docs = make_docs(1)
    conf = rag_service._calculate_confidence("This is a sufficiently long answer.", docs)
    assert 0.6 < conf < 0.85

def test_calculate_confidence_forbidden_phrase(rag_service):
    docs = make_docs(2)
    conf = rag_service._calculate_confidence("Generally, you should...", docs)
    assert conf < 0.85

def test_calculate_confidence_empty(rag_service):
    docs = make_docs(2)
    assert rag_service._calculate_confidence("", docs) == 0.0

def test_answer_question_safety_blocked(rag_service):
    query = make_query("How to make a bomb?")
    result = rag_service.answer_question(query)
    assert "violates safety guidelines" in result.answer
    assert result.confidence_score == 1.0
    assert result.sources == []

def test_answer_question_no_retriever(rag_service, mock_vector_store_service):
    mock_vector_store_service.as_retriever.return_value = None
    query = make_query("How to ship a package?")
    result = rag_service.answer_question(query)
    assert "cannot find any relevant information" in result.answer
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_answer_question_no_docs(rag_service, mock_vector_store_service):
    retriever = MagicMock()
    retriever.invoke.return_value = []
    mock_vector_store_service.as_retriever.return_value = retriever
    query = make_query("How to ship a package?")
    result = rag_service.answer_question(query)
    assert "cannot find the answer" in result.answer
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_answer_question_happy_path(rag_service, mock_vector_store_service):
    docs = make_docs(2)
    retriever = MagicMock()
    retriever.invoke.return_value = docs
    # Patch the chain to return a deterministic answer
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = "The answer is in the documents."
    # Patch the RAG chain construction
    with patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
         patch.object(rag_service, "_runnable_passthrough") as mock_passthrough:
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = fake_chain
        mock_vector_store_service.as_retriever.return_value = retriever
        query = make_query("How to ship a package?")
        result = rag_service.answer_question(query)
        assert result.answer == "The answer is in the documents."
        assert 0.0 < result.confidence_score <= 1.0
        assert len(result.sources) == 2
        assert isinstance(result.sources[0], Chunk)
        assert result.sources[0].text == docs[0].page_content

def test_answer_question_chain_exception(rag_service, mock_vector_store_service):
    docs = make_docs(2)
    retriever = MagicMock()
    retriever.invoke.return_value = docs
    # Patch the chain to raise
    fake_chain = MagicMock()
    fake_chain.invoke.side_effect = RuntimeError("fail")
    with patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
         patch.object(rag_service, "_runnable_passthrough") as mock_passthrough:
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = fake_chain
        mock_vector_store_service.as_retriever.return_value = retriever
        query = make_query("How to ship a package?")
        result = rag_service.answer_question(query)
        assert "error occurred" in result.answer
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_forbidden_phrase_penalty(rag_service, mock_vector_store_service):
    docs = make_docs(2)
    retriever = MagicMock()
    retriever.invoke.return_value = docs
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = "Generally, you should..."
    with patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
         patch.object(rag_service, "_runnable_passthrough") as mock_passthrough:
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = fake_chain
        mock_vector_store_service.as_retriever.return_value = retriever
        query = make_query("How to ship a package?")
        result = rag_service.answer_question(query)
        assert "Generally" in result.answer
        assert result.confidence_score < 0.85

def test_answer_question_minimum_confidence_bounds(rag_service, mock_vector_store_service):
    docs = make_docs(2)
    retriever = MagicMock()
    retriever.invoke.return_value = docs
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = ""
    with patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
         patch.object(rag_service, "_runnable_passthrough") as mock_passthrough:
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = fake_chain
        mock_vector_store_service.as_retriever.return_value = retriever
        query = make_query("How to ship a package?")
        result = rag_service.answer_question(query)
        assert result.confidence_score == 0.0

def test_answer_question_maximum_confidence_bounds(rag_service, mock_vector_store_service):
    docs = make_docs(3)
    retriever = MagicMock()
    retriever.invoke.return_value = docs
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = "This is a long and detailed answer that should yield high confidence."
    with patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
         patch.object(rag_service, "_runnable_passthrough") as mock_passthrough:
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = fake_chain
        mock_vector_store_service.as_retriever.return_value = retriever
        query = make_query("How to ship a package?")
        result = rag_service.answer_question(query)
        assert 0.8 <= result.confidence_score <= 1.0
