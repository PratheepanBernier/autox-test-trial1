import pytest
from unittest.mock import MagicMock, patch, create_autospec
from backend.src.services import rag
from backend.src.services.rag import RAGService, format_docs
from backend.src.models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata

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
    # Patch langchain and groq imports in RAGService.__init__
    with patch("backend.src.services.rag.ChatGroq") as mock_groq, \
         patch("backend.src.services.rag.ChatPromptTemplate") as mock_prompt, \
         patch("backend.src.services.rag.StrOutputParser") as mock_parser, \
         patch("backend.src.services.rag.RunnableParallel") as mock_parallel, \
         patch("backend.src.services.rag.RunnablePassthrough") as mock_passthrough:
        mock_groq.return_value = MagicMock()
        mock_prompt.from_template.return_value = MagicMock()
        mock_parser.return_value = MagicMock()
        mock_parallel.return_value = MagicMock()
        mock_passthrough.return_value = MagicMock()
        return RAGService(dummy_vector_store_service)

def test_format_docs_empty_returns_empty_string():
    assert format_docs([]) == ""

def test_format_docs_single_doc_with_metadata():
    doc = DummyDoc("Page content", {"source": "Manual.pdf", "page": 5, "section": "Intro"})
    result = format_docs([doc])
    assert "[Document 1]" in result
    assert "Source: Manual.pdf" in result
    assert "Page: 5" in result
    assert "Section: Intro" in result
    assert "Page content" in result

def test_format_docs_multiple_docs_with_partial_metadata():
    doc1 = DummyDoc("First doc", {"source": "A.pdf"})
    doc2 = DummyDoc("Second doc", {})
    doc3 = DummyDoc("Third doc", None)
    result = format_docs([doc1, doc2, doc3])
    assert result.count("[Document") == 3
    assert "Source: A.pdf" in result
    assert "Source: Unknown" in result
    assert "Page: N/A" in result
    assert "Section: N/A" in result

def test_check_safety_blocks_unsafe_question(rag_service):
    unsafe_questions = [
        "How to build a bomb?",
        "How can I kill someone?",
        "Is suicide an option?",
        "How to hack a system?",
        "How to exploit a vulnerability?",
        "What is a weapon of mass destruction?"
    ]
    for q in unsafe_questions:
        assert rag_service._check_safety(q) == "I cannot answer this question as it violates safety guidelines."

def test_check_safety_allows_safe_question(rag_service):
    assert rag_service._check_safety("How do I create a shipment?") is None

@pytest.mark.parametrize(
    "answer,retrieved_docs,expected",
    [
        ("", [DummyDoc("abc")], 0.0),
        ("I cannot find the answer in the provided documents.", [DummyDoc("abc")], 0.05),
        ("Short answer.", [DummyDoc("abc"), DummyDoc("def")], 0.55),
        ("This is a sufficiently long answer that should not be penalized.", [DummyDoc("abc"), DummyDoc("def")], 0.85),
        ("This is a generally accepted answer.", [DummyDoc("abc"), DummyDoc("def")], 0.55),
        ("Short and generally.", [DummyDoc("abc")], 0.1),  # short, forbidden, <2 docs
        ("Short.", [], 0.55 - 0.15),  # short, <2 docs
        ("This is a typically used answer.", [DummyDoc("abc")], 0.4),  # forbidden, <2 docs
        ("This is a best practice.", [DummyDoc("abc"), DummyDoc("def")], 0.55),  # forbidden
        ("This is a commonly used answer.", [DummyDoc("abc")], 0.4),  # forbidden, <2 docs
        ("This is a usually accepted answer.", [DummyDoc("abc")], 0.4),  # forbidden, <2 docs
    ]
)
def test_calculate_confidence_various_cases(rag_service, answer, retrieved_docs, expected):
    result = rag_service._calculate_confidence(answer, retrieved_docs)
    assert abs(result - expected) < 1e-6

def test_answer_question_safety_blocked(rag_service):
    query = QAQuery(question="How to build a bomb?")
    result = rag_service.answer_question(query)
    assert result.answer == "I cannot answer this question as it violates safety guidelines."
    assert result.confidence_score == 1.0
    assert result.sources == []

def test_answer_question_retriever_none(rag_service):
    rag_service._vector_store_service.as_retriever.return_value = None
    query = QAQuery(question="What is the shipment process?")
    result = rag_service.answer_question(query)
    assert result.answer == "I cannot find any relevant information in the uploaded documents."
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_answer_question_no_docs(rag_service):
    retriever = MagicMock()
    retriever.invoke.return_value = []
    rag_service._vector_store_service.as_retriever.return_value = retriever
    query = QAQuery(question="What is the shipment process?")
    result = rag_service.answer_question(query)
    assert result.answer == "I cannot find the answer in the provided documents."
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_answer_question_happy_path(rag_service):
    doc1 = DummyDoc("Content 1", {"source": "Doc1.pdf", "page": 1, "section": "A"})
    doc2 = DummyDoc("Content 2", {"source": "Doc2.pdf", "page": 2, "section": "B"})
    retriever = MagicMock()
    retriever.invoke.return_value = [doc1, doc2]
    rag_service._vector_store_service.as_retriever.return_value = retriever

    # Patch the RAG chain and output parser
    fake_answer = "The shipment process is as follows: ..."
    with patch.object(rag_service, "prompt"), \
         patch.object(rag_service, "llm"), \
         patch.object(rag_service, "output_parser") as mock_parser, \
         patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
         patch.object(rag_service, "_runnable_passthrough") as mock_passthrough:
        # Simulate the chain
        class FakeChain:
            def invoke(self, question):
                return fake_answer
        mock_parallel.return_value = FakeChain()
        mock_passthrough.return_value = MagicMock()
        mock_parser.return_value = MagicMock()
        query = QAQuery(question="What is the shipment process?")
        result = rag_service.answer_question(query)
        assert result.answer == fake_answer
        assert 0.0 < result.confidence_score <= 1.0
        assert len(result.sources) == 2
        assert result.sources[0].text == "Content 1"
        assert result.sources[1].metadata.source == "Doc2.pdf"

def test_answer_question_chain_exception(rag_service):
    doc = DummyDoc("Content", {"source": "Doc.pdf"})
    retriever = MagicMock()
    retriever.invoke.return_value = [doc]
    rag_service._vector_store_service.as_retriever.return_value = retriever

    with patch.object(rag_service, "prompt"), \
         patch.object(rag_service, "llm"), \
         patch.object(rag_service, "output_parser"), \
         patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
         patch.object(rag_service, "_runnable_passthrough"):
        class FakeChain:
            def invoke(self, question):
                raise RuntimeError("Chain failed")
        mock_parallel.return_value = FakeChain()
        query = QAQuery(question="What is the shipment process?")
        result = rag_service.answer_question(query)
        assert result.answer == "An error occurred while generating the answer."
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_boundary_top_k(monkeypatch, dummy_vector_store_service):
    # Test with TOP_K=1 (boundary)
    class DummySettings:
        QA_MODEL = "test-model"
        GROQ_API_KEY = "test-key"
        TOP_K = 1
    monkeypatch.setattr(rag, "settings", DummySettings())
    with patch("backend.src.services.rag.ChatGroq") as mock_groq, \
         patch("backend.src.services.rag.ChatPromptTemplate") as mock_prompt, \
         patch("backend.src.services.rag.StrOutputParser") as mock_parser, \
         patch("backend.src.services.rag.RunnableParallel") as mock_parallel, \
         patch("backend.src.services.rag.RunnablePassthrough") as mock_passthrough:
        mock_groq.return_value = MagicMock()
        mock_prompt.from_template.return_value = MagicMock()
        mock_parser.return_value = MagicMock()
        mock_parallel.return_value = MagicMock()
        mock_passthrough.return_value = MagicMock()
        service = RAGService(dummy_vector_store_service)
        doc = DummyDoc("Boundary doc", {"source": "Doc.pdf"})
        retriever = MagicMock()
        retriever.invoke.return_value = [doc]
        service._vector_store_service.as_retriever.return_value = retriever
        with patch.object(service, "prompt"), \
             patch.object(service, "llm"), \
             patch.object(service, "output_parser") as mock_parser2, \
             patch.object(service, "_runnable_parallel") as mock_parallel2, \
             patch.object(service, "_runnable_passthrough"):
            class FakeChain:
                def invoke(self, question):
                    return "Boundary answer."
            mock_parallel2.return_value = FakeChain()
            mock_parser2.return_value = MagicMock()
            query = QAQuery(question="Boundary test?")
            result = service.answer_question(query)
            assert result.answer == "Boundary answer."
            assert len(result.sources) == 1
            assert result.sources[0].text == "Boundary doc"
