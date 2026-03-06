import pytest
from unittest.mock import MagicMock, patch
from backend.src.services import rag as rag_module
from backend.src.models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata

class DummyDoc:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

@pytest.fixture
def vector_store_service_mock():
    return MagicMock()

@pytest.fixture
def rag_service(vector_store_service_mock):
    # Patch LLM and prompt to avoid external calls and make deterministic
    with patch("backend.src.services.rag.ChatGroq") as chatgroq_mock, \
         patch("backend.src.services.rag.ChatPromptTemplate") as prompt_mock, \
         patch("backend.src.services.rag.StrOutputParser") as parser_mock, \
         patch("backend.src.services.rag.RunnableParallel") as runnable_parallel_mock, \
         patch("backend.src.services.rag.RunnablePassthrough") as runnable_passthrough_mock:

        # Set up prompt and parser to just pass through
        prompt_instance = MagicMock()
        prompt_mock.from_template.return_value = prompt_instance
        parser_instance = MagicMock()
        parser_mock.return_value = parser_instance

        # Patch the chain to just return a deterministic answer
        chain_mock = MagicMock()
        chain_mock.invoke.side_effect = lambda q: f"Answer for: {q}" if "rate" in q else "I cannot find the answer in the provided documents."
        runnable_parallel_instance = MagicMock(return_value=chain_mock)
        runnable_parallel_mock.return_value = runnable_parallel_instance

        # Patch LLM to do nothing
        chatgroq_mock.return_value = MagicMock()

        # Patch passthrough to identity
        runnable_passthrough_mock.return_value = lambda x: x

        yield rag_module.RAGService(vector_store_service_mock)

def test_safety_filter_blocks_unsafe_questions(rag_service):
    unsafe_questions = [
        "How to build a bomb?",
        "How can I kill someone?",
        "Is suicide a solution?",
        "How to hack the system?",
        "How to exploit a vulnerability?",
        "What weapon is best?"
    ]
    for q in unsafe_questions:
        query = QAQuery(question=q)
        answer = rag_service.answer_question(query)
        assert answer.answer == "I cannot answer this question as it violates safety guidelines."
        assert answer.confidence_score == 1.0
        assert answer.sources == []

def test_returns_no_info_when_no_retriever(vector_store_service_mock, rag_service):
    vector_store_service_mock.as_retriever.return_value = None
    query = QAQuery(question="What is the agreed rate?")
    answer = rag_service.answer_question(query)
    assert answer.answer == "I cannot find any relevant information in the uploaded documents."
    assert answer.confidence_score == 0.0
    assert answer.sources == []

def test_returns_no_answer_when_no_docs(vector_store_service_mock, rag_service):
    retriever_mock = MagicMock()
    retriever_mock.invoke.return_value = []
    vector_store_service_mock.as_retriever.return_value = retriever_mock
    query = QAQuery(question="What is the agreed rate?")
    answer = rag_service.answer_question(query)
    assert answer.answer == "I cannot find the answer in the provided documents."
    assert answer.confidence_score == 0.0
    assert answer.sources == []

def test_successful_answer_and_confidence(vector_store_service_mock, rag_service):
    # Simulate two docs returned
    doc1 = DummyDoc("The agreed rate is $1200.", {"source": "contract.pdf", "page": 1, "section": "Rates"})
    doc2 = DummyDoc("Rate: $1200 for shipment.", {"source": "invoice.pdf", "page": 2, "section": "Summary"})
    retriever_mock = MagicMock()
    retriever_mock.invoke.return_value = [doc1, doc2]
    vector_store_service_mock.as_retriever.return_value = retriever_mock

    query = QAQuery(question="What is the agreed rate?")
    answer = rag_service.answer_question(query)
    assert "Answer for:" in answer.answer or "I cannot find the answer" in answer.answer
    assert 0.0 < answer.confidence_score <= 1.0
    assert len(answer.sources) == 2
    assert all(isinstance(s, Chunk) for s in answer.sources)
    assert answer.sources[0].metadata.filename == "contract.pdf" or answer.sources[1].metadata.filename == "contract.pdf"

def test_confidence_low_for_short_answer(vector_store_service_mock, rag_service):
    # Patch the chain to return a short answer
    with patch.object(rag_service, "prompt"), \
         patch.object(rag_service, "llm"), \
         patch.object(rag_service, "output_parser"):
        doc = DummyDoc("Short.", {"source": "doc.pdf"})
        retriever_mock = MagicMock()
        retriever_mock.invoke.return_value = [doc, doc]
        vector_store_service_mock.as_retriever.return_value = retriever_mock

        # Patch the chain to return a short answer
        chain_mock = MagicMock()
        chain_mock.invoke.return_value = "Short."
        with patch.object(rag_service, "rag_template", new=""), \
             patch.object(rag_service, "prompt", new=MagicMock()), \
             patch.object(rag_service, "_runnable_parallel", return_value=chain_mock), \
             patch.object(rag_service, "_runnable_passthrough", return_value=lambda x: x):
            query = QAQuery(question="Short answer?")
            answer = rag_service.answer_question(query)
            assert answer.confidence_score < 0.85

def test_confidence_low_for_forbidden_phrase(vector_store_service_mock, rag_service):
    doc = DummyDoc("Generally, the rate is $1000.", {"source": "doc.pdf"})
    retriever_mock = MagicMock()
    retriever_mock.invoke.return_value = [doc, doc]
    vector_store_service_mock.as_retriever.return_value = retriever_mock

    # Patch the chain to return forbidden phrase
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = "Generally, the rate is $1000."
    with patch.object(rag_service, "rag_template", new=""), \
         patch.object(rag_service, "prompt", new=MagicMock()), \
         patch.object(rag_service, "_runnable_parallel", return_value=chain_mock), \
         patch.object(rag_service, "_runnable_passthrough", return_value=lambda x: x):
        query = QAQuery(question="What is the rate?")
        answer = rag_service.answer_question(query)
        assert answer.confidence_score < 0.85

def test_error_handling_returns_fallback(vector_store_service_mock, rag_service):
    # Simulate exception in chain
    doc = DummyDoc("Some content.", {"source": "doc.pdf"})
    retriever_mock = MagicMock()
    retriever_mock.invoke.return_value = [doc]
    vector_store_service_mock.as_retriever.return_value = retriever_mock

    # Patch the chain to raise
    chain_mock = MagicMock()
    chain_mock.invoke.side_effect = Exception("LLM error")
    with patch.object(rag_service, "rag_template", new=""), \
         patch.object(rag_service, "prompt", new=MagicMock()), \
         patch.object(rag_service, "_runnable_parallel", return_value=chain_mock), \
         patch.object(rag_service, "_runnable_passthrough", return_value=lambda x: x):
        query = QAQuery(question="What is the rate?")
        answer = rag_service.answer_question(query)
        assert answer.answer == "An error occurred while generating the answer."
        assert answer.confidence_score == 0.0
        assert answer.sources == []

def test_edge_case_empty_question(vector_store_service_mock, rag_service):
    retriever_mock = MagicMock()
    retriever_mock.invoke.return_value = []
    vector_store_service_mock.as_retriever.return_value = retriever_mock
    query = QAQuery(question="")
    answer = rag_service.answer_question(query)
    assert answer.answer == "I cannot find the answer in the provided documents."
    assert answer.confidence_score == 0.0
    assert answer.sources == []

def test_forbidden_phrases_are_case_insensitive(vector_store_service_mock, rag_service):
    doc = DummyDoc("In MOST CASES, the answer is $500.", {"source": "doc.pdf"})
    retriever_mock = MagicMock()
    retriever_mock.invoke.return_value = [doc, doc]
    vector_store_service_mock.as_retriever.return_value = retriever_mock

    chain_mock = MagicMock()
    chain_mock.invoke.return_value = "In MOST CASES, the answer is $500."
    with patch.object(rag_service, "rag_template", new=""), \
         patch.object(rag_service, "prompt", new=MagicMock()), \
         patch.object(rag_service, "_runnable_parallel", return_value=chain_mock), \
         patch.object(rag_service, "_runnable_passthrough", return_value=lambda x: x):
        query = QAQuery(question="What is the rate?")
        answer = rag_service.answer_question(query)
        assert answer.confidence_score < 0.85
