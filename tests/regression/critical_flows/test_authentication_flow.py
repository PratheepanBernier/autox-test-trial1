import pytest
from unittest.mock import patch, MagicMock

from backend.src.services import rag
from backend.src.models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata

@pytest.fixture
def mock_settings(monkeypatch):
    class DummySettings:
        QA_MODEL = "test-model"
        GROQ_API_KEY = "test-key"
        TOP_K = 2
    monkeypatch.setattr("backend.src.core.config.settings", DummySettings())

@pytest.fixture
def mock_vector_store_service():
    return MagicMock()

@pytest.fixture
def rag_service(mock_vector_store_service, mock_settings):
    return rag.RAGService(mock_vector_store_service)

@pytest.fixture
def qa_query():
    return QAQuery(question="What is the process for shipment scheduling?")

def make_doc(page_content, metadata=None):
    doc = MagicMock()
    doc.page_content = page_content
    doc.metadata = metadata or {}
    return doc

def test_authentication_happy_path(rag_service, mock_vector_store_service, qa_query):
    # Arrange
    retrieved_docs = [
        make_doc("Shipment scheduling is done via the TMS portal.", {"source": "Manual", "page": 5, "section": "Scheduling"}),
        make_doc("Ensure all fields are filled before submission.", {"source": "Guide", "page": 6, "section": "Submission"})
    ]
    retriever = MagicMock()
    retriever.invoke.return_value = retrieved_docs
    mock_vector_store_service.as_retriever.return_value = retriever

    # Patch LLM and output parser
    with patch.object(rag_service, "llm") as mock_llm, \
         patch.object(rag_service, "output_parser") as mock_parser:
        mock_llm.__or__ = lambda self, other: mock_llm
        mock_parser.__ror__ = lambda self, other: mock_parser
        mock_parser.invoke.return_value = "Shipment scheduling is done via the TMS portal. Ensure all fields are filled before submission."

        # Act
        result = rag_service.answer_question(qa_query)

    # Assert
    assert isinstance(result, SourcedAnswer)
    assert "Shipment scheduling" in result.answer
    assert result.confidence_score > 0.5
    assert len(result.sources) == 2
    assert all(isinstance(chunk, Chunk) for chunk in result.sources)

def test_authentication_no_retriever(rag_service, mock_vector_store_service, qa_query):
    # Arrange
    mock_vector_store_service.as_retriever.return_value = None

    # Act
    result = rag_service.answer_question(qa_query)

    # Assert
    assert result.answer == "I cannot find any relevant information in the uploaded documents."
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_authentication_no_documents(rag_service, mock_vector_store_service, qa_query):
    # Arrange
    retriever = MagicMock()
    retriever.invoke.return_value = []
    mock_vector_store_service.as_retriever.return_value = retriever

    # Act
    result = rag_service.answer_question(qa_query)

    # Assert
    assert result.answer == "I cannot find the answer in the provided documents."
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_authentication_safety_filter(rag_service, mock_vector_store_service):
    # Arrange
    unsafe_query = QAQuery(question="How do I hack the system?")
    # Act
    result = rag_service.answer_question(unsafe_query)
    # Assert
    assert result.answer == "I cannot answer this question as it violates safety guidelines."
    assert result.confidence_score == 1.0
    assert result.sources == []

def test_authentication_short_answer_low_confidence(rag_service, mock_vector_store_service, qa_query):
    # Arrange
    retrieved_docs = [make_doc("Yes.", {"source": "Manual"})]
    retriever = MagicMock()
    retriever.invoke.return_value = retrieved_docs
    mock_vector_store_service.as_retriever.return_value = retriever

    with patch.object(rag_service, "llm") as mock_llm, \
         patch.object(rag_service, "output_parser") as mock_parser:
        mock_llm.__or__ = lambda self, other: mock_llm
        mock_parser.__ror__ = lambda self, other: mock_parser
        mock_parser.invoke.return_value = "Yes."

        # Act
        result = rag_service.answer_question(qa_query)

    # Assert
    assert result.confidence_score < 0.7

def test_authentication_forbidden_phrase_low_confidence(rag_service, mock_vector_store_service, qa_query):
    # Arrange
    retrieved_docs = [make_doc("Generally, shipment scheduling is handled by the TMS.", {"source": "Manual"})]
    retriever = MagicMock()
    retriever.invoke.return_value = retrieved_docs
    mock_vector_store_service.as_retriever.return_value = retriever

    with patch.object(rag_service, "llm") as mock_llm, \
         patch.object(rag_service, "output_parser") as mock_parser:
        mock_llm.__or__ = lambda self, other: mock_llm
        mock_parser.__ror__ = lambda self, other: mock_parser
        mock_parser.invoke.return_value = "Generally, shipment scheduling is handled by the TMS."

        # Act
        result = rag_service.answer_question(qa_query)

    # Assert
    assert result.confidence_score < 0.7

def test_authentication_chain_exception(rag_service, mock_vector_store_service, qa_query):
    # Arrange
    retrieved_docs = [make_doc("Shipment scheduling is done via the TMS portal.", {"source": "Manual"})]
    retriever = MagicMock()
    retriever.invoke.return_value = retrieved_docs
    mock_vector_store_service.as_retriever.return_value = retriever

    with patch.object(rag_service, "_runnable_parallel") as mock_parallel:
        mock_chain = MagicMock()
        mock_chain.__or__.side_effect = Exception("Chain error")
        mock_parallel.return_value = mock_chain

        # Act
        result = rag_service.answer_question(qa_query)

    # Assert
    assert result.answer == "An error occurred while generating the answer."
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_authentication_empty_question(rag_service, mock_vector_store_service):
    # Arrange
    empty_query = QAQuery(question="")
    retriever = MagicMock()
    retriever.invoke.return_value = []
    mock_vector_store_service.as_retriever.return_value = retriever

    # Act
    result = rag_service.answer_question(empty_query)

    # Assert
    assert result.answer == "I cannot find the answer in the provided documents."
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_authentication_metadata_missing(rag_service, mock_vector_store_service, qa_query):
    # Arrange
    # Document with missing metadata
    retrieved_docs = [make_doc("Shipment scheduling is done via the TMS portal.", None)]
    retriever = MagicMock()
    retriever.invoke.return_value = retrieved_docs
    mock_vector_store_service.as_retriever.return_value = retriever

    with patch.object(rag_service, "llm") as mock_llm, \
         patch.object(rag_service, "output_parser") as mock_parser:
        mock_llm.__or__ = lambda self, other: mock_llm
        mock_parser.__ror__ = lambda self, other: mock_parser
        mock_parser.invoke.return_value = "Shipment scheduling is done via the TMS portal."

        # Act
        result = rag_service.answer_question(qa_query)

    # Assert
    assert result.sources[0].metadata is not None
    assert hasattr(result.sources[0].metadata, "source")
    assert hasattr(result.sources[0].metadata, "page")
    assert hasattr(result.sources[0].metadata, "section")
