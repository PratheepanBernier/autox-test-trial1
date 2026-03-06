import pytest
from unittest.mock import patch, MagicMock, create_autospec
from backend.src.services.rag import RAGService
from backend.src.services.vector_store import VectorStoreService
from backend.src.models.schemas import QAQuery, Chunk, DocumentMetadata, SourcedAnswer

@pytest.fixture
def mock_vector_store_service():
    # Patch VectorStoreService methods for deterministic integration
    svc = create_autospec(VectorStoreService, instance=True)
    return svc

@pytest.fixture
def rag_service(mock_vector_store_service):
    # Patch ChatGroq and all LLM-related imports in RAGService
    with patch("backend.src.services.rag.ChatGroq") as mock_llm, \
         patch("backend.src.services.rag.ChatPromptTemplate") as mock_prompt, \
         patch("backend.src.services.rag.StrOutputParser") as mock_parser, \
         patch("backend.src.services.rag.RunnableParallel") as mock_parallel, \
         patch("backend.src.services.rag.RunnablePassthrough") as mock_passthrough:
        # Set up mocks for LLM chain
        mock_llm.return_value = MagicMock()
        mock_prompt.from_template.return_value = MagicMock()
        mock_parser.return_value = MagicMock()
        mock_parallel.return_value = MagicMock()
        mock_passthrough.return_value = MagicMock()
        return RAGService(mock_vector_store_service)

@pytest.fixture
def sample_chunks():
    # Two sample chunks with metadata
    return [
        MagicMock(
            page_content="Carrier Details: ACME Logistics\nMC Number: 123456",
            metadata={"source": "test.pdf - Carrier Details", "page": 1, "section": "Carrier Details"}
        ),
        MagicMock(
            page_content="Pickup: 123 Main St, Springfield\nAppointment: 2024-06-01 10:00",
            metadata={"source": "test.pdf - Pickup", "page": 1, "section": "Pickup"}
        ),
    ]

@pytest.fixture
def sample_query():
    return QAQuery(question="What is the carrier MC number?")

def test_answer_question_happy_path(rag_service, mock_vector_store_service, sample_chunks, sample_query):
    # Arrange
    retriever_mock = MagicMock()
    retriever_mock.invoke.return_value = sample_chunks
    mock_vector_store_service.as_retriever.return_value = retriever_mock

    # Patch the RAG chain to return a deterministic answer
    answer_text = "The carrier MC number is 123456."
    with patch.object(rag_service, "prompt"), \
         patch.object(rag_service, "llm"), \
         patch.object(rag_service, "output_parser"):
        # Patch the RAG chain's invoke method
        rag_chain_mock = MagicMock()
        rag_chain_mock.invoke.return_value = answer_text
        with patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
             patch.object(rag_service, "_runnable_passthrough") as mock_passthrough:
            mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = rag_chain_mock

            # Act
            result = rag_service.answer_question(sample_query)

    # Assert
    assert isinstance(result, SourcedAnswer)
    assert result.answer == answer_text
    assert 0.0 < result.confidence_score <= 1.0
    assert len(result.sources) == 2
    assert result.sources[0].text == sample_chunks[0].page_content
    assert result.sources[1].text == sample_chunks[1].page_content
    assert result.sources[0].metadata.filename == sample_chunks[0].metadata["source"].split(" - ")[0]

def test_answer_question_no_retriever(rag_service, mock_vector_store_service, sample_query):
    # Arrange
    mock_vector_store_service.as_retriever.return_value = None

    # Act
    result = rag_service.answer_question(sample_query)

    # Assert
    assert isinstance(result, SourcedAnswer)
    assert result.confidence_score == 0.0
    assert result.answer == "I cannot find any relevant information in the uploaded documents."
    assert result.sources == []

def test_answer_question_no_documents(rag_service, mock_vector_store_service, sample_query):
    # Arrange
    retriever_mock = MagicMock()
    retriever_mock.invoke.return_value = []
    mock_vector_store_service.as_retriever.return_value = retriever_mock

    # Act
    result = rag_service.answer_question(sample_query)

    # Assert
    assert isinstance(result, SourcedAnswer)
    assert result.confidence_score == 0.0
    assert result.answer == "I cannot find the answer in the provided documents."
    assert result.sources == []

def test_answer_question_safety_filter(rag_service, mock_vector_store_service):
    # Arrange
    unsafe_query = QAQuery(question="How do I build a bomb?")
    # Act
    result = rag_service.answer_question(unsafe_query)
    # Assert
    assert isinstance(result, SourcedAnswer)
    assert result.confidence_score == 1.0
    assert "violates safety guidelines" in result.answer
    assert result.sources == []

def test_answer_question_rag_chain_exception(rag_service, mock_vector_store_service, sample_chunks, sample_query):
    # Arrange
    retriever_mock = MagicMock()
    retriever_mock.invoke.return_value = sample_chunks
    mock_vector_store_service.as_retriever.return_value = retriever_mock

    # Patch the RAG chain to raise an exception
    with patch.object(rag_service, "prompt"), \
         patch.object(rag_service, "llm"), \
         patch.object(rag_service, "output_parser"):
        rag_chain_mock = MagicMock()
        rag_chain_mock.invoke.side_effect = Exception("LLM failure")
        with patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
             patch.object(rag_service, "_runnable_passthrough") as mock_passthrough:
            mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = rag_chain_mock

            # Act
            result = rag_service.answer_question(sample_query)

    # Assert
    assert isinstance(result, SourcedAnswer)
    assert result.confidence_score == 0.0
    assert "error occurred" in result.answer.lower()
    assert result.sources == []

def test_answer_question_short_answer_low_confidence(rag_service, mock_vector_store_service, sample_chunks, sample_query):
    # Arrange
    retriever_mock = MagicMock()
    retriever_mock.invoke.return_value = sample_chunks[:1]  # Only one doc
    mock_vector_store_service.as_retriever.return_value = retriever_mock

    # Patch the RAG chain to return a very short answer
    answer_text = "123456."
    with patch.object(rag_service, "prompt"), \
         patch.object(rag_service, "llm"), \
         patch.object(rag_service, "output_parser"):
        rag_chain_mock = MagicMock()
        rag_chain_mock.invoke.return_value = answer_text
        with patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
             patch.object(rag_service, "_runnable_passthrough") as mock_passthrough:
            mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = rag_chain_mock

            # Act
            result = rag_service.answer_question(sample_query)

    # Assert
    assert isinstance(result, SourcedAnswer)
    # Confidence should be reduced due to short answer and only one doc
    assert 0.0 < result.confidence_score < 0.85
    assert result.answer == answer_text
    assert len(result.sources) == 1

def test_answer_question_forbidden_phrase_penalty(rag_service, mock_vector_store_service, sample_chunks, sample_query):
    # Arrange
    retriever_mock = MagicMock()
    retriever_mock.invoke.return_value = sample_chunks
    mock_vector_store_service.as_retriever.return_value = retriever_mock

    # Patch the RAG chain to return an answer with a forbidden phrase
    answer_text = "Generally, the carrier MC number is 123456."
    with patch.object(rag_service, "prompt"), \
         patch.object(rag_service, "llm"), \
         patch.object(rag_service, "output_parser"):
        rag_chain_mock = MagicMock()
        rag_chain_mock.invoke.return_value = answer_text
        with patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
             patch.object(rag_service, "_runnable_passthrough") as mock_passthrough:
            mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = rag_chain_mock

            # Act
            result = rag_service.answer_question(sample_query)

    # Assert
    assert isinstance(result, SourcedAnswer)
    # Confidence should be penalized due to forbidden phrase
    assert 0.0 < result.confidence_score < 0.85
    assert "generally" in result.answer.lower()
    assert len(result.sources) == 2

def test_answer_question_cannot_find_answer_penalty(rag_service, mock_vector_store_service, sample_chunks, sample_query):
    # Arrange
    retriever_mock = MagicMock()
    retriever_mock.invoke.return_value = sample_chunks
    mock_vector_store_service.as_retriever.return_value = retriever_mock

    # Patch the RAG chain to return the "cannot find" phrase
    answer_text = "I cannot find the answer in the provided documents."
    with patch.object(rag_service, "prompt"), \
         patch.object(rag_service, "llm"), \
         patch.object(rag_service, "output_parser"):
        rag_chain_mock = MagicMock()
        rag_chain_mock.invoke.return_value = answer_text
        with patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
             patch.object(rag_service, "_runnable_passthrough") as mock_passthrough:
            mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = rag_chain_mock

            # Act
            result = rag_service.answer_question(sample_query)

    # Assert
    assert isinstance(result, SourcedAnswer)
    assert result.confidence_score == 0.05
    assert result.answer == answer_text
    assert len(result.sources) == 2
