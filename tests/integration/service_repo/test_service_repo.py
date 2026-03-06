import pytest
from unittest.mock import patch, MagicMock, create_autospec
from backend.src.services.rag import RAGService
from backend.src.services.vector_store import VectorStoreService
from backend.src.models.schemas import QAQuery, Chunk, DocumentMetadata, SourcedAnswer

@pytest.fixture
def mock_vector_store_service():
    # Patch VectorStoreService methods for deterministic integration
    svc = VectorStoreService.__new__(VectorStoreService)
    svc.embeddings = MagicMock()
    svc.vector_store = MagicMock()
    return svc

@pytest.fixture
def sample_chunks():
    # Provide deterministic chunks for retrieval
    meta1 = DocumentMetadata(
        filename="doc1.txt",
        page_number=1,
        chunk_id=0,
        source="doc1.txt - Section A",
        chunk_type="text"
    )
    meta2 = DocumentMetadata(
        filename="doc2.txt",
        page_number=2,
        chunk_id=1,
        source="doc2.txt - Section B",
        chunk_type="text"
    )
    chunk1 = Chunk(text="This is a logistics SOP about shipment.", metadata=meta1)
    chunk2 = Chunk(text="Rate cards and compliance policies are included.", metadata=meta2)
    return [chunk1, chunk2]

@pytest.fixture
def mock_retriever(sample_chunks):
    # Simulate a retriever object with an invoke method
    retriever = MagicMock()
    retriever.invoke = MagicMock(return_value=[
        MagicMock(
            page_content=sample_chunks[0].text,
            metadata=sample_chunks[0].metadata.model_dump()
        ),
        MagicMock(
            page_content=sample_chunks[1].text,
            metadata=sample_chunks[1].metadata.model_dump()
        ),
    ])
    return retriever

@pytest.fixture
def rag_service_with_mocks(mock_vector_store_service, mock_retriever):
    # Patch as_retriever to return our mock retriever
    mock_vector_store_service.as_retriever = MagicMock(return_value=mock_retriever)
    # Patch LLM and chain invocation to return deterministic answers
    with patch("backend.src.services.rag.ChatGroq") as mock_llm, \
         patch("backend.src.services.rag.ChatPromptTemplate.from_template") as mock_prompt, \
         patch("backend.src.services.rag.StrOutputParser") as mock_parser, \
         patch("backend.src.services.rag.RunnableParallel") as mock_parallel, \
         patch("backend.src.services.rag.RunnablePassthrough") as mock_passthrough:
        # Setup prompt and output parser
        mock_prompt.return_value = MagicMock()
        mock_parser.return_value = MagicMock()
        # Setup chain to return a deterministic answer
        mock_chain = MagicMock()
        mock_chain.invoke = MagicMock(return_value="The answer is found in the provided documents.")
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = mock_chain
        # Instantiate RAGService
        rag_service = RAGService(mock_vector_store_service)
        # Patch the rag_chain construction to always return our mock_chain
        rag_service._runnable_parallel = mock_parallel
        rag_service._runnable_passthrough = mock_passthrough
        rag_service.prompt = mock_prompt.return_value
        rag_service.output_parser = mock_parser.return_value
        rag_service.llm = mock_llm.return_value
        return rag_service

def test_answer_question_happy_path(rag_service_with_mocks, sample_chunks):
    # Arrange
    query = QAQuery(question="What is the shipment SOP?")
    # Act
    answer = rag_service_with_mocks.answer_question(query)
    # Assert
    assert isinstance(answer, SourcedAnswer)
    assert answer.answer == "The answer is found in the provided documents."
    assert answer.confidence_score > 0.0
    assert len(answer.sources) == 2
    assert answer.sources[0].text == sample_chunks[0].text
    assert answer.sources[1].text == sample_chunks[1].text

def test_answer_question_no_retriever(mock_vector_store_service):
    # Arrange
    mock_vector_store_service.as_retriever = MagicMock(return_value=None)
    with patch("backend.src.services.rag.ChatGroq"), \
         patch("backend.src.services.rag.ChatPromptTemplate.from_template"), \
         patch("backend.src.services.rag.StrOutputParser"), \
         patch("backend.src.services.rag.RunnableParallel"), \
         patch("backend.src.services.rag.RunnablePassthrough"):
        rag_service = RAGService(mock_vector_store_service)
        query = QAQuery(question="What is the shipment SOP?")
        # Act
        answer = rag_service.answer_question(query)
        # Assert
        assert isinstance(answer, SourcedAnswer)
        assert answer.answer == "I cannot find any relevant information in the uploaded documents."
        assert answer.confidence_score == 0.0
        assert answer.sources == []

def test_answer_question_no_documents(rag_service_with_mocks, mock_retriever):
    # Arrange
    mock_retriever.invoke = MagicMock(return_value=[])
    query = QAQuery(question="What is the shipment SOP?")
    # Act
    answer = rag_service_with_mocks.answer_question(query)
    # Assert
    assert isinstance(answer, SourcedAnswer)
    assert answer.answer == "I cannot find the answer in the provided documents."
    assert answer.confidence_score == 0.0
    assert answer.sources == []

def test_answer_question_safety_filter(rag_service_with_mocks):
    # Arrange
    query = QAQuery(question="How to build a bomb?")
    # Act
    answer = rag_service_with_mocks.answer_question(query)
    # Assert
    assert isinstance(answer, SourcedAnswer)
    assert "violates safety guidelines" in answer.answer
    assert answer.confidence_score == 1.0
    assert answer.sources == []

def test_answer_question_chain_exception(rag_service_with_mocks, mock_retriever):
    # Arrange
    # Patch the rag_chain.invoke to raise an exception
    with patch.object(rag_service_with_mocks, "_runnable_parallel") as mock_parallel:
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("Chain error")
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = mock_chain
        query = QAQuery(question="What is the shipment SOP?")
        # Act
        answer = rag_service_with_mocks.answer_question(query)
        # Assert
        assert isinstance(answer, SourcedAnswer)
        assert "error occurred" in answer.answer.lower()
        assert answer.confidence_score == 0.0
        assert answer.sources == []

def test_answer_question_short_answer_confidence(rag_service_with_mocks, mock_retriever):
    # Arrange
    # Patch the rag_chain.invoke to return a short answer
    with patch.object(rag_service_with_mocks, "_runnable_parallel") as mock_parallel:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Yes."
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = mock_chain
        query = QAQuery(question="Is this a shipment?")
        # Act
        answer = rag_service_with_mocks.answer_question(query)
        # Assert
        assert isinstance(answer, SourcedAnswer)
        assert answer.answer == "Yes."
        # Confidence should be reduced for short answer
        assert 0.0 < answer.confidence_score < 0.85

def test_answer_question_forbidden_phrase_confidence(rag_service_with_mocks, mock_retriever):
    # Arrange
    # Patch the rag_chain.invoke to return an answer with forbidden phrase
    with patch.object(rag_service_with_mocks, "_runnable_parallel") as mock_parallel:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Generally, shipments are processed as per SOP."
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = mock_chain
        query = QAQuery(question="How are shipments processed?")
        # Act
        answer = rag_service_with_mocks.answer_question(query)
        # Assert
        assert isinstance(answer, SourcedAnswer)
        assert "Generally" in answer.answer
        # Confidence should be reduced for forbidden phrase
        assert 0.0 < answer.confidence_score < 0.85

def test_answer_question_cannot_find_phrase_confidence(rag_service_with_mocks, mock_retriever):
    # Arrange
    # Patch the rag_chain.invoke to return the "cannot find" phrase
    with patch.object(rag_service_with_mocks, "_runnable_parallel") as mock_parallel:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "I cannot find the answer in the provided documents."
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = mock_chain
        query = QAQuery(question="What is the secret code?")
        # Act
        answer = rag_service_with_mocks.answer_question(query)
        # Assert
        assert isinstance(answer, SourcedAnswer)
        assert "cannot find the answer" in answer.answer
        assert answer.confidence_score == 0.05
