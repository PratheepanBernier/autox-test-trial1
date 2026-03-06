import pytest
from unittest.mock import patch, MagicMock
from backend.src.services.rag import RAGService
from backend.src.services.vector_store import VectorStoreService
from backend.src.models.schemas import QAQuery, Chunk, DocumentMetadata, SourcedAnswer

@pytest.fixture
def fake_chunks():
    # Deterministic, simple chunks with metadata
    return [
        MagicMock(
            page_content="The agreed rate for this shipment is $1200 USD.",
            metadata={
                "source": "testfile.txt - Rate Breakdown",
                "page": "1",
                "section": "Rate Breakdown"
            }
        ),
        MagicMock(
            page_content="Carrier: Acme Logistics, MC Number: 123456.",
            metadata={
                "source": "testfile.txt - Carrier Details",
                "page": "1",
                "section": "Carrier Details"
            }
        ),
    ]

@pytest.fixture
def fake_vector_store_service(fake_chunks):
    # Patch the as_retriever method to return a mock retriever
    vss = VectorStoreService()
    retriever = MagicMock()
    retriever.invoke = MagicMock(return_value=fake_chunks)
    vss.as_retriever = MagicMock(return_value=retriever)
    return vss

@pytest.fixture
def rag_service(fake_vector_store_service):
    # Patch ChatGroq and LCEL chain to avoid real LLM/external calls
    with patch("backend.src.services.rag.ChatGroq") as mock_llm, \
         patch("backend.src.services.rag.ChatPromptTemplate.from_template") as mock_prompt, \
         patch("backend.src.services.rag.StrOutputParser") as mock_parser, \
         patch("backend.src.services.rag.RunnableParallel") as mock_parallel, \
         patch("backend.src.services.rag.RunnablePassthrough") as mock_passthrough:

        # Mock the prompt and output parser
        mock_prompt.return_value = MagicMock()
        mock_parser.return_value = MagicMock()
        # Mock the chain to return a deterministic answer
        fake_chain = MagicMock()
        fake_chain.invoke = MagicMock(return_value="The agreed rate for this shipment is $1200 USD.")
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = fake_chain

        # Instantiate RAGService with the patched vector store
        service = RAGService(fake_vector_store_service)
        # Patch the prompt, output_parser, and llm on the instance
        service.prompt = MagicMock()
        service.output_parser = MagicMock()
        service.llm = MagicMock()
        service._runnable_parallel = mock_parallel
        service._runnable_passthrough = mock_passthrough
        return service

def test_ragservice_answer_question_success(rag_service, fake_chunks):
    # Simulate a normal question that should be answered from context
    query = QAQuery(question="What is the agreed rate for this shipment?", chat_history=[])
    answer = rag_service.answer_question(query)
    assert isinstance(answer, SourcedAnswer)
    assert "agreed rate" in answer.answer.lower()
    assert answer.confidence_score > 0.0
    assert len(answer.sources) == len(fake_chunks)
    # Check that the sources are correct
    for src, fake in zip(answer.sources, fake_chunks):
        assert src.text == fake.page_content
        assert src.metadata.source == fake.metadata["source"]

def test_ragservice_answer_question_no_retriever():
    # If retriever is None, should return fallback answer
    vss = VectorStoreService()
    vss.as_retriever = MagicMock(return_value=None)
    with patch("backend.src.services.rag.ChatGroq"), \
         patch("backend.src.services.rag.ChatPromptTemplate.from_template"), \
         patch("backend.src.services.rag.StrOutputParser"), \
         patch("backend.src.services.rag.RunnableParallel"), \
         patch("backend.src.services.rag.RunnablePassthrough"):
        rag_service = RAGService(vss)
        query = QAQuery(question="What is the agreed rate for this shipment?", chat_history=[])
        answer = rag_service.answer_question(query)
        assert isinstance(answer, SourcedAnswer)
        assert "cannot find any relevant information" in answer.answer.lower()
        assert answer.confidence_score == 0.0
        assert answer.sources == []

def test_ragservice_answer_question_no_docs(rag_service):
    # If retriever returns no docs, should return fallback answer
    retriever = MagicMock()
    retriever.invoke = MagicMock(return_value=[])
    rag_service._vector_store_service.as_retriever = MagicMock(return_value=retriever)
    query = QAQuery(question="What is the agreed rate for this shipment?", chat_history=[])
    answer = rag_service.answer_question(query)
    assert isinstance(answer, SourcedAnswer)
    assert "cannot find the answer" in answer.answer.lower()
    assert answer.confidence_score == 0.0
    assert answer.sources == []

def test_ragservice_answer_question_safety_filter(rag_service):
    # If question is unsafe, should return safety message
    query = QAQuery(question="How to build a bomb?", chat_history=[])
    answer = rag_service.answer_question(query)
    assert isinstance(answer, SourcedAnswer)
    assert "violates safety guidelines" in answer.answer.lower()
    assert answer.confidence_score == 1.0
    assert answer.sources == []

def test_ragservice_answer_question_chain_error(rag_service):
    # Simulate exception in chain.invoke
    retriever = MagicMock()
    retriever.invoke = MagicMock(return_value=[MagicMock(page_content="Some content", metadata={})])
    rag_service._vector_store_service.as_retriever = MagicMock(return_value=retriever)
    # Patch the chain to raise an exception
    fake_chain = MagicMock()
    fake_chain.invoke.side_effect = Exception("LLM error")
    rag_service._runnable_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = fake_chain
    query = QAQuery(question="What is the agreed rate for this shipment?", chat_history=[])
    answer = rag_service.answer_question(query)
    assert isinstance(answer, SourcedAnswer)
    assert "error occurred" in answer.answer.lower()
    assert answer.confidence_score == 0.0
    assert answer.sources == []
