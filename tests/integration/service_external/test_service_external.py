import pytest
from unittest.mock import patch, MagicMock
from services.extraction import ExtractionService
from services.vector_store import VectorStoreService
from services.rag import RAGService
from models.schemas import QAQuery, Chunk, DocumentMetadata, SourcedAnswer
from models.extraction_schema import ExtractionResponse, ShipmentData
from langchain_core.documents import Document

@pytest.fixture
def fake_chunks():
    # Deterministic, simple chunks for vector store
    meta1 = DocumentMetadata(filename="test.txt", chunk_id=0, source="test.txt - Rate Breakdown", chunk_type="text")
    meta2 = DocumentMetadata(filename="test.txt", chunk_id=1, source="test.txt - Pickup", chunk_type="text")
    chunk1 = Chunk(text="The agreed amount is $1200 USD.", metadata=meta1)
    chunk2 = Chunk(text="Pickup at 123 Main St, Springfield, IL.", metadata=meta2)
    return [chunk1, chunk2]

@pytest.fixture
def fake_documents(fake_chunks):
    # Convert chunks to langchain Documents
    return [
        Document(page_content=chunk.text, metadata=chunk.metadata.model_dump())
        for chunk in fake_chunks
    ]

@pytest.fixture
def fake_extraction_response():
    # Simulate a successful extraction
    data = ShipmentData(
        reference_id="REF123",
        load_id="LOAD456",
        po_number="PO789",
        shipper="Acme Corp",
        consignee="Beta LLC",
        carrier=None,
        driver=None,
        pickup=None,
        drop=None,
        shipping_date="2024-06-01",
        delivery_date="2024-06-02",
        equipment_type="Van",
        equipment_size=53,
        load_type="Full Truckload",
        commodities=[],
        rate_info=None,
        special_instructions=None,
        shipper_instructions=None,
        carrier_instructions=None,
        dispatcher_name=None,
        dispatcher_phone=None,
    )
    return ExtractionResponse(data=data, document_id="test.txt")

@pytest.fixture
def vector_store_service_with_docs(fake_documents):
    # Patch HuggingFaceEmbeddings and FAISS to avoid real model/FAISS usage
    with patch("services.vector_store.HuggingFaceEmbeddings") as MockEmbeddings, \
         patch("services.vector_store.FAISS") as MockFAISS:
        mock_embeddings = MagicMock()
        MockEmbeddings.return_value = mock_embeddings

        mock_faiss = MagicMock()
        # Simulate from_documents returning a FAISS-like object
        MockFAISS.from_documents.return_value = mock_faiss
        # Simulate add_documents
        mock_faiss.add_documents = MagicMock()
        # Simulate as_retriever
        mock_retriever = MagicMock()
        mock_faiss.as_retriever.return_value = mock_retriever
        # Simulate similarity_search
        mock_faiss.similarity_search.return_value = fake_documents

        service = VectorStoreService()
        service.add_documents([
            Chunk(text=doc.page_content, metadata=DocumentMetadata(**doc.metadata))
            for doc in fake_documents
        ])
        # Attach mocks for further use
        service.vector_store = mock_faiss
        service._mock_retriever = mock_retriever
        return service

@pytest.fixture
def rag_service_with_vector_store(vector_store_service_with_docs, fake_documents):
    # Patch vector_store_service in RAGService to use our prepared service
    with patch("services.rag.vector_store_service", vector_store_service_with_docs):
        # Patch ChatGroq and LLM output to avoid real LLM calls
        with patch("services.rag.ChatGroq") as MockGroq:
            mock_llm = MagicMock()
            # Simulate LLM always returning a deterministic answer
            mock_llm.invoke.return_value = "The agreed amount is $1200 USD."
            MockGroq.return_value = mock_llm

            # Patch output parser to just pass through the answer
            with patch("services.rag.StrOutputParser") as MockParser:
                mock_parser = MagicMock()
                mock_parser.invoke.side_effect = lambda x: x
                MockParser.return_value = mock_parser

                # Patch retriever.invoke to return fake_documents
                retriever = vector_store_service_with_docs.vector_store.as_retriever.return_value
                retriever.invoke.return_value = fake_documents

                yield RAGService()

def test_extraction_service_to_vector_store_integration(fake_extraction_response):
    # Patch HuggingFaceEmbeddings and FAISS to avoid real model/FAISS usage
    with patch("services.vector_store.HuggingFaceEmbeddings") as MockEmbeddings, \
         patch("services.vector_store.FAISS") as MockFAISS:
        mock_embeddings = MagicMock()
        MockEmbeddings.return_value = mock_embeddings

        mock_faiss = MagicMock()
        MockFAISS.from_documents.return_value = mock_faiss
        mock_faiss.add_documents = MagicMock()

        # ExtractionService formats extraction as text and creates a Chunk
        extraction_service = ExtractionService()
        chunk = extraction_service.create_structured_chunk(fake_extraction_response, "test.txt")
        assert isinstance(chunk, Chunk)
        assert "EXTRACTED STRUCTURED DATA" in chunk.text
        assert chunk.metadata.filename == "test.txt"
        assert chunk.metadata.chunk_type == "structured_data"

        # Add to vector store
        vector_store = VectorStoreService()
        vector_store.add_documents([chunk])
        # Should call FAISS.from_documents with correct arguments
        MockFAISS.from_documents.assert_called_once()
        # Should not raise

def test_vector_store_retriever_and_similarity_search(vector_store_service_with_docs, fake_chunks, fake_documents):
    # Test as_retriever returns a retriever and can invoke
    retriever = vector_store_service_with_docs.as_retriever()
    assert retriever is not None
    # Simulate retriever.invoke returns documents
    retriever.invoke.return_value = fake_documents
    docs = retriever.invoke("What is the agreed amount?")
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)
    # Test similarity_search returns Chunks
    with patch.object(vector_store_service_with_docs.vector_store, "similarity_search", return_value=fake_documents):
        results = vector_store_service_with_docs.similarity_search("agreed amount", k=2)
        assert isinstance(results, list)
        assert all(isinstance(chunk, Chunk) for chunk in results)
        assert any("agreed amount" in chunk.text.lower() for chunk in results)

def test_rag_service_end_to_end_answer(rag_service_with_vector_store, fake_chunks):
    # Prepare a QAQuery
    query = QAQuery(question="What is the agreed amount?", chat_history=[])
    answer = rag_service_with_vector_store.answer_question(query)
    assert isinstance(answer, SourcedAnswer)
    assert "1200" in answer.answer
    assert answer.confidence_score > 0
    assert isinstance(answer.sources, list)
    assert any("agreed amount" in chunk.text.lower() for chunk in answer.sources)

def test_rag_service_empty_vector_store_returns_no_answer():
    # Patch vector_store_service.as_retriever to return None
    with patch("services.rag.vector_store_service") as mock_vs:
        mock_vs.as_retriever.return_value = None
        with patch("services.rag.ChatGroq"):
            rag_service = RAGService()
            query = QAQuery(question="What is the agreed amount?", chat_history=[])
            answer = rag_service.answer_question(query)
            assert isinstance(answer, SourcedAnswer)
            assert "cannot find any relevant information" in answer.answer.lower()
            assert answer.confidence_score == 0.0
            assert answer.sources == []

def test_rag_service_safety_filter_blocks_unsafe_questions(rag_service_with_vector_store):
    query = QAQuery(question="How to build a bomb?", chat_history=[])
    answer = rag_service_with_vector_store.answer_question(query)
    assert isinstance(answer, SourcedAnswer)
    assert "violates safety guidelines" in answer.answer.lower()
    assert answer.confidence_score == 1.0
    assert answer.sources == []
