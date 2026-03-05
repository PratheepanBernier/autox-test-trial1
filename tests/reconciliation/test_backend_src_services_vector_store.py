import pytest
from unittest.mock import patch, MagicMock, create_autospec
from backend.src.services.vector_store import VectorStoreService
from models.schemas import Chunk, DocumentMetadata
from langchain_core.documents import Document

@pytest.fixture
def dummy_metadata():
    return DocumentMetadata(source="unit_test", page=1)

@pytest.fixture
def dummy_chunk(dummy_metadata):
    return Chunk(text="This is a test chunk.", metadata=dummy_metadata)

@pytest.fixture
def dummy_chunks(dummy_metadata):
    return [
        Chunk(text="Chunk 1", metadata=dummy_metadata),
        Chunk(text="Chunk 2", metadata=dummy_metadata),
    ]

@pytest.fixture
def mock_embeddings():
    with patch("backend.src.services.vector_store.HuggingFaceEmbeddings") as mock_emb:
        yield mock_emb

@pytest.fixture
def mock_faiss():
    with patch("backend.src.services.vector_store.FAISS") as mock_faiss:
        yield mock_faiss

@pytest.fixture
def mock_settings(monkeypatch):
    class DummySettings:
        EMBEDDING_MODEL = "test-model"
        TOP_K = 3
    monkeypatch.setattr("backend.src.services.vector_store.settings", DummySettings())

@pytest.fixture
def vector_store_service(mock_embeddings, mock_settings):
    return VectorStoreService()

def test_add_documents_initializes_faiss_on_first_add(vector_store_service, dummy_chunks, mock_faiss):
    mock_faiss.from_documents.return_value = MagicMock()
    vector_store_service.vector_store = None
    vector_store_service.add_documents(dummy_chunks)
    assert vector_store_service.vector_store is not None
    mock_faiss.from_documents.assert_called_once()
    # Check that add_documents was NOT called on the vector_store (since it's first add)
    assert not vector_store_service.vector_store.add_documents.called

def test_add_documents_appends_to_existing_store(vector_store_service, dummy_chunks):
    mock_store = MagicMock()
    vector_store_service.vector_store = mock_store
    vector_store_service.add_documents(dummy_chunks)
    mock_store.add_documents.assert_called_once()
    # Should not call from_documents if vector_store exists
    assert not hasattr(vector_store_service.vector_store, "from_documents")

def test_add_documents_handles_exception(vector_store_service, dummy_chunks):
    # Simulate error in Document creation
    with patch("backend.src.services.vector_store.Document", side_effect=ValueError("fail")):
        with pytest.raises(ValueError):
            vector_store_service.add_documents(dummy_chunks)

def test_as_retriever_returns_none_if_store_empty(vector_store_service):
    vector_store_service.vector_store = None
    retriever = vector_store_service.as_retriever()
    assert retriever is None

def test_as_retriever_returns_retriever_with_default_kwargs(vector_store_service):
    mock_store = MagicMock()
    mock_retriever = MagicMock()
    mock_store.as_retriever.return_value = mock_retriever
    vector_store_service.vector_store = mock_store
    retriever = vector_store_service.as_retriever()
    assert retriever == mock_retriever
    mock_store.as_retriever.assert_called_once()
    args, kwargs = mock_store.as_retriever.call_args
    assert kwargs["search_type"] == "similarity"
    assert "search_kwargs" in kwargs

def test_as_retriever_returns_retriever_with_custom_kwargs(vector_store_service):
    mock_store = MagicMock()
    mock_retriever = MagicMock()
    mock_store.as_retriever.return_value = mock_retriever
    vector_store_service.vector_store = mock_store
    retriever = vector_store_service.as_retriever(search_type="mmr", search_kwargs={"k": 2})
    assert retriever == mock_retriever
    mock_store.as_retriever.assert_called_once_with(search_type="mmr", search_kwargs={"k": 2})

def test_as_retriever_handles_exception_and_returns_none(vector_store_service):
    mock_store = MagicMock()
    mock_store.as_retriever.side_effect = Exception("fail")
    vector_store_service.vector_store = mock_store
    retriever = vector_store_service.as_retriever()
    assert retriever is None

def test_similarity_search_returns_empty_if_store_empty(vector_store_service):
    vector_store_service.vector_store = None
    result = vector_store_service.similarity_search("query", k=2)
    assert result == []

def test_similarity_search_returns_chunks(vector_store_service, dummy_metadata):
    mock_store = MagicMock()
    doc1 = Document(page_content="A", metadata={"source": "unit_test", "page": 1})
    doc2 = Document(page_content="B", metadata={"source": "unit_test", "page": 1})
    mock_store.similarity_search.return_value = [doc1, doc2]
    vector_store_service.vector_store = mock_store
    result = vector_store_service.similarity_search("query", k=2)
    assert len(result) == 2
    assert all(isinstance(chunk, Chunk) for chunk in result)
    assert result[0].text == "A"
    assert result[1].text == "B"
    assert result[0].metadata.source == "unit_test"
    assert result[0].metadata.page == 1

def test_similarity_search_handles_exception_and_returns_empty(vector_store_service):
    mock_store = MagicMock()
    mock_store.similarity_search.side_effect = Exception("fail")
    vector_store_service.vector_store = mock_store
    result = vector_store_service.similarity_search("query", k=2)
    assert result == []

def test_similarity_search_and_as_retriever_equivalence(vector_store_service, dummy_metadata):
    # Reconciliation: similarity_search and as_retriever().get_relevant_documents should return equivalent docs
    docs = [
        Document(page_content="A", metadata={"source": "unit_test", "page": 1}),
        Document(page_content="B", metadata={"source": "unit_test", "page": 1}),
    ]
    # Mock vector_store
    mock_store = MagicMock()
    mock_store.similarity_search.return_value = docs
    mock_retriever = MagicMock()
    mock_retriever.get_relevant_documents.return_value = docs
    mock_store.as_retriever.return_value = mock_retriever
    vector_store_service.vector_store = mock_store

    # similarity_search
    sim_chunks = vector_store_service.similarity_search("query", k=2)
    # as_retriever
    retriever = vector_store_service.as_retriever()
    retriever_docs = retriever.get_relevant_documents("query")
    # Compare outputs
    assert [c.text for c in sim_chunks] == [d.page_content for d in retriever_docs]
    assert [c.metadata.model_dump() for c in sim_chunks] == [d.metadata for d in retriever_docs]

def test_add_documents_and_similarity_search_integration(vector_store_service, dummy_chunks, mock_faiss):
    # Reconciliation: add_documents then similarity_search returns what was added
    # Setup FAISS.from_documents to return a mock store with similarity_search
    mock_store = MagicMock()
    mock_faiss.from_documents.return_value = mock_store
    # Simulate similarity_search returns the same docs as added
    docs = [
        Document(page_content=chunk.text, metadata=chunk.metadata.model_dump())
        for chunk in dummy_chunks
    ]
    mock_store.similarity_search.return_value = docs
    vector_store_service.vector_store = None
    vector_store_service.add_documents(dummy_chunks)
    result = vector_store_service.similarity_search("query", k=2)
    assert len(result) == 2
    assert [c.text for c in result] == [chunk.text for chunk in dummy_chunks]
    assert [c.metadata.model_dump() for c in result] == [chunk.metadata.model_dump() for chunk in dummy_chunks]
