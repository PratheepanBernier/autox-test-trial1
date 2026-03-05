import pytest
from unittest.mock import patch, MagicMock, create_autospec
from backend.src.services.vector_store import VectorStoreService
from models.schemas import Chunk, DocumentMetadata
from langchain_core.documents import Document

@pytest.fixture
def mock_settings(monkeypatch):
    class DummySettings:
        EMBEDDING_MODEL = "test-model"
        TOP_K = 3
    monkeypatch.setattr("core.config.settings", DummySettings())

@pytest.fixture
def dummy_metadata():
    return DocumentMetadata(source="test_source", page=1)

@pytest.fixture
def dummy_chunk(dummy_metadata):
    return Chunk(text="test text", metadata=dummy_metadata)

@pytest.fixture
def dummy_chunks(dummy_metadata):
    return [
        Chunk(text="text 1", metadata=dummy_metadata),
        Chunk(text="text 2", metadata=dummy_metadata),
    ]

@pytest.fixture
def mock_embeddings():
    return MagicMock(name="HuggingFaceEmbeddings")

@pytest.fixture
def mock_faiss_cls():
    return MagicMock(name="FAISS")

@pytest.fixture
def mock_document_cls():
    return MagicMock(name="Document")

@pytest.fixture(autouse=True)
def patch_external_deps(mock_settings):
    with patch("backend.src.services.vector_store.HuggingFaceEmbeddings") as emb, \
         patch("backend.src.services.vector_store.FAISS") as faiss, \
         patch("backend.src.services.vector_store.Document") as doc:
        yield {"embeddings": emb, "faiss": faiss, "document": doc}

def test_init_success_creates_embeddings_and_logs(patch_external_deps):
    patch_external_deps["embeddings"].return_value = MagicMock()
    service = VectorStoreService()
    patch_external_deps["embeddings"].assert_called_once()
    assert service.embeddings is not None
    assert service.vector_store is None

def test_init_failure_raises_and_logs(patch_external_deps):
    patch_external_deps["embeddings"].side_effect = Exception("fail")
    with pytest.raises(Exception) as excinfo:
        VectorStoreService()
    assert "fail" in str(excinfo.value)

def test_add_documents_initializes_vector_store_and_adds_documents(patch_external_deps, dummy_chunks):
    service = VectorStoreService()
    mock_faiss = patch_external_deps["faiss"]
    mock_faiss.from_documents.return_value = MagicMock()
    service.add_documents(dummy_chunks)
    assert service.vector_store is not None
    mock_faiss.from_documents.assert_called_once()
    # Should not call add_documents on first add
    assert not service.vector_store.add_documents.called

def test_add_documents_adds_to_existing_vector_store(patch_external_deps, dummy_chunks):
    service = VectorStoreService()
    mock_vector_store = MagicMock()
    service.vector_store = mock_vector_store
    service.add_documents(dummy_chunks)
    mock_vector_store.add_documents.assert_called_once()
    patch_external_deps["faiss"].from_documents.assert_not_called()

def test_add_documents_handles_exception_and_logs(patch_external_deps, dummy_chunks):
    service = VectorStoreService()
    patch_external_deps["faiss"].from_documents.side_effect = Exception("vector fail")
    with pytest.raises(Exception) as excinfo:
        service.add_documents(dummy_chunks)
    assert "vector fail" in str(excinfo.value)

def test_as_retriever_returns_none_if_vector_store_empty(patch_external_deps):
    service = VectorStoreService()
    service.vector_store = None
    retriever = service.as_retriever()
    assert retriever is None

def test_as_retriever_returns_retriever_with_default_kwargs(patch_external_deps):
    service = VectorStoreService()
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever
    service.vector_store = mock_vector_store
    retriever = service.as_retriever()
    mock_vector_store.as_retriever.assert_called_once()
    assert retriever == mock_retriever

def test_as_retriever_returns_retriever_with_custom_kwargs(patch_external_deps):
    service = VectorStoreService()
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever
    service.vector_store = mock_vector_store
    retriever = service.as_retriever(search_type="mmr", search_kwargs={"k": 2})
    mock_vector_store.as_retriever.assert_called_once_with(search_type="mmr", search_kwargs={"k": 2})
    assert retriever == mock_retriever

def test_as_retriever_handles_exception_and_returns_none(patch_external_deps):
    service = VectorStoreService()
    mock_vector_store = MagicMock()
    mock_vector_store.as_retriever.side_effect = Exception("retriever fail")
    service.vector_store = mock_vector_store
    retriever = service.as_retriever()
    assert retriever is None

def test_similarity_search_returns_empty_if_vector_store_empty(patch_external_deps):
    service = VectorStoreService()
    service.vector_store = None
    results = service.similarity_search("query", k=2)
    assert results == []

def test_similarity_search_returns_chunks_from_documents(patch_external_deps, dummy_metadata):
    service = VectorStoreService()
    mock_vector_store = MagicMock()
    doc1 = MagicMock()
    doc1.page_content = "doc1 text"
    doc1.metadata = dummy_metadata.model_dump()
    doc2 = MagicMock()
    doc2.page_content = "doc2 text"
    doc2.metadata = dummy_metadata.model_dump()
    mock_vector_store.similarity_search.return_value = [doc1, doc2]
    service.vector_store = mock_vector_store
    results = service.similarity_search("query", k=2)
    assert len(results) == 2
    assert all(isinstance(chunk, Chunk) for chunk in results)
    assert results[0].text == "doc1 text"
    assert results[1].text == "doc2 text"
    assert results[0].metadata == dummy_metadata

def test_similarity_search_handles_exception_and_returns_empty(patch_external_deps):
    service = VectorStoreService()
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.side_effect = Exception("search fail")
    service.vector_store = mock_vector_store
    results = service.similarity_search("query", k=2)
    assert results == []

def test_add_documents_with_empty_list_does_not_fail(patch_external_deps):
    service = VectorStoreService()
    service.add_documents([])
    # Should not initialize vector store or call add_documents
    assert service.vector_store is None
    patch_external_deps["faiss"].from_documents.assert_not_called()

def test_as_retriever_with_empty_search_kwargs_uses_default(monkeypatch, patch_external_deps):
    # This test ensures that if search_kwargs is None, settings.TOP_K is used
    class DummySettings:
        EMBEDDING_MODEL = "test-model"
        TOP_K = 42
    monkeypatch.setattr("core.config.settings", DummySettings())
    service = VectorStoreService()
    mock_vector_store = MagicMock()
    service.vector_store = mock_vector_store
    mock_vector_store.as_retriever.return_value = "retriever"
    retriever = service.as_retriever(search_kwargs=None)
    mock_vector_store.as_retriever.assert_called_once_with(search_type="similarity", search_kwargs={"k": 42})
    assert retriever == "retriever"

def test_similarity_search_with_k_zero_returns_empty_list(patch_external_deps):
    service = VectorStoreService()
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.return_value = []
    service.vector_store = mock_vector_store
    results = service.similarity_search("query", k=0)
    assert results == []
    mock_vector_store.similarity_search.assert_called_once_with("query", k=0)
