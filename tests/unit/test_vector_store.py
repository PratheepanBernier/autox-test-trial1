import pytest
from unittest.mock import patch, MagicMock, create_autospec
from backend.src.services.vector_store import VectorStoreService
from models.schemas import Chunk, DocumentMetadata
from langchain_core.documents import Document

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
def service(mock_embeddings, mock_settings):
    return VectorStoreService()

def test_init_success_logs_and_sets_embeddings(mock_embeddings, mock_settings, caplog):
    caplog.set_level("INFO")
    service = VectorStoreService()
    assert hasattr(service, "embeddings")
    assert "Initialized HuggingFace embeddings with model" in caplog.text

def test_init_failure_logs_and_raises(monkeypatch, caplog, mock_settings):
    caplog.set_level("ERROR")
    def raise_exc(*a, **kw): raise RuntimeError("fail")
    monkeypatch.setattr("backend.src.services.vector_store.HuggingFaceEmbeddings", raise_exc)
    with pytest.raises(RuntimeError):
        VectorStoreService()
    assert "Failed to initialize embeddings" in caplog.text

def test_add_documents_initializes_faiss_and_adds_documents(service, mock_faiss, dummy_chunks):
    mock_vector_store = MagicMock()
    mock_faiss.from_documents.return_value = mock_vector_store
    service.vector_store = None
    service.embeddings = MagicMock()
    service.add_documents(dummy_chunks)
    mock_faiss.from_documents.assert_called_once()
    assert service.vector_store == mock_vector_store

def test_add_documents_adds_to_existing_vector_store(service, dummy_chunks):
    mock_vector_store = MagicMock()
    service.vector_store = mock_vector_store
    service.embeddings = MagicMock()
    service.add_documents(dummy_chunks)
    mock_vector_store.add_documents.assert_called_once()
    # Should not call FAISS.from_documents
    assert not hasattr(service.vector_store, "from_documents")

def test_add_documents_handles_exception_and_logs(service, dummy_chunks, caplog):
    caplog.set_level("ERROR")
    service.vector_store = None
    service.embeddings = MagicMock()
    with patch("backend.src.services.vector_store.FAISS.from_documents", side_effect=Exception("fail")):
        with pytest.raises(Exception):
            service.add_documents(dummy_chunks)
    assert "Error adding documents to vector store" in caplog.text

def test_as_retriever_returns_none_if_vector_store_is_none(service, caplog):
    caplog.set_level("WARNING")
    service.vector_store = None
    retriever = service.as_retriever()
    assert retriever is None
    assert "Vector store is empty, cannot create retriever." in caplog.text

def test_as_retriever_uses_default_kwargs(service):
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever
    service.vector_store = mock_vector_store
    result = service.as_retriever()
    mock_vector_store.as_retriever.assert_called_once()
    assert result == mock_retriever

def test_as_retriever_uses_custom_kwargs_and_type(service):
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever
    service.vector_store = mock_vector_store
    kwargs = {"k": 7, "score_threshold": 0.5}
    result = service.as_retriever(search_type="mmr", search_kwargs=kwargs)
    mock_vector_store.as_retriever.assert_called_once_with(search_type="mmr", search_kwargs=kwargs)
    assert result == mock_retriever

def test_as_retriever_handles_exception_and_logs(service, caplog):
    caplog.set_level("ERROR")
    mock_vector_store = MagicMock()
    mock_vector_store.as_retriever.side_effect = Exception("fail")
    service.vector_store = mock_vector_store
    result = service.as_retriever()
    assert result is None
    assert "Error creating retriever" in caplog.text

def test_similarity_search_returns_empty_if_vector_store_none(service, caplog):
    caplog.set_level("WARNING")
    service.vector_store = None
    result = service.similarity_search("query", k=2)
    assert result == []
    assert "Vector store is empty, returning no results." in caplog.text

def test_similarity_search_returns_chunks(service, dummy_metadata):
    mock_vector_store = MagicMock()
    doc1 = Document(page_content="abc", metadata=dummy_metadata.model_dump())
    doc2 = Document(page_content="def", metadata=dummy_metadata.model_dump())
    mock_vector_store.similarity_search.return_value = [doc1, doc2]
    service.vector_store = mock_vector_store
    result = service.similarity_search("query", k=2)
    assert isinstance(result, list)
    assert all(isinstance(chunk, Chunk) for chunk in result)
    assert result[0].text == "abc"
    assert result[1].text == "def"

def test_similarity_search_handles_exception_and_logs(service, caplog):
    caplog.set_level("ERROR")
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.side_effect = Exception("fail")
    service.vector_store = mock_vector_store
    result = service.similarity_search("query", k=2)
    assert result == []
    assert "Error during similarity search" in caplog.text

def test_add_documents_empty_list(service, mock_faiss):
    service.vector_store = None
    service.embeddings = MagicMock()
    service.add_documents([])
    mock_faiss.from_documents.assert_called_once_with([], service.embeddings)

def test_similarity_search_zero_k(service, dummy_metadata):
    mock_vector_store = MagicMock()
    doc = Document(page_content="abc", metadata=dummy_metadata.model_dump())
    mock_vector_store.similarity_search.return_value = [doc]
    service.vector_store = mock_vector_store
    result = service.similarity_search("query", k=0)
    # Should still call similarity_search with k=0
    mock_vector_store.similarity_search.assert_called_once_with("query", k=0)
    assert isinstance(result, list)

def test_as_retriever_with_none_kwargs(service):
    mock_vector_store = MagicMock()
    mock_vector_store.as_retriever.return_value = "retriever"
    service.vector_store = mock_vector_store
    result = service.as_retriever(search_kwargs=None)
    assert result == "retriever"
    mock_vector_store.as_retriever.assert_called_once()
