# source_hash: 5d518cf45d49b218
import pytest
from unittest.mock import patch, MagicMock, create_autospec
from backend.src.services.vector_store import VectorStoreService
from models.schemas import Chunk, DocumentMetadata
from langchain_core.documents import Document

@pytest.fixture
def mock_settings(monkeypatch):
    class DummySettings:
        EMBEDDING_MODEL = "test-embedding-model"
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
def patch_langchain(monkeypatch, mock_embeddings, mock_faiss_cls):
    monkeypatch.setattr("backend.src.services.vector_store.HuggingFaceEmbeddings", lambda model_name: mock_embeddings)
    monkeypatch.setattr("backend.src.services.vector_store.FAISS", mock_faiss_cls)

@pytest.fixture
def service(mock_settings, patch_langchain):
    return VectorStoreService()

def test_init_success_logs_and_sets_embeddings(service, mock_embeddings):
    assert service.embeddings is mock_embeddings
    assert service.vector_store is None

def test_init_failure_logs_and_raises(monkeypatch, mock_settings):
    def raise_exc(model_name):
        raise RuntimeError("fail")
    monkeypatch.setattr("backend.src.services.vector_store.HuggingFaceEmbeddings", raise_exc)
    with pytest.raises(RuntimeError):
        VectorStoreService()

def test_add_documents_initializes_faiss_when_none(service, dummy_chunks, mock_faiss_cls):
    mock_faiss_cls.from_documents.return_value = "mock_vector_store"
    service.vector_store = None
    service.add_documents(dummy_chunks)
    assert service.vector_store == "mock_vector_store"
    assert mock_faiss_cls.from_documents.called
    docs_arg = mock_faiss_cls.from_documents.call_args[0][0]
    assert all(isinstance(doc, Document) for doc in docs_arg)
    assert docs_arg[0].page_content == "text 1"

def test_add_documents_adds_to_existing_vector_store(service, dummy_chunks, mock_faiss_cls):
    mock_vector_store = MagicMock()
    service.vector_store = mock_vector_store
    service.add_documents(dummy_chunks)
    assert mock_vector_store.add_documents.called
    docs_arg = mock_vector_store.add_documents.call_args[0][0]
    assert all(isinstance(doc, Document) for doc in docs_arg)
    assert docs_arg[1].page_content == "text 2"

def test_add_documents_handles_exception(service, dummy_chunks, mock_faiss_cls):
    mock_faiss_cls.from_documents.side_effect = Exception("fail add")
    service.vector_store = None
    with pytest.raises(Exception):
        service.add_documents(dummy_chunks)

def test_as_retriever_returns_none_if_vector_store_empty(service):
    service.vector_store = None
    retriever = service.as_retriever()
    assert retriever is None

def test_as_retriever_returns_retriever_with_defaults(service, mock_faiss_cls, mock_settings):
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever
    service.vector_store = mock_vector_store
    result = service.as_retriever()
    assert result == mock_retriever
    mock_vector_store.as_retriever.assert_called_once()
    args, kwargs = mock_vector_store.as_retriever.call_args
    assert kwargs["search_type"] == "similarity"
    assert kwargs["search_kwargs"]["k"] == mock_settings.TOP_K

def test_as_retriever_accepts_custom_search_type_and_kwargs(service):
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever
    service.vector_store = mock_vector_store
    result = service.as_retriever(search_type="mmr", search_kwargs={"k": 2, "score_threshold": 0.5})
    assert result == mock_retriever
    mock_vector_store.as_retriever.assert_called_once_with(
        search_type="mmr", search_kwargs={"k": 2, "score_threshold": 0.5}
    )

def test_as_retriever_handles_exception_and_returns_none(service):
    mock_vector_store = MagicMock()
    mock_vector_store.as_retriever.side_effect = Exception("fail retriever")
    service.vector_store = mock_vector_store
    result = service.as_retriever()
    assert result is None

def test_similarity_search_returns_empty_if_vector_store_none(service):
    service.vector_store = None
    result = service.similarity_search("query", k=2)
    assert result == []

def test_similarity_search_returns_chunks(service, dummy_metadata):
    mock_vector_store = MagicMock()
    doc1 = Document(page_content="doc1", metadata=dummy_metadata.model_dump())
    doc2 = Document(page_content="doc2", metadata=dummy_metadata.model_dump())
    mock_vector_store.similarity_search.return_value = [doc1, doc2]
    service.vector_store = mock_vector_store
    result = service.similarity_search("query", k=2)
    assert len(result) == 2
    assert all(isinstance(chunk, Chunk) for chunk in result)
    assert result[0].text == "doc1"
    assert result[1].metadata.source == dummy_metadata.source

def test_similarity_search_handles_exception_and_returns_empty(service):
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.side_effect = Exception("fail search")
    service.vector_store = mock_vector_store
    result = service.similarity_search("query", k=2)
    assert result == []

def test_add_documents_with_empty_list_does_not_fail(service, mock_faiss_cls):
    service.vector_store = None
    service.add_documents([])
    # Should not raise, and FAISS.from_documents should be called with empty list
    assert mock_faiss_cls.from_documents.called
    docs_arg = mock_faiss_cls.from_documents.call_args[0][0]
    assert docs_arg == []

def test_similarity_search_with_zero_k_returns_empty(service):
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.return_value = []
    service.vector_store = mock_vector_store
    result = service.similarity_search("query", k=0)
    assert result == []

def test_as_retriever_with_empty_kwargs_uses_default_k(service, mock_settings):
    mock_vector_store = MagicMock()
    mock_vector_store.as_retriever.return_value = "retriever"
    service.vector_store = mock_vector_store
    result = service.as_retriever(search_kwargs={})
    assert result == "retriever"
    mock_vector_store.as_retriever.assert_called_once()
    # Should use empty dict, not default TOP_K, since search_kwargs is provided
    args, kwargs = mock_vector_store.as_retriever.call_args
    assert kwargs["search_kwargs"] == {}

def test_add_documents_preserves_metadata(service, dummy_metadata, mock_faiss_cls):
    service.vector_store = None
    chunk = Chunk(text="abc", metadata=dummy_metadata)
    service.add_documents([chunk])
    docs_arg = mock_faiss_cls.from_documents.call_args[0][0]
    assert docs_arg[0].metadata == dummy_metadata.model_dump()
