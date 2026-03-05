# source_hash: 5d518cf45d49b218
# import_target: backend.src.services.vector_store
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import pytest
from unittest.mock import patch, MagicMock, create_autospec
from backend.src.services import vector_store as vector_store_module
from backend.src.services.vector_store import VectorStoreService
from backend.src.models.schemas import Chunk, DocumentMetadata

@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    class DummySettings:
        EMBEDDING_MODEL = "test-embedding-model"
        TOP_K = 3
    monkeypatch.setattr(vector_store_module, "settings", DummySettings())
    yield

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
        Chunk(text="text 3", metadata=dummy_metadata),
    ]

@pytest.fixture
def mock_embeddings():
    with patch("backend.src.services.vector_store.HuggingFaceEmbeddings") as mock_cls:
        yield mock_cls

@pytest.fixture
def mock_faiss():
    with patch("backend.src.services.vector_store.FAISS") as mock_cls:
        yield mock_cls

@pytest.fixture
def service_with_mocks(mock_embeddings, mock_faiss):
    return VectorStoreService()

def test_init_success_logs_and_sets_embeddings(mock_embeddings):
    with patch("backend.src.services.vector_store.logger") as mock_logger:
        service = VectorStoreService()
        assert hasattr(service, "embeddings")
        mock_logger.info.assert_any_call("Initialized HuggingFace embeddings with model: test-embedding-model")

def test_init_failure_logs_and_raises(monkeypatch):
    def raise_exc(*a, **kw):
        raise RuntimeError("fail!")
    monkeypatch.setattr(vector_store_module, "HuggingFaceEmbeddings", raise_exc)
    with patch("backend.src.services.vector_store.logger") as mock_logger:
        with pytest.raises(RuntimeError):
            VectorStoreService()
        assert mock_logger.error.call_count > 0

def test_add_documents_initializes_faiss_and_adds_documents(service_with_mocks, dummy_chunks, mock_faiss):
    mock_faiss.from_documents.return_value = MagicMock()
    service = service_with_mocks
    service.vector_store = None
    service.embeddings = MagicMock()
    with patch("backend.src.services.vector_store.logger") as mock_logger:
        service.add_documents(dummy_chunks)
        mock_faiss.from_documents.assert_called_once()
        assert service.vector_store is not None
        mock_logger.info.assert_any_call("Initializing new FAISS vector store.")
        mock_logger.info.assert_any_call(f"Added {len(dummy_chunks)} documents to vector store.")

def test_add_documents_adds_to_existing_vector_store(service_with_mocks, dummy_chunks):
    mock_vector_store = MagicMock()
    service = service_with_mocks
    service.vector_store = mock_vector_store
    service.embeddings = MagicMock()
    with patch("backend.src.services.vector_store.logger") as mock_logger:
        service.add_documents(dummy_chunks)
        mock_vector_store.add_documents.assert_called_once()
        mock_logger.info.assert_any_call(f"Added {len(dummy_chunks)} documents to vector store.")

def test_add_documents_handles_exception(service_with_mocks, dummy_chunks):
    service = service_with_mocks
    service.embeddings = MagicMock()
    service.vector_store = None
    with patch("backend.src.services.vector_store.FAISS.from_documents", side_effect=Exception("fail!")), \
         patch("backend.src.services.vector_store.logger") as mock_logger:
        with pytest.raises(Exception):
            service.add_documents(dummy_chunks)
        assert mock_logger.error.call_count > 0

def test_as_retriever_returns_none_if_vector_store_empty(service_with_mocks):
    service = service_with_mocks
    service.vector_store = None
    with patch("backend.src.services.vector_store.logger") as mock_logger:
        retriever = service.as_retriever()
        assert retriever is None
        mock_logger.warning.assert_any_call("Vector store is empty, cannot create retriever.")

def test_as_retriever_returns_retriever_with_defaults(service_with_mocks):
    mock_retriever = MagicMock()
    mock_vector_store = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever
    service = service_with_mocks
    service.vector_store = mock_vector_store
    with patch("backend.src.services.vector_store.logger") as mock_logger:
        retriever = service.as_retriever()
        assert retriever is mock_retriever
        mock_vector_store.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

def test_as_retriever_returns_retriever_with_custom_args(service_with_mocks):
    mock_retriever = MagicMock()
    mock_vector_store = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever
    service = service_with_mocks
    service.vector_store = mock_vector_store
    retriever = service.as_retriever(search_type="mmr", search_kwargs={"k": 2, "foo": "bar"})
    assert retriever is mock_retriever
    mock_vector_store.as_retriever.assert_called_once_with(
        search_type="mmr",
        search_kwargs={"k": 2, "foo": "bar"}
    )

def test_as_retriever_handles_exception_and_returns_none(service_with_mocks):
    mock_vector_store = MagicMock()
    mock_vector_store.as_retriever.side_effect = Exception("fail!")
    service = service_with_mocks
    service.vector_store = mock_vector_store
    with patch("backend.src.services.vector_store.logger") as mock_logger:
        retriever = service.as_retriever()
        assert retriever is None
        assert mock_logger.error.call_count > 0

def test_similarity_search_returns_empty_if_vector_store_none(service_with_mocks):
    service = service_with_mocks
    service.vector_store = None
    with patch("backend.src.services.vector_store.logger") as mock_logger:
        result = service.similarity_search("query", k=2)
        assert result == []
        mock_logger.warning.assert_any_call("Vector store is empty, returning no results.")

def test_similarity_search_returns_chunks(service_with_mocks, dummy_metadata):
    mock_doc1 = MagicMock()
    mock_doc1.page_content = "doc1"
    mock_doc1.metadata = {"source": "test_source", "page": 1}
    mock_doc2 = MagicMock()
    mock_doc2.page_content = "doc2"
    mock_doc2.metadata = {"source": "test_source", "page": 1}
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.return_value = [mock_doc1, mock_doc2]
    service = service_with_mocks
    service.vector_store = mock_vector_store
    result = service.similarity_search("query", k=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(chunk, Chunk) for chunk in result)
    assert result[0].text == "doc1"
    assert result[1].text == "doc2"
    assert result[0].metadata.source == "test_source"
    assert result[0].metadata.page == 1

def test_similarity_search_handles_exception_and_returns_empty(service_with_mocks):
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.side_effect = Exception("fail!")
    service = service_with_mocks
    service.vector_store = mock_vector_store
    with patch("backend.src.services.vector_store.logger") as mock_logger:
        result = service.similarity_search("query", k=2)
        assert result == []
        assert mock_logger.error.call_count > 0

def test_add_documents_with_empty_list(service_with_mocks):
    service = service_with_mocks
    service.vector_store = None
    service.embeddings = MagicMock()
    with patch("backend.src.services.vector_store.FAISS.from_documents") as mock_from_docs, \
         patch("backend.src.services.vector_store.logger") as mock_logger:
        service.add_documents([])
        mock_from_docs.assert_called_once_with([], service.embeddings)
        mock_logger.info.assert_any_call("Initializing new FAISS vector store.")
        mock_logger.info.assert_any_call("Added 0 documents to vector store.")

def test_similarity_search_with_zero_k_returns_empty_list(service_with_mocks):
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.return_value = []
    service = service_with_mocks
    service.vector_store = mock_vector_store
    result = service.similarity_search("query", k=0)
    assert result == []

def test_as_retriever_with_zero_k(service_with_mocks):
    mock_retriever = MagicMock()
    mock_vector_store = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever
    service = service_with_mocks
    service.vector_store = mock_vector_store
    retriever = service.as_retriever(search_kwargs={"k": 0})
    assert retriever is mock_retriever
    mock_vector_store.as_retriever.assert_called_once_with(
        search_type="similarity",
        search_kwargs={"k": 0}
    )
