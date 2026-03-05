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

@pytest.fixture
def mock_settings(monkeypatch):
    class DummySettings:
        EMBEDDING_MODEL = "test-model"
        TOP_K = 3
    monkeypatch.setattr(vector_store_module, "settings", DummySettings())

@pytest.fixture
def dummy_metadata():
    return DocumentMetadata(source="test_source", page=1)

@pytest.fixture
def dummy_chunk(dummy_metadata):
    return Chunk(text="test text", metadata=dummy_metadata)

@pytest.fixture
def dummy_chunks(dummy_metadata):
    return [
        Chunk(text="text1", metadata=dummy_metadata),
        Chunk(text="text2", metadata=dummy_metadata),
    ]

@pytest.fixture
def mock_embeddings(monkeypatch):
    mock_embed = MagicMock(name="HuggingFaceEmbeddings")
    monkeypatch.setattr(vector_store_module, "HuggingFaceEmbeddings", MagicMock(return_value=mock_embed))
    return mock_embed

@pytest.fixture
def mock_faiss(monkeypatch):
    mock_faiss_cls = MagicMock(name="FAISS")
    monkeypatch.setattr(vector_store_module, "FAISS", mock_faiss_cls)
    return mock_faiss_cls

@pytest.fixture
def service_with_mocks(mock_settings, mock_embeddings, mock_faiss):
    return VectorStoreService()

def test_embeddings_initialization_success(mock_settings, mock_embeddings):
    service = VectorStoreService()
    assert service.embeddings is mock_embeddings.return_value

def test_embeddings_initialization_failure(mock_settings, monkeypatch):
    monkeypatch.setattr(vector_store_module, "HuggingFaceEmbeddings", MagicMock(side_effect=RuntimeError("fail")))
    with pytest.raises(RuntimeError):
        VectorStoreService()

def test_add_documents_initializes_vector_store(service_with_mocks, dummy_chunks, mock_faiss):
    service = service_with_mocks
    service.vector_store = None
    mock_vector_store = MagicMock()
    mock_faiss.from_documents.return_value = mock_vector_store

    service.add_documents(dummy_chunks)
    assert service.vector_store is mock_vector_store
    mock_faiss.from_documents.assert_called_once()
    mock_vector_store.add_documents.assert_not_called()

def test_add_documents_appends_to_existing_vector_store(service_with_mocks, dummy_chunks):
    service = service_with_mocks
    mock_vector_store = MagicMock()
    service.vector_store = mock_vector_store

    service.add_documents(dummy_chunks)
    mock_vector_store.add_documents.assert_called_once()
    assert service.vector_store is mock_vector_store

def test_add_documents_handles_exception(service_with_mocks, dummy_chunks, mock_faiss):
    service = service_with_mocks
    mock_faiss.from_documents.side_effect = Exception("fail")
    service.vector_store = None
    with pytest.raises(Exception):
        service.add_documents(dummy_chunks)

def test_as_retriever_returns_none_when_vector_store_empty(service_with_mocks):
    service = service_with_mocks
    service.vector_store = None
    retriever = service.as_retriever()
    assert retriever is None

def test_as_retriever_returns_retriever_with_default_kwargs(service_with_mocks, mock_settings):
    service = service_with_mocks
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever
    service.vector_store = mock_vector_store

    retriever = service.as_retriever()
    mock_vector_store.as_retriever.assert_called_once_with(
        search_type="similarity",
        search_kwargs={"k": mock_settings.TOP_K}
    )
    assert retriever is mock_retriever

def test_as_retriever_returns_retriever_with_custom_kwargs(service_with_mocks):
    service = service_with_mocks
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever
    service.vector_store = mock_vector_store

    retriever = service.as_retriever(search_type="mmr", search_kwargs={"k": 2, "score_threshold": 0.5})
    mock_vector_store.as_retriever.assert_called_once_with(
        search_type="mmr",
        search_kwargs={"k": 2, "score_threshold": 0.5}
    )
    assert retriever is mock_retriever

def test_as_retriever_handles_exception_and_returns_none(service_with_mocks):
    service = service_with_mocks
    mock_vector_store = MagicMock()
    mock_vector_store.as_retriever.side_effect = Exception("fail")
    service.vector_store = mock_vector_store

    retriever = service.as_retriever()
    assert retriever is None

def test_similarity_search_returns_empty_when_vector_store_empty(service_with_mocks):
    service = service_with_mocks
    service.vector_store = None
    results = service.similarity_search("query", k=2)
    assert results == []

def test_similarity_search_returns_chunks(service_with_mocks, dummy_metadata):
    service = service_with_mocks
    mock_vector_store = MagicMock()
    doc1 = MagicMock()
    doc1.page_content = "doc1"
    doc1.metadata = {"source": "test_source", "page": 1}
    doc2 = MagicMock()
    doc2.page_content = "doc2"
    doc2.metadata = {"source": "test_source", "page": 1}
    mock_vector_store.similarity_search.return_value = [doc1, doc2]
    service.vector_store = mock_vector_store

    results = service.similarity_search("query", k=2)
    assert len(results) == 2
    assert all(isinstance(chunk, Chunk) for chunk in results)
    assert results[0].text == "doc1"
    assert results[1].text == "doc2"
    assert results[0].metadata.source == "test_source"
    assert results[1].metadata.page == 1

def test_similarity_search_handles_exception_and_returns_empty(service_with_mocks):
    service = service_with_mocks
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.side_effect = Exception("fail")
    service.vector_store = mock_vector_store

    results = service.similarity_search("query", k=2)
    assert results == []

def test_reconciliation_similarity_search_and_as_retriever_equivalence(service_with_mocks, dummy_metadata):
    service = service_with_mocks
    mock_vector_store = MagicMock()
    doc = MagicMock()
    doc.page_content = "doc"
    doc.metadata = {"source": "test_source", "page": 1}
    mock_vector_store.similarity_search.return_value = [doc]
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [doc]
    mock_vector_store.as_retriever.return_value = mock_retriever
    service.vector_store = mock_vector_store

    # similarity_search path
    legacy_chunks = service.similarity_search("query", k=1)

    # as_retriever path
    retriever = service.as_retriever()
    docs = retriever.invoke("query")
    chunks = []
    for d in docs:
        metadata = DocumentMetadata(**d.metadata)
        chunks.append(Chunk(text=d.page_content, metadata=metadata))

    assert len(legacy_chunks) == len(chunks)
    for c1, c2 in zip(legacy_chunks, chunks):
        assert c1.text == c2.text
        assert c1.metadata.source == c2.metadata.source
        assert c1.metadata.page == c2.metadata.page

def test_add_documents_and_similarity_search_integration(service_with_mocks, dummy_chunks, mock_faiss):
    service = service_with_mocks
    mock_vector_store = MagicMock()
    mock_faiss.from_documents.return_value = mock_vector_store
    doc = MagicMock()
    doc.page_content = "text1"
    doc.metadata = {"source": "test_source", "page": 1}
    mock_vector_store.similarity_search.return_value = [doc]
    service.add_documents([dummy_chunks[0]])
    results = service.similarity_search("text1", k=1)
    assert len(results) == 1
    assert results[0].text == "text1"
    assert results[0].metadata.source == "test_source"
    assert results[0].metadata.page == 1

def test_add_documents_with_empty_list(service_with_mocks, mock_faiss):
    service = service_with_mocks
    service.vector_store = None
    service.add_documents([])
    # Should not raise, and should initialize vector_store with empty docs
    mock_faiss.from_documents.assert_called_once()
