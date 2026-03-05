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
from backend.src.services.vector_store import VectorStoreService
from backend.src.services.vector_store import vector_store_service
from backend.src.services.vector_store import Document
from backend.src.services.vector_store import logger
from models.schemas import Chunk, DocumentMetadata

class DummyMetadata:
    def model_dump(self):
        return {"source": "dummy", "page": 1}

@pytest.fixture
def dummy_chunk():
    meta = DocumentMetadata(source="dummy", page=1)
    return Chunk(text="test text", metadata=meta)

@pytest.fixture
def dummy_chunks():
    meta1 = DocumentMetadata(source="dummy1", page=1)
    meta2 = DocumentMetadata(source="dummy2", page=2)
    return [
        Chunk(text="text1", metadata=meta1),
        Chunk(text="text2", metadata=meta2)
    ]

@pytest.fixture
def mock_embeddings():
    with patch("backend.src.services.vector_store.HuggingFaceEmbeddings") as emb:
        yield emb

@pytest.fixture
def mock_faiss():
    with patch("backend.src.services.vector_store.FAISS") as faiss:
        yield faiss

@pytest.fixture
def mock_settings(monkeypatch):
    class DummySettings:
        EMBEDDING_MODEL = "test-model"
        TOP_K = 3
    monkeypatch.setattr("backend.src.services.vector_store.settings", DummySettings())

def test_init_success_logs_and_sets_embeddings(mock_settings, mock_embeddings):
    with patch.object(logger, "info") as mock_log:
        service = VectorStoreService()
        assert hasattr(service, "embeddings")
        mock_log.assert_any_call("Initialized HuggingFace embeddings with model: test-model")

def test_init_failure_logs_and_raises(mock_settings):
    with patch("backend.src.services.vector_store.HuggingFaceEmbeddings", side_effect=RuntimeError("fail")):
        with patch.object(logger, "error") as mock_log:
            with pytest.raises(RuntimeError):
                VectorStoreService()
            assert mock_log.call_args[0][0].startswith("Failed to initialize embeddings:")

def test_add_documents_initializes_faiss_and_adds_documents(mock_settings, mock_embeddings, mock_faiss, dummy_chunks):
    service = VectorStoreService()
    service.vector_store = None
    mock_faiss.from_documents.return_value = MagicMock()
    with patch.object(logger, "info") as mock_log:
        service.add_documents(dummy_chunks)
        mock_faiss.from_documents.assert_called_once()
        assert service.vector_store is not None
        mock_log.assert_any_call("Initializing new FAISS vector store.")
        mock_log.assert_any_call(f"Added {len(dummy_chunks)} documents to vector store.")

def test_add_documents_adds_to_existing_vector_store(mock_settings, mock_embeddings, mock_faiss, dummy_chunks):
    service = VectorStoreService()
    mock_vs = MagicMock()
    service.vector_store = mock_vs
    with patch.object(logger, "info") as mock_log:
        service.add_documents(dummy_chunks)
        mock_vs.add_documents.assert_called_once()
        mock_log.assert_any_call(f"Added {len(dummy_chunks)} documents to vector store.")

def test_add_documents_handles_exception_and_logs_error(mock_settings, mock_embeddings, mock_faiss, dummy_chunks):
    service = VectorStoreService()
    service.vector_store = None
    mock_faiss.from_documents.side_effect = Exception("fail add")
    with patch.object(logger, "error") as mock_log:
        with pytest.raises(Exception):
            service.add_documents(dummy_chunks)
        assert "Error adding documents to vector store:" in mock_log.call_args[0][0]

def test_as_retriever_returns_none_if_vector_store_empty(mock_settings, mock_embeddings):
    service = VectorStoreService()
    service.vector_store = None
    with patch.object(logger, "warning") as mock_log:
        retriever = service.as_retriever()
        assert retriever is None
        mock_log.assert_called_with("Vector store is empty, cannot create retriever.")

def test_as_retriever_returns_retriever_with_defaults(mock_settings, mock_embeddings):
    service = VectorStoreService()
    mock_vs = MagicMock()
    mock_retriever = MagicMock()
    mock_vs.as_retriever.return_value = mock_retriever
    service.vector_store = mock_vs
    with patch.object(logger, "debug") as mock_log:
        retriever = service.as_retriever()
        assert retriever == mock_retriever
        mock_vs.as_retriever.assert_called_once()
        assert "Created retriever" in mock_log.call_args[0][0]

def test_as_retriever_returns_retriever_with_custom_kwargs(mock_settings, mock_embeddings):
    service = VectorStoreService()
    mock_vs = MagicMock()
    mock_retriever = MagicMock()
    mock_vs.as_retriever.return_value = mock_retriever
    service.vector_store = mock_vs
    search_kwargs = {"k": 7, "score_threshold": 0.5}
    retriever = service.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
    mock_vs.as_retriever.assert_called_once_with(search_type="mmr", search_kwargs=search_kwargs)
    assert retriever == mock_retriever

def test_as_retriever_handles_exception_and_logs_error(mock_settings, mock_embeddings):
    service = VectorStoreService()
    mock_vs = MagicMock()
    mock_vs.as_retriever.side_effect = Exception("fail retriever")
    service.vector_store = mock_vs
    with patch.object(logger, "error") as mock_log:
        retriever = service.as_retriever()
        assert retriever is None
        assert "Error creating retriever:" in mock_log.call_args[0][0]

def test_similarity_search_returns_empty_if_vector_store_none(mock_settings, mock_embeddings):
    service = VectorStoreService()
    service.vector_store = None
    with patch.object(logger, "warning") as mock_log:
        result = service.similarity_search("query", k=2)
        assert result == []
        mock_log.assert_called_with("Vector store is empty, returning no results.")

def test_similarity_search_returns_chunks_from_docs(mock_settings, mock_embeddings):
    service = VectorStoreService()
    mock_vs = MagicMock()
    doc1 = MagicMock()
    doc1.page_content = "doc1"
    doc1.metadata = {"source": "src1", "page": 1}
    doc2 = MagicMock()
    doc2.page_content = "doc2"
    doc2.metadata = {"source": "src2", "page": 2}
    mock_vs.similarity_search.return_value = [doc1, doc2]
    service.vector_store = mock_vs
    result = service.similarity_search("query", k=2)
    assert len(result) == 2
    assert isinstance(result[0], Chunk)
    assert result[0].text == "doc1"
    assert result[0].metadata.source == "src1"
    assert result[1].text == "doc2"
    assert result[1].metadata.page == 2

def test_similarity_search_handles_exception_and_logs_error(mock_settings, mock_embeddings):
    service = VectorStoreService()
    mock_vs = MagicMock()
    mock_vs.similarity_search.side_effect = Exception("fail search")
    service.vector_store = mock_vs
    with patch.object(logger, "error") as mock_log:
        result = service.similarity_search("query", k=2)
        assert result == []
        assert "Error during similarity search:" in mock_log.call_args[0][0]

def test_similarity_search_with_zero_k_returns_empty_list(mock_settings, mock_embeddings):
    service = VectorStoreService()
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = []
    service.vector_store = mock_vs
    result = service.similarity_search("query", k=0)
    assert result == []

def test_add_documents_with_empty_chunks_initializes_vector_store(mock_settings, mock_embeddings, mock_faiss):
    service = VectorStoreService()
    service.vector_store = None
    mock_faiss.from_documents.return_value = MagicMock()
    service.add_documents([])
    mock_faiss.from_documents.assert_called_once_with([], service.embeddings)

def test_as_retriever_with_empty_search_kwargs_uses_default_top_k(mock_settings, mock_embeddings):
    service = VectorStoreService()
    mock_vs = MagicMock()
    service.vector_store = mock_vs
    service.as_retriever(search_kwargs={})
    mock_vs.as_retriever.assert_called_once_with(search_type="similarity", search_kwargs={})

def test_vector_store_service_singleton_is_instance():
    from backend.src.services.vector_store import vector_store_service, VectorStoreService
    assert isinstance(vector_store_service, VectorStoreService)
