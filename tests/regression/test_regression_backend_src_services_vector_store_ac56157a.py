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

class DummyMetadata:
    def model_dump(self):
        return {"source": "dummy", "page": 1}

@pytest.fixture
def dummy_chunk():
    metadata = DocumentMetadata(source="dummy", page=1)
    return Chunk(text="test content", metadata=metadata)

@pytest.fixture
def dummy_chunks():
    return [
        Chunk(text=f"content {i}", metadata=DocumentMetadata(source="src", page=i))
        for i in range(3)
    ]

@pytest.fixture
def mock_embeddings():
    return MagicMock(name="HuggingFaceEmbeddings")

@pytest.fixture
def mock_faiss():
    return MagicMock(name="FAISS")

@pytest.fixture
def patch_settings(monkeypatch):
    class DummySettings:
        EMBEDDING_MODEL = "test-model"
        TOP_K = 2
    monkeypatch.setattr(vector_store_module, "settings", DummySettings())

@pytest.fixture
def patch_logger(monkeypatch):
    monkeypatch.setattr(vector_store_module, "logger", MagicMock())

def test_init_success_creates_embeddings_and_logs(patch_settings, patch_logger):
    with patch("backend.src.services.vector_store.HuggingFaceEmbeddings") as mock_emb:
        mock_emb.return_value = MagicMock()
        service = VectorStoreService()
        assert service.embeddings is mock_emb.return_value
        assert service.vector_store is None
        vector_store_module.logger.info.assert_called_with(
            "Initialized HuggingFace embeddings with model: test-model"
        )

def test_init_failure_logs_and_raises(patch_settings, patch_logger):
    with patch("backend.src.services.vector_store.HuggingFaceEmbeddings", side_effect=Exception("fail")):
        with pytest.raises(Exception) as excinfo:
            VectorStoreService()
        assert "fail" in str(excinfo.value)
        assert vector_store_module.logger.error.call_args[0][0].startswith("Failed to initialize embeddings:")

def test_add_documents_initializes_vector_store(monkeypatch, patch_settings, patch_logger, dummy_chunks):
    mock_embeddings = MagicMock()
    mock_faiss_cls = MagicMock()
    mock_faiss_instance = MagicMock()
    mock_faiss_cls.from_documents.return_value = mock_faiss_instance

    monkeypatch.setattr(vector_store_module, "HuggingFaceEmbeddings", lambda model_name: mock_embeddings)
    monkeypatch.setattr(vector_store_module, "FAISS", mock_faiss_cls)

    service = VectorStoreService()
    service.vector_store = None

    service.add_documents(dummy_chunks)
    mock_faiss_cls.from_documents.assert_called_once()
    assert service.vector_store == mock_faiss_instance
    vector_store_module.logger.info.assert_any_call("Initializing new FAISS vector store.")
    vector_store_module.logger.info.assert_any_call(f"Added {len(dummy_chunks)} documents to vector store.")

def test_add_documents_adds_to_existing_vector_store(monkeypatch, patch_settings, patch_logger, dummy_chunks):
    mock_embeddings = MagicMock()
    mock_faiss_cls = MagicMock()
    mock_vector_store = MagicMock()
    monkeypatch.setattr(vector_store_module, "HuggingFaceEmbeddings", lambda model_name: mock_embeddings)
    monkeypatch.setattr(vector_store_module, "FAISS", mock_faiss_cls)

    service = VectorStoreService()
    service.vector_store = mock_vector_store

    service.add_documents(dummy_chunks)
    mock_vector_store.add_documents.assert_called_once()
    vector_store_module.logger.info.assert_any_call(f"Added {len(dummy_chunks)} documents to vector store.")

def test_add_documents_handles_exception(monkeypatch, patch_settings, patch_logger, dummy_chunks):
    mock_embeddings = MagicMock()
    mock_faiss_cls = MagicMock()
    mock_faiss_cls.from_documents.side_effect = Exception("vector fail")
    monkeypatch.setattr(vector_store_module, "HuggingFaceEmbeddings", lambda model_name: mock_embeddings)
    monkeypatch.setattr(vector_store_module, "FAISS", mock_faiss_cls)

    service = VectorStoreService()
    service.vector_store = None

    with pytest.raises(Exception) as excinfo:
        service.add_documents(dummy_chunks)
    assert "vector fail" in str(excinfo.value)
    assert vector_store_module.logger.error.call_args[0][0].startswith("Error adding documents to vector store:")

def test_as_retriever_returns_none_if_vector_store_empty(patch_settings, patch_logger):
    with patch("backend.src.services.vector_store.HuggingFaceEmbeddings"):
        service = VectorStoreService()
        service.vector_store = None
        retriever = service.as_retriever()
        assert retriever is None
        vector_store_module.logger.warning.assert_called_with("Vector store is empty, cannot create retriever.")

def test_as_retriever_returns_retriever(monkeypatch, patch_settings, patch_logger):
    mock_embeddings = MagicMock()
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever
    monkeypatch.setattr(vector_store_module, "HuggingFaceEmbeddings", lambda model_name: mock_embeddings)

    service = VectorStoreService()
    service.vector_store = mock_vector_store

    retriever = service.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    mock_vector_store.as_retriever.assert_called_with(search_type="similarity", search_kwargs={"k": 5})
    assert retriever == mock_retriever
    vector_store_module.logger.debug.assert_called_with(
        "Created retriever with search_type=similarity, kwargs={'k': 5}"
    )

def test_as_retriever_uses_default_search_kwargs(monkeypatch, patch_settings, patch_logger):
    mock_embeddings = MagicMock()
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever
    monkeypatch.setattr(vector_store_module, "HuggingFaceEmbeddings", lambda model_name: mock_embeddings)

    service = VectorStoreService()
    service.vector_store = mock_vector_store

    retriever = service.as_retriever()
    mock_vector_store.as_retriever.assert_called_with(search_type="similarity", search_kwargs={"k": 2})
    assert retriever == mock_retriever

def test_as_retriever_handles_exception(monkeypatch, patch_settings, patch_logger):
    mock_embeddings = MagicMock()
    mock_vector_store = MagicMock()
    mock_vector_store.as_retriever.side_effect = Exception("retriever fail")
    monkeypatch.setattr(vector_store_module, "HuggingFaceEmbeddings", lambda model_name: mock_embeddings)

    service = VectorStoreService()
    service.vector_store = mock_vector_store

    retriever = service.as_retriever()
    assert retriever is None
    assert vector_store_module.logger.error.call_args[0][0].startswith("Error creating retriever:")

def test_similarity_search_returns_empty_if_vector_store_none(patch_settings, patch_logger):
    with patch("backend.src.services.vector_store.HuggingFaceEmbeddings"):
        service = VectorStoreService()
        service.vector_store = None
        result = service.similarity_search("query", k=2)
        assert result == []
        vector_store_module.logger.warning.assert_called_with("Vector store is empty, returning no results.")

def test_similarity_search_returns_chunks(monkeypatch, patch_settings, patch_logger):
    mock_embeddings = MagicMock()
    mock_vector_store = MagicMock()
    doc1 = MagicMock()
    doc1.page_content = "doc1"
    doc1.metadata = {"source": "src", "page": 1}
    doc2 = MagicMock()
    doc2.page_content = "doc2"
    doc2.metadata = {"source": "src", "page": 2}
    mock_vector_store.similarity_search.return_value = [doc1, doc2]
    monkeypatch.setattr(vector_store_module, "HuggingFaceEmbeddings", lambda model_name: mock_embeddings)

    service = VectorStoreService()
    service.vector_store = mock_vector_store

    result = service.similarity_search("query", k=2)
    assert len(result) == 2
    assert all(isinstance(chunk, Chunk) for chunk in result)
    assert result[0].text == "doc1"
    assert result[1].text == "doc2"

def test_similarity_search_handles_exception(monkeypatch, patch_settings, patch_logger):
    mock_embeddings = MagicMock()
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.side_effect = Exception("search fail")
    monkeypatch.setattr(vector_store_module, "HuggingFaceEmbeddings", lambda model_name: mock_embeddings)

    service = VectorStoreService()
    service.vector_store = mock_vector_store

    result = service.similarity_search("query", k=2)
    assert result == []
    assert vector_store_module.logger.error.call_args[0][0].startswith("Error during similarity search:")

def test_add_documents_with_empty_list(monkeypatch, patch_settings, patch_logger):
    mock_embeddings = MagicMock()
    mock_faiss_cls = MagicMock()
    mock_faiss_instance = MagicMock()
    mock_faiss_cls.from_documents.return_value = mock_faiss_instance
    monkeypatch.setattr(vector_store_module, "HuggingFaceEmbeddings", lambda model_name: mock_embeddings)
    monkeypatch.setattr(vector_store_module, "FAISS", mock_faiss_cls)

    service = VectorStoreService()
    service.vector_store = None

    service.add_documents([])
    mock_faiss_cls.from_documents.assert_called_once_with([], mock_embeddings)
    vector_store_module.logger.info.assert_any_call("Initializing new FAISS vector store.")
    vector_store_module.logger.info.assert_any_call("Added 0 documents to vector store.")

def test_similarity_search_with_zero_k(monkeypatch, patch_settings, patch_logger):
    mock_embeddings = MagicMock()
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.return_value = []
    monkeypatch.setattr(vector_store_module, "HuggingFaceEmbeddings", lambda model_name: mock_embeddings)

    service = VectorStoreService()
    service.vector_store = mock_vector_store

    result = service.similarity_search("query", k=0)
    assert result == []
    mock_vector_store.similarity_search.assert_called_with("query", k=0)
