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

@pytest.fixture
def mock_settings(monkeypatch):
    class DummySettings:
        EMBEDDING_MODEL = "test-model"
        TOP_K = 3
    monkeypatch.setattr("backend.src.services.vector_store.settings", DummySettings)

@pytest.fixture
def mock_embeddings():
    with patch("backend.src.services.vector_store.HuggingFaceEmbeddings") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_faiss():
    with patch("backend.src.services.vector_store.FAISS") as mock_cls:
        yield mock_cls

@pytest.fixture
def mock_logger():
    with patch("backend.src.services.vector_store.logger") as mock_logger:
        yield mock_logger

@pytest.fixture
def mock_document():
    with patch("backend.src.services.vector_store.Document") as mock_doc:
        yield mock_doc

@pytest.fixture
def mock_chunk_and_metadata():
    # Simulate Chunk and DocumentMetadata from models.schemas
    class DummyMetadata:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        def model_dump(self):
            return {"foo": "bar"}
    class DummyChunk:
        def __init__(self, text, metadata):
            self.text = text
            self.metadata = metadata
    return DummyChunk, DummyMetadata

def test_init_success(mock_settings, mock_embeddings, mock_logger):
    service = VectorStoreService()
    assert service.embeddings is mock_embeddings
    assert service.vector_store is None
    mock_logger.info.assert_called_with("Initialized HuggingFace embeddings with model: test-model")

def test_init_failure_logs_and_raises(mock_settings, mock_logger):
    with patch("backend.src.services.vector_store.HuggingFaceEmbeddings", side_effect=RuntimeError("fail")):
        with pytest.raises(RuntimeError):
            VectorStoreService()
    assert mock_logger.error.call_count == 1
    assert "Failed to initialize embeddings" in str(mock_logger.error.call_args[0][0])

def test_add_documents_initializes_faiss(mock_settings, mock_embeddings, mock_faiss, mock_logger, mock_document, mock_chunk_and_metadata):
    DummyChunk, DummyMetadata = mock_chunk_and_metadata
    service = VectorStoreService()
    chunk = DummyChunk("text1", DummyMetadata(foo="bar"))
    mock_doc_instance = MagicMock()
    mock_document.return_value = mock_doc_instance
    mock_faiss.from_documents.return_value = "FAISS_INSTANCE"
    service.add_documents([chunk])
    mock_document.assert_called_once_with(page_content="text1", metadata={"foo": "bar"})
    mock_faiss.from_documents.assert_called_once_with([mock_doc_instance], mock_embeddings)
    assert service.vector_store == "FAISS_INSTANCE"
    mock_logger.info.assert_any_call("Initializing new FAISS vector store.")
    mock_logger.info.assert_any_call("Added 1 documents to vector store.")

def test_add_documents_adds_to_existing_store(mock_settings, mock_embeddings, mock_faiss, mock_logger, mock_document, mock_chunk_and_metadata):
    DummyChunk, DummyMetadata = mock_chunk_and_metadata
    service = VectorStoreService()
    service.vector_store = MagicMock()
    chunk = DummyChunk("text2", DummyMetadata(foo="baz"))
    mock_doc_instance = MagicMock()
    mock_document.return_value = mock_doc_instance
    service.add_documents([chunk])
    service.vector_store.add_documents.assert_called_once_with([mock_doc_instance])
    mock_logger.info.assert_any_call("Added 1 documents to vector store.")

def test_add_documents_handles_exception(mock_settings, mock_embeddings, mock_faiss, mock_logger, mock_document, mock_chunk_and_metadata):
    DummyChunk, DummyMetadata = mock_chunk_and_metadata
    service = VectorStoreService()
    mock_document.side_effect = Exception("doc fail")
    chunk = DummyChunk("text3", DummyMetadata(foo="baz"))
    with pytest.raises(Exception):
        service.add_documents([chunk])
    assert mock_logger.error.call_count == 1
    assert "Error adding documents to vector store" in str(mock_logger.error.call_args[0][0])

def test_as_retriever_returns_none_if_empty(mock_settings, mock_embeddings, mock_logger):
    service = VectorStoreService()
    service.vector_store = None
    result = service.as_retriever()
    assert result is None
    mock_logger.warning.assert_called_with("Vector store is empty, cannot create retriever.")

def test_as_retriever_returns_retriever_with_defaults(mock_settings, mock_embeddings, mock_logger):
    service = VectorStoreService()
    mock_vs = MagicMock()
    retriever = MagicMock()
    mock_vs.as_retriever.return_value = retriever
    service.vector_store = mock_vs
    result = service.as_retriever()
    assert result == retriever
    mock_vs.as_retriever.assert_called_once_with(search_type="similarity", search_kwargs={"k": 3})

def test_as_retriever_with_custom_args(mock_settings, mock_embeddings, mock_logger):
    service = VectorStoreService()
    mock_vs = MagicMock()
    retriever = MagicMock()
    mock_vs.as_retriever.return_value = retriever
    service.vector_store = mock_vs
    result = service.as_retriever(search_type="mmr", search_kwargs={"k": 2, "foo": "bar"})
    assert result == retriever
    mock_vs.as_retriever.assert_called_once_with(search_type="mmr", search_kwargs={"k": 2, "foo": "bar"})

def test_as_retriever_handles_exception_and_logs(mock_settings, mock_embeddings, mock_logger):
    service = VectorStoreService()
    mock_vs = MagicMock()
    mock_vs.as_retriever.side_effect = Exception("fail retriever")
    service.vector_store = mock_vs
    result = service.as_retriever()
    assert result is None
    assert mock_logger.error.call_count == 1
    assert "Error creating retriever" in str(mock_logger.error.call_args[0][0])

def test_similarity_search_returns_empty_if_no_store(mock_settings, mock_embeddings, mock_logger):
    service = VectorStoreService()
    service.vector_store = None
    result = service.similarity_search("query", k=2)
    assert result == []
    mock_logger.warning.assert_called_with("Vector store is empty, returning no results.")

def test_similarity_search_returns_chunks(mock_settings, mock_embeddings, mock_logger, mock_chunk_and_metadata):
    DummyChunk, DummyMetadata = mock_chunk_and_metadata
    service = VectorStoreService()
    mock_vs = MagicMock()
    doc1 = MagicMock()
    doc1.page_content = "foo"
    doc1.metadata = {"foo": "bar"}
    doc2 = MagicMock()
    doc2.page_content = "baz"
    doc2.metadata = {"foo": "qux"}
    mock_vs.similarity_search.return_value = [doc1, doc2]
    service.vector_store = mock_vs
    # Patch DocumentMetadata and Chunk
    with patch("backend.src.services.vector_store.Chunk") as mock_chunk_cls, \
         patch("backend.src.services.vector_store.DocumentMetadata") as mock_meta_cls:
        mock_meta1 = MagicMock()
        mock_meta2 = MagicMock()
        mock_meta_cls.side_effect = [mock_meta1, mock_meta2]
        mock_chunk1 = MagicMock()
        mock_chunk2 = MagicMock()
        mock_chunk_cls.side_effect = [mock_chunk1, mock_chunk2]
        result = service.similarity_search("query", k=2)
        assert result == [mock_chunk1, mock_chunk2]
        mock_vs.similarity_search.assert_called_once_with("query", k=2)
        mock_meta_cls.assert_any_call(**doc1.metadata)
        mock_meta_cls.assert_any_call(**doc2.metadata)
        mock_chunk_cls.assert_any_call(text="foo", metadata=mock_meta1)
        mock_chunk_cls.assert_any_call(text="baz", metadata=mock_meta2)

def test_similarity_search_handles_exception_and_logs(mock_settings, mock_embeddings, mock_logger):
    service = VectorStoreService()
    mock_vs = MagicMock()
    mock_vs.similarity_search.side_effect = Exception("fail search")
    service.vector_store = mock_vs
    result = service.similarity_search("query", k=2)
    assert result == []
    assert mock_logger.error.call_count == 1
    assert "Error during similarity search" in str(mock_logger.error.call_args[0][0])

def test_similarity_search_empty_result(mock_settings, mock_embeddings, mock_logger):
    service = VectorStoreService()
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = []
    service.vector_store = mock_vs
    with patch("backend.src.services.vector_store.Chunk") as mock_chunk_cls, \
         patch("backend.src.services.vector_store.DocumentMetadata") as mock_meta_cls:
        result = service.similarity_search("query", k=2)
        assert result == []
        mock_vs.similarity_search.assert_called_once_with("query", k=2)
        mock_chunk_cls.assert_not_called()
        mock_meta_cls.assert_not_called()
