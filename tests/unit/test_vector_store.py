# source_hash: 5d518cf45d49b218
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
    monkeypatch.setattr("backend.src.services.vector_store.settings", DummySettings())

@pytest.fixture
def mock_embeddings():
    with patch("backend.src.services.vector_store.HuggingFaceEmbeddings") as mock_emb:
        yield mock_emb

@pytest.fixture
def mock_faiss():
    with patch("backend.src.services.vector_store.FAISS") as mock_faiss_cls:
        yield mock_faiss_cls

@pytest.fixture
def dummy_chunks():
    meta1 = DocumentMetadata(source="file1", page=1)
    meta2 = DocumentMetadata(source="file2", page=2)
    return [
        Chunk(text="chunk1", metadata=meta1),
        Chunk(text="chunk2", metadata=meta2)
    ]

def test_init_successful_embeddings(mock_settings, mock_embeddings):
    mock_embeddings.return_value = MagicMock()
    service = VectorStoreService()
    assert hasattr(service, "embeddings")
    assert service.vector_store is None

def test_init_embeddings_failure(mock_settings):
    with patch("backend.src.services.vector_store.HuggingFaceEmbeddings", side_effect=Exception("fail")):
        with pytest.raises(Exception) as excinfo:
            VectorStoreService()
        assert "fail" in str(excinfo.value)

def test_add_documents_initializes_vector_store(mock_settings, mock_embeddings, mock_faiss, dummy_chunks):
    mock_embeddings.return_value = MagicMock()
    mock_faiss_instance = MagicMock()
    mock_faiss.from_documents.return_value = mock_faiss_instance

    service = VectorStoreService()
    service.add_documents(dummy_chunks)
    assert service.vector_store == mock_faiss_instance
    mock_faiss.from_documents.assert_called_once()
    # Check that the correct number of documents is passed
    docs_arg = mock_faiss.from_documents.call_args[0][0]
    assert len(docs_arg) == len(dummy_chunks)
    assert all(isinstance(doc, Document) for doc in docs_arg)

def test_add_documents_appends_to_existing_vector_store(mock_settings, mock_embeddings, mock_faiss, dummy_chunks):
    mock_embeddings.return_value = MagicMock()
    service = VectorStoreService()
    service.vector_store = MagicMock()
    service.add_documents(dummy_chunks)
    service.vector_store.add_documents.assert_called_once()
    docs_arg = service.vector_store.add_documents.call_args[0][0]
    assert len(docs_arg) == len(dummy_chunks)

def test_add_documents_handles_exception(mock_settings, mock_embeddings, mock_faiss, dummy_chunks):
    mock_embeddings.return_value = MagicMock()
    mock_faiss.from_documents.side_effect = Exception("vector error")
    service = VectorStoreService()
    with pytest.raises(Exception) as excinfo:
        service.add_documents(dummy_chunks)
    assert "vector error" in str(excinfo.value)

def test_add_documents_empty_chunks(mock_settings, mock_embeddings, mock_faiss):
    mock_embeddings.return_value = MagicMock()
    mock_faiss.from_documents.return_value = MagicMock()
    service = VectorStoreService()
    service.add_documents([])
    # Should still initialize vector store with empty list
    mock_faiss.from_documents.assert_called_once()
    docs_arg = mock_faiss.from_documents.call_args[0][0]
    assert docs_arg == []

def test_as_retriever_returns_none_if_vector_store_empty(mock_settings, mock_embeddings):
    mock_embeddings.return_value = MagicMock()
    service = VectorStoreService()
    assert service.as_retriever() is None

def test_as_retriever_returns_retriever_with_defaults(mock_settings, mock_embeddings):
    mock_embeddings.return_value = MagicMock()
    service = VectorStoreService()
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever
    service.vector_store = mock_vector_store
    result = service.as_retriever()
    mock_vector_store.as_retriever.assert_called_once_with(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    assert result == mock_retriever

def test_as_retriever_with_custom_args(mock_settings, mock_embeddings):
    mock_embeddings.return_value = MagicMock()
    service = VectorStoreService()
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    service.vector_store = mock_vector_store
    mock_vector_store.as_retriever.return_value = mock_retriever
    result = service.as_retriever(search_type="mmr", search_kwargs={"k": 2, "score_threshold": 0.5})
    mock_vector_store.as_retriever.assert_called_once_with(
        search_type="mmr",
        search_kwargs={"k": 2, "score_threshold": 0.5}
    )
    assert result == mock_retriever

def test_as_retriever_handles_exception_and_returns_none(mock_settings, mock_embeddings):
    mock_embeddings.return_value = MagicMock()
    service = VectorStoreService()
    mock_vector_store = MagicMock()
    mock_vector_store.as_retriever.side_effect = Exception("retriever error")
    service.vector_store = mock_vector_store
    result = service.as_retriever()
    assert result is None

def test_similarity_search_returns_empty_if_vector_store_none(mock_settings, mock_embeddings):
    mock_embeddings.return_value = MagicMock()
    service = VectorStoreService()
    result = service.similarity_search("query", k=2)
    assert result == []

def test_similarity_search_returns_chunks(mock_settings, mock_embeddings):
    mock_embeddings.return_value = MagicMock()
    service = VectorStoreService()
    mock_vector_store = MagicMock()
    doc1 = MagicMock()
    doc1.page_content = "doc1"
    doc1.metadata = {"source": "file1", "page": 1}
    doc2 = MagicMock()
    doc2.page_content = "doc2"
    doc2.metadata = {"source": "file2", "page": 2}
    mock_vector_store.similarity_search.return_value = [doc1, doc2]
    service.vector_store = mock_vector_store
    result = service.similarity_search("query", k=2)
    assert len(result) == 2
    assert all(isinstance(chunk, Chunk) for chunk in result)
    assert result[0].text == "doc1"
    assert result[0].metadata.source == "file1"
    assert result[1].text == "doc2"
    assert result[1].metadata.page == 2

def test_similarity_search_handles_exception_and_returns_empty(mock_settings, mock_embeddings):
    mock_embeddings.return_value = MagicMock()
    service = VectorStoreService()
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.side_effect = Exception("search error")
    service.vector_store = mock_vector_store
    result = service.similarity_search("query", k=2)
    assert result == []

def test_similarity_search_with_zero_k_returns_empty_list(mock_settings, mock_embeddings):
    mock_embeddings.return_value = MagicMock()
    service = VectorStoreService()
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.return_value = []
    service.vector_store = mock_vector_store
    result = service.similarity_search("query", k=0)
    assert result == []

def test_similarity_search_with_non_string_query(mock_settings, mock_embeddings):
    mock_embeddings.return_value = MagicMock()
    service = VectorStoreService()
    mock_vector_store = MagicMock()
    doc = MagicMock()
    doc.page_content = "doc"
    doc.metadata = {"source": "file", "page": 1}
    mock_vector_store.similarity_search.return_value = [doc]
    service.vector_store = mock_vector_store
    # Should not raise, even if query is not a string (edge case)
    result = service.similarity_search(12345, k=1)
    assert len(result) == 1
    assert result[0].text == "doc"
