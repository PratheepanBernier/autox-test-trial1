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
        Chunk(text="Hello world", metadata=meta1),
        Chunk(text="Goodbye world", metadata=meta2),
    ]

def test_init_success(mock_settings, mock_embeddings):
    mock_embeddings.return_value = MagicMock()
    service = VectorStoreService()
    assert hasattr(service, "embeddings")
    assert service.vector_store is None

def test_init_failure_logs_and_raises(mock_settings):
    with patch("backend.src.services.vector_store.HuggingFaceEmbeddings", side_effect=RuntimeError("fail")):
        with patch("backend.src.services.vector_store.logger") as mock_logger:
            with pytest.raises(RuntimeError):
                VectorStoreService()
            assert mock_logger.error.called
            assert "Failed to initialize embeddings" in str(mock_logger.error.call_args)

def test_add_documents_initializes_vector_store(mock_settings, mock_embeddings, mock_faiss, dummy_chunks):
    mock_embeddings.return_value = MagicMock()
    mock_faiss.from_documents.return_value = MagicMock()
    service = VectorStoreService()
    service.add_documents(dummy_chunks)
    assert service.vector_store is not None
    mock_faiss.from_documents.assert_called_once()
    # Check correct number of documents passed
    docs_arg = mock_faiss.from_documents.call_args[0][0]
    assert all(isinstance(doc, Document) for doc in docs_arg)
    assert len(docs_arg) == 2

def test_add_documents_appends_to_existing_vector_store(mock_settings, mock_embeddings, mock_faiss, dummy_chunks):
    mock_embeddings.return_value = MagicMock()
    mock_vector_store = MagicMock()
    service = VectorStoreService()
    service.vector_store = mock_vector_store
    service.add_documents(dummy_chunks)
    mock_vector_store.add_documents.assert_called_once()
    docs_arg = mock_vector_store.add_documents.call_args[0][0]
    assert all(isinstance(doc, Document) for doc in docs_arg)
    assert len(docs_arg) == 2

def test_add_documents_handles_exception_and_logs(mock_settings, mock_embeddings, mock_faiss, dummy_chunks):
    mock_embeddings.return_value = MagicMock()
    mock_faiss.from_documents.side_effect = Exception("vector store error")
    service = VectorStoreService()
    with patch("backend.src.services.vector_store.logger") as mock_logger:
        with pytest.raises(Exception):
            service.add_documents(dummy_chunks)
        assert mock_logger.error.called
        assert "Error adding documents to vector store" in str(mock_logger.error.call_args)

def test_add_documents_empty_list(mock_settings, mock_embeddings, mock_faiss):
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
    with patch("backend.src.services.vector_store.logger") as mock_logger:
        retriever = service.as_retriever()
        assert retriever is None
        assert mock_logger.warning.called
        assert "Vector store is empty" in str(mock_logger.warning.call_args)

def test_as_retriever_returns_retriever_with_defaults(mock_settings, mock_embeddings, mock_faiss):
    mock_embeddings.return_value = MagicMock()
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever
    service = VectorStoreService()
    service.vector_store = mock_vector_store
    result = service.as_retriever()
    mock_vector_store.as_retriever.assert_called_once_with(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    assert result == mock_retriever

def test_as_retriever_with_custom_args(mock_settings, mock_embeddings, mock_faiss):
    mock_embeddings.return_value = MagicMock()
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever
    service = VectorStoreService()
    service.vector_store = mock_vector_store
    result = service.as_retriever(search_type="mmr", search_kwargs={"k": 2, "score_threshold": 0.5})
    mock_vector_store.as_retriever.assert_called_once_with(
        search_type="mmr",
        search_kwargs={"k": 2, "score_threshold": 0.5}
    )
    assert result == mock_retriever

def test_as_retriever_handles_exception_and_logs(mock_settings, mock_embeddings, mock_faiss):
    mock_embeddings.return_value = MagicMock()
    mock_vector_store = MagicMock()
    mock_vector_store.as_retriever.side_effect = Exception("retriever error")
    service = VectorStoreService()
    service.vector_store = mock_vector_store
    with patch("backend.src.services.vector_store.logger") as mock_logger:
        result = service.as_retriever()
        assert result is None
        assert mock_logger.error.called
        assert "Error creating retriever" in str(mock_logger.error.call_args)

def test_similarity_search_returns_empty_if_vector_store_none(mock_settings, mock_embeddings):
    mock_embeddings.return_value = MagicMock()
    service = VectorStoreService()
    with patch("backend.src.services.vector_store.logger") as mock_logger:
        result = service.similarity_search("query", k=2)
        assert result == []
        assert mock_logger.warning.called
        assert "Vector store is empty" in str(mock_logger.warning.call_args)

def test_similarity_search_returns_chunks(mock_settings, mock_embeddings):
    mock_embeddings.return_value = MagicMock()
    mock_vector_store = MagicMock()
    doc1 = Document(page_content="A", metadata={"source": "f1", "page": 1})
    doc2 = Document(page_content="B", metadata={"source": "f2", "page": 2})
    mock_vector_store.similarity_search.return_value = [doc1, doc2]
    service = VectorStoreService()
    service.vector_store = mock_vector_store
    result = service.similarity_search("query", k=2)
    assert len(result) == 2
    assert all(isinstance(chunk, Chunk) for chunk in result)
    assert result[0].text == "A"
    assert result[0].metadata.source == "f1"
    assert result[1].text == "B"
    assert result[1].metadata.page == 2

def test_similarity_search_handles_exception_and_logs(mock_settings, mock_embeddings):
    mock_embeddings.return_value = MagicMock()
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.side_effect = Exception("search error")
    service = VectorStoreService()
    service.vector_store = mock_vector_store
    with patch("backend.src.services.vector_store.logger") as mock_logger:
        result = service.similarity_search("query", k=2)
        assert result == []
        assert mock_logger.error.called
        assert "Error during similarity search" in str(mock_logger.error.call_args)

def test_similarity_search_with_zero_k(mock_settings, mock_embeddings):
    mock_embeddings.return_value = MagicMock()
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.return_value = []
    service = VectorStoreService()
    service.vector_store = mock_vector_store
    result = service.similarity_search("query", k=0)
    assert result == []
    mock_vector_store.similarity_search.assert_called_once_with("query", k=0)

def test_similarity_search_with_large_k(mock_settings, mock_embeddings):
    mock_embeddings.return_value = MagicMock()
    mock_vector_store = MagicMock()
    docs = [Document(page_content=f"Doc{i}", metadata={"source": f"s{i}", "page": i}) for i in range(10)]
    mock_vector_store.similarity_search.return_value = docs
    service = VectorStoreService()
    service.vector_store = mock_vector_store
    result = service.similarity_search("query", k=10)
    assert len(result) == 10
    for i, chunk in enumerate(result):
        assert chunk.text == f"Doc{i}"
        assert chunk.metadata.page == i
