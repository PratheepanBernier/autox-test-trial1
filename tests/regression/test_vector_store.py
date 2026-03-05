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
def dummy_chunk():
    meta = DocumentMetadata(source="test", page=1)
    return Chunk(text="sample text", metadata=meta)

@pytest.fixture
def dummy_chunks():
    return [
        Chunk(text="text1", metadata=DocumentMetadata(source="src1", page=1)),
        Chunk(text="text2", metadata=DocumentMetadata(source="src2", page=2)),
    ]

def test_init_success(mock_settings, mock_embeddings):
    mock_embeddings.return_value = MagicMock()
    service = VectorStoreService()
    assert hasattr(service, "embeddings")
    assert service.vector_store is None

def test_init_failure(mock_settings):
    with patch("backend.src.services.vector_store.HuggingFaceEmbeddings", side_effect=RuntimeError("fail")):
        with pytest.raises(RuntimeError):
            VectorStoreService()

def test_add_documents_initializes_vector_store(mock_settings, mock_embeddings, mock_faiss, dummy_chunks):
    mock_embeddings.return_value = MagicMock()
    mock_faiss_instance = MagicMock()
    mock_faiss.from_documents.return_value = mock_faiss_instance

    service = VectorStoreService()
    service.add_documents(dummy_chunks)
    assert service.vector_store == mock_faiss_instance
    mock_faiss.from_documents.assert_called_once()
    # Check that the correct number of documents are passed
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
    assert all(isinstance(doc, Document) for doc in docs_arg)

def test_add_documents_handles_exception(mock_settings, mock_embeddings, mock_faiss, dummy_chunks):
    mock_embeddings.return_value = MagicMock()
    mock_faiss.from_documents.side_effect = Exception("vector store error")
    service = VectorStoreService()
    with pytest.raises(Exception):
        service.add_documents(dummy_chunks)

def test_as_retriever_returns_none_if_vector_store_empty(mock_settings, mock_embeddings):
    mock_embeddings.return_value = MagicMock()
    service = VectorStoreService()
    assert service.as_retriever() is None

def test_as_retriever_returns_retriever_with_default_kwargs(mock_settings, mock_embeddings, mock_faiss):
    mock_embeddings.return_value = MagicMock()
    mock_retriever = MagicMock()
    mock_vector_store = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever

    service = VectorStoreService()
    service.vector_store = mock_vector_store
    result = service.as_retriever()
    assert result == mock_retriever
    mock_vector_store.as_retriever.assert_called_once()
    args, kwargs = mock_vector_store.as_retriever.call_args
    assert kwargs["search_type"] == "similarity"
    assert kwargs["search_kwargs"] == {"k": 3}

def test_as_retriever_respects_custom_kwargs(mock_settings, mock_embeddings, mock_faiss):
    mock_embeddings.return_value = MagicMock()
    mock_retriever = MagicMock()
    mock_vector_store = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever

    service = VectorStoreService()
    service.vector_store = mock_vector_store
    custom_kwargs = {"k": 10, "score_threshold": 0.5}
    result = service.as_retriever(search_type="mmr", search_kwargs=custom_kwargs)
    assert result == mock_retriever
    mock_vector_store.as_retriever.assert_called_once_with(
        search_type="mmr", search_kwargs=custom_kwargs
    )

def test_as_retriever_handles_exception_and_returns_none(mock_settings, mock_embeddings, mock_faiss):
    mock_embeddings.return_value = MagicMock()
    mock_vector_store = MagicMock()
    mock_vector_store.as_retriever.side_effect = Exception("fail")
    service = VectorStoreService()
    service.vector_store = mock_vector_store
    assert service.as_retriever() is None

def test_similarity_search_returns_empty_if_vector_store_none(mock_settings, mock_embeddings):
    mock_embeddings.return_value = MagicMock()
    service = VectorStoreService()
    result = service.similarity_search("query", k=2)
    assert result == []

def test_similarity_search_returns_chunks(mock_settings, mock_embeddings, dummy_chunk):
    mock_embeddings.return_value = MagicMock()
    mock_vector_store = MagicMock()
    doc1 = MagicMock()
    doc1.page_content = "abc"
    doc1.metadata = {"source": "src", "page": 1}
    doc2 = MagicMock()
    doc2.page_content = "def"
    doc2.metadata = {"source": "src2", "page": 2}
    mock_vector_store.similarity_search.return_value = [doc1, doc2]

    service = VectorStoreService()
    service.vector_store = mock_vector_store
    result = service.similarity_search("query", k=2)
    assert len(result) == 2
    assert all(isinstance(chunk, Chunk) for chunk in result)
    assert result[0].text == "abc"
    assert result[0].metadata.source == "src"
    assert result[1].text == "def"
    assert result[1].metadata.page == 2

def test_similarity_search_handles_exception_and_returns_empty(mock_settings, mock_embeddings):
    mock_embeddings.return_value = MagicMock()
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.side_effect = Exception("fail")
    service = VectorStoreService()
    service.vector_store = mock_vector_store
    result = service.similarity_search("query", k=2)
    assert result == []

def test_add_documents_with_empty_list(mock_settings, mock_embeddings, mock_faiss):
    mock_embeddings.return_value = MagicMock()
    mock_faiss.from_documents.return_value = MagicMock()
    service = VectorStoreService()
    service.add_documents([])
    # Should still initialize vector store, but with empty docs
    mock_faiss.from_documents.assert_called_once()
    docs_arg = mock_faiss.from_documents.call_args[0][0]
    assert docs_arg == []

def test_similarity_search_with_zero_k(mock_settings, mock_embeddings):
    mock_embeddings.return_value = MagicMock()
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.return_value = []
    service = VectorStoreService()
    service.vector_store = mock_vector_store
    result = service.similarity_search("query", k=0)
    assert result == []

def test_as_retriever_with_boundary_k(mock_settings, mock_embeddings, mock_faiss):
    mock_embeddings.return_value = MagicMock()
    mock_retriever = MagicMock()
    mock_vector_store = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever
    service = VectorStoreService()
    service.vector_store = mock_vector_store
    result = service.as_retriever(search_kwargs={"k": 0})
    assert result == mock_retriever
    mock_vector_store.as_retriever.assert_called_once_with(
        search_type="similarity", search_kwargs={"k": 0}
    )
