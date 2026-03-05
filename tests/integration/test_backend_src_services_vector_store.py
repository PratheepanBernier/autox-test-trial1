import pytest
from unittest.mock import patch, MagicMock, create_autospec
from backend.src.services.vector_store import VectorStoreService
from backend.src.models.schemas import Chunk, DocumentMetadata
from backend.src.core import config

@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    monkeypatch.setattr(config.settings, "EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setattr(config.settings, "TOP_K", 3)

@pytest.fixture
def fake_metadata():
    return DocumentMetadata(source="unit-test", page=1)

@pytest.fixture
def fake_chunk(fake_metadata):
    return Chunk(text="This is a test chunk.", metadata=fake_metadata)

@pytest.fixture
def fake_chunks(fake_metadata):
    return [
        Chunk(text="Chunk one.", metadata=fake_metadata),
        Chunk(text="Chunk two.", metadata=fake_metadata),
        Chunk(text="Chunk three.", metadata=fake_metadata),
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
def mock_document():
    with patch("backend.src.services.vector_store.Document") as doc:
        yield doc

def test_init_success(mock_embeddings):
    service = VectorStoreService()
    mock_embeddings.assert_called_once_with(model_name="test-embedding-model")
    assert service.embeddings is mock_embeddings.return_value
    assert service.vector_store is None

def test_init_failure(monkeypatch):
    def raise_exc(*args, **kwargs):
        raise RuntimeError("fail!")
    monkeypatch.setattr("backend.src.services.vector_store.HuggingFaceEmbeddings", raise_exc)
    with pytest.raises(RuntimeError):
        VectorStoreService()

def test_add_documents_initializes_vector_store(
    mock_embeddings, mock_faiss, mock_document, fake_chunks
):
    service = VectorStoreService()
    service.vector_store = None
    mock_faiss.from_documents.return_value = MagicMock()
    mock_document.side_effect = lambda page_content, metadata: MagicMock(page_content=page_content, metadata=metadata)
    service.add_documents(fake_chunks)
    assert service.vector_store == mock_faiss.from_documents.return_value
    assert mock_faiss.from_documents.called
    assert mock_document.call_count == len(fake_chunks)

def test_add_documents_appends_to_existing_vector_store(
    mock_embeddings, mock_faiss, mock_document, fake_chunks
):
    service = VectorStoreService()
    mock_vs = MagicMock()
    service.vector_store = mock_vs
    mock_document.side_effect = lambda page_content, metadata: MagicMock(page_content=page_content, metadata=metadata)
    service.add_documents(fake_chunks)
    mock_vs.add_documents.assert_called_once()
    assert mock_document.call_count == len(fake_chunks)

def test_add_documents_raises_on_error(
    mock_embeddings, mock_faiss, mock_document, fake_chunks
):
    service = VectorStoreService()
    mock_document.side_effect = Exception("doc fail")
    with pytest.raises(Exception):
        service.add_documents(fake_chunks)

def test_as_retriever_returns_none_if_vector_store_empty(mock_embeddings):
    service = VectorStoreService()
    service.vector_store = None
    retriever = service.as_retriever()
    assert retriever is None

def test_as_retriever_returns_retriever_with_defaults(
    mock_embeddings, mock_faiss
):
    service = VectorStoreService()
    mock_vs = MagicMock()
    service.vector_store = mock_vs
    retriever_mock = MagicMock()
    mock_vs.as_retriever.return_value = retriever_mock
    result = service.as_retriever()
    mock_vs.as_retriever.assert_called_once_with(
        search_type="similarity", search_kwargs={"k": 3}
    )
    assert result == retriever_mock

def test_as_retriever_with_custom_args(mock_embeddings):
    service = VectorStoreService()
    mock_vs = MagicMock()
    service.vector_store = mock_vs
    retriever_mock = MagicMock()
    mock_vs.as_retriever.return_value = retriever_mock
    result = service.as_retriever(search_type="mmr", search_kwargs={"k": 2, "score_threshold": 0.5})
    mock_vs.as_retriever.assert_called_once_with(
        search_type="mmr", search_kwargs={"k": 2, "score_threshold": 0.5}
    )
    assert result == retriever_mock

def test_as_retriever_returns_none_on_error(mock_embeddings):
    service = VectorStoreService()
    mock_vs = MagicMock()
    service.vector_store = mock_vs
    mock_vs.as_retriever.side_effect = Exception("fail")
    result = service.as_retriever()
    assert result is None

def test_similarity_search_returns_empty_if_vector_store_empty(mock_embeddings):
    service = VectorStoreService()
    service.vector_store = None
    results = service.similarity_search("query", k=2)
    assert results == []

def test_similarity_search_returns_chunks(
    mock_embeddings, fake_metadata
):
    service = VectorStoreService()
    mock_vs = MagicMock()
    service.vector_store = mock_vs
    doc1 = MagicMock(page_content="A", metadata={"source": "unit-test", "page": 1})
    doc2 = MagicMock(page_content="B", metadata={"source": "unit-test", "page": 1})
    mock_vs.similarity_search.return_value = [doc1, doc2]
    results = service.similarity_search("query", k=2)
    assert len(results) == 2
    assert all(isinstance(chunk, Chunk) for chunk in results)
    assert results[0].text == "A"
    assert results[1].text == "B"
    assert results[0].metadata.source == "unit-test"
    assert results[0].metadata.page == 1

def test_similarity_search_handles_exception(mock_embeddings):
    service = VectorStoreService()
    mock_vs = MagicMock()
    service.vector_store = mock_vs
    mock_vs.similarity_search.side_effect = Exception("fail")
    results = service.similarity_search("query", k=2)
    assert results == []

def test_add_documents_with_empty_list(mock_embeddings, mock_faiss, mock_document):
    service = VectorStoreService()
    service.vector_store = None
    mock_faiss.from_documents.return_value = MagicMock()
    service.add_documents([])
    # Should still initialize vector store with empty docs
    assert service.vector_store == mock_faiss.from_documents.return_value
    mock_faiss.from_documents.assert_called_once_with([], mock_embeddings.return_value)
    assert mock_document.call_count == 0

def test_similarity_search_with_zero_k(mock_embeddings):
    service = VectorStoreService()
    mock_vs = MagicMock()
    service.vector_store = mock_vs
    mock_vs.similarity_search.return_value = []
    results = service.similarity_search("query", k=0)
    assert results == []
    mock_vs.similarity_search.assert_called_once_with("query", k=0)
