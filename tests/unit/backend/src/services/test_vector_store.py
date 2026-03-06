import pytest
from unittest.mock import patch, MagicMock, create_autospec
from backend.src.services.vector_store import VectorStoreService
from backend.src.models.schemas import Chunk, DocumentMetadata

@pytest.fixture
def mock_settings(monkeypatch):
    class DummySettings:
        EMBEDDING_MODEL = "test-embedding-model"
        TOP_K = 3
    monkeypatch.setattr("backend.src.core.config.settings", DummySettings())

@pytest.fixture
def dummy_chunk():
    return Chunk(
        text="Sample chunk text",
        metadata=DocumentMetadata(source="test_source", page=1)
    )

@pytest.fixture
def dummy_chunks():
    return [
        Chunk(
            text=f"Chunk {i}",
            metadata=DocumentMetadata(source=f"source_{i}", page=i)
        ) for i in range(2)
    ]

@pytest.fixture
def mock_embeddings(monkeypatch):
    mock_embed = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "langchain_huggingface", MagicMock(HuggingFaceEmbeddings=MagicMock(return_value=mock_embed)))
    return mock_embed

@pytest.fixture
def mock_faiss(monkeypatch):
    mock_faiss_cls = MagicMock()
    mock_faiss_mod = MagicMock(FAISS=mock_faiss_cls)
    monkeypatch.setitem(__import__("sys").modules, "langchain_community.vectorstores", mock_faiss_mod)
    return mock_faiss_cls

@pytest.fixture
def mock_document(monkeypatch):
    mock_doc_cls = MagicMock()
    mock_doc_mod = MagicMock(Document=mock_doc_cls)
    monkeypatch.setitem(__import__("sys").modules, "langchain_core.documents", mock_doc_mod)
    return mock_doc_cls

def test_init_success(mock_settings, mock_embeddings):
    service = VectorStoreService()
    assert hasattr(service, "embeddings")
    assert service.embeddings is mock_embeddings
    assert service.vector_store is None

def test_init_failure(monkeypatch, mock_settings):
    def raise_import(*args, **kwargs):
        raise ImportError("fail")
    monkeypatch.setitem(__import__("sys").modules, "langchain_huggingface", MagicMock(HuggingFaceEmbeddings=raise_import))
    with pytest.raises(ImportError):
        VectorStoreService()

def test_add_documents_initializes_vector_store(monkeypatch, mock_settings, mock_embeddings, mock_faiss, mock_document, dummy_chunks):
    # Patch Document to return a simple object with page_content and metadata
    def doc_side_effect(page_content, metadata):
        return MagicMock(page_content=page_content, metadata=metadata)
    mock_document.Document.side_effect = doc_side_effect

    service = VectorStoreService()
    service.embeddings = mock_embeddings
    service.vector_store = None

    mock_faiss.from_documents.return_value = "mock_vector_store"
    service.add_documents(dummy_chunks)
    assert service.vector_store == "mock_vector_store"
    assert mock_faiss.from_documents.called
    # Check correct number of documents created
    assert mock_document.Document.call_count == len(dummy_chunks)

def test_add_documents_adds_to_existing_vector_store(monkeypatch, mock_settings, mock_embeddings, mock_faiss, mock_document, dummy_chunks):
    service = VectorStoreService()
    service.embeddings = mock_embeddings
    mock_vector_store = MagicMock()
    service.vector_store = mock_vector_store

    def doc_side_effect(page_content, metadata):
        return MagicMock(page_content=page_content, metadata=metadata)
    mock_document.Document.side_effect = doc_side_effect

    service.add_documents(dummy_chunks)
    mock_vector_store.add_documents.assert_called_once()
    assert mock_document.Document.call_count == len(dummy_chunks)

def test_add_documents_error(monkeypatch, mock_settings, mock_embeddings, mock_document, dummy_chunks):
    service = VectorStoreService()
    service.embeddings = mock_embeddings
    # Patch FAISS to raise error
    monkeypatch.setitem(__import__("sys").modules, "langchain_community.vectorstores", MagicMock(FAISS=MagicMock(from_documents=MagicMock(side_effect=RuntimeError("fail")))))
    mock_document.Document.side_effect = lambda page_content, metadata: MagicMock(page_content=page_content, metadata=metadata)
    service.vector_store = None
    with pytest.raises(RuntimeError):
        service.add_documents(dummy_chunks)

def test_as_retriever_returns_none_if_vector_store_empty(mock_settings, mock_embeddings):
    service = VectorStoreService()
    service.embeddings = mock_embeddings
    service.vector_store = None
    retriever = service.as_retriever()
    assert retriever is None

def test_as_retriever_returns_retriever_with_defaults(mock_settings, mock_embeddings):
    service = VectorStoreService()
    service.embeddings = mock_embeddings
    mock_vs = MagicMock()
    mock_retriever = MagicMock()
    mock_vs.as_retriever.return_value = mock_retriever
    service.vector_store = mock_vs
    result = service.as_retriever()
    assert result == mock_retriever
    mock_vs.as_retriever.assert_called_once()
    args, kwargs = mock_vs.as_retriever.call_args
    assert kwargs["search_type"] == "similarity"
    assert kwargs["search_kwargs"]["k"] == 3  # settings.TOP_K

def test_as_retriever_with_custom_args(mock_settings, mock_embeddings):
    service = VectorStoreService()
    service.embeddings = mock_embeddings
    mock_vs = MagicMock()
    mock_retriever = MagicMock()
    mock_vs.as_retriever.return_value = mock_retriever
    service.vector_store = mock_vs
    result = service.as_retriever(search_type="mmr", search_kwargs={"k": 2, "score_threshold": 0.5})
    assert result == mock_retriever
    mock_vs.as_retriever.assert_called_once_with(search_type="mmr", search_kwargs={"k": 2, "score_threshold": 0.5})

def test_as_retriever_error_returns_none(mock_settings, mock_embeddings):
    service = VectorStoreService()
    service.embeddings = mock_embeddings
    mock_vs = MagicMock()
    mock_vs.as_retriever.side_effect = Exception("fail")
    service.vector_store = mock_vs
    result = service.as_retriever()
    assert result is None

def test_similarity_search_returns_empty_if_vector_store_empty(mock_settings, mock_embeddings):
    service = VectorStoreService()
    service.embeddings = mock_embeddings
    service.vector_store = None
    result = service.similarity_search("query", k=2)
    assert result == []

def test_similarity_search_happy_path(monkeypatch, mock_settings, mock_embeddings):
    service = VectorStoreService()
    service.embeddings = mock_embeddings
    mock_vs = MagicMock()
    # Simulate two docs returned by similarity_search
    doc1 = MagicMock(page_content="doc1 text", metadata={"source": "src1", "page": 1})
    doc2 = MagicMock(page_content="doc2 text", metadata={"source": "src2", "page": 2})
    mock_vs.similarity_search.return_value = [doc1, doc2]
    service.vector_store = mock_vs

    result = service.similarity_search("query", k=2)
    assert len(result) == 2
    assert all(isinstance(chunk, Chunk) for chunk in result)
    assert result[0].text == "doc1 text"
    assert result[0].metadata.source == "src1"
    assert result[0].metadata.page == 1

def test_similarity_search_error_returns_empty(mock_settings, mock_embeddings):
    service = VectorStoreService()
    service.embeddings = mock_embeddings
    mock_vs = MagicMock()
    mock_vs.similarity_search.side_effect = Exception("fail")
    service.vector_store = mock_vs
    result = service.similarity_search("query", k=2)
    assert result == []
