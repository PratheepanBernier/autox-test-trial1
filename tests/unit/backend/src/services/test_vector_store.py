import pytest
from unittest.mock import patch, MagicMock, create_autospec
from backend.src.services.vector_store import VectorStoreService
from backend.src.models.schemas import Chunk, DocumentMetadata
from backend.src.core import config

@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    monkeypatch.setattr(config, "settings", MagicMock(
        EMBEDDING_MODEL="test-embedding-model",
        TOP_K=3
    ))

@pytest.fixture
def sample_chunks():
    return [
        Chunk(text="chunk1", metadata=DocumentMetadata(source="file1", page=1)),
        Chunk(text="chunk2", metadata=DocumentMetadata(source="file2", page=2)),
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

def test_init_success(monkeypatch):
    mock_emb = MagicMock()
    with patch("backend.src.services.vector_store.HuggingFaceEmbeddings", return_value=mock_emb) as emb_cls:
        svc = VectorStoreService()
        assert svc.embeddings is mock_emb
        assert svc.vector_store is None
        emb_cls.assert_called_once_with(model_name="test-embedding-model")

def test_init_failure(monkeypatch):
    with patch("backend.src.services.vector_store.HuggingFaceEmbeddings", side_effect=RuntimeError("fail")):
        with pytest.raises(RuntimeError):
            VectorStoreService()

def test_add_documents_initializes_vector_store(sample_chunks, mock_embeddings, mock_faiss, mock_document):
    svc = VectorStoreService()
    svc.embeddings = MagicMock()
    svc.vector_store = None

    mock_doc_instance = MagicMock()
    mock_document.side_effect = lambda page_content, metadata: mock_doc_instance
    mock_faiss.from_documents.return_value = "vector_store_obj"

    svc.add_documents(sample_chunks)

    assert svc.vector_store == "vector_store_obj"
    mock_faiss.from_documents.assert_called_once()
    assert mock_document.call_count == len(sample_chunks)

def test_add_documents_adds_to_existing_vector_store(sample_chunks, mock_embeddings, mock_faiss, mock_document):
    svc = VectorStoreService()
    svc.embeddings = MagicMock()
    svc.vector_store = MagicMock()
    mock_doc_instance = MagicMock()
    mock_document.side_effect = lambda page_content, metadata: mock_doc_instance

    svc.add_documents(sample_chunks)

    svc.vector_store.add_documents.assert_called_once()
    assert mock_document.call_count == len(sample_chunks)

def test_add_documents_raises_on_error(sample_chunks, mock_embeddings, mock_faiss, mock_document):
    svc = VectorStoreService()
    svc.embeddings = MagicMock()
    svc.vector_store = None
    mock_document.side_effect = Exception("doc error")

    with pytest.raises(Exception):
        svc.add_documents(sample_chunks)

def test_as_retriever_returns_none_if_vector_store_empty(mock_embeddings):
    svc = VectorStoreService()
    svc.vector_store = None
    retriever = svc.as_retriever()
    assert retriever is None

def test_as_retriever_returns_retriever_with_defaults(mock_embeddings):
    svc = VectorStoreService()
    mock_vs = MagicMock()
    svc.vector_store = mock_vs
    retriever_obj = MagicMock()
    mock_vs.as_retriever.return_value = retriever_obj

    result = svc.as_retriever()
    assert result is retriever_obj
    mock_vs.as_retriever.assert_called_once_with(
        search_type="similarity",
        search_kwargs={"k": config.settings.TOP_K}
    )

def test_as_retriever_with_custom_args(mock_embeddings):
    svc = VectorStoreService()
    mock_vs = MagicMock()
    svc.vector_store = mock_vs
    retriever_obj = MagicMock()
    mock_vs.as_retriever.return_value = retriever_obj

    result = svc.as_retriever(search_type="mmr", search_kwargs={"k": 2, "score_threshold": 0.5})
    assert result is retriever_obj
    mock_vs.as_retriever.assert_called_once_with(
        search_type="mmr",
        search_kwargs={"k": 2, "score_threshold": 0.5}
    )

def test_as_retriever_returns_none_on_exception(mock_embeddings):
    svc = VectorStoreService()
    mock_vs = MagicMock()
    svc.vector_store = mock_vs
    mock_vs.as_retriever.side_effect = Exception("fail")

    result = svc.as_retriever()
    assert result is None

def test_similarity_search_returns_empty_if_vector_store_none(mock_embeddings):
    svc = VectorStoreService()
    svc.vector_store = None
    result = svc.similarity_search("query", k=2)
    assert result == []

def test_similarity_search_returns_chunks(sample_chunks, mock_embeddings):
    svc = VectorStoreService()
    mock_vs = MagicMock()
    svc.vector_store = mock_vs

    # Prepare mock docs returned by similarity_search
    doc1 = MagicMock()
    doc1.page_content = "text1"
    doc1.metadata = {"source": "file1", "page": 1}
    doc2 = MagicMock()
    doc2.page_content = "text2"
    doc2.metadata = {"source": "file2", "page": 2}
    mock_vs.similarity_search.return_value = [doc1, doc2]

    result = svc.similarity_search("query", k=2)
    assert len(result) == 2
    assert isinstance(result[0], Chunk)
    assert result[0].text == "text1"
    assert result[0].metadata.source == "file1"
    assert result[0].metadata.page == 1

def test_similarity_search_returns_empty_on_exception(mock_embeddings):
    svc = VectorStoreService()
    mock_vs = MagicMock()
    svc.vector_store = mock_vs
    mock_vs.similarity_search.side_effect = Exception("fail")

    result = svc.similarity_search("query", k=2)
    assert result == []

def test_similarity_search_boundary_k_zero(mock_embeddings):
    svc = VectorStoreService()
    mock_vs = MagicMock()
    svc.vector_store = mock_vs
    mock_vs.similarity_search.return_value = []

    result = svc.similarity_search("query", k=0)
    assert result == []

def test_add_documents_empty_list(mock_embeddings, mock_faiss, mock_document):
    svc = VectorStoreService()
    svc.embeddings = MagicMock()
    svc.vector_store = None
    mock_faiss.from_documents.return_value = "vector_store_obj"
    svc.add_documents([])
    mock_faiss.from_documents.assert_called_once_with([], svc.embeddings)
    assert svc.vector_store == "vector_store_obj"
