import pytest
from unittest.mock import patch, MagicMock, create_autospec
from backend.src.services.vector_store import VectorStoreService
from backend.src.models.schemas import Chunk, DocumentMetadata
from backend.src.core import config

@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    monkeypatch.setattr(config, "settings", MagicMock(EMBEDDING_MODEL="test-model", TOP_K=3))


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
        emb_cls.assert_called_once_with(model_name="test-model")


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
    mock_faiss.from_documents.return_value = "vector_store_instance"

    svc.add_documents(sample_chunks)

    assert svc.vector_store == "vector_store_instance"
    assert mock_faiss.from_documents.called
    assert mock_document.call_count == len(sample_chunks)


def test_add_documents_appends_to_existing_vector_store(sample_chunks, mock_embeddings, mock_faiss, mock_document):
    svc = VectorStoreService()
    svc.embeddings = MagicMock()
    svc.vector_store = MagicMock()
    mock_doc_instance = MagicMock()
    mock_document.side_effect = lambda page_content, metadata: mock_doc_instance

    svc.add_documents(sample_chunks)

    svc.vector_store.add_documents.assert_called_once()
    assert mock_document.call_count == len(sample_chunks)
    mock_faiss.from_documents.assert_not_called()


def test_add_documents_raises_on_failure(sample_chunks, mock_embeddings, mock_faiss, mock_document):
    svc = VectorStoreService()
    svc.embeddings = MagicMock()
    svc.vector_store = None
    mock_document.side_effect = Exception("doc error")

    with pytest.raises(Exception):
        svc.add_documents(sample_chunks)


def test_as_retriever_returns_none_when_vector_store_empty(mock_embeddings):
    svc = VectorStoreService()
    svc.vector_store = None
    retriever = svc.as_retriever()
    assert retriever is None


def test_as_retriever_returns_retriever_with_defaults(mock_embeddings):
    svc = VectorStoreService()
    mock_vs = MagicMock()
    retriever_obj = MagicMock()
    mock_vs.as_retriever.return_value = retriever_obj
    svc.vector_store = mock_vs

    result = svc.as_retriever()
    assert result is retriever_obj
    mock_vs.as_retriever.assert_called_once_with(
        search_type="similarity",
        search_kwargs={"k": 3}
    )


def test_as_retriever_with_custom_args(mock_embeddings):
    svc = VectorStoreService()
    mock_vs = MagicMock()
    retriever_obj = MagicMock()
    mock_vs.as_retriever.return_value = retriever_obj
    svc.vector_store = mock_vs

    result = svc.as_retriever(search_type="mmr", search_kwargs={"k": 2, "score_threshold": 0.7})
    assert result is retriever_obj
    mock_vs.as_retriever.assert_called_once_with(
        search_type="mmr",
        search_kwargs={"k": 2, "score_threshold": 0.7}
    )


def test_as_retriever_returns_none_on_exception(mock_embeddings):
    svc = VectorStoreService()
    mock_vs = MagicMock()
    mock_vs.as_retriever.side_effect = Exception("fail")
    svc.vector_store = mock_vs

    result = svc.as_retriever()
    assert result is None


def test_similarity_search_returns_empty_when_vector_store_empty(mock_embeddings):
    svc = VectorStoreService()
    svc.vector_store = None
    result = svc.similarity_search("query", k=2)
    assert result == []


def test_similarity_search_returns_chunks(sample_chunks, mock_embeddings):
    svc = VectorStoreService()
    mock_vs = MagicMock()
    svc.vector_store = mock_vs

    # Prepare fake docs returned by similarity_search
    doc1 = MagicMock()
    doc1.page_content = "foo"
    doc1.metadata = {"source": "file1", "page": 1}
    doc2 = MagicMock()
    doc2.page_content = "bar"
    doc2.metadata = {"source": "file2", "page": 2}
    mock_vs.similarity_search.return_value = [doc1, doc2]

    result = svc.similarity_search("query", k=2)
    assert len(result) == 2
    assert all(isinstance(chunk, Chunk) for chunk in result)
    assert result[0].text == "foo"
    assert result[0].metadata.source == "file1"
    assert result[1].text == "bar"
    assert result[1].metadata.page == 2


def test_similarity_search_returns_empty_on_exception(mock_embeddings):
    svc = VectorStoreService()
    mock_vs = MagicMock()
    mock_vs.similarity_search.side_effect = Exception("fail")
    svc.vector_store = mock_vs

    result = svc.similarity_search("query", k=2)
    assert result == []


def test_add_documents_with_empty_list(mock_embeddings, mock_faiss, mock_document):
    svc = VectorStoreService()
    svc.embeddings = MagicMock()
    svc.vector_store = None
    mock_faiss.from_documents.return_value = "vector_store_instance"
    svc.add_documents([])
    assert svc.vector_store == "vector_store_instance"
    mock_faiss.from_documents.assert_called_once_with([], svc.embeddings)
    mock_document.assert_not_called()


def test_similarity_search_with_zero_k_returns_empty(mock_embeddings):
    svc = VectorStoreService()
    mock_vs = MagicMock()
    svc.vector_store = mock_vs
    mock_vs.similarity_search.return_value = []

    result = svc.similarity_search("query", k=0)
    assert result == []
    mock_vs.similarity_search.assert_called_once_with("query", k=0)
