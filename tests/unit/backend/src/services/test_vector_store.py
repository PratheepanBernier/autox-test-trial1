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
    monkeypatch.setattr("core.config.settings", DummySettings())

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
def vector_store_service(mock_settings, mock_embeddings):
    # Patch logger to avoid noisy output
    with patch("backend.src.services.vector_store.logger"):
        return VectorStoreService()

def make_chunk(text="foo", author="alice", doc_id="doc1"):
    return Chunk(
        text=text,
        metadata=DocumentMetadata(author=author, document_id=doc_id)
    )

def test_init_success(mock_settings):
    with patch("backend.src.services.vector_store.HuggingFaceEmbeddings") as mock_embed, \
         patch("backend.src.services.vector_store.logger"):
        mock_embed.return_value = MagicMock()
        svc = VectorStoreService()
        assert svc.embeddings is mock_embed.return_value
        assert svc.vector_store is None

def test_init_failure_logs_and_raises(mock_settings):
    with patch("backend.src.services.vector_store.HuggingFaceEmbeddings", side_effect=RuntimeError("fail")), \
         patch("backend.src.services.vector_store.logger") as mock_logger:
        with pytest.raises(RuntimeError):
            VectorStoreService()
        assert mock_logger.error.called

def test_add_documents_initializes_faiss(vector_store_service, mock_faiss, mock_embeddings):
    chunk1 = make_chunk("foo", "alice", "doc1")
    chunk2 = make_chunk("bar", "bob", "doc2")
    docs = [
        Document(page_content=chunk1.text, metadata=chunk1.metadata.model_dump()),
        Document(page_content=chunk2.text, metadata=chunk2.metadata.model_dump())
    ]
    mock_faiss.from_documents.return_value = MagicMock()
    vector_store_service.add_documents([chunk1, chunk2])
    mock_faiss.from_documents.assert_called_once()
    args, kwargs = mock_faiss.from_documents.call_args
    assert [d.page_content for d in args[0]] == [chunk1.text, chunk2.text]
    assert args[1] is mock_embeddings

def test_add_documents_adds_to_existing_store(vector_store_service, mock_faiss, mock_embeddings):
    chunk = make_chunk("baz", "carol", "doc3")
    mock_vector_store = MagicMock()
    vector_store_service.vector_store = mock_vector_store
    vector_store_service.add_documents([chunk])
    mock_vector_store.add_documents.assert_called_once()
    docs = mock_vector_store.add_documents.call_args[0][0]
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == chunk.text

def test_add_documents_error_logs_and_raises(vector_store_service):
    # Simulate error in add_documents
    with patch("backend.src.services.vector_store.Document", side_effect=ValueError("bad doc")), \
         patch("backend.src.services.vector_store.logger") as mock_logger:
        with pytest.raises(ValueError):
            vector_store_service.add_documents([make_chunk()])
        assert mock_logger.error.called

def test_as_retriever_returns_none_if_empty(vector_store_service):
    with patch("backend.src.services.vector_store.logger") as mock_logger:
        retriever = vector_store_service.as_retriever()
        assert retriever is None
        assert mock_logger.warning.called

def test_as_retriever_returns_retriever_with_defaults(vector_store_service):
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever
    vector_store_service.vector_store = mock_vector_store
    with patch("backend.src.services.vector_store.settings") as mock_settings:
        mock_settings.TOP_K = 5
        retriever = vector_store_service.as_retriever()
        assert retriever is mock_retriever
        mock_vector_store.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

def test_as_retriever_with_custom_kwargs(vector_store_service):
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    vector_store_service.vector_store = mock_vector_store
    mock_vector_store.as_retriever.return_value = mock_retriever
    retriever = vector_store_service.as_retriever(search_type="mmr", search_kwargs={"k": 2, "foo": "bar"})
    assert retriever is mock_retriever
    mock_vector_store.as_retriever.assert_called_once_with(
        search_type="mmr",
        search_kwargs={"k": 2, "foo": "bar"}
    )

def test_as_retriever_error_logs_and_returns_none(vector_store_service):
    mock_vector_store = MagicMock()
    mock_vector_store.as_retriever.side_effect = Exception("fail")
    vector_store_service.vector_store = mock_vector_store
    with patch("backend.src.services.vector_store.logger") as mock_logger:
        retriever = vector_store_service.as_retriever()
        assert retriever is None
        assert mock_logger.error.called

def test_similarity_search_returns_chunks(vector_store_service):
    chunk = make_chunk("baz", "carol", "doc3")
    doc = Document(page_content=chunk.text, metadata=chunk.metadata.model_dump())
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.return_value = [doc]
    vector_store_service.vector_store = mock_vector_store
    results = vector_store_service.similarity_search("baz", k=1)
    assert len(results) == 1
    assert isinstance(results[0], Chunk)
    assert results[0].text == chunk.text
    assert results[0].metadata.author == chunk.metadata.author
    assert results[0].metadata.document_id == chunk.metadata.document_id

def test_similarity_search_empty_store_returns_empty(vector_store_service):
    with patch("backend.src.services.vector_store.logger") as mock_logger:
        results = vector_store_service.similarity_search("foo")
        assert results == []
        assert mock_logger.warning.called

def test_similarity_search_error_logs_and_returns_empty(vector_store_service):
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.side_effect = Exception("fail")
    vector_store_service.vector_store = mock_vector_store
    with patch("backend.src.services.vector_store.logger") as mock_logger:
        results = vector_store_service.similarity_search("foo")
        assert results == []
        assert mock_logger.error.called

def test_similarity_search_returns_empty_on_no_results(vector_store_service):
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.return_value = []
    vector_store_service.vector_store = mock_vector_store
    results = vector_store_service.similarity_search("foo")
    assert results == []
