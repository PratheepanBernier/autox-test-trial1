import pytest
from unittest.mock import patch, MagicMock, create_autospec
from backend.src.services.vector_store import VectorStoreService
from backend.src.models.schemas import Chunk, DocumentMetadata

@pytest.fixture
def dummy_chunks():
    return [
        Chunk(text="foo", metadata=DocumentMetadata(source="src1", page=1)),
        Chunk(text="bar", metadata=DocumentMetadata(source="src2", page=2)),
    ]

@pytest.fixture
def dummy_documents(dummy_chunks):
    # Simulate langchain_core.documents.Document objects
    Document = MagicMock()
    docs = []
    for chunk in dummy_chunks:
        doc = MagicMock()
        doc.page_content = chunk.text
        doc.metadata = chunk.metadata.model_dump()
        docs.append(doc)
    return docs

@pytest.fixture
def patch_embeddings():
    with patch("backend.src.services.vector_store.HuggingFaceEmbeddings") as emb_cls:
        emb_instance = MagicMock()
        emb_cls.return_value = emb_instance
        yield emb_instance

@pytest.fixture
def patch_faiss_and_document(dummy_documents):
    with patch("backend.src.services.vector_store.FAISS") as faiss_cls, \
         patch("backend.src.services.vector_store.Document") as doc_cls:
        doc_cls.side_effect = dummy_documents
        faiss_instance = MagicMock()
        faiss_cls.from_documents.return_value = faiss_instance
        yield faiss_cls, faiss_instance, doc_cls

@pytest.fixture
def patch_settings(monkeypatch):
    class DummySettings:
        EMBEDDING_MODEL = "test-embedding-model"
        TOP_K = 3
    monkeypatch.setattr("backend.src.services.vector_store.settings", DummySettings())

@pytest.fixture
def vector_store_service(patch_settings, patch_embeddings):
    return VectorStoreService()

def test_init_success_logs_and_sets_embeddings(patch_settings):
    with patch("backend.src.services.vector_store.HuggingFaceEmbeddings") as emb_cls, \
         patch("backend.src.services.vector_store.logger") as logger_mock:
        emb_instance = MagicMock()
        emb_cls.return_value = emb_instance
        svc = VectorStoreService()
        assert svc.embeddings is emb_instance
        logger_mock.info.assert_any_call("Initialized HuggingFace embeddings with model: test-embedding-model")
        assert svc.vector_store is None

def test_init_failure_logs_and_raises(patch_settings):
    with patch("backend.src.services.vector_store.HuggingFaceEmbeddings", side_effect=RuntimeError("fail")), \
         patch("backend.src.services.vector_store.logger") as logger_mock:
        with pytest.raises(RuntimeError):
            VectorStoreService()
        logger_mock.error.assert_called()
        assert "Failed to initialize embeddings" in logger_mock.error.call_args[0][0]

def test_add_documents_initializes_faiss_on_first_add(vector_store_service, dummy_chunks, patch_faiss_and_document):
    faiss_cls, faiss_instance, doc_cls = patch_faiss_and_document
    svc = vector_store_service
    svc.vector_store = None
    with patch("backend.src.services.vector_store.logger") as logger_mock:
        svc.add_documents(dummy_chunks)
        faiss_cls.from_documents.assert_called_once()
        assert svc.vector_store is faiss_instance
        logger_mock.info.assert_any_call("Initializing new FAISS vector store.")
        logger_mock.info.assert_any_call(f"Added {len(dummy_chunks)} documents to vector store.")

def test_add_documents_adds_to_existing_store(vector_store_service, dummy_chunks, patch_faiss_and_document):
    faiss_cls, faiss_instance, doc_cls = patch_faiss_and_document
    svc = vector_store_service
    svc.vector_store = faiss_instance
    with patch("backend.src.services.vector_store.logger") as logger_mock:
        svc.add_documents(dummy_chunks)
        faiss_instance.add_documents.assert_called_once()
        faiss_cls.from_documents.assert_not_called()
        logger_mock.info.assert_any_call(f"Added {len(dummy_chunks)} documents to vector store.")

def test_add_documents_handles_exception(vector_store_service, dummy_chunks):
    svc = vector_store_service
    with patch("backend.src.services.vector_store.FAISS", side_effect=Exception("fail")), \
         patch("backend.src.services.vector_store.Document"), \
         patch("backend.src.services.vector_store.logger") as logger_mock:
        with pytest.raises(Exception):
            svc.add_documents(dummy_chunks)
        logger_mock.error.assert_called()
        assert "Error adding documents to vector store" in logger_mock.error.call_args[0][0]

def test_as_retriever_returns_none_if_store_empty(vector_store_service):
    svc = vector_store_service
    svc.vector_store = None
    with patch("backend.src.services.vector_store.logger") as logger_mock:
        result = svc.as_retriever()
        assert result is None
        logger_mock.warning.assert_called_with("Vector store is empty, cannot create retriever.")

def test_as_retriever_uses_default_kwargs(vector_store_service, patch_settings):
    svc = vector_store_service
    mock_vs = MagicMock()
    svc.vector_store = mock_vs
    retriever = MagicMock()
    mock_vs.as_retriever.return_value = retriever
    with patch("backend.src.services.vector_store.logger") as logger_mock:
        result = svc.as_retriever()
        assert result is retriever
        mock_vs.as_retriever.assert_called_once_with(search_type="similarity", search_kwargs={"k": 3})
        logger_mock.debug.assert_called()

def test_as_retriever_uses_custom_kwargs_and_type(vector_store_service):
    svc = vector_store_service
    mock_vs = MagicMock()
    svc.vector_store = mock_vs
    retriever = MagicMock()
    mock_vs.as_retriever.return_value = retriever
    custom_kwargs = {"k": 7, "score_threshold": 0.5}
    result = svc.as_retriever(search_type="mmr", search_kwargs=custom_kwargs)
    assert result is retriever
    mock_vs.as_retriever.assert_called_once_with(search_type="mmr", search_kwargs=custom_kwargs)

def test_as_retriever_handles_exception(vector_store_service):
    svc = vector_store_service
    mock_vs = MagicMock()
    svc.vector_store = mock_vs
    mock_vs.as_retriever.side_effect = Exception("fail")
    with patch("backend.src.services.vector_store.logger") as logger_mock:
        result = svc.as_retriever()
        assert result is None
        logger_mock.error.assert_called()
        assert "Error creating retriever" in logger_mock.error.call_args[0][0]

def test_similarity_search_returns_empty_if_store_empty(vector_store_service):
    svc = vector_store_service
    svc.vector_store = None
    with patch("backend.src.services.vector_store.logger") as logger_mock:
        result = svc.similarity_search("query", k=2)
        assert result == []
        logger_mock.warning.assert_called_with("Vector store is empty, returning no results.")

def test_similarity_search_happy_path(vector_store_service):
    svc = vector_store_service
    mock_vs = MagicMock()
    svc.vector_store = mock_vs
    doc1 = MagicMock()
    doc1.page_content = "foo"
    doc1.metadata = {"source": "src1", "page": 1}
    doc2 = MagicMock()
    doc2.page_content = "bar"
    doc2.metadata = {"source": "src2", "page": 2}
    mock_vs.similarity_search.return_value = [doc1, doc2]
    result = svc.similarity_search("query", k=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].text == "foo"
    assert result[0].metadata.source == "src1"
    assert result[0].metadata.page == 1
    assert result[1].text == "bar"
    assert result[1].metadata.source == "src2"
    assert result[1].metadata.page == 2

def test_similarity_search_handles_exception(vector_store_service):
    svc = vector_store_service
    mock_vs = MagicMock()
    svc.vector_store = mock_vs
    mock_vs.similarity_search.side_effect = Exception("fail")
    with patch("backend.src.services.vector_store.logger") as logger_mock:
        result = svc.similarity_search("query", k=2)
        assert result == []
        logger_mock.error.assert_called()
        assert "Error during similarity search" in logger_mock.error.call_args[0][0]

def test_similarity_search_boundary_k_zero(vector_store_service):
    svc = vector_store_service
    mock_vs = MagicMock()
    svc.vector_store = mock_vs
    mock_vs.similarity_search.return_value = []
    result = svc.similarity_search("query", k=0)
    assert result == []
    mock_vs.similarity_search.assert_called_once_with("query", k=0)

def test_similarity_search_invalid_metadata(vector_store_service):
    svc = vector_store_service
    mock_vs = MagicMock()
    svc.vector_store = mock_vs
    doc = MagicMock()
    doc.page_content = "foo"
    doc.metadata = {"bad": "data"}
    mock_vs.similarity_search.return_value = [doc]
    # Should raise TypeError due to DocumentMetadata(**doc.metadata) failing
    with patch("backend.src.services.vector_store.logger") as logger_mock:
        result = svc.similarity_search("query", k=1)
        assert result == []
        logger_mock.error.assert_called()
        assert "Error during similarity search" in logger_mock.error.call_args[0][0]
