import pytest
from unittest.mock import patch, MagicMock, call
from backend.src.services.vector_store import VectorStoreService
from backend.src.models.schemas import Chunk, DocumentMetadata
from backend.src.core.config import settings

@pytest.fixture
def dummy_metadata():
    return DocumentMetadata(source="test_source", page=1)

@pytest.fixture
def dummy_chunk(dummy_metadata):
    return Chunk(text="This is a test chunk.", metadata=dummy_metadata)

@pytest.fixture
def dummy_chunks(dummy_metadata):
    return [
        Chunk(text="Chunk 1", metadata=dummy_metadata),
        Chunk(text="Chunk 2", metadata=dummy_metadata),
    ]

@pytest.fixture
def mock_embeddings():
    with patch("backend.src.services.vector_store.HuggingFaceEmbeddings") as mock:
        yield mock

@pytest.fixture
def mock_faiss():
    with patch("backend.src.services.vector_store.FAISS") as mock:
        yield mock

@pytest.fixture
def mock_document():
    with patch("backend.src.services.vector_store.Document") as mock:
        yield mock

@pytest.fixture
def vector_store_service(mock_embeddings):
    # Patch settings.EMBEDDING_MODEL to a deterministic value
    with patch("backend.src.services.vector_store.settings") as mock_settings:
        mock_settings.EMBEDDING_MODEL = "test-model"
        mock_settings.TOP_K = 4
        service = VectorStoreService()
        yield service

def test_init_success_logs_and_sets_embeddings(mock_embeddings):
    with patch("backend.src.services.vector_store.settings") as mock_settings:
        mock_settings.EMBEDDING_MODEL = "test-model"
        mock_settings.TOP_K = 4
        with patch("backend.src.services.vector_store.logger") as mock_logger:
            service = VectorStoreService()
            assert hasattr(service, "embeddings")
            mock_embeddings.assert_called_once_with(model_name="test-model")
            mock_logger.info.assert_any_call("Initialized HuggingFace embeddings with model: test-model")
            assert service.vector_store is None

def test_init_failure_logs_and_raises():
    with patch("backend.src.services.vector_store.HuggingFaceEmbeddings", side_effect=RuntimeError("fail")):
        with patch("backend.src.services.vector_store.settings") as mock_settings:
            mock_settings.EMBEDDING_MODEL = "test-model"
            mock_settings.TOP_K = 4
            with patch("backend.src.services.vector_store.logger") as mock_logger:
                with pytest.raises(RuntimeError):
                    VectorStoreService()
                assert mock_logger.error.call_args[0][0].startswith("Failed to initialize embeddings:")

def test_add_documents_initializes_faiss_and_adds_documents(vector_store_service, dummy_chunks, mock_faiss, mock_document):
    # Simulate Document creation
    mock_document.side_effect = lambda page_content, metadata: MagicMock(page_content=page_content, metadata=metadata)
    mock_faiss.from_documents.return_value = MagicMock()
    vector_store_service.vector_store = None

    with patch("backend.src.services.vector_store.logger") as mock_logger:
        vector_store_service.add_documents(dummy_chunks)
        # Should initialize FAISS
        mock_faiss.from_documents.assert_called_once()
        assert vector_store_service.vector_store is not None
        mock_logger.info.assert_any_call("Initializing new FAISS vector store.")
        mock_logger.info.assert_any_call(f"Added {len(dummy_chunks)} documents to vector store.")

def test_add_documents_adds_to_existing_vector_store(vector_store_service, dummy_chunks, mock_faiss, mock_document):
    # Simulate Document creation
    mock_document.side_effect = lambda page_content, metadata: MagicMock(page_content=page_content, metadata=metadata)
    mock_vector_store = MagicMock()
    vector_store_service.vector_store = mock_vector_store

    with patch("backend.src.services.vector_store.logger") as mock_logger:
        vector_store_service.add_documents(dummy_chunks)
        mock_vector_store.add_documents.assert_called_once()
        mock_faiss.from_documents.assert_not_called()
        mock_logger.info.assert_any_call(f"Added {len(dummy_chunks)} documents to vector store.")

def test_add_documents_handles_exception(vector_store_service, dummy_chunks, mock_document):
    # Simulate Document creation raising an exception
    mock_document.side_effect = Exception("doc error")
    with patch("backend.src.services.vector_store.logger") as mock_logger:
        with pytest.raises(Exception):
            vector_store_service.add_documents(dummy_chunks)
        assert mock_logger.error.call_args[0][0].startswith("Error adding documents to vector store:")

def test_as_retriever_returns_none_if_vector_store_empty(vector_store_service):
    vector_store_service.vector_store = None
    with patch("backend.src.services.vector_store.logger") as mock_logger:
        retriever = vector_store_service.as_retriever()
        assert retriever is None
        mock_logger.warning.assert_called_with("Vector store is empty, cannot create retriever.")

def test_as_retriever_returns_retriever_with_default_kwargs(vector_store_service):
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever
    vector_store_service.vector_store = mock_vector_store

    with patch("backend.src.services.vector_store.logger") as mock_logger:
        with patch("backend.src.services.vector_store.settings") as mock_settings:
            mock_settings.TOP_K = 4
            retriever = vector_store_service.as_retriever()
            assert retriever == mock_retriever
            mock_vector_store.as_retriever.assert_called_once_with(
                search_type="similarity", search_kwargs={"k": 4}
            )

def test_as_retriever_returns_retriever_with_custom_kwargs(vector_store_service):
    mock_vector_store = MagicMock()
    mock_retriever = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever
    vector_store_service.vector_store = mock_vector_store

    retriever = vector_store_service.as_retriever(search_type="mmr", search_kwargs={"k": 2, "score_threshold": 0.5})
    assert retriever == mock_retriever
    mock_vector_store.as_retriever.assert_called_once_with(
        search_type="mmr", search_kwargs={"k": 2, "score_threshold": 0.5}
    )

def test_as_retriever_handles_exception_and_returns_none(vector_store_service):
    mock_vector_store = MagicMock()
    mock_vector_store.as_retriever.side_effect = Exception("fail")
    vector_store_service.vector_store = mock_vector_store

    with patch("backend.src.services.vector_store.logger") as mock_logger:
        retriever = vector_store_service.as_retriever()
        assert retriever is None
        assert mock_logger.error.call_args[0][0].startswith("Error creating retriever:")

def test_similarity_search_returns_empty_if_vector_store_none(vector_store_service):
    vector_store_service.vector_store = None
    with patch("backend.src.services.vector_store.logger") as mock_logger:
        result = vector_store_service.similarity_search("query", k=2)
        assert result == []
        mock_logger.warning.assert_called_with("Vector store is empty, returning no results.")

def test_similarity_search_returns_chunks(vector_store_service, dummy_metadata):
    mock_vector_store = MagicMock()
    doc1 = MagicMock()
    doc1.page_content = "doc1"
    doc1.metadata = {"source": "test_source", "page": 1}
    doc2 = MagicMock()
    doc2.page_content = "doc2"
    doc2.metadata = {"source": "test_source", "page": 1}
    mock_vector_store.similarity_search.return_value = [doc1, doc2]
    vector_store_service.vector_store = mock_vector_store

    result = vector_store_service.similarity_search("query", k=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(chunk, Chunk) for chunk in result)
    assert result[0].text == "doc1"
    assert result[1].text == "doc2"
    assert result[0].metadata.source == "test_source"
    assert result[0].metadata.page == 1

def test_similarity_search_handles_exception_and_returns_empty(vector_store_service):
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.side_effect = Exception("fail")
    vector_store_service.vector_store = mock_vector_store

    with patch("backend.src.services.vector_store.logger") as mock_logger:
        result = vector_store_service.similarity_search("query", k=2)
        assert result == []
        assert mock_logger.error.call_args[0][0].startswith("Error during similarity search:")

def test_similarity_search_with_k_zero(vector_store_service):
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.return_value = []
    vector_store_service.vector_store = mock_vector_store

    result = vector_store_service.similarity_search("query", k=0)
    assert result == []
    mock_vector_store.similarity_search.assert_called_once_with("query", k=0)

def test_add_documents_with_empty_list(vector_store_service, mock_faiss, mock_document):
    # Should still initialize FAISS with empty documents
    mock_document.side_effect = lambda page_content, metadata: MagicMock(page_content=page_content, metadata=metadata)
    mock_faiss.from_documents.return_value = MagicMock()
    vector_store_service.vector_store = None

    with patch("backend.src.services.vector_store.logger") as mock_logger:
        vector_store_service.add_documents([])
        mock_faiss.from_documents.assert_called_once_with([], vector_store_service.embeddings)
        mock_logger.info.assert_any_call("Initializing new FAISS vector store.")
        mock_logger.info.assert_any_call("Added 0 documents to vector store.")
