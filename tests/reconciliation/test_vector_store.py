import pytest
from unittest.mock import patch, MagicMock, create_autospec
from backend.src.services.vector_store import VectorStoreService
from models.schemas import Chunk, DocumentMetadata
from langchain_core.documents import Document

@pytest.fixture
def dummy_metadata():
    return DocumentMetadata(source="test_source", page=1, extra_info="meta")

@pytest.fixture
def dummy_chunk(dummy_metadata):
    return Chunk(text="This is a test chunk.", metadata=dummy_metadata)

@pytest.fixture
def dummy_chunks(dummy_metadata):
    return [
        Chunk(text=f"Chunk {i}", metadata=dummy_metadata)
        for i in range(3)
    ]

@pytest.fixture
def mock_embeddings():
    with patch("backend.src.services.vector_store.HuggingFaceEmbeddings") as mock_emb:
        yield mock_emb

@pytest.fixture
def mock_faiss():
    with patch("backend.src.services.vector_store.FAISS") as mock_faiss:
        yield mock_faiss

@pytest.fixture
def mock_settings(monkeypatch):
    class DummySettings:
        EMBEDDING_MODEL = "test-model"
        TOP_K = 4
    monkeypatch.setattr("backend.src.services.vector_store.settings", DummySettings())

@pytest.fixture
def service(mock_embeddings, mock_faiss, mock_settings):
    return VectorStoreService()

def test_add_documents_initializes_vector_store(service, dummy_chunks, mock_faiss):
    mock_vs = MagicMock()
    mock_faiss.from_documents.return_value = mock_vs
    service.vector_store = None

    service.add_documents(dummy_chunks)

    assert service.vector_store == mock_vs
    mock_faiss.from_documents.assert_called_once()
    # Check correct number of documents passed
    docs = mock_faiss.from_documents.call_args[0][0]
    assert len(docs) == len(dummy_chunks)
    for doc, chunk in zip(docs, dummy_chunks):
        assert doc.page_content == chunk.text
        assert doc.metadata == chunk.metadata.model_dump()

def test_add_documents_appends_to_existing_vector_store(service, dummy_chunks):
    mock_vs = MagicMock()
    service.vector_store = mock_vs

    service.add_documents(dummy_chunks)

    mock_vs.add_documents.assert_called_once()
    docs = mock_vs.add_documents.call_args[0][0]
    assert len(docs) == len(dummy_chunks)

def test_add_documents_raises_on_error(service, dummy_chunks):
    # Simulate error in document creation
    with patch("backend.src.services.vector_store.Document", side_effect=ValueError("fail")):
        with pytest.raises(ValueError):
            service.add_documents(dummy_chunks)

def test_as_retriever_returns_none_if_vector_store_empty(service):
    service.vector_store = None
    retriever = service.as_retriever()
    assert retriever is None

def test_as_retriever_returns_retriever_with_default_kwargs(service):
    mock_vs = MagicMock()
    mock_retriever = MagicMock()
    mock_vs.as_retriever.return_value = mock_retriever
    service.vector_store = mock_vs

    retriever = service.as_retriever()
    assert retriever == mock_retriever
    mock_vs.as_retriever.assert_called_once()
    args, kwargs = mock_vs.as_retriever.call_args
    assert kwargs["search_type"] == "similarity"
    assert kwargs["search_kwargs"]["k"] == 4

def test_as_retriever_returns_retriever_with_custom_kwargs(service):
    mock_vs = MagicMock()
    mock_retriever = MagicMock()
    mock_vs.as_retriever.return_value = mock_retriever
    service.vector_store = mock_vs

    retriever = service.as_retriever(search_type="mmr", search_kwargs={"k": 2, "score_threshold": 0.5})
    assert retriever == mock_retriever
    mock_vs.as_retriever.assert_called_once_with(
        search_type="mmr",
        search_kwargs={"k": 2, "score_threshold": 0.5}
    )

def test_as_retriever_returns_none_on_exception(service):
    mock_vs = MagicMock()
    mock_vs.as_retriever.side_effect = RuntimeError("fail")
    service.vector_store = mock_vs

    retriever = service.as_retriever()
    assert retriever is None

def test_similarity_search_returns_empty_if_vector_store_empty(service):
    service.vector_store = None
    result = service.similarity_search("query", k=2)
    assert result == []

def test_similarity_search_returns_chunks(service, dummy_metadata):
    mock_vs = MagicMock()
    doc1 = Document(page_content="A", metadata=dummy_metadata.model_dump())
    doc2 = Document(page_content="B", metadata=dummy_metadata.model_dump())
    mock_vs.similarity_search.return_value = [doc1, doc2]
    service.vector_store = mock_vs

    result = service.similarity_search("query", k=2)
    assert len(result) == 2
    assert all(isinstance(chunk, Chunk) for chunk in result)
    assert result[0].text == "A"
    assert result[1].text == "B"
    assert result[0].metadata == dummy_metadata

def test_similarity_search_returns_empty_on_exception(service):
    mock_vs = MagicMock()
    mock_vs.similarity_search.side_effect = Exception("fail")
    service.vector_store = mock_vs

    result = service.similarity_search("query", k=2)
    assert result == []

def test_similarity_search_and_as_retriever_equivalence(service, dummy_metadata):
    # Reconciliation: similarity_search and as_retriever("similarity") should return equivalent results
    mock_vs = MagicMock()
    doc = Document(page_content="Recon", metadata=dummy_metadata.model_dump())
    mock_vs.similarity_search.return_value = [doc]
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [doc]
    mock_vs.as_retriever.return_value = mock_retriever
    service.vector_store = mock_vs

    # similarity_search path
    chunks_legacy = service.similarity_search("Recon", k=1)
    # as_retriever path
    retriever = service.as_retriever()
    docs = retriever.invoke("Recon", k=1)
    # Convert docs to chunks as similarity_search does
    chunks_retriever = [
        Chunk(text=d.page_content, metadata=DocumentMetadata(**d.metadata))
        for d in docs
    ]
    assert len(chunks_legacy) == len(chunks_retriever)
    assert chunks_legacy[0].text == chunks_retriever[0].text
    assert chunks_legacy[0].metadata == chunks_retriever[0].metadata

def test_add_documents_and_similarity_search_integration(service, dummy_chunks, dummy_metadata, mock_faiss):
    # Reconciliation: add_documents then similarity_search should return the added chunk
    mock_vs = MagicMock()
    doc = Document(page_content=dummy_chunks[0].text, metadata=dummy_metadata.model_dump())
    mock_vs.similarity_search.return_value = [doc]
    mock_faiss.from_documents.return_value = mock_vs
    service.vector_store = None

    service.add_documents([dummy_chunks[0]])
    result = service.similarity_search(dummy_chunks[0].text, k=1)
    assert len(result) == 1
    assert result[0].text == dummy_chunks[0].text
    assert result[0].metadata == dummy_metadata

def test_add_documents_with_empty_list(service, mock_faiss):
    service.vector_store = None
    service.add_documents([])
    # Should not raise, and vector_store should be initialized with empty docs
    mock_faiss.from_documents.assert_called_once()
    docs = mock_faiss.from_documents.call_args[0][0]
    assert docs == []

def test_add_documents_with_boundary_conditions(service, dummy_metadata, mock_faiss):
    # Edge: Add a chunk with empty text and minimal metadata
    empty_chunk = Chunk(text="", metadata=dummy_metadata)
    mock_vs = MagicMock()
    mock_faiss.from_documents.return_value = mock_vs
    service.vector_store = None

    service.add_documents([empty_chunk])
    mock_faiss.from_documents.assert_called_once()
    docs = mock_faiss.from_documents.call_args[0][0]
    assert docs[0].page_content == ""
    assert docs[0].metadata == dummy_metadata.model_dump()
