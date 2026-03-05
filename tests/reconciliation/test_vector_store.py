# source_hash: 5d518cf45d49b218
import pytest
from unittest.mock import patch, MagicMock, create_autospec
from backend.src.services.vector_store import VectorStoreService
from models.schemas import Chunk, DocumentMetadata
from langchain_core.documents import Document

@pytest.fixture
def dummy_metadata():
    return DocumentMetadata(source="test_source", page=1)

@pytest.fixture
def dummy_chunk(dummy_metadata):
    return Chunk(text="test text", metadata=dummy_metadata)

@pytest.fixture
def dummy_chunks(dummy_metadata):
    return [
        Chunk(text="text 1", metadata=dummy_metadata),
        Chunk(text="text 2", metadata=dummy_metadata),
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
def mock_settings(monkeypatch):
    class DummySettings:
        EMBEDDING_MODEL = "test-model"
        TOP_K = 3
    monkeypatch.setattr("backend.src.services.vector_store.settings", DummySettings())

@pytest.fixture
def service(mock_embeddings, mock_settings):
    return VectorStoreService()

def test_add_documents_initializes_vector_store(service, mock_faiss, dummy_chunks):
    mock_vs = MagicMock()
    mock_faiss.from_documents.return_value = mock_vs
    service.vector_store = None

    service.add_documents(dummy_chunks)

    assert service.vector_store == mock_vs
    mock_faiss.from_documents.assert_called_once()
    # Ensure correct number of documents passed
    docs_arg = mock_faiss.from_documents.call_args[0][0]
    assert len(docs_arg) == len(dummy_chunks)
    assert all(isinstance(doc, Document) for doc in docs_arg)

def test_add_documents_appends_to_existing_vector_store(service, mock_faiss, dummy_chunks):
    mock_vs = MagicMock()
    service.vector_store = mock_vs

    service.add_documents(dummy_chunks)

    mock_vs.add_documents.assert_called_once()
    docs_arg = mock_vs.add_documents.call_args[0][0]
    assert len(docs_arg) == len(dummy_chunks)
    assert all(isinstance(doc, Document) for doc in docs_arg)
    mock_faiss.from_documents.assert_not_called()

def test_add_documents_handles_exception(service, mock_faiss, dummy_chunks):
    mock_faiss.from_documents.side_effect = Exception("fail!")
    service.vector_store = None
    with pytest.raises(Exception) as exc:
        service.add_documents(dummy_chunks)
    assert "fail!" in str(exc.value)

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
    assert "search_kwargs" in kwargs

def test_as_retriever_passes_custom_kwargs(service):
    mock_vs = MagicMock()
    mock_retriever = MagicMock()
    mock_vs.as_retriever.return_value = mock_retriever
    service.vector_store = mock_vs

    retriever = service.as_retriever(search_type="mmr", search_kwargs={"k": 7})
    assert retriever == mock_retriever
    mock_vs.as_retriever.assert_called_once_with(search_type="mmr", search_kwargs={"k": 7})

def test_as_retriever_handles_exception_and_returns_none(service):
    mock_vs = MagicMock()
    mock_vs.as_retriever.side_effect = Exception("fail!")
    service.vector_store = mock_vs

    retriever = service.as_retriever()
    assert retriever is None

def test_similarity_search_returns_empty_if_vector_store_empty(service):
    service.vector_store = None
    results = service.similarity_search("query", k=2)
    assert results == []

def test_similarity_search_returns_chunks(service, dummy_metadata):
    mock_vs = MagicMock()
    doc1 = Document(page_content="foo", metadata=dummy_metadata.model_dump())
    doc2 = Document(page_content="bar", metadata=dummy_metadata.model_dump())
    mock_vs.similarity_search.return_value = [doc1, doc2]
    service.vector_store = mock_vs

    results = service.similarity_search("query", k=2)
    assert len(results) == 2
    assert all(isinstance(chunk, Chunk) for chunk in results)
    assert results[0].text == "foo"
    assert results[1].text == "bar"
    assert results[0].metadata.source == dummy_metadata.source

def test_similarity_search_handles_exception_and_returns_empty(service):
    mock_vs = MagicMock()
    mock_vs.similarity_search.side_effect = Exception("fail!")
    service.vector_store = mock_vs

    results = service.similarity_search("query", k=2)
    assert results == []

def test_similarity_search_and_as_retriever_equivalence(service, dummy_metadata):
    # Reconciliation: similarity_search vs as_retriever().get_relevant_documents
    mock_vs = MagicMock()
    doc1 = Document(page_content="foo", metadata=dummy_metadata.model_dump())
    doc2 = Document(page_content="bar", metadata=dummy_metadata.model_dump())
    mock_vs.similarity_search.return_value = [doc1, doc2]

    # as_retriever returns a retriever whose get_relevant_documents returns same docs
    mock_retriever = MagicMock()
    mock_retriever.get_relevant_documents.return_value = [doc1, doc2]
    mock_vs.as_retriever.return_value = mock_retriever
    service.vector_store = mock_vs

    # similarity_search
    sim_chunks = service.similarity_search("query", k=2)
    # as_retriever
    retriever = service.as_retriever()
    docs = retriever.get_relevant_documents("query")
    # Convert docs to Chunks as similarity_search does
    chunks_from_retriever = [
        Chunk(text=doc.page_content, metadata=DocumentMetadata(**doc.metadata))
        for doc in docs
    ]
    # Compare outputs
    assert [c.text for c in sim_chunks] == [c.text for c in chunks_from_retriever]
    assert [c.metadata.source for c in sim_chunks] == [c.metadata.source for c in chunks_from_retriever]

def test_add_documents_with_empty_list(service, mock_faiss):
    service.vector_store = None
    service.add_documents([])
    # Should initialize vector store with empty docs
    mock_faiss.from_documents.assert_called_once()
    docs_arg = mock_faiss.from_documents.call_args[0][0]
    assert docs_arg == []

def test_add_documents_with_boundary_conditions(service, mock_faiss, dummy_metadata):
    # Test with a single chunk (boundary)
    chunk = Chunk(text="", metadata=dummy_metadata)
    service.vector_store = None
    service.add_documents([chunk])
    docs_arg = mock_faiss.from_documents.call_args[0][0]
    assert len(docs_arg) == 1
    assert docs_arg[0].page_content == ""
    assert docs_arg[0].metadata == dummy_metadata.model_dump()
