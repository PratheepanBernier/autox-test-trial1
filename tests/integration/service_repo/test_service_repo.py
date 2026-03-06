import pytest

from backend.src.services.ingestion import DocumentIngestionService
from backend.src.services.vector_store import VectorStoreService
from backend.src.models.schemas import Chunk, DocumentMetadata
from backend.src.core import config

import types

class DummyEmbeddings:
    def embed_documents(self, texts):
        # Return deterministic embeddings (fixed-length vectors)
        return [[float(i)] * 5 for i in range(len(texts))]
    def embed_query(self, text):
        return [0.1] * 5

class DummyFAISS:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings
        self._docs = list(documents)
    @classmethod
    def from_documents(cls, documents, embeddings):
        return cls(documents, embeddings)
    def add_documents(self, documents):
        self._docs.extend(documents)
    def as_retriever(self, search_type="similarity", search_kwargs=None):
        # Return a dummy retriever that always returns the first k docs
        class DummyRetriever:
            def __init__(self, docs):
                self.docs = docs
            def invoke(self, query):
                k = search_kwargs.get("k", 4) if search_kwargs else 4
                return self.docs[:k]
        return DummyRetriever(self._docs)
    def similarity_search(self, query, k=4):
        return self._docs[:k]

@pytest.fixture(autouse=True)
def patch_vector_store(monkeypatch):
    # Patch HuggingFaceEmbeddings and FAISS in vector_store
    monkeypatch.setattr("backend.src.services.vector_store.HuggingFaceEmbeddings", lambda model_name: DummyEmbeddings())
    monkeypatch.setattr("backend.src.services.vector_store.FAISS", DummyFAISS)
    yield

@pytest.fixture
def ingestion_service():
    # Patch RecursiveCharacterTextSplitter to avoid langchain dependency
    class DummySplitter:
        def __init__(self, **kwargs):
            pass
        def split_text(self, text):
            # Split by double newlines for deterministic chunking
            return [t.strip() for t in text.split("\n\n") if t.strip()]
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("backend.src.services.ingestion.RecursiveCharacterTextSplitter", DummySplitter)
    yield DocumentIngestionService()
    monkeypatch.undo()

@pytest.fixture
def vector_store_service():
    return VectorStoreService()

def test_ingestion_to_vector_store_txt(ingestion_service, vector_store_service):
    # Prepare a deterministic logistics document as bytes
    doc_text = (
        "Carrier Details\nName: Acme Logistics\nMC: 123456\n"
        "Driver Details\nName: John Doe\nPhone: 555-1234\n"
        "Pickup\nLocation: Warehouse A\nCity: Dallas\n"
        "Drop\nLocation: Store B\nCity: Houston\n"
        "Rate Breakdown\nTotal: $1200\n"
        "Commodity\nWidgets\nWeight: 1000 lbs\n"
        "Standing Instructions\nCall before delivery."
    )
    file_content = doc_text.encode("utf-8")
    filename = "test_doc.txt"

    # Ingestion: process file and get chunks
    chunks = ingestion_service.process_file(file_content, filename)
    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert any("Carrier Details" in c.text for c in chunks)
    assert any("Pickup" in c.text for c in chunks)
    assert any("Rate Breakdown" in c.text for c in chunks)
    # Chunks should have metadata with correct filename
    for c in chunks:
        assert c.metadata.filename == filename
        assert c.metadata.chunk_type == "text"

    # Vector store: add chunks
    vector_store_service.add_documents(chunks)
    # The vector store should now be initialized
    assert vector_store_service.vector_store is not None

    # Test retriever returns the correct number of docs and content
    retriever = vector_store_service.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    assert retriever is not None
    retrieved = retriever.invoke("Carrier")
    assert isinstance(retrieved, list)
    assert len(retrieved) == 2
    # The retrieved docs should have the same content as the original chunks
    assert any("Carrier Details" in doc.page_content for doc in retrieved)

    # Test similarity_search legacy method
    results = vector_store_service.similarity_search("Pickup", k=3)
    assert isinstance(results, list)
    assert len(results) == 3
    assert all(isinstance(r, Chunk) for r in results)
    assert any("Pickup" in r.text for r in results)

def test_vector_store_add_and_retrieve_multiple_calls(ingestion_service, vector_store_service):
    # Prepare two different documents
    doc1 = "Carrier Details\nName: Alpha\nPickup\nLocation: X\nDrop\nLocation: Y"
    doc2 = "Carrier Details\nName: Beta\nPickup\nLocation: Z\nDrop\nLocation: W"
    file1 = "alpha.txt"
    file2 = "beta.txt"

    chunks1 = ingestion_service.process_file(doc1.encode("utf-8"), file1)
    chunks2 = ingestion_service.process_file(doc2.encode("utf-8"), file2)

    # Add first document
    vector_store_service.add_documents(chunks1)
    # Add second document (should append, not overwrite)
    vector_store_service.add_documents(chunks2)

    # Retrieve with retriever, should see chunks from both docs
    retriever = vector_store_service.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    retrieved = retriever.invoke("Carrier")
    sources = [doc.metadata["filename"] for doc in retrieved]
    assert file1 in sources
    assert file2 in sources

    # Similarity search should also return chunks from both docs
    results = vector_store_service.similarity_search("Pickup", k=5)
    filenames = [r.metadata.filename for r in results]
    assert file1 in filenames
    assert file2 in filenames

def test_vector_store_empty_behavior(vector_store_service):
    # No documents added yet
    retriever = vector_store_service.as_retriever()
    assert retriever is None

    results = vector_store_service.similarity_search("anything", k=2)
    assert results == []
