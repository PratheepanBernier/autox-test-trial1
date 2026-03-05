import io
import sys
import types
import pytest
import builtins

import streamlit_app

from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def patch_streamlit(monkeypatch):
    # Patch all streamlit UI calls to no-op or dummy
    dummy = MagicMock()
    monkeypatch.setattr(streamlit_app.st, "set_page_config", lambda *a, **k: None)
    monkeypatch.setattr(streamlit_app.st, "title", lambda *a, **k: None)
    monkeypatch.setattr(streamlit_app.st, "info", lambda *a, **k: None)
    monkeypatch.setattr(streamlit_app.st, "error", lambda *a, **k: None)
    monkeypatch.setattr(streamlit_app.st, "stop", lambda: (_ for _ in ()).throw(SystemExit))
    monkeypatch.setattr(streamlit_app.st, "tabs", lambda x: [dummy, dummy, dummy])
    monkeypatch.setattr(streamlit_app.st, "header", lambda *a, **k: None)
    monkeypatch.setattr(streamlit_app.st, "file_uploader", lambda *a, **k: None)
    monkeypatch.setattr(streamlit_app.st, "button", lambda *a, **k: False)
    monkeypatch.setattr(streamlit_app.st, "spinner", lambda *a, **k: dummy)
    monkeypatch.setattr(streamlit_app.st, "success", lambda *a, **k: None)
    monkeypatch.setattr(streamlit_app.st, "text_input", lambda *a, **k: "")
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda *a, **k: None)
    monkeypatch.setattr(streamlit_app.st, "metric", lambda *a, **k: None)
    monkeypatch.setattr(streamlit_app.st, "expander", lambda *a, **k: dummy)
    monkeypatch.setattr(streamlit_app.st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(streamlit_app.st, "write", lambda *a, **k: None)
    monkeypatch.setattr(streamlit_app.st, "divider", lambda *a, **k: None)
    monkeypatch.setattr(streamlit_app.st, "warning", lambda *a, **k: None)
    monkeypatch.setattr(streamlit_app.st, "json", lambda *a, **k: None)
    monkeypatch.setattr(streamlit_app.st, "session_state", {})
    yield

@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    # Patch environment variables for deterministic test
    monkeypatch.setenv("GROQ_API_KEY", "dummy-key")
    monkeypatch.setenv("QA_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    monkeypatch.setenv("CHUNK_SIZE", "1000")
    monkeypatch.setenv("CHUNK_OVERLAP", "200")
    monkeypatch.setenv("TOP_K", "4")
    yield

@pytest.fixture
def ingestion_service():
    return streamlit_app.DocumentIngestionService()

@pytest.fixture
def dummy_chunks():
    # Return a list of two dummy chunks
    meta1 = streamlit_app.DocumentMetadata(filename="a.txt", chunk_id=0, source="a.txt - Part 1")
    meta2 = streamlit_app.DocumentMetadata(filename="a.txt", chunk_id=1, source="a.txt - Part 2")
    return [
        streamlit_app.Chunk(text="First chunk text", metadata=meta1),
        streamlit_app.Chunk(text="Second chunk text", metadata=meta2),
    ]

@pytest.fixture
def vector_store_service(monkeypatch):
    # Patch HuggingFaceEmbeddings and FAISS
    dummy_embeddings = MagicMock()
    dummy_faiss = MagicMock()
    monkeypatch.setattr(streamlit_app, "HuggingFaceEmbeddings", lambda model_name: dummy_embeddings)
    monkeypatch.setattr(streamlit_app, "FAISS", MagicMock(return_value=dummy_faiss))
    monkeypatch.setattr(dummy_faiss, "from_documents", MagicMock(return_value=dummy_faiss))
    monkeypatch.setattr(dummy_faiss, "add_documents", MagicMock())
    return streamlit_app.VectorStoreService()

@pytest.fixture
def rag_service(monkeypatch, vector_store_service, dummy_chunks):
    # Patch ChatGroq, ChatPromptTemplate, StrOutputParser, Document
    dummy_llm = MagicMock()
    dummy_prompt = MagicMock()
    dummy_parser = MagicMock()
    dummy_chain = MagicMock()
    dummy_doc1 = MagicMock()
    dummy_doc2 = MagicMock()
    dummy_doc1.page_content = "First chunk text"
    dummy_doc1.metadata = dummy_chunks[0].metadata.model_dump()
    dummy_doc2.page_content = "Second chunk text"
    dummy_doc2.metadata = dummy_chunks[1].metadata.model_dump()
    dummy_retriever = MagicMock()
    dummy_retriever.invoke = MagicMock(return_value=[dummy_doc1, dummy_doc2])
    dummy_vs = MagicMock()
    dummy_vs.as_retriever = MagicMock(return_value=dummy_retriever)
    vector_store_service.vector_store = dummy_vs

    monkeypatch.setattr(streamlit_app, "ChatGroq", MagicMock(return_value=dummy_llm))
    monkeypatch.setattr(streamlit_app, "ChatPromptTemplate", MagicMock())
    streamlit_app.ChatPromptTemplate.from_template = MagicMock(return_value=dummy_prompt)
    dummy_prompt.__or__ = lambda self, other: dummy_chain
    dummy_chain.invoke = MagicMock(return_value="The answer is 42.")
    monkeypatch.setattr(streamlit_app, "StrOutputParser", MagicMock(return_value=dummy_parser))
    dummy_parser.__or__ = lambda self, other: dummy_chain

    return streamlit_app.RAGService(vector_store_service)

@pytest.fixture
def extraction_service(monkeypatch):
    dummy_llm = MagicMock()
    dummy_parser = MagicMock()
    dummy_prompt = MagicMock()
    dummy_chain = MagicMock()
    monkeypatch.setattr(streamlit_app, "ChatGroq", MagicMock(return_value=dummy_llm))
    monkeypatch.setattr(streamlit_app, "PydanticOutputParser", MagicMock(return_value=dummy_parser))
    monkeypatch.setattr(streamlit_app, "ChatPromptTemplate", MagicMock())
    streamlit_app.ChatPromptTemplate.from_template = MagicMock(return_value=dummy_prompt)
    dummy_parser.get_format_instructions = MagicMock(return_value="FORMAT")
    dummy_prompt.__or__ = lambda self, other: dummy_chain
    dummy_chain.invoke = MagicMock(return_value=streamlit_app.ShipmentData(reference_id="ABC123"))
    dummy_parser.__or__ = lambda self, other: dummy_chain
    return streamlit_app.ExtractionService()

def test_process_file_pdf_and_txt_equivalence(monkeypatch, ingestion_service):
    # Patch pymupdf.open and docx.Document
    dummy_pdf = MagicMock()
    dummy_page = MagicMock()
    dummy_page.get_text.return_value = "Carrier Details: ACME\nRate Breakdown: $1000"
    dummy_pdf.__iter__ = lambda self: iter([dummy_page])
    monkeypatch.setattr(streamlit_app.pymupdf, "open", MagicMock(return_value=dummy_pdf))
    # PDF
    pdf_bytes = b"%PDF-1.4 dummy"
    pdf_chunks = ingestion_service.process_file(pdf_bytes, "test.pdf")
    # TXT
    txt_bytes = b"Carrier Details: ACME\nRate Breakdown: $1000"
    txt_chunks = ingestion_service.process_file(txt_bytes, "test.txt")
    # Reconciliation: Both should produce similar chunk text content (ignoring chunking differences)
    pdf_texts = [c.text for c in pdf_chunks]
    txt_texts = [c.text for c in txt_chunks]
    # Both should contain the semantic markers
    assert any("## Carrier Details" in t for t in pdf_texts)
    assert any("## Carrier Details" in t for t in txt_texts)
    assert any("## Rate Breakdown" in t for t in pdf_texts)
    assert any("## Rate Breakdown" in t for t in txt_texts)
    # The text content should be similar
    assert "ACME" in "".join(pdf_texts)
    assert "ACME" in "".join(txt_texts)

def test_process_file_docx_and_txt_equivalence(monkeypatch, ingestion_service):
    # Patch docx.Document
    dummy_docx = MagicMock()
    dummy_para1 = MagicMock()
    dummy_para1.text = "Carrier Details: ACME"
    dummy_para2 = MagicMock()
    dummy_para2.text = "Rate Breakdown: $1000"
    dummy_docx.paragraphs = [dummy_para1, dummy_para2]
    monkeypatch.setattr(streamlit_app.docx, "Document", MagicMock(return_value=dummy_docx))
    docx_bytes = b"dummy docx"
    docx_chunks = ingestion_service.process_file(docx_bytes, "test.docx")
    txt_bytes = b"Carrier Details: ACME\nRate Breakdown: $1000"
    txt_chunks = ingestion_service.process_file(txt_bytes, "test.txt")
    docx_texts = [c.text for c in docx_chunks]
    txt_texts = [c.text for c in txt_chunks]
    assert any("## Carrier Details" in t for t in docx_texts)
    assert any("## Carrier Details" in t for t in txt_texts)
    assert any("## Rate Breakdown" in t for t in docx_texts)
    assert any("## Rate Breakdown" in t for t in txt_texts)
    assert "ACME" in "".join(docx_texts)
    assert "ACME" in "".join(txt_texts)

def test_process_file_empty(monkeypatch, ingestion_service):
    # Empty file should produce at least one chunk (possibly empty)
    empty_chunks = ingestion_service.process_file(b"", "empty.txt")
    assert isinstance(empty_chunks, list)
    assert len(empty_chunks) >= 1
    # All chunk texts should be empty or whitespace
    for c in empty_chunks:
        assert isinstance(c, streamlit_app.Chunk)
        assert c.text.strip() == ""

def test_vector_store_add_documents_creates_and_adds(monkeypatch, dummy_chunks):
    # Patch HuggingFaceEmbeddings and FAISS
    dummy_embeddings = MagicMock()
    dummy_faiss = MagicMock()
    monkeypatch.setattr(streamlit_app, "HuggingFaceEmbeddings", lambda model_name: dummy_embeddings)
    monkeypatch.setattr(streamlit_app, "FAISS", MagicMock())
    streamlit_app.FAISS.from_documents = MagicMock(return_value=dummy_faiss)
    dummy_faiss.add_documents = MagicMock()
    vs = streamlit_app.VectorStoreService()
    # First call: should create vector_store
    vs.add_documents(dummy_chunks)
    streamlit_app.FAISS.from_documents.assert_called_once()
    # Second call: should add to existing vector_store
    vs.add_documents(dummy_chunks)
    dummy_faiss.add_documents.assert_called()

def test_rag_service_answer_equivalence(rag_service, dummy_chunks):
    # The answer method should return SourcedAnswer with correct fields
    result = rag_service.answer("What is the answer?")
    assert isinstance(result, streamlit_app.SourcedAnswer)
    assert result.answer == "The answer is 42."
    assert result.confidence_score == 0.9
    # Sources should match dummy_chunks content
    assert len(result.sources) == 2
    assert result.sources[0].text == "First chunk text"
    assert result.sources[1].text == "Second chunk text"

def test_rag_service_answer_low_confidence(rag_service, dummy_chunks):
    # Patch chain.invoke to return "I cannot find the answer in the provided documents."
    rag_service.llm = MagicMock()
    rag_service.prompt = MagicMock()
    dummy_chain = MagicMock()
    dummy_chain.invoke = MagicMock(return_value="I cannot find the answer in the provided documents.")
    rag_service.prompt.__or__ = lambda self, other: dummy_chain
    rag_service.llm.__or__ = lambda self, other: dummy_chain
    rag_service.vs.vector_store.as_retriever().invoke = MagicMock(return_value=[
        MagicMock(page_content="irrelevant", metadata=dummy_chunks[0].metadata.model_dump())
    ])
    result = rag_service.answer("Unknown question?")
    assert result.confidence_score == 0.1
    assert "cannot find" in result.answer.lower()

def test_extraction_service_extract_equivalence(extraction_service):
    # Should return a ShipmentData object with reference_id set
    result = extraction_service.extract("Reference ID: ABC123")
    assert isinstance(result, streamlit_app.ShipmentData)
    assert result.reference_id == "ABC123"

def test_document_metadata_boundary_conditions():
    # Test with only required fields
    meta = streamlit_app.DocumentMetadata(filename="file.txt", chunk_id=0, source="src")
    assert meta.filename == "file.txt"
    assert meta.chunk_id == 0
    assert meta.source == "src"
    assert meta.chunk_type == "text"
    assert meta.page_number is None

def test_chunk_and_sourcedanswer_models():
    meta = streamlit_app.DocumentMetadata(filename="file.txt", chunk_id=1, source="src")
    chunk = streamlit_app.Chunk(text="abc", metadata=meta)
    assert chunk.text == "abc"
    assert chunk.metadata == meta
    answer = streamlit_app.SourcedAnswer(answer="42", confidence_score=0.8, sources=[chunk])
    assert answer.answer == "42"
    assert answer.confidence_score == 0.8
    assert answer.sources[0] == chunk

def test_shipment_data_edge_cases():
    # All optional fields None
    data = streamlit_app.ShipmentData()
    assert data.reference_id is None
    assert data.carrier is None
    # Partial data
    data2 = streamlit_app.ShipmentData(reference_id="X", shipper="Y")
    assert data2.reference_id == "X"
    assert data2.shipper == "Y"
    # Nested models
    carrier = streamlit_app.CarrierInfo(carrier_name="CarrierX")
    data3 = streamlit_app.ShipmentData(carrier=carrier)
    assert data3.carrier.carrier_name == "CarrierX"
