import io
import os
import sys
import types
import pytest

import streamlit_app

from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def patch_streamlit(monkeypatch):
    # Patch all streamlit UI calls to no-ops or mocks for integration testing
    st_mock = MagicMock()
    monkeypatch.setattr(streamlit_app, "st", st_mock)
    st_mock.session_state = {}
    st_mock.file_uploader.return_value = []
    st_mock.button.return_value = False
    st_mock.text_input.return_value = ""
    st_mock.spinner.return_value.__enter__.return_value = None
    st_mock.spinner.return_value.__exit__.return_value = None
    st_mock.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    st_mock.set_page_config.return_value = None
    st_mock.title.return_value = None
    st_mock.info.return_value = None
    st_mock.error.return_value = None
    st_mock.stop.side_effect = Exception("Streamlit stopped")
    st_mock.header.return_value = None
    st_mock.success.return_value = None
    st_mock.warning.return_value = None
    st_mock.metric.return_value = None
    st_mock.markdown.return_value = None
    st_mock.caption.return_value = None
    st_mock.write.return_value = None
    st_mock.divider.return_value = None
    st_mock.expander.return_value.__enter__.return_value = None
    st_mock.expander.return_value.__exit__.return_value = None
    st_mock.json.return_value = None
    return st_mock

@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    # Patch environment variables for deterministic test
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("QA_MODEL", "test-qa-model")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "10")
    monkeypatch.setenv("TOP_K", "2")

@pytest.fixture
def sample_pdf_bytes():
    # Minimal PDF file bytes (header only, not a valid PDF but enough for mock)
    return b"%PDF-1.4\n%EOF"

@pytest.fixture
def sample_docx_bytes():
    # Minimal docx file bytes (not a valid docx, but enough for mock)
    return b"PK\x03\x04docx-content"

@pytest.fixture
def sample_txt_bytes():
    return b"Carrier Details\nJohn Doe\nRate Breakdown\n$1000\nPickup\nNew York\nDrop\nChicago"

@pytest.fixture
def ingestion_service():
    return streamlit_app.DocumentIngestionService()

@pytest.fixture
def vector_store_service():
    # Patch HuggingFaceEmbeddings and FAISS for deterministic, isolated test
    with patch.object(streamlit_app, "HuggingFaceEmbeddings") as emb_mock, \
         patch.object(streamlit_app, "FAISS") as faiss_mock:
        emb_mock.return_value = MagicMock()
        faiss_instance = MagicMock()
        faiss_mock.from_documents.return_value = faiss_instance
        faiss_instance.as_retriever.return_value.invoke.return_value = []
        yield streamlit_app.VectorStoreService()

@pytest.fixture
def rag_service(vector_store_service):
    # Patch ChatGroq, ChatPromptTemplate, StrOutputParser
    with patch.object(streamlit_app, "ChatGroq") as chat_mock, \
         patch.object(streamlit_app, "ChatPromptTemplate") as prompt_mock, \
         patch.object(streamlit_app, "StrOutputParser") as parser_mock:
        chat_instance = MagicMock()
        chat_instance.invoke.return_value = "Test answer"
        chat_mock.return_value = chat_instance
        prompt_instance = MagicMock()
        prompt_instance.__or__.return_value = lambda x: "Test answer"
        prompt_mock.from_template.return_value = prompt_instance
        parser_instance = MagicMock()
        parser_instance.invoke.return_value = "Test answer"
        parser_mock.return_value = parser_instance
        yield streamlit_app.RAGService(vector_store_service)

@pytest.fixture
def extraction_service():
    # Patch ChatGroq, ChatPromptTemplate, PydanticOutputParser
    with patch.object(streamlit_app, "ChatGroq") as chat_mock, \
         patch.object(streamlit_app, "ChatPromptTemplate") as prompt_mock, \
         patch.object(streamlit_app, "PydanticOutputParser") as parser_mock:
        chat_instance = MagicMock()
        chat_instance.invoke.return_value = {"reference_id": "REF123"}
        chat_mock.return_value = chat_instance
        prompt_instance = MagicMock()
        prompt_instance.__or__.return_value = lambda x: streamlit_app.ShipmentData(reference_id="REF123")
        prompt_mock.from_template.return_value = prompt_instance
        parser_instance = MagicMock()
        parser_instance.invoke.return_value = streamlit_app.ShipmentData(reference_id="REF123")
        parser_instance.get_format_instructions.return_value = "format"
        parser_mock.return_value = parser_instance
        yield streamlit_app.ExtractionService()

def test_process_file_txt_happy_path(ingestion_service, sample_txt_bytes):
    chunks = ingestion_service.process_file(sample_txt_bytes, "test.txt")
    assert isinstance(chunks, list)
    assert all(isinstance(c, streamlit_app.Chunk) for c in chunks)
    assert any("Carrier Details" in c.text for c in chunks)
    assert all(c.metadata.filename == "test.txt" for c in chunks)
    assert all(c.metadata.chunk_type == "text" for c in chunks)
    # Check chunk_id increments
    ids = [c.metadata.chunk_id for c in chunks]
    assert ids == list(range(len(chunks)))

def test_process_file_docx_happy_path(ingestion_service, sample_docx_bytes):
    # Patch docx.Document to return paragraphs
    with patch.object(streamlit_app.docx, "Document") as docx_mock:
        doc_mock = MagicMock()
        doc_mock.paragraphs = [MagicMock(text="Carrier Details"), MagicMock(text="Pickup")]
        docx_mock.return_value = doc_mock
        chunks = ingestion_service.process_file(sample_docx_bytes, "test.docx")
        assert any("Carrier Details" in c.text for c in chunks)
        assert any("Pickup" in c.text for c in chunks)

def test_process_file_pdf_happy_path(ingestion_service, sample_pdf_bytes):
    # Patch pymupdf.open to return a doc with pages
    with patch.object(streamlit_app.pymupdf, "open") as pdf_open:
        page_mock = MagicMock()
        page_mock.get_text.return_value = "Carrier Details\nPickup"
        doc_mock = [page_mock, page_mock]
        pdf_open.return_value = doc_mock
        chunks = ingestion_service.process_file(sample_pdf_bytes, "test.pdf")
        assert any("Carrier Details" in c.text for c in chunks)
        assert any("Page 1" in c.text or "Page 2" in c.text for c in chunks)

def test_process_file_empty_file(ingestion_service):
    chunks = ingestion_service.process_file(b"", "empty.txt")
    assert isinstance(chunks, list)
    assert all(isinstance(c, streamlit_app.Chunk) for c in chunks)
    # Should still produce at least one chunk (even if empty)
    assert len(chunks) >= 1

def test_process_file_unknown_extension(ingestion_service):
    # Should not raise, just skip extraction
    chunks = ingestion_service.process_file(b"Some data", "file.unknown")
    assert isinstance(chunks, list)
    assert all(isinstance(c, streamlit_app.Chunk) for c in chunks)

def test_vector_store_add_documents_creates_store(vector_store_service, ingestion_service, sample_txt_bytes):
    # Patch FAISS.from_documents to check call
    with patch.object(streamlit_app.FAISS, "from_documents") as from_docs:
        from_docs.return_value = MagicMock()
        chunks = ingestion_service.process_file(sample_txt_bytes, "test.txt")
        vector_store_service.add_documents(chunks)
        from_docs.assert_called_once()

def test_vector_store_add_documents_appends(vector_store_service, ingestion_service, sample_txt_bytes):
    # Simulate existing vector_store
    chunks = ingestion_service.process_file(sample_txt_bytes, "test.txt")
    vector_store_service.vector_store = MagicMock()
    vector_store_service.add_documents(chunks)
    vector_store_service.vector_store.add_documents.assert_called()

def test_rag_service_answer_happy_path(rag_service, vector_store_service):
    # Patch retriever.invoke to return mock docs
    doc_mock = MagicMock()
    doc_mock.page_content = "Carrier Details: John Doe"
    doc_mock.metadata = {
        "filename": "test.txt",
        "chunk_id": 0,
        "source": "test.txt - Part 1",
        "chunk_type": "text"
    }
    vector_store_service.vector_store = MagicMock()
    retriever = MagicMock()
    retriever.invoke.return_value = [doc_mock]
    vector_store_service.vector_store.as_retriever.return_value = retriever
    result = rag_service.answer("Who is the carrier?")
    assert isinstance(result, streamlit_app.SourcedAnswer)
    assert "Test answer" in result.answer
    assert result.confidence_score == 0.9
    assert len(result.sources) == 1
    assert result.sources[0].text == "Carrier Details: John Doe"

def test_rag_service_answer_not_found(rag_service, vector_store_service):
    # Patch retriever.invoke to return empty and answer with "cannot find"
    doc_mock = MagicMock()
    doc_mock.page_content = ""
    doc_mock.metadata = {
        "filename": "test.txt",
        "chunk_id": 0,
        "source": "test.txt - Part 1",
        "chunk_type": "text"
    }
    vector_store_service.vector_store = MagicMock()
    retriever = MagicMock()
    retriever.invoke.return_value = [doc_mock]
    vector_store_service.vector_store.as_retriever.return_value = retriever
    # Patch chain.invoke to return "I cannot find the answer in the provided documents."
    with patch.object(rag_service, "prompt") as prompt_mock, \
         patch.object(rag_service, "llm") as llm_mock, \
         patch.object(streamlit_app, "StrOutputParser") as parser_mock:
        chain = MagicMock()
        chain.invoke.return_value = "I cannot find the answer in the provided documents."
        prompt_mock.__or__.return_value = chain
        llm_mock.__or__.return_value = chain
        parser_mock.return_value = chain
        result = rag_service.answer("Unknown question?")
        assert result.confidence_score == 0.1
        assert "cannot find" in result.answer.lower()

def test_extraction_service_extract_happy_path(extraction_service):
    data = extraction_service.extract("Carrier: ACME\nPickup: NYC")
    assert isinstance(data, streamlit_app.ShipmentData)
    assert data.reference_id == "REF123"

def test_extraction_service_extract_missing_fields(extraction_service):
    # Should fill missing fields with None
    data = extraction_service.extract("")
    assert isinstance(data, streamlit_app.ShipmentData)
    assert data.reference_id == "REF123"

def test_document_metadata_boundary_conditions():
    # Test with minimal and maximal values
    meta = streamlit_app.DocumentMetadata(filename="a"*255, chunk_id=2**31-1, source="src", chunk_type="text")
    assert meta.filename == "a"*255
    assert meta.chunk_id == 2**31-1
    assert meta.source == "src"
    assert meta.chunk_type == "text"
    # page_number is optional
    meta2 = streamlit_app.DocumentMetadata(filename="b", chunk_id=0, source="src2")
    assert meta2.page_number is None

def test_chunk_and_sourcedanswer_repr():
    meta = streamlit_app.DocumentMetadata(filename="f", chunk_id=1, source="s")
    chunk = streamlit_app.Chunk(text="abc", metadata=meta)
    answer = streamlit_app.SourcedAnswer(answer="ans", confidence_score=0.5, sources=[chunk])
    assert chunk.text == "abc"
    assert answer.answer == "ans"
    assert answer.confidence_score == 0.5
    assert answer.sources[0] == chunk

def test_shipment_data_model_fields():
    # Test all optional fields
    data = streamlit_app.ShipmentData(
        reference_id="RID",
        shipper="SHP",
        consignee="CNEE",
        carrier=streamlit_app.CarrierInfo(carrier_name="Carrier", mc_number="123", phone="555-5555"),
        driver=streamlit_app.DriverInfo(driver_name="Driver", cell_number="555-0000", truck_number="TRK1"),
        pickup=streamlit_app.Location(name="Pickup", address="Addr", city="City", state="ST", zip_code="00000", appointment_time="10:00"),
        drop=streamlit_app.Location(name="Drop", address="Addr2", city="City2", state="ST2", zip_code="11111", appointment_time="12:00"),
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Van",
        rate_info=streamlit_app.RateInfo(total_rate=1000.0, currency="USD", rate_breakdown={"base": 900, "fuel": 100}),
        special_instructions="None"
    )
    assert data.reference_id == "RID"
    assert data.carrier.carrier_name == "Carrier"
    assert data.driver.truck_number == "TRK1"
    assert data.pickup.city == "City"
    assert data.drop.zip_code == "11111"
    assert data.rate_info.total_rate == 1000.0
    assert data.special_instructions == "None"

def test_section_groups_are_defined(ingestion_service):
    # Ensure all expected section groups are present
    expected = {'carrier_info', 'customer_info', 'location_info', 'rate_info', 'commodity_info', 'instructions'}
    assert set(ingestion_service.section_groups.keys()) == expected

def test_text_splitter_config(ingestion_service):
    splitter = ingestion_service.text_splitter
    assert splitter.chunk_size == 200  # CHUNK_SIZE * 2 from env
    assert splitter.chunk_overlap == 10
    assert splitter.separators == ["\n## ", "\n\n", "\n", ". ", " ", ""]

def test_error_on_missing_groq_key(monkeypatch):
    # Remove GROQ_API_KEY and check st.error and st.stop are called
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with patch.object(streamlit_app, "st") as st_mock:
        st_mock.session_state = {}
        st_mock.error = MagicMock()
        st_mock.stop = MagicMock(side_effect=Exception("Streamlit stopped"))
        with pytest.raises(Exception, match="Streamlit stopped"):
            # Re-import to trigger the check
            import importlib
            importlib.reload(streamlit_app)
        st_mock.error.assert_called()
        st_mock.stop.assert_called()
