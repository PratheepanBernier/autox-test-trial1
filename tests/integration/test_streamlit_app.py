# source_hash: 5fab573753f93532
import io
import os
import sys
import types
import pytest

from unittest.mock import patch, MagicMock, call

import streamlit_app

@pytest.fixture(autouse=True)
def patch_streamlit(monkeypatch):
    # Patch all streamlit UI functions to be no-ops or record calls
    st_mock = MagicMock()
    monkeypatch.setattr(streamlit_app, "st", st_mock)
    # Patch set_page_config to avoid side effects
    st_mock.set_page_config.return_value = None
    st_mock.session_state = {}
    st_mock.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    st_mock.file_uploader.return_value = []
    st_mock.button.return_value = False
    st_mock.text_input.return_value = ""
    st_mock.spinner.return_value = MagicMock()
    st_mock.header.return_value = None
    st_mock.title.return_value = None
    st_mock.info.return_value = None
    st_mock.error.return_value = None
    st_mock.stop.side_effect = Exception("Streamlit stopped")
    st_mock.success.return_value = None
    st_mock.warning.return_value = None
    st_mock.metric.return_value = None
    st_mock.markdown.return_value = None
    st_mock.caption.return_value = None
    st_mock.write.return_value = None
    st_mock.divider.return_value = None
    st_mock.expander.return_value = MagicMock()
    st_mock.json.return_value = None
    return st_mock

@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    # Patch environment variables for deterministic tests
    monkeypatch.setenv("GROQ_API_KEY", "dummy-key")
    monkeypatch.setenv("QA_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    monkeypatch.setenv("CHUNK_SIZE", "1000")
    monkeypatch.setenv("CHUNK_OVERLAP", "200")
    monkeypatch.setenv("TOP_K", "4")

@pytest.fixture
def dummy_pdf_bytes():
    # Minimal valid PDF header and one page text
    return b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<<>>\nstartxref\n9\n%%EOF"

@pytest.fixture
def dummy_docx_bytes():
    # Not a real docx, but enough for docx.Document(BytesIO(...)) to fail gracefully
    return b"PK\x03\x04dummy-docx-content"

@pytest.fixture
def dummy_txt_bytes():
    return b"Carrier Details\nJohn Doe\nRate Breakdown\n$1000\nPickup\nNYC\nDrop\nLA\nCommodity\nWidgets\nSpecial Instructions\nNone"

@pytest.fixture
def ingestion_service():
    return streamlit_app.DocumentIngestionService()

@pytest.fixture
def vector_store_service(monkeypatch):
    # Patch HuggingFaceEmbeddings and FAISS
    embeddings_mock = MagicMock()
    monkeypatch.setattr(streamlit_app, "HuggingFaceEmbeddings", MagicMock(return_value=embeddings_mock))
    faiss_mock = MagicMock()
    faiss_instance = MagicMock()
    faiss_mock.from_documents.return_value = faiss_instance
    monkeypatch.setattr(streamlit_app, "FAISS", faiss_mock)
    return streamlit_app.VectorStoreService()

@pytest.fixture
def rag_service(vector_store_service, monkeypatch):
    # Patch ChatGroq, ChatPromptTemplate, StrOutputParser
    llm_mock = MagicMock()
    monkeypatch.setattr(streamlit_app, "ChatGroq", MagicMock(return_value=llm_mock))
    prompt_mock = MagicMock()
    monkeypatch.setattr(streamlit_app, "ChatPromptTemplate", MagicMock())
    streamlit_app.ChatPromptTemplate.from_template.return_value = prompt_mock
    str_output_parser_mock = MagicMock()
    monkeypatch.setattr(streamlit_app, "StrOutputParser", MagicMock(return_value=str_output_parser_mock))
    return streamlit_app.RAGService(vector_store_service)

@pytest.fixture
def extraction_service(monkeypatch):
    llm_mock = MagicMock()
    monkeypatch.setattr(streamlit_app, "ChatGroq", MagicMock(return_value=llm_mock))
    parser_mock = MagicMock()
    monkeypatch.setattr(streamlit_app, "PydanticOutputParser", MagicMock(return_value=parser_mock))
    prompt_mock = MagicMock()
    monkeypatch.setattr(streamlit_app, "ChatPromptTemplate", MagicMock())
    streamlit_app.ChatPromptTemplate.from_template.return_value = prompt_mock
    return streamlit_app.ExtractionService()

def test_process_file_txt_happy_path(ingestion_service, dummy_txt_bytes):
    chunks = ingestion_service.process_file(dummy_txt_bytes, "test.txt")
    assert isinstance(chunks, list)
    assert all(isinstance(c, streamlit_app.Chunk) for c in chunks)
    # Should have at least one chunk and semantic markers inserted
    assert any("## Carrier Details" in c.text for c in chunks)
    assert any("## Rate Breakdown" in c.text for c in chunks)
    assert any("## Pickup" in c.text for c in chunks)
    assert any("## Drop" in c.text for c in chunks)
    assert any("## Commodity" in c.text for c in chunks)
    assert any("## Special Instructions" in c.text for c in chunks)

def test_process_file_pdf_edge(monkeypatch, ingestion_service, dummy_pdf_bytes):
    # Patch pymupdf.open to simulate PDF reading
    page_mock = MagicMock()
    page_mock.get_text.return_value = "Carrier Details\nJohn Doe"
    doc_mock = [page_mock]
    pymupdf_mock = MagicMock()
    pymupdf_mock.open.return_value = doc_mock
    monkeypatch.setattr(streamlit_app, "pymupdf", pymupdf_mock)
    chunks = ingestion_service.process_file(dummy_pdf_bytes, "test.pdf")
    assert isinstance(chunks, list)
    assert any("Carrier Details" in c.text for c in chunks)

def test_process_file_docx_boundary(monkeypatch, ingestion_service, dummy_docx_bytes):
    # Patch docx.Document to simulate docx reading
    para_mock = MagicMock()
    para_mock.text = "Carrier Details\nJohn Doe"
    docx_doc_mock = MagicMock()
    docx_doc_mock.paragraphs = [para_mock]
    docx_mock = MagicMock()
    docx_mock.Document.return_value = docx_doc_mock
    monkeypatch.setattr(streamlit_app, "docx", docx_mock)
    chunks = ingestion_service.process_file(dummy_docx_bytes, "test.docx")
    assert isinstance(chunks, list)
    assert any("Carrier Details" in c.text for c in chunks)

def test_process_file_empty(monkeypatch, ingestion_service):
    # Empty file should produce at least one chunk (possibly empty)
    chunks = ingestion_service.process_file(b"", "empty.txt")
    assert isinstance(chunks, list)
    assert all(isinstance(c, streamlit_app.Chunk) for c in chunks)

def test_vector_store_add_documents_creates_store(monkeypatch, vector_store_service):
    # Patch Document and FAISS
    doc_mock = MagicMock()
    monkeypatch.setattr(streamlit_app, "Document", MagicMock(return_value=doc_mock))
    chunks = [streamlit_app.Chunk(text="foo", metadata=streamlit_app.DocumentMetadata(filename="a", chunk_id=0, source="a", chunk_type="text"))]
    vector_store_service.add_documents(chunks)
    assert vector_store_service.vector_store is not None
    # Should call FAISS.from_documents
    streamlit_app.FAISS.from_documents.assert_called_once()

def test_vector_store_add_documents_appends(monkeypatch, vector_store_service):
    # Simulate existing vector_store
    vector_store_service.vector_store = MagicMock()
    doc_mock = MagicMock()
    monkeypatch.setattr(streamlit_app, "Document", MagicMock(return_value=doc_mock))
    chunks = [streamlit_app.Chunk(text="bar", metadata=streamlit_app.DocumentMetadata(filename="b", chunk_id=1, source="b", chunk_type="text"))]
    vector_store_service.add_documents(chunks)
    vector_store_service.vector_store.add_documents.assert_called_once()

def test_rag_service_answer_happy_path(monkeypatch, rag_service):
    # Patch retriever and chain
    doc1 = MagicMock()
    doc1.page_content = "Carrier Details: John Doe"
    doc1.metadata = {"filename": "a", "chunk_id": 0, "source": "a", "chunk_type": "text"}
    retriever_mock = MagicMock()
    retriever_mock.invoke.return_value = [doc1]
    rag_service.vs.vector_store = MagicMock()
    rag_service.vs.vector_store.as_retriever.return_value = retriever_mock
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = "The carrier is John Doe."
    rag_service.prompt.__or__.return_value = chain_mock
    # Patch StrOutputParser to just return chain_mock
    monkeypatch.setattr(streamlit_app, "StrOutputParser", MagicMock(return_value=chain_mock))
    answer = rag_service.answer("Who is the carrier?")
    assert isinstance(answer, streamlit_app.SourcedAnswer)
    assert answer.confidence_score > 0.5
    assert "John Doe" in answer.answer
    assert len(answer.sources) == 1

def test_rag_service_answer_not_found(monkeypatch, rag_service):
    doc1 = MagicMock()
    doc1.page_content = "No relevant info"
    doc1.metadata = {"filename": "a", "chunk_id": 0, "source": "a", "chunk_type": "text"}
    retriever_mock = MagicMock()
    retriever_mock.invoke.return_value = [doc1]
    rag_service.vs.vector_store = MagicMock()
    rag_service.vs.vector_store.as_retriever.return_value = retriever_mock
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = "I cannot find the answer in the provided documents."
    rag_service.prompt.__or__.return_value = chain_mock
    monkeypatch.setattr(streamlit_app, "StrOutputParser", MagicMock(return_value=chain_mock))
    answer = rag_service.answer("What is the moon made of?")
    assert isinstance(answer, streamlit_app.SourcedAnswer)
    assert answer.confidence_score < 0.5
    assert "cannot find" in answer.answer.lower()

def test_extraction_service_extract_happy_path(monkeypatch, extraction_service):
    # Patch chain.invoke to return a ShipmentData instance
    shipment_data = streamlit_app.ShipmentData(reference_id="123", shipper="Acme", consignee="Beta")
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = shipment_data
    extraction_service.prompt.__or__.return_value = chain_mock
    extraction_service.parser.get_format_instructions.return_value = "format"
    result = extraction_service.extract("Reference: 123\nShipper: Acme\nConsignee: Beta")
    assert isinstance(result, streamlit_app.ShipmentData)
    assert result.reference_id == "123"
    assert result.shipper == "Acme"
    assert result.consignee == "Beta"

def test_extraction_service_extract_missing_fields(monkeypatch, extraction_service):
    # Patch chain.invoke to return ShipmentData with None fields
    shipment_data = streamlit_app.ShipmentData()
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = shipment_data
    extraction_service.prompt.__or__.return_value = chain_mock
    extraction_service.parser.get_format_instructions.return_value = "format"
    result = extraction_service.extract("No relevant info")
    assert isinstance(result, streamlit_app.ShipmentData)
    assert result.reference_id is None
    assert result.shipper is None
    assert result.consignee is None

def test_document_metadata_boundary_conditions():
    # Test DocumentMetadata with and without optional fields
    meta1 = streamlit_app.DocumentMetadata(filename="a", chunk_id=1, source="src")
    assert meta1.filename == "a"
    assert meta1.page_number is None
    meta2 = streamlit_app.DocumentMetadata(filename="b", chunk_id=2, source="src", page_number=5)
    assert meta2.page_number == 5

def test_chunk_and_sourced_answer_repr():
    meta = streamlit_app.DocumentMetadata(filename="a", chunk_id=1, source="src")
    chunk = streamlit_app.Chunk(text="foo", metadata=meta)
    answer = streamlit_app.SourcedAnswer(answer="bar", confidence_score=0.8, sources=[chunk])
    assert chunk.text == "foo"
    assert answer.answer == "bar"
    assert answer.confidence_score == 0.8
    assert answer.sources[0] == chunk

def test_location_and_commodityitem_models():
    loc = streamlit_app.Location(name="Depot", address="123 Main", city="NY", state="NY", zip_code="10001", appointment_time="10:00")
    assert loc.name == "Depot"
    item = streamlit_app.CommodityItem(commodity_name="Widgets", weight="100kg", quantity="10")
    assert item.commodity_name == "Widgets"

def test_carrierinfo_and_driverinfo_models():
    carrier = streamlit_app.CarrierInfo(carrier_name="CarrierX", mc_number="12345", phone="555-1234")
    assert carrier.carrier_name == "CarrierX"
    driver = streamlit_app.DriverInfo(driver_name="Bob", cell_number="555-5678", truck_number="TX-100")
    assert driver.driver_name == "Bob"

def test_rateinfo_and_shipmentdata_models():
    rate = streamlit_app.RateInfo(total_rate=1000.0, currency="USD", rate_breakdown={"base": 900, "fuel": 100})
    assert rate.total_rate == 1000.0
    shipment = streamlit_app.ShipmentData(reference_id="REF", shipper="S", consignee="C", carrier=None, driver=None, pickup=None, drop=None, shipping_date=None, delivery_date=None, equipment_type=None, rate_info=None, special_instructions=None)
    assert shipment.reference_id == "REF"
    assert shipment.shipper == "S"
    assert shipment.consignee == "C"

def test_document_ingestion_service_section_groups():
    service = streamlit_app.DocumentIngestionService()
    assert "carrier_info" in service.section_groups
    assert isinstance(service.text_splitter, streamlit_app.RecursiveCharacterTextSplitter)
