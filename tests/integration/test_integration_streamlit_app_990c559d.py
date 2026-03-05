# source_hash: 5fab573753f93532
# import_target: streamlit_app
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import io
import types
import pytest
import streamlit_app
from unittest.mock import patch, MagicMock, call

@pytest.fixture(autouse=True)
def patch_streamlit(monkeypatch):
    # Patch all streamlit UI functions to no-op or capture calls
    st_mock = MagicMock()
    monkeypatch.setattr(streamlit_app, "st", st_mock)
    return st_mock

@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    # Patch os.getenv to always return a dummy GROQ_API_KEY for tests
    monkeypatch.setattr(streamlit_app.os, "getenv", lambda k, d=None: "dummy-key" if k == "GROQ_API_KEY" else d)

@pytest.fixture
def dummy_pdf_bytes():
    # Minimal PDF bytes for pymupdf
    return b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"

@pytest.fixture
def dummy_docx_bytes():
    # Minimal docx file bytes
    from docx import Document
    buf = io.BytesIO()
    doc = Document()
    doc.add_paragraph("Carrier Details\nCarrier: ACME\nRate Breakdown\n$1000\nPickup\nNYC\nDrop\nLA\nCommodity\nWidgets\nSpecial Instructions\nNone")
    doc.save(buf)
    return buf.getvalue()

@pytest.fixture
def dummy_txt_bytes():
    return b"Carrier Details\nCarrier: ACME\nRate Breakdown\n$1000\nPickup\nNYC\nDrop\nLA\nCommodity\nWidgets\nSpecial Instructions\nNone"

@pytest.fixture
def ingestion_service():
    return streamlit_app.DocumentIngestionService()

@pytest.fixture
def vector_store_service(monkeypatch):
    # Patch HuggingFaceEmbeddings and FAISS
    embeddings_mock = MagicMock()
    faiss_mock = MagicMock()
    monkeypatch.setattr(streamlit_app, "HuggingFaceEmbeddings", MagicMock(return_value=embeddings_mock))
    monkeypatch.setattr(streamlit_app, "FAISS", MagicMock())
    return streamlit_app.VectorStoreService()

@pytest.fixture
def rag_service(vector_store_service, monkeypatch):
    # Patch ChatGroq, ChatPromptTemplate, StrOutputParser
    llm_mock = MagicMock()
    prompt_mock = MagicMock()
    parser_mock = MagicMock()
    monkeypatch.setattr(streamlit_app, "ChatGroq", MagicMock(return_value=llm_mock))
    monkeypatch.setattr(streamlit_app.ChatPromptTemplate, "from_template", MagicMock(return_value=prompt_mock))
    monkeypatch.setattr(streamlit_app, "StrOutputParser", MagicMock(return_value=parser_mock))
    return streamlit_app.RAGService(vector_store_service)

@pytest.fixture
def extraction_service(monkeypatch):
    llm_mock = MagicMock()
    parser_mock = MagicMock()
    prompt_mock = MagicMock()
    monkeypatch.setattr(streamlit_app, "ChatGroq", MagicMock(return_value=llm_mock))
    monkeypatch.setattr(streamlit_app, "PydanticOutputParser", MagicMock(return_value=parser_mock))
    monkeypatch.setattr(streamlit_app.ChatPromptTemplate, "from_template", MagicMock(return_value=prompt_mock))
    return streamlit_app.ExtractionService()

def test_process_file_pdf_happy_path(ingestion_service, monkeypatch, dummy_pdf_bytes):
    # Patch pymupdf.open to return a doc with one page with get_text
    page_mock = MagicMock()
    page_mock.get_text.return_value = "Carrier Details\nCarrier: ACME"
    doc_mock = [page_mock]
    monkeypatch.setattr(streamlit_app.pymupdf, "open", MagicMock(return_value=doc_mock))
    chunks = ingestion_service.process_file(dummy_pdf_bytes, "test.pdf")
    assert isinstance(chunks, list)
    assert any("Carrier Details" in c.text for c in chunks)
    assert all(isinstance(c, streamlit_app.Chunk) for c in chunks)
    assert all(c.metadata.filename == "test.pdf" for c in chunks)

def test_process_file_docx_happy_path(ingestion_service, dummy_docx_bytes):
    chunks = ingestion_service.process_file(dummy_docx_bytes, "test.docx")
    assert isinstance(chunks, list)
    assert any("Carrier Details" in c.text for c in chunks)
    assert all(isinstance(c, streamlit_app.Chunk) for c in chunks)
    assert all(c.metadata.filename == "test.docx" for c in chunks)

def test_process_file_txt_happy_path(ingestion_service, dummy_txt_bytes):
    chunks = ingestion_service.process_file(dummy_txt_bytes, "test.txt")
    assert isinstance(chunks, list)
    assert any("Carrier Details" in c.text for c in chunks)
    assert all(isinstance(c, streamlit_app.Chunk) for c in chunks)
    assert all(c.metadata.filename == "test.txt" for c in chunks)

def test_process_file_empty(monkeypatch, ingestion_service):
    # Should handle empty file gracefully
    monkeypatch.setattr(streamlit_app.pymupdf, "open", MagicMock(return_value=[]))
    chunks = ingestion_service.process_file(b"", "empty.pdf")
    assert isinstance(chunks, list)
    assert all(isinstance(c, streamlit_app.Chunk) for c in chunks)

def test_process_file_unsupported_extension(ingestion_service):
    # Should return empty chunks for unsupported extension
    chunks = ingestion_service.process_file(b"irrelevant", "file.xyz")
    assert isinstance(chunks, list)
    assert len(chunks) == 0

def test_vector_store_add_documents_creates_store(monkeypatch, vector_store_service):
    # Patch FAISS.from_documents to return a mock vector store
    faiss_instance = MagicMock()
    streamlit_app.FAISS.from_documents.return_value = faiss_instance
    chunk = streamlit_app.Chunk(text="test", metadata=streamlit_app.DocumentMetadata(filename="f", chunk_id=0, source="f", chunk_type="text"))
    vector_store_service.add_documents([chunk])
    assert vector_store_service.vector_store == faiss_instance

def test_vector_store_add_documents_appends(monkeypatch, vector_store_service):
    # If vector_store exists, should call add_documents
    faiss_instance = MagicMock()
    vector_store_service.vector_store = faiss_instance
    chunk = streamlit_app.Chunk(text="test", metadata=streamlit_app.DocumentMetadata(filename="f", chunk_id=0, source="f", chunk_type="text"))
    vector_store_service.add_documents([chunk])
    faiss_instance.add_documents.assert_called()

def test_rag_service_answer_happy_path(monkeypatch, rag_service):
    # Patch retriever.invoke and chain.invoke
    doc_mock = MagicMock()
    doc_mock.page_content = "Carrier: ACME"
    doc_mock.metadata = {"filename": "f", "chunk_id": 0, "source": "f", "chunk_type": "text"}
    retriever_mock = MagicMock()
    retriever_mock.invoke.return_value = [doc_mock]
    rag_service.vs.vector_store = MagicMock()
    rag_service.vs.vector_store.as_retriever.return_value = retriever_mock
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = "The carrier is ACME."
    rag_service.prompt.__or__.return_value = chain_mock
    res = rag_service.answer("Who is the carrier?")
    assert isinstance(res, streamlit_app.SourcedAnswer)
    assert "ACME" in res.answer
    assert res.confidence_score > 0.5
    assert isinstance(res.sources, list)
    assert res.sources[0].text == "Carrier: ACME"

def test_rag_service_answer_not_found(monkeypatch, rag_service):
    doc_mock = MagicMock()
    doc_mock.page_content = "No relevant info"
    doc_mock.metadata = {"filename": "f", "chunk_id": 0, "source": "f", "chunk_type": "text"}
    retriever_mock = MagicMock()
    retriever_mock.invoke.return_value = [doc_mock]
    rag_service.vs.vector_store = MagicMock()
    rag_service.vs.vector_store.as_retriever.return_value = retriever_mock
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = "I cannot find the answer in the provided documents."
    rag_service.prompt.__or__.return_value = chain_mock
    res = rag_service.answer("What is the delivery date?")
    assert isinstance(res, streamlit_app.SourcedAnswer)
    assert "cannot find" in res.answer.lower()
    assert res.confidence_score < 0.5

def test_extraction_service_extract_happy_path(extraction_service):
    # Patch chain.invoke to return a ShipmentData instance
    shipment_data = streamlit_app.ShipmentData(reference_id="123", shipper="ACME", consignee="XYZ")
    chain_mock = MagicMock()
    extraction_service.prompt.__or__.return_value = chain_mock
    chain_mock.invoke.return_value = shipment_data
    res = extraction_service.extract("Carrier: ACME\nShipper: XYZ")
    assert isinstance(res, streamlit_app.ShipmentData)
    assert res.shipper == "ACME" or res.consignee == "XYZ"

def test_extraction_service_extract_missing_fields(extraction_service):
    # Patch chain.invoke to return ShipmentData with None fields
    shipment_data = streamlit_app.ShipmentData()
    chain_mock = MagicMock()
    extraction_service.prompt.__or__.return_value = chain_mock
    chain_mock.invoke.return_value = shipment_data
    res = extraction_service.extract("No relevant info")
    assert isinstance(res, streamlit_app.ShipmentData)
    assert res.reference_id is None
    assert res.shipper is None

def test_document_metadata_boundary_conditions():
    # Test DocumentMetadata with/without optional page_number
    meta1 = streamlit_app.DocumentMetadata(filename="f", chunk_id=0, source="s")
    meta2 = streamlit_app.DocumentMetadata(filename="f", chunk_id=1, source="s", page_number=2)
    assert meta1.page_number is None
    assert meta2.page_number == 2

def test_chunk_and_sourced_answer_model():
    meta = streamlit_app.DocumentMetadata(filename="f", chunk_id=0, source="s")
    chunk = streamlit_app.Chunk(text="abc", metadata=meta)
    answer = streamlit_app.SourcedAnswer(answer="foo", confidence_score=0.8, sources=[chunk])
    assert answer.answer == "foo"
    assert answer.confidence_score == 0.8
    assert answer.sources[0].text == "abc"

def test_shipment_data_model_fields():
    carrier = streamlit_app.CarrierInfo(carrier_name="ACME", mc_number="123", phone="555-1234")
    driver = streamlit_app.DriverInfo(driver_name="John", cell_number="555-5678", truck_number="T-1")
    pickup = streamlit_app.Location(name="Warehouse", address="123 St", city="NYC", state="NY", zip_code="10001", appointment_time="10:00")
    drop = streamlit_app.Location(name="Store", address="456 Ave", city="LA", state="CA", zip_code="90001", appointment_time="18:00")
    rate = streamlit_app.RateInfo(total_rate=1000.0, currency="USD", rate_breakdown={"linehaul": 900, "fuel": 100})
    shipment = streamlit_app.ShipmentData(
        reference_id="REF123",
        shipper="ACME",
        consignee="XYZ",
        carrier=carrier,
        driver=driver,
        pickup=pickup,
        drop=drop,
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Van",
        rate_info=rate,
        special_instructions="None"
    )
    assert shipment.reference_id == "REF123"
    assert shipment.carrier.carrier_name == "ACME"
    assert shipment.driver.driver_name == "John"
    assert shipment.pickup.city == "NYC"
    assert shipment.drop.city == "LA"
    assert shipment.rate_info.total_rate == 1000.0
    assert shipment.special_instructions == "None"

def test_document_ingestion_section_markers(ingestion_service, dummy_txt_bytes):
    # Ensure section markers are inserted
    chunks = ingestion_service.process_file(dummy_txt_bytes, "test.txt")
    found_marker = any("## Carrier Details" in c.text for c in chunks)
    assert found_marker

def test_document_ingestion_chunking_boundary(ingestion_service):
    # Test chunking with text just at chunk_size boundary
    text = "A" * streamlit_app.CHUNK_SIZE
    chunks = ingestion_service.process_file(text.encode(), "test.txt")
    assert isinstance(chunks, list)
    assert all(isinstance(c, streamlit_app.Chunk) for c in chunks)
    # Should be at least one chunk, possibly more if overlap
    assert len(chunks) >= 1

def test_vector_store_service_add_documents_empty(monkeypatch, vector_store_service):
    # Should not fail on empty input
    streamlit_app.FAISS.from_documents.return_value = MagicMock()
    vector_store_service.add_documents([])
    # Should not raise

def test_rag_service_answer_no_vector_store(rag_service):
    # Should handle missing vector_store gracefully
    rag_service.vs.vector_store = None
    with pytest.raises(AttributeError):
        rag_service.answer("Any question")

def test_extraction_service_extract_invalid(monkeypatch, extraction_service):
    # Should propagate exceptions from chain.invoke
    chain_mock = MagicMock()
    extraction_service.prompt.__or__.return_value = chain_mock
    chain_mock.invoke.side_effect = Exception("LLM error")
    with pytest.raises(Exception):
        extraction_service.extract("text")
