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

@pytest.fixture
def dummy_pdf_bytes():
    return b"%PDF-1.4\n%Fake PDF content"

@pytest.fixture
def dummy_docx_bytes():
    return b"PK\x03\x04\x14\x00\x06\x00"

@pytest.fixture
def dummy_txt_bytes():
    return b"Carrier Details\nJohn Doe\nRate Breakdown\n$1000\nPickup\nLocation A\nDrop\nLocation B\nCommodity\nWidgets\nSpecial Instructions\nNone"

@pytest.fixture
def dummy_filename_pdf():
    return "test.pdf"

@pytest.fixture
def dummy_filename_docx():
    return "test.docx"

@pytest.fixture
def dummy_filename_txt():
    return "test.txt"

@pytest.fixture
def ingestion_service():
    return streamlit_app.DocumentIngestionService()

@pytest.fixture
def vector_store_service():
    svc = streamlit_app.VectorStoreService()
    svc.embeddings = MagicMock()
    svc.vector_store = MagicMock()
    return svc

@pytest.fixture
def rag_service(vector_store_service):
    svc = streamlit_app.RAGService(vector_store_service)
    svc.llm = MagicMock()
    svc.prompt = MagicMock()
    return svc

@pytest.fixture
def extraction_service():
    svc = streamlit_app.ExtractionService()
    svc.llm = MagicMock()
    svc.prompt = MagicMock()
    svc.parser = MagicMock()
    return svc

def test_process_file_pdf_happy_path(ingestion_service, dummy_pdf_bytes, dummy_filename_pdf):
    with patch("streamlit_app.pymupdf.open") as mock_open:
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Carrier Details\nJohn Doe"
        mock_doc.__iter__.return_value = [mock_page]
        mock_open.return_value = mock_doc
        chunks = ingestion_service.process_file(dummy_pdf_bytes, dummy_filename_pdf)
        assert isinstance(chunks, list)
        assert all(isinstance(c, streamlit_app.Chunk) for c in chunks)
        assert any("Carrier Details" in c.text for c in chunks)
        assert all(c.metadata.filename == dummy_filename_pdf for c in chunks)

def test_process_file_docx_happy_path(ingestion_service, dummy_docx_bytes, dummy_filename_docx):
    with patch("streamlit_app.docx.Document") as mock_docx:
        mock_doc = MagicMock()
        mock_doc.paragraphs = [MagicMock(text="Carrier Details"), MagicMock(text="John Doe")]
        mock_docx.return_value = mock_doc
        chunks = ingestion_service.process_file(dummy_docx_bytes, dummy_filename_docx)
        assert isinstance(chunks, list)
        assert any("Carrier Details" in c.text for c in chunks)
        assert all(c.metadata.filename == dummy_filename_docx for c in chunks)

def test_process_file_txt_happy_path(ingestion_service, dummy_txt_bytes, dummy_filename_txt):
    chunks = ingestion_service.process_file(dummy_txt_bytes, dummy_filename_txt)
    assert isinstance(chunks, list)
    assert any("Carrier Details" in c.text for c in chunks)
    assert all(c.metadata.filename == dummy_filename_txt for c in chunks)

def test_process_file_empty_txt(ingestion_service):
    chunks = ingestion_service.process_file(b"", "empty.txt")
    assert isinstance(chunks, list)
    assert all(isinstance(c, streamlit_app.Chunk) for c in chunks)
    # Should produce at least one chunk (even if empty)
    assert len(chunks) >= 1

def test_process_file_unsupported_extension(ingestion_service):
    # Should not raise, just produce empty chunks
    chunks = ingestion_service.process_file(b"irrelevant", "file.xyz")
    assert isinstance(chunks, list)
    assert all(isinstance(c, streamlit_app.Chunk) for c in chunks)

def test_vector_store_add_documents_creates_store(monkeypatch):
    svc = streamlit_app.VectorStoreService()
    svc.embeddings = MagicMock()
    dummy_chunks = [
        streamlit_app.Chunk(
            text="Carrier Details",
            metadata=streamlit_app.DocumentMetadata(filename="a.txt", chunk_id=0, source="a.txt - Part 1")
        )
    ]
    with patch("streamlit_app.FAISS.from_documents") as mock_from_docs:
        mock_vs = MagicMock()
        mock_from_docs.return_value = mock_vs
        svc.add_documents(dummy_chunks)
        assert svc.vector_store is mock_vs
        mock_from_docs.assert_called_once()

def test_vector_store_add_documents_appends(monkeypatch):
    svc = streamlit_app.VectorStoreService()
    svc.embeddings = MagicMock()
    svc.vector_store = MagicMock()
    dummy_chunks = [
        streamlit_app.Chunk(
            text="Carrier Details",
            metadata=streamlit_app.DocumentMetadata(filename="a.txt", chunk_id=0, source="a.txt - Part 1")
        )
    ]
    svc.add_documents(dummy_chunks)
    svc.vector_store.add_documents.assert_called_once()

def test_rag_service_answer_happy_path(rag_service):
    mock_retriever = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "Carrier Details: John Doe"
    mock_doc.metadata = {"filename": "a.txt", "chunk_id": 0, "source": "a.txt - Part 1"}
    rag_service.vs.vector_store.as_retriever.return_value = mock_retriever
    mock_retriever.invoke.return_value = [mock_doc]
    rag_service.prompt.__or__.return_value = lambda x: "The answer is John Doe"
    with patch("streamlit_app.StrOutputParser") as mock_parser:
        mock_parser.return_value = lambda x: "The answer is John Doe"
        res = rag_service.answer("Who is the carrier?")
        assert isinstance(res, streamlit_app.SourcedAnswer)
        assert "John Doe" in res.answer
        assert res.confidence_score > 0.5
        assert len(res.sources) == 1

def test_rag_service_answer_no_answer(rag_service):
    mock_retriever = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "No relevant info"
    mock_doc.metadata = {"filename": "a.txt", "chunk_id": 0, "source": "a.txt - Part 1"}
    rag_service.vs.vector_store.as_retriever.return_value = mock_retriever
    mock_retriever.invoke.return_value = [mock_doc]
    rag_service.prompt.__or__.return_value = lambda x: "I cannot find the answer in the provided documents."
    with patch("streamlit_app.StrOutputParser") as mock_parser:
        mock_parser.return_value = lambda x: "I cannot find the answer in the provided documents."
        res = rag_service.answer("What is the moon made of?")
        assert isinstance(res, streamlit_app.SourcedAnswer)
        assert "cannot find" in res.answer.lower()
        assert res.confidence_score < 0.5

def test_extraction_service_extract_happy_path(extraction_service):
    mock_chain = MagicMock()
    expected_data = streamlit_app.ShipmentData(reference_id="REF123", shipper="ACME", consignee="XYZ")
    extraction_service.prompt.__or__.return_value = mock_chain
    extraction_service.parser.get_format_instructions.return_value = "format"
    mock_chain.invoke.return_value = expected_data
    result = extraction_service.extract("Carrier Details: ACME\nConsignee: XYZ\nReference: REF123")
    assert isinstance(result, streamlit_app.ShipmentData)
    assert result.reference_id == "REF123"
    assert result.shipper == "ACME"
    assert result.consignee == "XYZ"

def test_extraction_service_extract_missing_fields(extraction_service):
    mock_chain = MagicMock()
    expected_data = streamlit_app.ShipmentData(reference_id=None, shipper=None, consignee=None)
    extraction_service.prompt.__or__.return_value = mock_chain
    extraction_service.parser.get_format_instructions.return_value = "format"
    mock_chain.invoke.return_value = expected_data
    result = extraction_service.extract("No relevant info")
    assert isinstance(result, streamlit_app.ShipmentData)
    assert result.reference_id is None
    assert result.shipper is None
    assert result.consignee is None

def test_document_metadata_boundary_conditions():
    # chunk_id at 0, page_number None, minimal fields
    meta = streamlit_app.DocumentMetadata(filename="a.txt", chunk_id=0, source="src")
    assert meta.filename == "a.txt"
    assert meta.chunk_id == 0
    assert meta.page_number is None
    assert meta.chunk_type == "text"

def test_chunk_and_sourced_answer_types():
    meta = streamlit_app.DocumentMetadata(filename="a.txt", chunk_id=1, source="src")
    chunk = streamlit_app.Chunk(text="abc", metadata=meta)
    answer = streamlit_app.SourcedAnswer(answer="foo", confidence_score=0.8, sources=[chunk])
    assert chunk.text == "abc"
    assert answer.answer == "foo"
    assert answer.confidence_score == 0.8
    assert answer.sources[0] == chunk

def test_location_and_commodityitem_edge_cases():
    loc = streamlit_app.Location()
    assert loc.name is None
    assert loc.address is None
    item = streamlit_app.CommodityItem()
    assert item.commodity_name is None
    assert item.weight is None
    assert item.quantity is None

def test_rateinfo_and_shipmentdata_defaults():
    rate = streamlit_app.RateInfo()
    assert rate.total_rate is None
    assert rate.currency is None
    assert rate.rate_breakdown is None
    shipment = streamlit_app.ShipmentData()
    assert shipment.reference_id is None
    assert shipment.carrier is None
    assert shipment.driver is None
    assert shipment.pickup is None
    assert shipment.drop is None
    assert shipment.rate_info is None
    assert shipment.special_instructions is None

def test_document_ingestion_service_section_groups():
    svc = streamlit_app.DocumentIngestionService()
    assert isinstance(svc.section_groups, dict)
    assert "carrier_info" in svc.section_groups
    assert "instructions" in svc.section_groups

def test_ingestion_service_chunking_boundary(ingestion_service):
    # Large text to test chunking boundary
    text = "Carrier Details\n" + ("A" * (streamlit_app.CHUNK_SIZE * 3))
    chunks = ingestion_service.process_file(text.encode("utf-8"), "big.txt")
    # Should produce at least 2 chunks for large input
    assert len(chunks) >= 2

def test_ingestion_service_semantic_markers(ingestion_service):
    # Ensure semantic markers are inserted
    text = "Carrier Details\nRate Breakdown\nPickup\nDrop\nCommodity\nSpecial Instructions"
    chunks = ingestion_service.process_file(text.encode("utf-8"), "semantics.txt")
    # At least one chunk should have the marker
    assert any("## Carrier Details" in c.text for c in chunks)
    assert any("## Rate Breakdown" in c.text for c in chunks)
    assert any("## Pickup" in c.text for c in chunks)
    assert any("## Drop" in c.text for c in chunks)
    assert any("## Commodity" in c.text for c in chunks)
    assert any("## Special Instructions" in c.text for c in chunks)
