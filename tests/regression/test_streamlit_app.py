import pytest
from unittest.mock import patch, MagicMock, call
from streamlit_app import (
    DocumentIngestionService,
    VectorStoreService,
    RAGService,
    ExtractionService,
    DocumentMetadata,
    Chunk,
    SourcedAnswer,
    ShipmentData,
    Location,
    CommodityItem,
    CarrierInfo,
    DriverInfo,
    RateInfo,
)
import io

@pytest.fixture
def sample_pdf_bytes():
    # Simulate a PDF file as bytes
    return b"%PDF-1.4\n%Fake PDF content\n"

@pytest.fixture
def sample_docx_bytes():
    # Simulate a DOCX file as bytes
    return b"PK\x03\x04Fake DOCX content"

@pytest.fixture
def sample_txt_bytes():
    return b"Carrier Details\nJohn Doe\nRate Breakdown\n$1000\nPickup\n123 Main St\nDrop\n456 Elm St\nCommodity\nWidgets\nSpecial Instructions\nHandle with care"

@pytest.fixture
def ingestion_service():
    return DocumentIngestionService()

@pytest.fixture
def vector_store_service():
    with patch("streamlit_app.HuggingFaceEmbeddings") as mock_emb:
        mock_emb.return_value = MagicMock()
        return VectorStoreService()

@pytest.fixture
def rag_service(vector_store_service):
    with patch("streamlit_app.ChatGroq") as mock_llm:
        mock_llm.return_value = MagicMock()
        return RAGService(vector_store_service)

@pytest.fixture
def extraction_service():
    with patch("streamlit_app.ChatGroq") as mock_llm:
        mock_llm.return_value = MagicMock()
        return ExtractionService()

def test_process_file_pdf_happy_path(ingestion_service, sample_pdf_bytes):
    # Mock pymupdf.open and page.get_text
    with patch("streamlit_app.pymupdf.open") as mock_open:
        mock_doc = [MagicMock(), MagicMock()]
        mock_doc[0].get_text.return_value = "Carrier Details\nJohn Doe"
        mock_doc[1].get_text.return_value = "Rate Breakdown\n$1000"
        mock_open.return_value = mock_doc
        chunks = ingestion_service.process_file(sample_pdf_bytes, "test.pdf")
        assert len(chunks) > 0
        assert any("Carrier Details" in c.text for c in chunks)
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.metadata.filename == "test.pdf" for c in chunks)

def test_process_file_docx_happy_path(ingestion_service, sample_docx_bytes):
    # Mock docx.Document and paragraphs
    with patch("streamlit_app.docx.Document") as mock_docx:
        mock_doc = MagicMock()
        mock_doc.paragraphs = [MagicMock(text="Carrier Details"), MagicMock(text="John Doe")]
        mock_docx.return_value = mock_doc
        chunks = ingestion_service.process_file(sample_docx_bytes, "test.docx")
        assert len(chunks) > 0
        assert any("Carrier Details" in c.text for c in chunks)
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.metadata.filename == "test.docx" for c in chunks)

def test_process_file_txt_happy_path(ingestion_service, sample_txt_bytes):
    chunks = ingestion_service.process_file(sample_txt_bytes, "test.txt")
    assert len(chunks) > 0
    assert any("Carrier Details" in c.text for c in chunks)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.metadata.filename == "test.txt" for c in chunks)

def test_process_file_empty_txt(ingestion_service):
    chunks = ingestion_service.process_file(b"", "empty.txt")
    assert len(chunks) == 1
    assert chunks[0].text == ""
    assert chunks[0].metadata.filename == "empty.txt"

def test_process_file_unsupported_extension(ingestion_service):
    # Should not raise, but produce empty chunk
    chunks = ingestion_service.process_file(b"irrelevant", "file.xyz")
    assert len(chunks) == 1
    assert chunks[0].text == ""
    assert chunks[0].metadata.filename == "file.xyz"

def test_vector_store_add_documents_creates_store(vector_store_service):
    chunk = Chunk(text="test text", metadata=DocumentMetadata(filename="f.txt", chunk_id=0, source="f.txt - Part 1"))
    with patch("streamlit_app.FAISS") as mock_faiss, patch("streamlit_app.Document") as mock_doc:
        mock_faiss.from_documents.return_value = MagicMock()
        vector_store_service.add_documents([chunk])
        assert vector_store_service.vector_store is not None
        mock_faiss.from_documents.assert_called_once()

def test_vector_store_add_documents_appends(vector_store_service):
    chunk = Chunk(text="test text", metadata=DocumentMetadata(filename="f.txt", chunk_id=0, source="f.txt - Part 1"))
    with patch("streamlit_app.FAISS") as mock_faiss, patch("streamlit_app.Document") as mock_doc:
        mock_vs = MagicMock()
        vector_store_service.vector_store = mock_vs
        vector_store_service.add_documents([chunk])
        mock_vs.add_documents.assert_called_once()

def test_rag_service_answer_happy_path(rag_service):
    # Setup vector store and retriever
    mock_vs = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "Carrier Details: John Doe"
    mock_doc.metadata = {"filename": "f.txt", "chunk_id": 0, "source": "f.txt - Part 1"}
    mock_vs.as_retriever.return_value.invoke.return_value = [mock_doc]
    rag_service.vs.vector_store = mock_vs
    # Patch LLM chain
    rag_service.llm.__or__.return_value.invoke.return_value = "John Doe is the carrier."
    with patch.object(rag_service.prompt, "__or__", return_value=rag_service.llm):
        answer = rag_service.answer("Who is the carrier?")
        assert isinstance(answer, SourcedAnswer)
        assert answer.confidence_score == 0.9
        assert "John Doe" in answer.answer
        assert len(answer.sources) == 1
        assert answer.sources[0].metadata.filename == "f.txt"

def test_rag_service_answer_not_found(rag_service):
    mock_vs = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "No relevant info"
    mock_doc.metadata = {"filename": "f.txt", "chunk_id": 0, "source": "f.txt - Part 1"}
    mock_vs.as_retriever.return_value.invoke.return_value = [mock_doc]
    rag_service.vs.vector_store = mock_vs
    rag_service.llm.__or__.return_value.invoke.return_value = "I cannot find the answer in the provided documents."
    with patch.object(rag_service.prompt, "__or__", return_value=rag_service.llm):
        answer = rag_service.answer("What is the delivery date?")
        assert isinstance(answer, SourcedAnswer)
        assert answer.confidence_score == 0.1
        assert "cannot find" in answer.answer.lower()

def test_extraction_service_extract_happy_path(extraction_service):
    # Patch the chain to return a ShipmentData instance
    shipment_data = ShipmentData(
        reference_id="REF123",
        shipper="Acme Corp",
        consignee="Beta LLC",
        carrier=CarrierInfo(carrier_name="CarrierX"),
        driver=DriverInfo(driver_name="John Doe"),
        pickup=Location(name="Warehouse A"),
        drop=Location(name="Warehouse B"),
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Flatbed",
        rate_info=RateInfo(total_rate=1000.0, currency="USD"),
        special_instructions="Handle with care"
    )
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = shipment_data
    extraction_service.prompt.__or__.return_value = extraction_service.llm
    extraction_service.llm.__or__.return_value = extraction_service.parser
    extraction_service.parser.__call__.return_value = shipment_data
    with patch.object(extraction_service.prompt, "__or__", return_value=extraction_service.llm), \
         patch.object(extraction_service.llm, "__or__", return_value=extraction_service.parser), \
         patch.object(extraction_service.parser, "invoke", return_value=shipment_data), \
         patch.object(extraction_service.parser, "get_format_instructions", return_value="FORMAT"):
        result = extraction_service.extract("Carrier: CarrierX")
        assert isinstance(result, ShipmentData)
        assert result.carrier.carrier_name == "CarrierX"
        assert result.reference_id == "REF123"

def test_extraction_service_extract_missing_fields(extraction_service):
    shipment_data = ShipmentData()
    with patch.object(extraction_service.prompt, "__or__", return_value=extraction_service.llm), \
         patch.object(extraction_service.llm, "__or__", return_value=extraction_service.parser), \
         patch.object(extraction_service.parser, "invoke", return_value=shipment_data), \
         patch.object(extraction_service.parser, "get_format_instructions", return_value="FORMAT"):
        result = extraction_service.extract("")
        assert isinstance(result, ShipmentData)
        # All fields should be None
        for field in ShipmentData.model_fields:
            assert getattr(result, field) is None

def test_document_metadata_boundary_conditions():
    # Test with only required fields
    meta = DocumentMetadata(filename="f.txt", chunk_id=0, source="src")
    assert meta.filename == "f.txt"
    assert meta.chunk_id == 0
    assert meta.source == "src"
    assert meta.chunk_type == "text"
    assert meta.page_number is None

def test_chunk_and_sourced_answer_repr():
    meta = DocumentMetadata(filename="f.txt", chunk_id=0, source="src")
    chunk = Chunk(text="abc", metadata=meta)
    answer = SourcedAnswer(answer="42", confidence_score=0.5, sources=[chunk])
    assert answer.answer == "42"
    assert answer.confidence_score == 0.5
    assert answer.sources[0].text == "abc"

def test_shipment_data_edge_cases():
    # All optional fields None
    data = ShipmentData()
    assert data.reference_id is None
    # Partial fill
    data2 = ShipmentData(reference_id="X", shipper="Y")
    assert data2.reference_id == "X"
    assert data2.shipper == "Y"
    # Nested
    data3 = ShipmentData(carrier=CarrierInfo(carrier_name="Z"))
    assert data3.carrier.carrier_name == "Z"

def test_reconciliation_equivalent_paths(ingestion_service, sample_txt_bytes):
    # TXT and DOCX with same content should produce equivalent chunks
    with patch("streamlit_app.docx.Document") as mock_docx:
        mock_doc = MagicMock()
        mock_doc.paragraphs = [MagicMock(text=line) for line in sample_txt_bytes.decode().splitlines()]
        mock_docx.return_value = mock_doc
        txt_chunks = ingestion_service.process_file(sample_txt_bytes, "test.txt")
        docx_chunks = ingestion_service.process_file(sample_docx_bytes(), "test.docx")
        # Compare chunk texts
        txt_texts = [c.text for c in txt_chunks]
        docx_texts = [c.text for c in docx_chunks]
        # Allow for possible chunking differences, but at least some overlap
        assert any(t in docx_texts for t in txt_texts)
