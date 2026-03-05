# source_hash: 5fab573753f93532
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
    CarrierInfo,
    DriverInfo,
    Location,
    RateInfo,
)
import io

@pytest.fixture
def sample_pdf_bytes():
    # Simulate a minimal PDF file in bytes
    return b"%PDF-1.4\n%Fake PDF content\n"

@pytest.fixture
def sample_docx_bytes():
    # Simulate a minimal DOCX file in bytes
    return b"PK\x03\x04Fake DOCX content"

@pytest.fixture
def sample_txt_bytes():
    return b"Carrier Details\nJohn Doe\nRate Breakdown\n$1000\nPickup\nNew York\nDrop\nLos Angeles"

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
    # Patch pymupdf.open and page.get_text
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
    assert len(chunks) == 0 or all(c.text == "" for c in chunks)

def test_process_file_unsupported_extension(ingestion_service):
    # Should not raise, but produce no chunks
    chunks = ingestion_service.process_file(b"irrelevant", "file.xyz")
    assert len(chunks) == 0 or all(isinstance(c, Chunk) for c in chunks)

def test_vector_store_add_documents_creates_store(vector_store_service):
    # Patch FAISS.from_documents
    with patch("streamlit_app.FAISS") as mock_faiss:
        mock_faiss.from_documents.return_value = MagicMock()
        chunk = Chunk(text="test", metadata=DocumentMetadata(filename="f.txt", chunk_id=0, source="f.txt", chunk_type="text"))
        vector_store_service.add_documents([chunk])
        assert vector_store_service.vector_store is not None
        mock_faiss.from_documents.assert_called_once()

def test_vector_store_add_documents_appends(vector_store_service):
    # Patch FAISS.from_documents and add_documents
    with patch("streamlit_app.FAISS") as mock_faiss:
        mock_vs = MagicMock()
        mock_faiss.from_documents.return_value = mock_vs
        chunk = Chunk(text="test", metadata=DocumentMetadata(filename="f.txt", chunk_id=0, source="f.txt", chunk_type="text"))
        vector_store_service.add_documents([chunk])
        # Now add again, should call add_documents
        vector_store_service.add_documents([chunk])
        mock_vs.add_documents.assert_called()

def test_rag_service_answer_happy_path(rag_service):
    # Patch retriever and chain
    mock_doc = MagicMock()
    mock_doc.page_content = "Carrier Details: John Doe"
    mock_doc.metadata = {"filename": "f.txt", "chunk_id": 0, "source": "f.txt", "chunk_type": "text"}
    rag_service.vs.vector_store = MagicMock()
    retriever = MagicMock()
    retriever.invoke.return_value = [mock_doc]
    rag_service.vs.vector_store.as_retriever.return_value = retriever

    # Patch prompt | llm | StrOutputParser chain
    with patch.object(rag_service, "prompt") as mock_prompt:
        chain = MagicMock()
        chain.invoke.return_value = "John Doe is the carrier."
        mock_prompt.__or__.return_value = chain
        answer = rag_service.answer("Who is the carrier?")
        assert isinstance(answer, SourcedAnswer)
        assert answer.confidence_score == 0.9
        assert "John Doe" in answer.answer
        assert len(answer.sources) == 1
        assert answer.sources[0].metadata.filename == "f.txt"

def test_rag_service_answer_not_found(rag_service):
    mock_doc = MagicMock()
    mock_doc.page_content = "No relevant info"
    mock_doc.metadata = {"filename": "f.txt", "chunk_id": 0, "source": "f.txt", "chunk_type": "text"}
    rag_service.vs.vector_store = MagicMock()
    retriever = MagicMock()
    retriever.invoke.return_value = [mock_doc]
    rag_service.vs.vector_store.as_retriever.return_value = retriever

    with patch.object(rag_service, "prompt") as mock_prompt:
        chain = MagicMock()
        chain.invoke.return_value = "I cannot find the answer in the provided documents."
        mock_prompt.__or__.return_value = chain
        answer = rag_service.answer("What is the delivery date?")
        assert isinstance(answer, SourcedAnswer)
        assert answer.confidence_score == 0.1
        assert "cannot find" in answer.answer.lower()

def test_extraction_service_extract_happy_path(extraction_service):
    # Patch prompt | llm | parser chain
    with patch.object(extraction_service, "prompt") as mock_prompt:
        chain = MagicMock()
        expected = ShipmentData(
            reference_id="REF123",
            shipper="Shipper Inc",
            consignee="Consignee LLC",
            carrier=CarrierInfo(carrier_name="CarrierX", mc_number="123456", phone="555-1234"),
            driver=DriverInfo(driver_name="Jane Doe", cell_number="555-5678", truck_number="TRK123"),
            pickup=Location(name="Warehouse", address="123 Main St", city="New York", state="NY", zip_code="10001", appointment_time="9:00 AM"),
            drop=Location(name="Store", address="456 Elm St", city="Los Angeles", state="CA", zip_code="90001", appointment_time="5:00 PM"),
            shipping_date="2024-06-01",
            delivery_date="2024-06-02",
            equipment_type="Van",
            rate_info=RateInfo(total_rate=1000.0, currency="USD", rate_breakdown={"base": 900, "fuel": 100}),
            special_instructions="Handle with care"
        )
        chain.invoke.return_value = expected
        mock_prompt.__or__.return_value = chain
        result = extraction_service.extract("Some logistics text")
        assert isinstance(result, ShipmentData)
        assert result.reference_id == "REF123"
        assert result.carrier.carrier_name == "CarrierX"
        assert result.pickup.city == "New York"
        assert result.rate_info.total_rate == 1000.0

def test_extraction_service_extract_missing_fields(extraction_service):
    with patch.object(extraction_service, "prompt") as mock_prompt:
        chain = MagicMock()
        expected = ShipmentData(
            reference_id=None,
            shipper=None,
            consignee=None,
            carrier=None,
            driver=None,
            pickup=None,
            drop=None,
            shipping_date=None,
            delivery_date=None,
            equipment_type=None,
            rate_info=None,
            special_instructions=None
        )
        chain.invoke.return_value = expected
        mock_prompt.__or__.return_value = chain
        result = extraction_service.extract("Text with no shipment data")
        assert isinstance(result, ShipmentData)
        assert result.reference_id is None
        assert result.carrier is None

def test_document_ingestion_section_markers(ingestion_service):
    # Ensure section markers are inserted
    text = b"Carrier Details\nRate Breakdown\nPickup\nDrop\nCommodity\nSpecial Instructions"
    chunks = ingestion_service.process_file(text, "test.txt")
    # Should have section markers like "\n## Carrier Details\n"
    found = any("\n## Carrier Details\n" in c.text for c in chunks)
    assert found

def test_document_ingestion_chunking_boundary(ingestion_service):
    # Create a long text to test chunking at boundary
    long_text = ("Carrier Details\n" + "A" * 2000 + "\nRate Breakdown\n" + "B" * 2000).encode("utf-8")
    chunks = ingestion_service.process_file(long_text, "test.txt")
    # Should produce more than one chunk
    assert len(chunks) > 1
    # Each chunk should not exceed the configured chunk size * 2
    for c in chunks:
        assert len(c.text) <= ingestion_service.text_splitter._chunk_size

def test_vector_store_add_documents_empty(vector_store_service):
    # Should not raise error when adding empty list
    with patch("streamlit_app.FAISS") as mock_faiss:
        vector_store_service.add_documents([])
        # Should not call from_documents or add_documents
        assert not mock_faiss.from_documents.called

def test_rag_service_answer_no_vector_store(rag_service):
    rag_service.vs.vector_store = None
    with pytest.raises(AttributeError):
        rag_service.answer("Any question")

def test_extraction_service_extract_error_handling(extraction_service):
    # Simulate chain.invoke raising an exception
    with patch.object(extraction_service, "prompt") as mock_prompt:
        chain = MagicMock()
        chain.invoke.side_effect = Exception("LLM error")
        mock_prompt.__or__.return_value = chain
        with pytest.raises(Exception):
            extraction_service.extract("Some text")
