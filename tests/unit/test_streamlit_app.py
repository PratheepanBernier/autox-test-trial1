import io
import os
import re
import types
import pytest
from unittest.mock import patch, MagicMock, call

import streamlit_app

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

@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy-key")
    monkeypatch.setenv("QA_MODEL", "dummy-model")
    monkeypatch.setenv("EMBEDDING_MODEL", "dummy-embedding")
    monkeypatch.setenv("CHUNK_SIZE", "1000")
    monkeypatch.setenv("CHUNK_OVERLAP", "200")
    monkeypatch.setenv("TOP_K", "4")

@pytest.fixture
def ingestion_service():
    return DocumentIngestionService()

@pytest.fixture
def vector_store_service():
    with patch("streamlit_app.HuggingFaceEmbeddings") as mock_embed:
        mock_embed.return_value = MagicMock()
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

def test_process_file_pdf_happy_path(ingestion_service):
    # Arrange
    fake_pdf_bytes = b"%PDF-1.4"
    filename = "test.pdf"
    fake_page = MagicMock()
    fake_page.get_text.return_value = "Carrier Details\nSome text\nRate Breakdown\nMore text"
    fake_doc = [fake_page, fake_page]
    with patch("streamlit_app.pymupdf.open", return_value=fake_doc):
        # Act
        chunks = ingestion_service.process_file(fake_pdf_bytes, filename)
    # Assert
    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert any("Carrier Details" in c.text for c in chunks)
    assert any("Rate Breakdown" in c.text for c in chunks)
    assert all(c.metadata.filename == filename for c in chunks)
    assert all(c.metadata.chunk_type == "text" for c in chunks)
    assert all(isinstance(c.metadata.chunk_id, int) for c in chunks)
    assert all(isinstance(c.text, str) for c in chunks)

def test_process_file_docx_happy_path(ingestion_service):
    # Arrange
    fake_docx_bytes = b"docx"
    filename = "test.docx"
    fake_doc = MagicMock()
    fake_doc.paragraphs = [MagicMock(text="Carrier Details"), MagicMock(text="Pickup")]
    with patch("streamlit_app.docx.Document", return_value=fake_doc):
        # Act
        chunks = ingestion_service.process_file(fake_docx_bytes, filename)
    # Assert
    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert any("Carrier Details" in c.text for c in chunks)
    assert any("Pickup" in c.text for c in chunks)

def test_process_file_txt_happy_path(ingestion_service):
    # Arrange
    text = "Carrier Details\nPickup\nDrop"
    filename = "test.txt"
    # Act
    chunks = ingestion_service.process_file(text.encode("utf-8"), filename)
    # Assert
    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert any("Carrier Details" in c.text for c in chunks)
    assert any("Pickup" in c.text for c in chunks)
    assert any("Drop" in c.text for c in chunks)

def test_process_file_empty_file(ingestion_service):
    # Arrange
    filename = "empty.txt"
    # Act
    chunks = ingestion_service.process_file(b"", filename)
    # Assert
    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks)
    # Should still produce at least one chunk (even if empty)
    assert len(chunks) >= 1

def test_process_file_unsupported_extension(ingestion_service):
    # Arrange
    filename = "file.xyz"
    # Act
    chunks = ingestion_service.process_file(b"irrelevant", filename)
    # Assert
    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks)
    # Should produce at least one chunk (even if empty)
    assert len(chunks) >= 1

def test_process_file_section_markers_inserted(ingestion_service):
    # Arrange
    text = "Carrier Details\nRate Breakdown\nPickup\nDrop\nCommodity\nSpecial Instructions"
    filename = "test.txt"
    # Act
    chunks = ingestion_service.process_file(text.encode("utf-8"), filename)
    # Assert
    joined = "\n".join(c.text for c in chunks)
    # Each section should be marked with \n## SectionName\n
    for section in ['Carrier Details', 'Rate Breakdown', 'Pickup', 'Drop', 'Commodity', 'Special Instructions']:
        assert re.search(rf"\n## {section}\n", joined, re.IGNORECASE)

def test_vector_store_add_documents_creates_store(vector_store_service):
    # Arrange
    chunk = Chunk(
        text="test text",
        metadata=DocumentMetadata(filename="f.txt", chunk_id=0, source="f.txt - Part 1", chunk_type="text")
    )
    with patch("streamlit_app.FAISS") as mock_faiss, \
         patch("streamlit_app.Document") as mock_doc:
        mock_faiss.from_documents.return_value = MagicMock()
        mock_doc.side_effect = lambda page_content, metadata: MagicMock(page_content=page_content, metadata=metadata)
        # Act
        vector_store_service.add_documents([chunk])
        # Assert
        assert vector_store_service.vector_store is not None
        mock_faiss.from_documents.assert_called_once()

def test_vector_store_add_documents_appends(vector_store_service):
    # Arrange
    chunk = Chunk(
        text="test text",
        metadata=DocumentMetadata(filename="f.txt", chunk_id=0, source="f.txt - Part 1", chunk_type="text")
    )
    with patch("streamlit_app.FAISS") as mock_faiss, \
         patch("streamlit_app.Document") as mock_doc:
        mock_vs = MagicMock()
        vector_store_service.vector_store = mock_vs
        mock_doc.side_effect = lambda page_content, metadata: MagicMock(page_content=page_content, metadata=metadata)
        # Act
        vector_store_service.add_documents([chunk])
        # Assert
        mock_vs.add_documents.assert_called_once()

def test_rag_service_answer_happy_path(rag_service):
    # Arrange
    mock_vs = MagicMock()
    mock_retriever = MagicMock()
    doc1 = MagicMock(page_content="answer1", metadata={"filename": "f.txt", "chunk_id": 0, "source": "src", "chunk_type": "text"})
    doc2 = MagicMock(page_content="answer2", metadata={"filename": "f.txt", "chunk_id": 1, "source": "src", "chunk_type": "text"})
    mock_retriever.invoke.return_value = [doc1, doc2]
    rag_service.vs.vector_store = MagicMock()
    rag_service.vs.vector_store.as_retriever.return_value = mock_retriever
    # Patch chain
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "This is the answer."
    with patch.object(rag_service, "prompt", create=True) as mock_prompt, \
         patch.object(rag_service, "llm", create=True) as mock_llm, \
         patch("streamlit_app.StrOutputParser") as mock_parser:
        mock_prompt.__or__.return_value = mock_chain
        mock_chain.__or__.return_value = mock_chain
        mock_parser.return_value = mock_chain
        # Act
        answer = rag_service.answer("What is the answer?")
    # Assert
    assert isinstance(answer, SourcedAnswer)
    assert answer.answer == "This is the answer."
    assert answer.confidence_score == 0.9
    assert len(answer.sources) == 2
    assert all(isinstance(s, Chunk) for s in answer.sources)

def test_rag_service_answer_low_confidence(rag_service):
    # Arrange
    mock_vs = MagicMock()
    mock_retriever = MagicMock()
    doc1 = MagicMock(page_content="no answer", metadata={"filename": "f.txt", "chunk_id": 0, "source": "src", "chunk_type": "text"})
    mock_retriever.invoke.return_value = [doc1]
    rag_service.vs.vector_store = MagicMock()
    rag_service.vs.vector_store.as_retriever.return_value = mock_retriever
    # Patch chain
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "I cannot find the answer in the provided documents."
    with patch.object(rag_service, "prompt", create=True) as mock_prompt, \
         patch.object(rag_service, "llm", create=True) as mock_llm, \
         patch("streamlit_app.StrOutputParser") as mock_parser:
        mock_prompt.__or__.return_value = mock_chain
        mock_chain.__or__.return_value = mock_chain
        mock_parser.return_value = mock_chain
        # Act
        answer = rag_service.answer("Unknown question?")
    # Assert
    assert isinstance(answer, SourcedAnswer)
    assert answer.confidence_score == 0.1
    assert "cannot find" in answer.answer.lower()

def test_extraction_service_extract_happy_path(extraction_service):
    # Arrange
    mock_chain = MagicMock()
    expected = ShipmentData(reference_id="123", shipper="ACME", consignee="XYZ")
    mock_chain.invoke.return_value = expected
    with patch.object(extraction_service, "prompt", create=True) as mock_prompt, \
         patch.object(extraction_service, "llm", create=True) as mock_llm, \
         patch.object(extraction_service, "parser", create=True) as mock_parser:
        mock_prompt.__or__.return_value = mock_chain
        mock_chain.__or__.return_value = mock_chain
        mock_parser.get_format_instructions.return_value = "format"
        # Act
        result = extraction_service.extract("some text")
    # Assert
    assert isinstance(result, ShipmentData)
    assert result.reference_id == "123"
    assert result.shipper == "ACME"
    assert result.consignee == "XYZ"

def test_extraction_service_extract_missing_fields(extraction_service):
    # Arrange
    mock_chain = MagicMock()
    expected = ShipmentData(reference_id=None, shipper=None, consignee=None)
    mock_chain.invoke.return_value = expected
    with patch.object(extraction_service, "prompt", create=True) as mock_prompt, \
         patch.object(extraction_service, "llm", create=True) as mock_llm, \
         patch.object(extraction_service, "parser", create=True) as mock_parser:
        mock_prompt.__or__.return_value = mock_chain
        mock_chain.__or__.return_value = mock_chain
        mock_parser.get_format_instructions.return_value = "format"
        # Act
        result = extraction_service.extract("irrelevant text")
    # Assert
    assert isinstance(result, ShipmentData)
    assert result.reference_id is None
    assert result.shipper is None
    assert result.consignee is None

def test_document_metadata_model_fields():
    # Arrange
    data = {
        "filename": "f.txt",
        "page_number": 1,
        "chunk_id": 0,
        "source": "src",
        "chunk_type": "text"
    }
    # Act
    meta = DocumentMetadata(**data)
    # Assert
    assert meta.filename == "f.txt"
    assert meta.page_number == 1
    assert meta.chunk_id == 0
    assert meta.source == "src"
    assert meta.chunk_type == "text"

def test_chunk_model_fields():
    # Arrange
    meta = DocumentMetadata(filename="f.txt", chunk_id=0, source="src", chunk_type="text")
    # Act
    chunk = Chunk(text="abc", metadata=meta)
    # Assert
    assert chunk.text == "abc"
    assert chunk.metadata == meta

def test_sourced_answer_model_fields():
    # Arrange
    meta = DocumentMetadata(filename="f.txt", chunk_id=0, source="src", chunk_type="text")
    chunk = Chunk(text="abc", metadata=meta)
    # Act
    answer = SourcedAnswer(answer="42", confidence_score=0.8, sources=[chunk])
    # Assert
    assert answer.answer == "42"
    assert answer.confidence_score == 0.8
    assert answer.sources == [chunk]

def test_shipment_data_model_fields():
    # Arrange
    carrier = CarrierInfo(carrier_name="Carrier", mc_number="123", phone="555-5555")
    driver = DriverInfo(driver_name="Driver", cell_number="555-1234", truck_number="T1")
    pickup = Location(name="Warehouse", address="123 St", city="City", state="ST", zip_code="12345", appointment_time="10:00")
    drop = Location(name="Store", address="456 Ave", city="Town", state="TS", zip_code="67890", appointment_time="14:00")
    rate_info = RateInfo(total_rate=1000.0, currency="USD", rate_breakdown={"linehaul": 900, "fuel": 100})
    # Act
    data = ShipmentData(
        reference_id="REF123",
        shipper="Shipper",
        consignee="Consignee",
        carrier=carrier,
        driver=driver,
        pickup=pickup,
        drop=drop,
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Van",
        rate_info=rate_info,
        special_instructions="None"
    )
    # Assert
    assert data.reference_id == "REF123"
    assert data.carrier == carrier
    assert data.driver == driver
    assert data.pickup == pickup
    assert data.drop == drop
    assert data.rate_info == rate_info
    assert data.special_instructions == "None"

def test_location_model_optional_fields():
    # Arrange
    # Only provide name
    loc = Location(name="Loc")
    # Assert
    assert loc.name == "Loc"
    assert loc.address is None
    assert loc.city is None
    assert loc.state is None
    assert loc.zip_code is None
    assert loc.appointment_time is None

def test_commodity_item_model_optional_fields():
    # Arrange
    item = CommodityItem(commodity_name="Widgets")
    # Assert
    assert item.commodity_name == "Widgets"
    assert item.weight is None
    assert item.quantity is None

def test_carrier_info_model_optional_fields():
    # Arrange
    info = CarrierInfo(carrier_name="Carrier")
    # Assert
    assert info.carrier_name == "Carrier"
    assert info.mc_number is None
    assert info.phone is None

def test_driver_info_model_optional_fields():
    # Arrange
    info = DriverInfo(driver_name="Driver")
    # Assert
    assert info.driver_name == "Driver"
    assert info.cell_number is None
    assert info.truck_number is None

def test_rate_info_model_optional_fields():
    # Arrange
    info = RateInfo(total_rate=123.45)
    # Assert
    assert info.total_rate == 123.45
    assert info.currency is None
    assert info.rate_breakdown is None

def test_shipment_data_model_all_optional():
    # Arrange
    data = ShipmentData()
    # Assert
    assert data.reference_id is None
    assert data.shipper is None
    assert data.consignee is None
    assert data.carrier is None
    assert data.driver is None
    assert data.pickup is None
    assert data.drop is None
    assert data.shipping_date is None
    assert data.delivery_date is None
    assert data.equipment_type is None
    assert data.rate_info is None
    assert data.special_instructions is None
