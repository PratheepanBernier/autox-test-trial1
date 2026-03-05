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
    CarrierInfo,
    DriverInfo,
    Location,
    RateInfo,
    CommodityItem,
)

@pytest.fixture
def sample_pdf_bytes():
    return b"%PDF-1.4 sample pdf content"

@pytest.fixture
def sample_docx_bytes():
    return b"PK\x03\x04 sample docx content"

@pytest.fixture
def sample_txt_bytes():
    return b"Carrier Details\nJohn Doe\nRate Breakdown\n$1000\nPickup\nLocation A\nDrop\nLocation B\nCommodity\nWidgets\nSpecial Instructions\nHandle with care."

@pytest.fixture
def sample_filename_pdf():
    return "test.pdf"

@pytest.fixture
def sample_filename_docx():
    return "test.docx"

@pytest.fixture
def sample_filename_txt():
    return "test.txt"

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
    with patch("streamlit_app.ChatGroq") as mock_llm, \
         patch("streamlit_app.PydanticOutputParser") as mock_parser, \
         patch("streamlit_app.ChatPromptTemplate") as mock_prompt:
        mock_llm.return_value = MagicMock()
        mock_parser.return_value = MagicMock()
        mock_prompt.from_template.return_value = MagicMock()
        return ExtractionService()

def test_process_file_pdf_happy_path(ingestion_service, sample_pdf_bytes, sample_filename_pdf):
    with patch("streamlit_app.pymupdf.open") as mock_open:
        mock_doc = [MagicMock(), MagicMock()]
        mock_doc[0].get_text.return_value = "Page 1 text"
        mock_doc[1].get_text.return_value = "Page 2 text"
        mock_open.return_value = mock_doc
        chunks = ingestion_service.process_file(sample_pdf_bytes, sample_filename_pdf)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)
        assert any("Page 1 text" in c.text for c in chunks)
        assert any("Page 2 text" in c.text for c in chunks)
        assert all(c.metadata.filename == sample_filename_pdf for c in chunks)

def test_process_file_docx_happy_path(ingestion_service, sample_docx_bytes, sample_filename_docx):
    with patch("streamlit_app.docx.Document") as mock_docx:
        mock_doc = MagicMock()
        mock_doc.paragraphs = [MagicMock(text="Para1"), MagicMock(text="Para2")]
        mock_docx.return_value = mock_doc
        chunks = ingestion_service.process_file(sample_docx_bytes, sample_filename_docx)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)
        assert any("Para1" in c.text for c in chunks)
        assert any("Para2" in c.text for c in chunks)
        assert all(c.metadata.filename == sample_filename_docx for c in chunks)

def test_process_file_txt_happy_path(ingestion_service, sample_txt_bytes, sample_filename_txt):
    chunks = ingestion_service.process_file(sample_txt_bytes, sample_filename_txt)
    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert any("Carrier Details" in c.text for c in chunks)
    assert all(c.metadata.filename == sample_filename_txt for c in chunks)

def test_process_file_empty_file(ingestion_service):
    chunks = ingestion_service.process_file(b"", "empty.txt")
    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks)
    # Should still produce at least one chunk (even if empty)
    assert len(chunks) >= 1

def test_process_file_unsupported_extension(ingestion_service):
    # Should not raise, just skip to chunking empty text
    chunks = ingestion_service.process_file(b"irrelevant", "file.xyz")
    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks)

def test_process_file_section_markers(ingestion_service, sample_txt_bytes, sample_filename_txt):
    # Should insert section markers for known sections
    chunks = ingestion_service.process_file(sample_txt_bytes, sample_filename_txt)
    found = any("## Carrier Details" in c.text for c in chunks)
    assert found

def test_vector_store_add_documents_creates_store(vector_store_service):
    chunk = Chunk(text="test text", metadata=DocumentMetadata(filename="f.txt", chunk_id=0, source="f.txt", chunk_type="text"))
    with patch("streamlit_app.FAISS") as mock_faiss, \
         patch("streamlit_app.Document") as mock_doc:
        mock_faiss.from_documents.return_value = MagicMock()
        vector_store_service.vector_store = None
        vector_store_service.add_documents([chunk])
        assert mock_faiss.from_documents.called

def test_vector_store_add_documents_appends(vector_store_service):
    chunk = Chunk(text="test text", metadata=DocumentMetadata(filename="f.txt", chunk_id=0, source="f.txt", chunk_type="text"))
    vector_store_service.vector_store = MagicMock()
    vector_store_service.vector_store.add_documents = MagicMock()
    with patch("streamlit_app.Document") as mock_doc:
        vector_store_service.add_documents([chunk])
        assert vector_store_service.vector_store.add_documents.called

def test_rag_service_answer_happy_path(rag_service):
    mock_retriever = MagicMock()
    mock_doc1 = MagicMock()
    mock_doc1.page_content = "context1"
    mock_doc1.metadata = {"filename": "f.txt", "chunk_id": 0, "source": "f.txt", "chunk_type": "text"}
    mock_doc2 = MagicMock()
    mock_doc2.page_content = "context2"
    mock_doc2.metadata = {"filename": "f.txt", "chunk_id": 1, "source": "f.txt", "chunk_type": "text"}
    rag_service.vs.vector_store = MagicMock()
    rag_service.vs.vector_store.as_retriever.return_value = mock_retriever
    mock_retriever.invoke.return_value = [mock_doc1, mock_doc2]
    rag_service.llm = MagicMock()
    rag_service.prompt = MagicMock()
    chain = MagicMock()
    chain.invoke.return_value = "This is the answer."
    rag_service.prompt.__or__.return_value = chain
    with patch("streamlit_app.StrOutputParser") as mock_parser:
        mock_parser.return_value = lambda x: x
        answer = rag_service.answer("What is the rate?")
        assert isinstance(answer, SourcedAnswer)
        assert answer.answer == "This is the answer."
        assert answer.confidence_score == 0.9
        assert len(answer.sources) == 2

def test_rag_service_answer_low_confidence(rag_service):
    mock_retriever = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "context"
    mock_doc.metadata = {"filename": "f.txt", "chunk_id": 0, "source": "f.txt", "chunk_type": "text"}
    rag_service.vs.vector_store = MagicMock()
    rag_service.vs.vector_store.as_retriever.return_value = mock_retriever
    mock_retriever.invoke.return_value = [mock_doc]
    rag_service.llm = MagicMock()
    rag_service.prompt = MagicMock()
    chain = MagicMock()
    chain.invoke.return_value = "I cannot find the answer in the provided documents."
    rag_service.prompt.__or__.return_value = chain
    with patch("streamlit_app.StrOutputParser") as mock_parser:
        mock_parser.return_value = lambda x: x
        answer = rag_service.answer("Unknown question?")
        assert answer.confidence_score == 0.1

def test_extraction_service_extract_happy_path():
    with patch("streamlit_app.ChatGroq") as mock_llm, \
         patch("streamlit_app.PydanticOutputParser") as mock_parser, \
         patch("streamlit_app.ChatPromptTemplate") as mock_prompt:
        mock_llm.return_value = MagicMock()
        mock_parser_instance = MagicMock()
        shipment_data = ShipmentData(reference_id="123", shipper="ShipperX")
        mock_parser_instance.get_format_instructions.return_value = "format"
        mock_parser_instance.__call__.return_value = shipment_data
        mock_parser_instance.invoke.return_value = shipment_data
        mock_parser.return_value = mock_parser_instance
        mock_prompt.from_template.return_value = MagicMock()
        extraction_service = ExtractionService()
        extraction_service.prompt = MagicMock()
        chain = MagicMock()
        chain.invoke.return_value = shipment_data
        extraction_service.prompt.__or__.return_value = chain
        result = extraction_service.extract("Some text")
        assert isinstance(result, ShipmentData)
        assert result.reference_id == "123"
        assert result.shipper == "ShipperX"

def test_extraction_service_extract_handles_missing_fields():
    with patch("streamlit_app.ChatGroq") as mock_llm, \
         patch("streamlit_app.PydanticOutputParser") as mock_parser, \
         patch("streamlit_app.ChatPromptTemplate") as mock_prompt:
        mock_llm.return_value = MagicMock()
        mock_parser_instance = MagicMock()
        shipment_data = ShipmentData()
        mock_parser_instance.get_format_instructions.return_value = "format"
        mock_parser_instance.__call__.return_value = shipment_data
        mock_parser_instance.invoke.return_value = shipment_data
        mock_parser.return_value = mock_parser_instance
        mock_prompt.from_template.return_value = MagicMock()
        extraction_service = ExtractionService()
        extraction_service.prompt = MagicMock()
        chain = MagicMock()
        chain.invoke.return_value = shipment_data
        extraction_service.prompt.__or__.return_value = chain
        result = extraction_service.extract("Some text")
        assert isinstance(result, ShipmentData)
        assert result.reference_id is None
        assert result.shipper is None

def test_document_metadata_model_fields():
    meta = DocumentMetadata(filename="f.txt", chunk_id=1, source="src", chunk_type="text")
    assert meta.filename == "f.txt"
    assert meta.chunk_id == 1
    assert meta.source == "src"
    assert meta.chunk_type == "text"
    assert meta.page_number is None

def test_chunk_model_fields():
    meta = DocumentMetadata(filename="f.txt", chunk_id=1, source="src", chunk_type="text")
    chunk = Chunk(text="abc", metadata=meta)
    assert chunk.text == "abc"
    assert chunk.metadata == meta

def test_sourced_answer_model_fields():
    meta = DocumentMetadata(filename="f.txt", chunk_id=1, source="src", chunk_type="text")
    chunk = Chunk(text="abc", metadata=meta)
    answer = SourcedAnswer(answer="42", confidence_score=0.8, sources=[chunk])
    assert answer.answer == "42"
    assert answer.confidence_score == 0.8
    assert answer.sources == [chunk]

def test_shipment_data_model_fields():
    carrier = CarrierInfo(carrier_name="CarrierX", mc_number="123", phone="555-1234")
    driver = DriverInfo(driver_name="DriverY", cell_number="555-5678", truck_number="T-100")
    pickup = Location(name="Warehouse", address="123 St", city="City", state="ST", zip_code="00000", appointment_time="10:00")
    drop = Location(name="Store", address="456 Ave", city="Town", state="TS", zip_code="11111", appointment_time="12:00")
    rate = RateInfo(total_rate=1000.0, currency="USD", rate_breakdown={"base": 900, "fuel": 100})
    shipment = ShipmentData(
        reference_id="REF1",
        shipper="ShipperA",
        consignee="ConsigneeB",
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
    assert shipment.reference_id == "REF1"
    assert shipment.carrier.carrier_name == "CarrierX"
    assert shipment.driver.driver_name == "DriverY"
    assert shipment.pickup.name == "Warehouse"
    assert shipment.drop.name == "Store"
    assert shipment.rate_info.total_rate == 1000.0
    assert shipment.special_instructions == "None"

def test_document_ingestion_service_section_groups():
    service = DocumentIngestionService()
    assert isinstance(service.section_groups, dict)
    assert "carrier_info" in service.section_groups

def test_document_ingestion_service_text_splitter_config():
    service = DocumentIngestionService()
    splitter = service.text_splitter
    assert hasattr(splitter, "chunk_size")
    assert hasattr(splitter, "chunk_overlap")
    assert hasattr(splitter, "separators")
    assert splitter.chunk_size == streamlit_app.CHUNK_SIZE * 2
    assert splitter.chunk_overlap == streamlit_app.CHUNK_OVERLAP

def test_vector_store_service_embeddings_model_name():
    with patch("streamlit_app.HuggingFaceEmbeddings") as mock_emb:
        mock_emb.return_value = MagicMock()
        service = VectorStoreService()
        mock_emb.assert_called_with(model_name=streamlit_app.EMBEDDING_MODEL)
