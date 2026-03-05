import io
import types
import pytest
from unittest.mock import patch, MagicMock, call

import streamlit_app

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
    return b"Carrier Details\nJohn Doe\nRate Breakdown\n$1000\nPickup\nNew York\nDrop\nChicago\nCommodity\nWidgets\nSpecial Instructions\nHandle with care."

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
    return streamlit_app.DocumentIngestionService()

@pytest.fixture
def vector_store_service():
    with patch("streamlit_app.HuggingFaceEmbeddings") as mock_emb:
        mock_emb.return_value = MagicMock()
        return streamlit_app.VectorStoreService()

@pytest.fixture
def rag_service(vector_store_service):
    with patch("streamlit_app.ChatGroq") as mock_llm:
        mock_llm.return_value = MagicMock()
        return streamlit_app.RAGService(vector_store_service)

@pytest.fixture
def extraction_service():
    with patch("streamlit_app.ChatGroq") as mock_llm:
        mock_llm.return_value = MagicMock()
        return streamlit_app.ExtractionService()

def test_process_file_pdf_happy_path(ingestion_service, sample_pdf_bytes, sample_filename_pdf):
    # Patch pymupdf.open and page.get_text
    fake_doc = [MagicMock(get_text=MagicMock(return_value="Carrier Details\nJohn Doe"))]
    with patch("streamlit_app.pymupdf.open", return_value=fake_doc):
        chunks = ingestion_service.process_file(sample_pdf_bytes, sample_filename_pdf)
        assert isinstance(chunks, list)
        assert all(isinstance(c, streamlit_app.Chunk) for c in chunks)
        # Should contain semantic markers
        assert any("## Carrier Details" in c.text for c in chunks)
        # Metadata checks
        for c in chunks:
            assert c.metadata.filename == sample_filename_pdf
            assert c.metadata.chunk_type == "text"

def test_process_file_docx_happy_path(ingestion_service, sample_docx_bytes, sample_filename_docx):
    # Patch docx.Document and paragraphs
    fake_doc = MagicMock()
    fake_doc.paragraphs = [MagicMock(text="Carrier Details"), MagicMock(text="John Doe")]
    with patch("streamlit_app.docx.Document", return_value=fake_doc):
        chunks = ingestion_service.process_file(sample_docx_bytes, sample_filename_docx)
        assert isinstance(chunks, list)
        assert all(isinstance(c, streamlit_app.Chunk) for c in chunks)
        assert any("## Carrier Details" in c.text for c in chunks)

def test_process_file_txt_happy_path(ingestion_service, sample_txt_bytes, sample_filename_txt):
    chunks = ingestion_service.process_file(sample_txt_bytes, sample_filename_txt)
    assert isinstance(chunks, list)
    assert all(isinstance(c, streamlit_app.Chunk) for c in chunks)
    assert any("## Carrier Details" in c.text for c in chunks)
    assert any("## Rate Breakdown" in c.text for c in chunks)

def test_process_file_empty_txt(ingestion_service):
    chunks = ingestion_service.process_file(b"", "empty.txt")
    assert isinstance(chunks, list)
    # Should still return at least one chunk (empty string)
    assert len(chunks) == 1
    assert chunks[0].text == ""

def test_process_file_unsupported_extension(ingestion_service):
    # Should not raise, just skip to chunking empty text
    chunks = ingestion_service.process_file(b"irrelevant", "file.xyz")
    assert isinstance(chunks, list)
    assert len(chunks) == 1
    assert chunks[0].text == ""

def test_vector_store_add_documents_creates_store(vector_store_service):
    # Patch FAISS.from_documents
    with patch("streamlit_app.FAISS.from_documents") as mock_faiss:
        mock_faiss.return_value = MagicMock()
        chunk = streamlit_app.Chunk(
            text="test",
            metadata=streamlit_app.DocumentMetadata(
                filename="f.txt", chunk_id=0, source="f.txt - Part 1"
            ),
        )
        vector_store_service.add_documents([chunk])
        assert vector_store_service.vector_store is not None
        mock_faiss.assert_called_once()

def test_vector_store_add_documents_appends(vector_store_service):
    # Patch FAISS.from_documents and add_documents
    fake_store = MagicMock()
    vector_store_service.vector_store = fake_store
    chunk = streamlit_app.Chunk(
        text="test",
        metadata=streamlit_app.DocumentMetadata(
            filename="f.txt", chunk_id=0, source="f.txt - Part 1"
        ),
    )
    vector_store_service.add_documents([chunk])
    fake_store.add_documents.assert_called_once()

def test_rag_service_answer_happy_path(rag_service):
    # Patch retriever.invoke and chain.invoke
    fake_doc = MagicMock()
    fake_doc.page_content = "Carrier Details: John Doe"
    fake_doc.metadata = {
        "filename": "f.txt",
        "chunk_id": 0,
        "source": "f.txt - Part 1",
        "chunk_type": "text",
        "page_number": None,
    }
    rag_service.vs.vector_store = MagicMock()
    retriever = MagicMock()
    retriever.invoke.return_value = [fake_doc]
    rag_service.vs.vector_store.as_retriever.return_value = retriever

    with patch.object(rag_service.prompt, "__or__", return_value=MagicMock(invoke=MagicMock(return_value="John Doe is the carrier."))):
        answer = rag_service.answer("Who is the carrier?")
        assert isinstance(answer, streamlit_app.SourcedAnswer)
        assert answer.answer == "John Doe is the carrier."
        assert answer.confidence_score == 0.9
        assert len(answer.sources) == 1
        assert answer.sources[0].text == "Carrier Details: John Doe"

def test_rag_service_answer_not_found(rag_service):
    fake_doc = MagicMock()
    fake_doc.page_content = "No relevant info"
    fake_doc.metadata = {
        "filename": "f.txt",
        "chunk_id": 0,
        "source": "f.txt - Part 1",
        "chunk_type": "text",
        "page_number": None,
    }
    rag_service.vs.vector_store = MagicMock()
    retriever = MagicMock()
    retriever.invoke.return_value = [fake_doc]
    rag_service.vs.vector_store.as_retriever.return_value = retriever

    with patch.object(rag_service.prompt, "__or__", return_value=MagicMock(invoke=MagicMock(return_value="I cannot find the answer in the provided documents."))):
        answer = rag_service.answer("What is the delivery date?")
        assert isinstance(answer, streamlit_app.SourcedAnswer)
        assert answer.confidence_score == 0.1
        assert "cannot find" in answer.answer.lower()

def test_extraction_service_extract_happy_path(extraction_service):
    # Patch chain.invoke to return a ShipmentData instance
    fake_data = streamlit_app.ShipmentData(reference_id="123", shipper="Acme", consignee="Beta")
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = fake_data
    with patch.object(extraction_service.prompt, "__or__", return_value=fake_chain):
        result = extraction_service.extract("Reference ID: 123\nShipper: Acme\nConsignee: Beta")
        assert isinstance(result, streamlit_app.ShipmentData)
        assert result.reference_id == "123"
        assert result.shipper == "Acme"
        assert result.consignee == "Beta"

def test_extraction_service_extract_missing_fields(extraction_service):
    # Patch chain.invoke to return a ShipmentData instance with None fields
    fake_data = streamlit_app.ShipmentData()
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = fake_data
    with patch.object(extraction_service.prompt, "__or__", return_value=fake_chain):
        result = extraction_service.extract("")
        assert isinstance(result, streamlit_app.ShipmentData)
        # All fields should be None
        for field in result.model_fields:
            assert getattr(result, field) is None

def test_document_metadata_model_boundary():
    # Test with minimal and maximal values
    meta = streamlit_app.DocumentMetadata(
        filename="a.pdf",
        chunk_id=0,
        source="a.pdf - Part 1",
        chunk_type="text",
        page_number=None
    )
    assert meta.filename == "a.pdf"
    assert meta.chunk_id == 0
    assert meta.source == "a.pdf - Part 1"
    assert meta.chunk_type == "text"
    assert meta.page_number is None

def test_chunk_model_and_dump():
    meta = streamlit_app.DocumentMetadata(
        filename="b.txt",
        chunk_id=1,
        source="b.txt - Part 2",
        chunk_type="text"
    )
    chunk = streamlit_app.Chunk(text="Some text", metadata=meta)
    dumped = chunk.metadata.model_dump()
    assert dumped["filename"] == "b.txt"
    assert dumped["chunk_id"] == 1

def test_sourced_answer_model():
    meta = streamlit_app.DocumentMetadata(
        filename="c.txt",
        chunk_id=2,
        source="c.txt - Part 3",
        chunk_type="text"
    )
    chunk = streamlit_app.Chunk(text="Answer text", metadata=meta)
    answer = streamlit_app.SourcedAnswer(answer="42", confidence_score=0.8, sources=[chunk])
    assert answer.answer == "42"
    assert answer.confidence_score == 0.8
    assert answer.sources[0].text == "Answer text"

def test_shipment_data_model_all_fields():
    carrier = streamlit_app.CarrierInfo(carrier_name="CarrierX", mc_number="12345", phone="555-1234")
    driver = streamlit_app.DriverInfo(driver_name="Alice", cell_number="555-5678", truck_number="TX123")
    pickup = streamlit_app.Location(name="Warehouse", address="123 St", city="Metropolis", state="NY", zip_code="10001", appointment_time="10:00")
    drop = streamlit_app.Location(name="Store", address="456 Ave", city="Gotham", state="NJ", zip_code="07001", appointment_time="16:00")
    rate = streamlit_app.RateInfo(total_rate=1000.0, currency="USD", rate_breakdown={"base": 900, "fuel": 100})
    data = streamlit_app.ShipmentData(
        reference_id="REF123",
        shipper="ShipperA",
        consignee="ConsigneeB",
        carrier=carrier,
        driver=driver,
        pickup=pickup,
        drop=drop,
        shipping_date="2024-06-01",
        delivery_date="2024-06-02",
        equipment_type="Van",
        rate_info=rate,
        special_instructions="Fragile"
    )
    assert data.reference_id == "REF123"
    assert data.carrier.carrier_name == "CarrierX"
    assert data.driver.driver_name == "Alice"
    assert data.pickup.city == "Metropolis"
    assert data.drop.city == "Gotham"
    assert data.rate_info.total_rate == 1000.0
    assert data.special_instructions == "Fragile"

def test_location_model_optional_fields():
    loc = streamlit_app.Location()
    assert loc.name is None
    assert loc.address is None
    assert loc.city is None
    assert loc.state is None
    assert loc.zip_code is None
    assert loc.appointment_time is None

def test_commodity_item_model_optional_fields():
    item = streamlit_app.CommodityItem()
    assert item.commodity_name is None
    assert item.weight is None
    assert item.quantity is None

def test_carrier_info_model_optional_fields():
    carrier = streamlit_app.CarrierInfo()
    assert carrier.carrier_name is None
    assert carrier.mc_number is None
    assert carrier.phone is None

def test_driver_info_model_optional_fields():
    driver = streamlit_app.DriverInfo()
    assert driver.driver_name is None
    assert driver.cell_number is None
    assert driver.truck_number is None

def test_rate_info_model_optional_fields():
    rate = streamlit_app.RateInfo()
    assert rate.total_rate is None
    assert rate.currency is None
    assert rate.rate_breakdown is None

def test_shipment_data_model_minimal():
    data = streamlit_app.ShipmentData()
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

def test_document_ingestion_service_section_groups():
    service = streamlit_app.DocumentIngestionService()
    assert "carrier_info" in service.section_groups
    assert isinstance(service.section_groups["carrier_info"], list)
    assert "Carrier Details" in service.section_groups["carrier_info"]

def test_document_ingestion_service_text_splitter_config():
    service = streamlit_app.DocumentIngestionService()
    splitter = service.text_splitter
    assert splitter.chunk_size == streamlit_app.CHUNK_SIZE * 2
    assert splitter.chunk_overlap == streamlit_app.CHUNK_OVERLAP
    assert isinstance(splitter.separators, list)
    assert "\n## " in splitter.separators

def test_vector_store_service_embeddings_model_name():
    with patch("streamlit_app.HuggingFaceEmbeddings") as mock_emb:
        mock_emb.return_value = MagicMock()
        service = streamlit_app.VectorStoreService()
        mock_emb.assert_called_with(model_name=streamlit_app.EMBEDDING_MODEL)
