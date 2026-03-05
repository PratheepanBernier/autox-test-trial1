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
    # Simulate a PDF file as bytes
    return b"%PDF-1.4\n%Fake PDF content"

@pytest.fixture
def sample_docx_bytes():
    # Simulate a DOCX file as bytes
    return b"PK\x03\x04Fake DOCX content"

@pytest.fixture
def sample_txt_bytes():
    return b"Carrier Details\nJohn Doe\nRate Breakdown\n$1000\nPickup\nLocation A\nDrop\nLocation B"

@pytest.fixture
def ingestion_service():
    return DocumentIngestionService()

def test_process_file_pdf_happy_path(ingestion_service, sample_pdf_bytes):
    with patch("streamlit_app.pymupdf.open") as mock_open:
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Carrier Details\nJohn Doe\nRate Breakdown\n$1000"
        mock_doc.__iter__.return_value = [mock_page, mock_page]
        mock_open.return_value = mock_doc
        chunks = ingestion_service.process_file(sample_pdf_bytes, "test.pdf")
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)
        assert any("Carrier Details" in c.text for c in chunks)
        assert any("Rate Breakdown" in c.text for c in chunks)
        assert all(c.metadata.filename == "test.pdf" for c in chunks)
        assert all(c.metadata.chunk_type == "text" for c in chunks)
        assert all(isinstance(c.metadata.chunk_id, int) for c in chunks)

def test_process_file_docx_happy_path(ingestion_service, sample_docx_bytes):
    with patch("streamlit_app.docx.Document") as mock_docx:
        mock_doc = MagicMock()
        mock_doc.paragraphs = [MagicMock(text="Carrier Details"), MagicMock(text="Rate Breakdown")]
        mock_docx.return_value = mock_doc
        chunks = ingestion_service.process_file(sample_docx_bytes, "test.docx")
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)
        assert any("Carrier Details" in c.text for c in chunks)
        assert any("Rate Breakdown" in c.text for c in chunks)
        assert all(c.metadata.filename == "test.docx" for c in chunks)

def test_process_file_txt_happy_path(ingestion_service, sample_txt_bytes):
    chunks = ingestion_service.process_file(sample_txt_bytes, "test.txt")
    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert any("Carrier Details" in c.text for c in chunks)
    assert any("Rate Breakdown" in c.text for c in chunks)
    assert all(c.metadata.filename == "test.txt" for c in chunks)

def test_process_file_unknown_extension_returns_empty(ingestion_service):
    chunks = ingestion_service.process_file(b"irrelevant", "test.unknown")
    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks) or len(chunks) == 0

def test_process_file_empty_content(ingestion_service):
    chunks = ingestion_service.process_file(b"", "test.txt")
    assert isinstance(chunks, list)
    # Should still return at least one chunk (empty string)
    assert len(chunks) >= 1

def test_process_file_semantic_markers_applied(ingestion_service):
    text = b"Carrier Details\nSome info\nRate Breakdown\nMore info"
    chunks = ingestion_service.process_file(text, "test.txt")
    # Should have inserted "\n## Carrier Details\n" and "\n## Rate Breakdown\n"
    assert any("## Carrier Details" in c.text for c in chunks)
    assert any("## Rate Breakdown" in c.text for c in chunks)

def test_vector_store_add_documents_creates_and_adds(monkeypatch):
    service = VectorStoreService()
    fake_embeddings = MagicMock()
    fake_faiss = MagicMock()
    fake_doc = MagicMock()
    monkeypatch.setattr("streamlit_app.HuggingFaceEmbeddings", lambda model_name: fake_embeddings)
    monkeypatch.setattr("streamlit_app.FAISS", fake_faiss)
    chunk = Chunk(text="test", metadata=DocumentMetadata(filename="f", chunk_id=0, source="s"))
    # First call: should create vector_store
    with patch("streamlit_app.Document") as mock_doc:
        mock_doc.return_value = fake_doc
        service.vector_store = None
        service.embeddings = fake_embeddings
        service.add_documents([chunk])
        assert fake_faiss.from_documents.called
    # Second call: should add to existing vector_store
    service.vector_store = MagicMock()
    with patch("streamlit_app.Document") as mock_doc:
        mock_doc.return_value = fake_doc
        service.add_documents([chunk])
        assert service.vector_store.add_documents.called

def test_rag_service_answer_happy(monkeypatch):
    # Setup
    vs = MagicMock()
    retriever = MagicMock()
    doc1 = MagicMock()
    doc1.page_content = "Carrier Details: John Doe"
    doc1.metadata = {"filename": "f", "chunk_id": 0, "source": "s"}
    doc2 = MagicMock()
    doc2.page_content = "Rate Breakdown: $1000"
    doc2.metadata = {"filename": "f", "chunk_id": 1, "source": "s"}
    retriever.invoke.return_value = [doc1, doc2]
    vs.vector_store.as_retriever.return_value = retriever
    # Patch LLM and prompt chain
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = "The carrier is John Doe and the rate is $1000."
    monkeypatch.setattr("streamlit_app.ChatPromptTemplate.from_template", lambda t: MagicMock())
    monkeypatch.setattr("streamlit_app.ChatGroq", lambda **kwargs: MagicMock())
    monkeypatch.setattr("streamlit_app.StrOutputParser", lambda: MagicMock())
    monkeypatch.setattr("streamlit_app.RunnablePassthrough", MagicMock())
    monkeypatch.setattr("streamlit_app.RunnableParallel", MagicMock())
    # Patch prompt | llm | parser to return fake_chain
    with patch("streamlit_app.ChatPromptTemplate.from_template") as mock_prompt:
        mock_prompt.return_value.__or__.return_value.__or__.return_value = fake_chain
        rag = RAGService(vs)
        answer = rag.answer("Who is the carrier?")
        assert isinstance(answer, SourcedAnswer)
        assert answer.confidence_score == 0.9
        assert "John Doe" in answer.answer
        assert len(answer.sources) == 2
        assert all(isinstance(s, Chunk) for s in answer.sources)

def test_rag_service_answer_low_confidence(monkeypatch):
    vs = MagicMock()
    retriever = MagicMock()
    doc = MagicMock()
    doc.page_content = "No relevant info"
    doc.metadata = {"filename": "f", "chunk_id": 0, "source": "s"}
    retriever.invoke.return_value = [doc]
    vs.vector_store.as_retriever.return_value = retriever
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = "I cannot find the answer in the provided documents."
    monkeypatch.setattr("streamlit_app.ChatPromptTemplate.from_template", lambda t: MagicMock())
    monkeypatch.setattr("streamlit_app.ChatGroq", lambda **kwargs: MagicMock())
    monkeypatch.setattr("streamlit_app.StrOutputParser", lambda: MagicMock())
    with patch("streamlit_app.ChatPromptTemplate.from_template") as mock_prompt:
        mock_prompt.return_value.__or__.return_value.__or__.return_value = fake_chain
        rag = RAGService(vs)
        answer = rag.answer("Unknown question?")
        assert isinstance(answer, SourcedAnswer)
        assert answer.confidence_score == 0.1
        assert "cannot find" in answer.answer.lower()

def test_extraction_service_extract_happy(monkeypatch):
    fake_llm = MagicMock()
    fake_parser = MagicMock()
    fake_prompt = MagicMock()
    fake_chain = MagicMock()
    shipment_data = ShipmentData(
        reference_id="REF123",
        shipper="Shipper Inc",
        consignee="Consignee LLC",
        carrier=CarrierInfo(carrier_name="CarrierX"),
        driver=DriverInfo(driver_name="DriverY"),
        pickup=Location(name="Warehouse A"),
        drop=Location(name="Warehouse B"),
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Flatbed",
        rate_info=RateInfo(total_rate=1000.0, currency="USD"),
        special_instructions="Handle with care"
    )
    fake_chain.invoke.return_value = shipment_data
    fake_parser.get_format_instructions.return_value = "FORMAT"
    monkeypatch.setattr("streamlit_app.ChatGroq", lambda **kwargs: fake_llm)
    monkeypatch.setattr("streamlit_app.PydanticOutputParser", lambda pydantic_object: fake_parser)
    monkeypatch.setattr("streamlit_app.ChatPromptTemplate.from_template", lambda t: fake_prompt)
    fake_prompt.__or__.return_value.__or__.return_value = fake_chain
    extractor = ExtractionService()
    result = extractor.extract("Carrier: CarrierX\nDriver: DriverY")
    assert isinstance(result, ShipmentData)
    assert result.carrier.carrier_name == "CarrierX"
    assert result.driver.driver_name == "DriverY"

def test_extraction_service_extract_missing_fields(monkeypatch):
    fake_llm = MagicMock()
    fake_parser = MagicMock()
    fake_prompt = MagicMock()
    fake_chain = MagicMock()
    shipment_data = ShipmentData(
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
    fake_chain.invoke.return_value = shipment_data
    fake_parser.get_format_instructions.return_value = "FORMAT"
    monkeypatch.setattr("streamlit_app.ChatGroq", lambda **kwargs: fake_llm)
    monkeypatch.setattr("streamlit_app.PydanticOutputParser", lambda pydantic_object: fake_parser)
    monkeypatch.setattr("streamlit_app.ChatPromptTemplate.from_template", lambda t: fake_prompt)
    fake_prompt.__or__.return_value.__or__.return_value = fake_chain
    extractor = ExtractionService()
    result = extractor.extract("No relevant info")
    assert isinstance(result, ShipmentData)
    assert result.carrier is None
    assert result.driver is None
    assert result.reference_id is None

def test_document_metadata_boundary_conditions():
    # Test with only required fields
    meta = DocumentMetadata(filename="f", chunk_id=0, source="s")
    assert meta.filename == "f"
    assert meta.chunk_id == 0
    assert meta.source == "s"
    assert meta.chunk_type == "text"
    # Test with all fields
    meta = DocumentMetadata(filename="f", chunk_id=1, source="s", page_number=2, chunk_type="custom")
    assert meta.page_number == 2
    assert meta.chunk_type == "custom"

def test_chunk_model_and_sourced_answer_model():
    meta = DocumentMetadata(filename="f", chunk_id=0, source="s")
    chunk = Chunk(text="abc", metadata=meta)
    assert chunk.text == "abc"
    assert chunk.metadata == meta
    answer = SourcedAnswer(answer="42", confidence_score=0.5, sources=[chunk])
    assert answer.answer == "42"
    assert answer.confidence_score == 0.5
    assert answer.sources == [chunk]

def test_shipment_data_model_edge_cases():
    # All fields None
    data = ShipmentData()
    assert data.reference_id is None
    # Partial fields
    data = ShipmentData(reference_id="X", carrier=CarrierInfo(carrier_name="Y"))
    assert data.reference_id == "X"
    assert data.carrier.carrier_name == "Y"
    # Nested models
    rate = RateInfo(total_rate=123.45, currency="USD", rate_breakdown={"base": 100, "fuel": 23.45})
    data = ShipmentData(rate_info=rate)
    assert data.rate_info.total_rate == 123.45
    assert data.rate_info.rate_breakdown["fuel"] == 23.45
