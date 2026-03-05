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
    Location,
    CarrierInfo,
    DriverInfo,
    RateInfo,
    CommodityItem,
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
    return b"Carrier Details\nJohn Doe\nRate Breakdown\n$1000\nPickup\nNYC\nDrop\nLA\nCommodity\nWidgets\nSpecial Instructions\nHandle with care"

@pytest.fixture
def ingestion_service():
    return DocumentIngestionService()

@pytest.fixture
def vector_store_service():
    with patch("streamlit_app.HuggingFaceEmbeddings") as mock_emb:
        mock_emb.return_value = MagicMock()
        with patch("streamlit_app.FAISS") as mock_faiss:
            mock_faiss.from_documents.return_value = MagicMock()
            yield VectorStoreService()

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

def test_process_file_unknown_extension_returns_empty(ingestion_service):
    chunks = ingestion_service.process_file(b"irrelevant", "test.unknown")
    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks) or len(chunks) == 0

def test_process_file_empty_content(ingestion_service):
    chunks = ingestion_service.process_file(b"", "test.txt")
    assert len(chunks) == 1 or len(chunks) == 0

def test_process_file_section_markers_inserted(ingestion_service, sample_txt_bytes):
    chunks = ingestion_service.process_file(sample_txt_bytes, "test.txt")
    # Should insert section markers like "\n## Carrier Details\n"
    assert any("## Carrier Details" in c.text for c in chunks)

def test_vector_store_add_documents_creates_store(vector_store_service):
    chunk = Chunk(text="test", metadata=DocumentMetadata(filename="f", chunk_id=0, source="f", chunk_type="text"))
    with patch("streamlit_app.FAISS") as mock_faiss:
        mock_faiss.from_documents.return_value = MagicMock()
        vector_store_service.vector_store = None
        vector_store_service.add_documents([chunk])
        assert vector_store_service.vector_store is not None
        mock_faiss.from_documents.assert_called_once()

def test_vector_store_add_documents_appends(vector_store_service):
    chunk = Chunk(text="test", metadata=DocumentMetadata(filename="f", chunk_id=0, source="f", chunk_type="text"))
    vector_store_service.vector_store = MagicMock()
    vector_store_service.add_documents([chunk])
    vector_store_service.vector_store.add_documents.assert_called_once()

def test_rag_service_answer_happy_path():
    # Setup
    mock_vs = MagicMock()
    mock_retriever = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "Carrier Details: John Doe"
    mock_doc.metadata = {"filename": "f", "chunk_id": 0, "source": "f", "chunk_type": "text"}
    mock_retriever.invoke.return_value = [mock_doc]
    mock_vs.vector_store.as_retriever.return_value = mock_retriever

    with patch("streamlit_app.ChatPromptTemplate.from_template") as mock_prompt, \
         patch("streamlit_app.ChatGroq") as mock_llm, \
         patch("streamlit_app.StrOutputParser") as mock_parser:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "John Doe is the carrier."
        mock_prompt.return_value = MagicMock()
        mock_llm.return_value = MagicMock()
        mock_parser.return_value = MagicMock()
        # Chain composition returns mock_chain
        (mock_prompt.return_value).__or__.return_value = mock_chain
        (mock_chain.__or__).return_value = mock_chain

        rag = RAGService(mock_vs)
        answer = rag.answer("Who is the carrier?")
        assert isinstance(answer, SourcedAnswer)
        assert answer.confidence_score > 0.5
        assert "John Doe" in answer.answer
        assert len(answer.sources) == 1
        assert answer.sources[0].text == "Carrier Details: John Doe"

def test_rag_service_answer_low_confidence():
    mock_vs = MagicMock()
    mock_retriever = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "No relevant info"
    mock_doc.metadata = {"filename": "f", "chunk_id": 0, "source": "f", "chunk_type": "text"}
    mock_retriever.invoke.return_value = [mock_doc]
    mock_vs.vector_store.as_retriever.return_value = mock_retriever

    with patch("streamlit_app.ChatPromptTemplate.from_template") as mock_prompt, \
         patch("streamlit_app.ChatGroq") as mock_llm, \
         patch("streamlit_app.StrOutputParser") as mock_parser:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "I cannot find the answer in the provided documents."
        mock_prompt.return_value = MagicMock()
        mock_llm.return_value = MagicMock()
        mock_parser.return_value = MagicMock()
        (mock_prompt.return_value).__or__.return_value = mock_chain
        (mock_chain.__or__).return_value = mock_chain

        rag = RAGService(mock_vs)
        answer = rag.answer("Unknown question?")
        assert answer.confidence_score < 0.5
        assert "cannot find" in answer.answer.lower()

def test_extraction_service_extract_happy_path():
    with patch("streamlit_app.ChatPromptTemplate.from_template") as mock_prompt, \
         patch("streamlit_app.ChatGroq") as mock_llm, \
         patch("streamlit_app.PydanticOutputParser") as mock_parser:
        mock_chain = MagicMock()
        expected_data = ShipmentData(
            reference_id="REF123",
            shipper="Shipper Inc",
            consignee="Consignee LLC",
            carrier=CarrierInfo(carrier_name="CarrierX"),
            driver=DriverInfo(driver_name="DriverY"),
            pickup=Location(name="NYC"),
            drop=Location(name="LA"),
            shipping_date="2024-01-01",
            delivery_date="2024-01-02",
            equipment_type="Flatbed",
            rate_info=RateInfo(total_rate=1000.0, currency="USD"),
            special_instructions="Handle with care"
        )
        mock_chain.invoke.return_value = expected_data
        mock_prompt.return_value = MagicMock()
        mock_llm.return_value = MagicMock()
        mock_parser.return_value = MagicMock()
        mock_parser.return_value.get_format_instructions.return_value = "FORMAT"
        (mock_prompt.return_value).__or__.return_value = mock_chain
        (mock_chain.__or__).return_value = mock_chain

        extractor = ExtractionService()
        result = extractor.extract("Carrier Details: CarrierX\nDriver: DriverY")
        assert isinstance(result, ShipmentData)
        assert result.carrier.carrier_name == "CarrierX"
        assert result.driver.driver_name == "DriverY"
        assert result.pickup.name == "NYC"
        assert result.drop.name == "LA"
        assert result.rate_info.total_rate == 1000.0

def test_extraction_service_extract_missing_fields():
    with patch("streamlit_app.ChatPromptTemplate.from_template") as mock_prompt, \
         patch("streamlit_app.ChatGroq") as mock_llm, \
         patch("streamlit_app.PydanticOutputParser") as mock_parser:
        mock_chain = MagicMock()
        expected_data = ShipmentData(
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
        mock_chain.invoke.return_value = expected_data
        mock_prompt.return_value = MagicMock()
        mock_llm.return_value = MagicMock()
        mock_parser.return_value = MagicMock()
        mock_parser.return_value.get_format_instructions.return_value = "FORMAT"
        (mock_prompt.return_value).__or__.return_value = mock_chain
        (mock_chain.__or__).return_value = mock_chain

        extractor = ExtractionService()
        result = extractor.extract("Random unrelated text")
        assert isinstance(result, ShipmentData)
        assert result.reference_id is None
        assert result.carrier is None

def test_document_metadata_model_dump_and_load():
    meta = DocumentMetadata(filename="f", chunk_id=1, source="src", chunk_type="text")
    dumped = meta.model_dump()
    loaded = DocumentMetadata(**dumped)
    assert loaded.filename == meta.filename
    assert loaded.chunk_id == meta.chunk_id
    assert loaded.source == meta.source
    assert loaded.chunk_type == meta.chunk_type

def test_chunk_model_dump_and_load():
    meta = DocumentMetadata(filename="f", chunk_id=1, source="src", chunk_type="text")
    chunk = Chunk(text="abc", metadata=meta)
    dumped = chunk.model_dump()
    loaded = Chunk(**dumped)
    assert loaded.text == chunk.text
    assert loaded.metadata.filename == chunk.metadata.filename

def test_sourced_answer_model_dump_and_load():
    meta = DocumentMetadata(filename="f", chunk_id=1, source="src", chunk_type="text")
    chunk = Chunk(text="abc", metadata=meta)
    answer = SourcedAnswer(answer="42", confidence_score=0.9, sources=[chunk])
    dumped = answer.model_dump()
    loaded = SourcedAnswer(**dumped)
    assert loaded.answer == answer.answer
    assert loaded.confidence_score == answer.confidence_score
    assert loaded.sources[0].text == chunk.text

def test_shipment_data_model_dump_and_load():
    data = ShipmentData(
        reference_id="REF",
        shipper="S",
        consignee="C",
        carrier=CarrierInfo(carrier_name="Carrier"),
        driver=DriverInfo(driver_name="Driver"),
        pickup=Location(name="A"),
        drop=Location(name="B"),
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Flatbed",
        rate_info=RateInfo(total_rate=1000.0, currency="USD"),
        special_instructions="None"
    )
    dumped = data.model_dump()
    loaded = ShipmentData(**dumped)
    assert loaded.reference_id == data.reference_id
    assert loaded.carrier.carrier_name == "Carrier"
    assert loaded.pickup.name == "A"
    assert loaded.rate_info.total_rate == 1000.0

def test_document_ingestion_service_section_groups():
    service = DocumentIngestionService()
    assert "carrier_info" in service.section_groups
    assert isinstance(service.section_groups["carrier_info"], list)
    assert "Carrier Details" in service.section_groups["carrier_info"]

def test_document_ingestion_service_text_splitter_config():
    service = DocumentIngestionService()
    splitter = service.text_splitter
    assert splitter.chunk_size == 2000
    assert splitter.chunk_overlap == 200
    assert isinstance(splitter.separators, list)
    assert "\n## " in splitter.separators

def test_vector_store_service_embeddings_and_store_init():
    with patch("streamlit_app.HuggingFaceEmbeddings") as mock_emb:
        mock_emb.return_value = MagicMock()
        with patch("streamlit_app.FAISS") as mock_faiss:
            mock_faiss.from_documents.return_value = MagicMock()
            vs = VectorStoreService()
            assert vs.embeddings is not None
            assert vs.vector_store is None

def test_rag_service_prompt_template_and_llm_init():
    with patch("streamlit_app.ChatPromptTemplate.from_template") as mock_prompt, \
         patch("streamlit_app.ChatGroq") as mock_llm:
        mock_prompt.return_value = MagicMock()
        mock_llm.return_value = MagicMock()
        vs = MagicMock()
        rag = RAGService(vs)
        assert rag.llm is not None
        assert rag.prompt is not None

def test_extraction_service_prompt_and_parser_init():
    with patch("streamlit_app.ChatPromptTemplate.from_template") as mock_prompt, \
         patch("streamlit_app.ChatGroq") as mock_llm, \
         patch("streamlit_app.PydanticOutputParser") as mock_parser:
        mock_prompt.return_value = MagicMock()
        mock_llm.return_value = MagicMock()
        mock_parser.return_value = MagicMock()
        extractor = ExtractionService()
        assert extractor.llm is not None
        assert extractor.prompt is not None
        assert extractor.parser is not None
