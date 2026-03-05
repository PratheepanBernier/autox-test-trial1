import pytest
from unittest.mock import patch, MagicMock, call
from streamlit_app import (
    DocumentIngestionService,
    VectorStoreService,
    RAGService,
    ExtractionService,
    Chunk,
    DocumentMetadata,
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
    return b"%PDF-1.4\n%Fake PDF content\n"

@pytest.fixture
def sample_docx_bytes():
    # Simulate a DOCX file as bytes
    return b"PK\x03\x04Fake DOCX content"

@pytest.fixture
def sample_txt_bytes():
    return b"Carrier Details\nJohn Doe\nRate Breakdown\n$1000\nPickup\n123 Main St\nDrop\n456 Elm St\nCommodity\nWidgets\nSpecial Instructions\nHandle with care."

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
    with patch("streamlit_app.ChatGroq") as mock_llm:
        mock_llm.return_value = MagicMock()
        return ExtractionService()

def test_process_file_equivalent_txt_and_docx(ingestion_service, sample_txt_bytes, sample_docx_bytes, sample_filename_txt, sample_filename_docx):
    # Mock docx.Document to return paragraphs matching the txt content
    paragraphs = [MagicMock(text=line) for line in sample_txt_bytes.decode().split('\n')]
    with patch("streamlit_app.docx.Document") as mock_docx:
        mock_docx.return_value.paragraphs = paragraphs
        txt_chunks = ingestion_service.process_file(sample_txt_bytes, sample_filename_txt)
        docx_chunks = ingestion_service.process_file(sample_docx_bytes, sample_filename_docx)
    # Reconciliation: The chunked text content should be equivalent for .txt and .docx with same content
    txt_texts = [c.text for c in txt_chunks]
    docx_texts = [c.text for c in docx_chunks]
    assert txt_texts == docx_texts
    assert all(isinstance(c, Chunk) for c in txt_chunks)
    assert all(isinstance(c, Chunk) for c in docx_chunks)

def test_process_file_pdf_and_txt_equivalence_on_content(ingestion_service, sample_pdf_bytes, sample_txt_bytes, sample_filename_pdf, sample_filename_txt):
    # Simulate a PDF with one page whose text matches the txt content
    class FakePage:
        def get_text(self):
            return sample_txt_bytes.decode()
    fake_doc = [FakePage()]
    with patch("streamlit_app.pymupdf.open") as mock_pdf_open:
        mock_pdf_open.return_value = fake_doc
        pdf_chunks = ingestion_service.process_file(sample_pdf_bytes, sample_filename_pdf)
    txt_chunks = ingestion_service.process_file(sample_txt_bytes, sample_filename_txt)
    # The first chunk of PDF should contain the same text as the txt
    pdf_texts = [c.text for c in pdf_chunks]
    txt_texts = [c.text for c in txt_chunks]
    # Allow for possible page marker in PDF
    assert any(txt_texts[0] in t or t in txt_texts[0] for t in pdf_texts)

def test_process_file_empty_file(ingestion_service):
    # Edge case: empty file
    empty_bytes = b""
    chunks = ingestion_service.process_file(empty_bytes, "empty.txt")
    assert len(chunks) == 1
    assert chunks[0].text == ""

def test_process_file_boundary_chunking(ingestion_service):
    # Boundary: file just at chunk size
    text = "A" * ingestion_service.text_splitter.chunk_size
    chunks = ingestion_service.process_file(text.encode(), "boundary.txt")
    assert len(chunks) == 1
    assert chunks[0].text == text

def test_vector_store_add_documents_equivalence(vector_store_service):
    # Reconciliation: Adding same chunks twice should not error and should call add_documents on vector_store
    chunk = Chunk(text="test", metadata=DocumentMetadata(filename="f", chunk_id=0, source="f-0"))
    with patch("streamlit_app.FAISS") as mock_faiss:
        mock_vs = MagicMock()
        mock_faiss.from_documents.return_value = mock_vs
        vector_store_service.add_documents([chunk])
        # Add again, should call add_documents
        vector_store_service.add_documents([chunk])
        assert mock_vs.add_documents.called

def test_rag_service_answer_equivalent_contexts(rag_service):
    # Reconciliation: Different context orderings should yield similar answers
    mock_retriever = MagicMock()
    doc1 = MagicMock(page_content="A", metadata={"filename": "f", "chunk_id": 0, "source": "f-0"})
    doc2 = MagicMock(page_content="B", metadata={"filename": "f", "chunk_id": 1, "source": "f-1"})
    rag_service.vs.vector_store = MagicMock()
    rag_service.vs.vector_store.as_retriever.return_value = mock_retriever
    # Simulate retriever returning docs in different order
    mock_retriever.invoke.side_effect = [
        [doc1, doc2],
        [doc2, doc1]
    ]
    with patch.object(rag_service.llm, "invoke", return_value="The answer is 42"):
        ans1 = rag_service.answer("Q")
        ans2 = rag_service.answer("Q")
    # The answer text should be the same regardless of context order
    assert ans1.answer == ans2.answer
    assert ans1.confidence_score == ans2.confidence_score

def test_rag_service_answer_handles_no_answer(rag_service):
    # Error handling: LLM returns "cannot find"
    mock_retriever = MagicMock()
    doc = MagicMock(page_content="irrelevant", metadata={"filename": "f", "chunk_id": 0, "source": "f-0"})
    rag_service.vs.vector_store = MagicMock()
    rag_service.vs.vector_store.as_retriever.return_value = mock_retriever
    mock_retriever.invoke.return_value = [doc]
    with patch.object(rag_service.llm, "invoke", return_value="I cannot find the answer in the provided documents."):
        ans = rag_service.answer("Q")
    assert ans.confidence_score == 0.1
    assert "cannot find" in ans.answer.lower()

def test_extraction_service_equivalent_texts(extraction_service):
    # Reconciliation: Extraction from equivalent texts should yield equivalent ShipmentData
    text1 = "Carrier: ACME\nDriver: John\nPickup: 123 Main\nDrop: 456 Elm\nRate: $1000"
    text2 = "Carrier: ACME\nDriver: John\nPickup: 123 Main\nDrop: 456 Elm\nRate: $1000"
    fake_data = ShipmentData(
        reference_id="REF123",
        shipper="ACME",
        consignee="XYZ",
        carrier=CarrierInfo(carrier_name="ACME"),
        driver=DriverInfo(driver_name="John"),
        pickup=Location(address="123 Main"),
        drop=Location(address="456 Elm"),
        shipping_date=None,
        delivery_date=None,
        equipment_type=None,
        rate_info=RateInfo(total_rate=1000.0, currency="USD"),
        special_instructions=None,
    )
    with patch.object(extraction_service.parser, "get_format_instructions", return_value="FORMAT"):
        with patch.object(extraction_service.llm, "invoke", return_value=fake_data):
            data1 = extraction_service.extract(text1)
            data2 = extraction_service.extract(text2)
    assert data1 == data2

def test_extraction_service_handles_missing_fields(extraction_service):
    # Edge case: text missing fields, should fill with None
    text = "Carrier: ACME"
    fake_data = ShipmentData(
        reference_id=None,
        shipper=None,
        consignee=None,
        carrier=CarrierInfo(carrier_name="ACME"),
        driver=None,
        pickup=None,
        drop=None,
        shipping_date=None,
        delivery_date=None,
        equipment_type=None,
        rate_info=None,
        special_instructions=None,
    )
    with patch.object(extraction_service.parser, "get_format_instructions", return_value="FORMAT"):
        with patch.object(extraction_service.llm, "invoke", return_value=fake_data):
            data = extraction_service.extract(text)
    assert data.carrier.carrier_name == "ACME"
    assert data.driver is None
    assert data.pickup is None

def test_extraction_service_handles_invalid_json(extraction_service):
    # Error handling: LLM returns invalid data, parser should raise
    text = "Carrier: ACME"
    with patch.object(extraction_service.parser, "get_format_instructions", return_value="FORMAT"):
        with patch.object(extraction_service.llm, "invoke", return_value="not a ShipmentData object"):
            with pytest.raises(Exception):
                extraction_service.extract(text)
