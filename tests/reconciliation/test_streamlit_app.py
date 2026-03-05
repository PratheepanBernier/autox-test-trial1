# source_hash: 5fab573753f93532
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
    Location,
    CarrierInfo,
    DriverInfo,
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
    return b"Carrier Details\nJohn Doe\nRate Breakdown\n$1000\nPickup\n123 Main St\nDrop\n456 Elm St\nCommodity\nWidgets\nSpecial Instructions\nHandle with care"

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

def test_process_file_equivalent_content_pdf_vs_txt(ingestion_service, sample_pdf_bytes, sample_txt_bytes, sample_filename_pdf, sample_filename_txt):
    # Mock pymupdf.open and page.get_text for PDF
    with patch("streamlit_app.pymupdf.open") as mock_pdf_open:
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Carrier Details\nJohn Doe\nRate Breakdown\n$1000\nPickup\n123 Main St\nDrop\n456 Elm St\nCommodity\nWidgets\nSpecial Instructions\nHandle with care"
        mock_doc.__iter__.return_value = [mock_page]
        mock_pdf_open.return_value = mock_doc

        pdf_chunks = ingestion_service.process_file(sample_pdf_bytes, sample_filename_pdf)

    # TXT path
    txt_chunks = ingestion_service.process_file(sample_txt_bytes, sample_filename_txt)

    # Reconciliation: The chunk texts should be equivalent (ignoring chunking artifacts)
    pdf_text = "\n".join([c.text for c in pdf_chunks])
    txt_text = "\n".join([c.text for c in txt_chunks])
    assert pdf_text.replace("\n", "").replace(" ", "") == txt_text.replace("\n", "").replace(" ", "")

def test_process_file_equivalent_content_docx_vs_txt(ingestion_service, sample_docx_bytes, sample_txt_bytes, sample_filename_docx, sample_filename_txt):
    # Mock docx.Document and paragraphs for DOCX
    with patch("streamlit_app.docx.Document") as mock_docx_doc:
        mock_doc = MagicMock()
        mock_doc.paragraphs = [
            MagicMock(text="Carrier Details"),
            MagicMock(text="John Doe"),
            MagicMock(text="Rate Breakdown"),
            MagicMock(text="$1000"),
            MagicMock(text="Pickup"),
            MagicMock(text="123 Main St"),
            MagicMock(text="Drop"),
            MagicMock(text="456 Elm St"),
            MagicMock(text="Commodity"),
            MagicMock(text="Widgets"),
            MagicMock(text="Special Instructions"),
            MagicMock(text="Handle with care"),
        ]
        mock_docx_doc.return_value = mock_doc

        docx_chunks = ingestion_service.process_file(sample_docx_bytes, sample_filename_docx)

    txt_chunks = ingestion_service.process_file(sample_txt_bytes, sample_filename_txt)

    docx_text = "\n".join([c.text for c in docx_chunks])
    txt_text = "\n".join([c.text for c in txt_chunks])
    assert docx_text.replace("\n", "").replace(" ", "") == txt_text.replace("\n", "").replace(" ", "")

def test_process_file_empty_file(ingestion_service):
    # Edge: empty file for each type
    for ext in [".pdf", ".docx", ".txt"]:
        filename = "empty" + ext
        if ext == ".pdf":
            with patch("streamlit_app.pymupdf.open") as mock_pdf_open:
                mock_doc = MagicMock()
                mock_doc.__iter__.return_value = []
                mock_pdf_open.return_value = mock_doc
                chunks = ingestion_service.process_file(b"", filename)
        elif ext == ".docx":
            with patch("streamlit_app.docx.Document") as mock_docx_doc:
                mock_doc = MagicMock()
                mock_doc.paragraphs = []
                mock_docx_doc.return_value = mock_doc
                chunks = ingestion_service.process_file(b"", filename)
        else:
            chunks = ingestion_service.process_file(b"", filename)
        assert len(chunks) == 1
        assert chunks[0].text == ""

def test_process_file_boundary_chunking(ingestion_service):
    # Test chunking at boundary: text exactly at chunk size
    ingestion_service.text_splitter.chunk_size = 10
    ingestion_service.text_splitter.chunk_overlap = 0
    text = "A" * 10 + "B" * 10 + "C" * 10  # 30 chars, should be 3 chunks
    chunks = ingestion_service.process_file(text.encode(), "test.txt")
    assert len(chunks) == 3
    assert chunks[0].text == "A" * 10
    assert chunks[1].text == "B" * 10
    assert chunks[2].text == "C" * 10

def test_vector_store_add_documents_equivalent(vector_store_service):
    # Two equivalent sets of chunks should result in equivalent vector store calls
    chunk1 = Chunk(text="foo", metadata=DocumentMetadata(filename="a.txt", chunk_id=0, source="a.txt - Part 1"))
    chunk2 = Chunk(text="bar", metadata=DocumentMetadata(filename="a.txt", chunk_id=1, source="a.txt - Part 2"))
    chunks1 = [chunk1, chunk2]
    chunks2 = [chunk1, chunk2]

    with patch("streamlit_app.FAISS") as mock_faiss:
        mock_vs = MagicMock()
        mock_faiss.from_documents.return_value = mock_vs
        vs1 = VectorStoreService()
        vs1.add_documents(chunks1)
        vs2 = VectorStoreService()
        vs2.add_documents(chunks2)
        # Both should call FAISS.from_documents with equivalent arguments
        assert mock_faiss.from_documents.call_count == 2
        args1, kwargs1 = mock_faiss.from_documents.call_args_list[0]
        args2, kwargs2 = mock_faiss.from_documents.call_args_list[1]
        # Compare the page_content of the first document in each call
        assert args1[0][0].page_content == args2[0][0].page_content

def test_rag_service_answer_equivalent_paths(rag_service):
    # Mock vector_store and retriever
    mock_retriever = MagicMock()
    doc1 = MagicMock(page_content="foo", metadata={"filename": "a.txt", "chunk_id": 0, "source": "a.txt - Part 1"})
    doc2 = MagicMock(page_content="bar", metadata={"filename": "a.txt", "chunk_id": 1, "source": "a.txt - Part 2"})
    mock_retriever.invoke.return_value = [doc1, doc2]
    rag_service.vs.vector_store = MagicMock()
    rag_service.vs.vector_store.as_retriever.return_value = mock_retriever

    # Patch the LLM chain to return a fixed answer
    with patch.object(rag_service.llm, "invoke", return_value="The answer is foo and bar"):
        with patch("streamlit_app.StrOutputParser") as mock_parser:
            mock_parser.return_value = lambda x: "The answer is foo and bar"
            answer1 = rag_service.answer("What is in the docs?")
            answer2 = rag_service.answer("What is in the docs?")
            assert answer1.answer == answer2.answer
            assert answer1.confidence_score == answer2.confidence_score
            assert [c.text for c in answer1.sources] == [c.text for c in answer2.sources]

def test_rag_service_answer_not_found_confidence(rag_service):
    # Simulate LLM returning "cannot find"
    mock_retriever = MagicMock()
    doc = MagicMock(page_content="irrelevant", metadata={"filename": "a.txt", "chunk_id": 0, "source": "a.txt - Part 1"})
    mock_retriever.invoke.return_value = [doc]
    rag_service.vs.vector_store = MagicMock()
    rag_service.vs.vector_store.as_retriever.return_value = mock_retriever

    with patch.object(rag_service.llm, "invoke", return_value="I cannot find the answer in the provided documents."):
        with patch("streamlit_app.StrOutputParser") as mock_parser:
            mock_parser.return_value = lambda x: "I cannot find the answer in the provided documents."
            answer = rag_service.answer("Unknown question?")
            assert answer.confidence_score == 0.1
            assert "cannot find" in answer.answer.lower()

def test_extraction_service_equivalent_texts(extraction_service):
    # Patch the LLM chain to return a fixed ShipmentData
    shipment_data = ShipmentData(
        reference_id="REF123",
        shipper="Acme",
        consignee="Beta",
        carrier=CarrierInfo(carrier_name="CarrierX"),
        driver=DriverInfo(driver_name="DriverY"),
        pickup=Location(name="Warehouse A"),
        drop=Location(name="Warehouse B"),
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Van",
        rate_info=RateInfo(total_rate=1000.0, currency="USD"),
        special_instructions="Fragile"
    )
    with patch.object(extraction_service.llm, "invoke", return_value=shipment_data.model_dump()):
        with patch("streamlit_app.PydanticOutputParser") as mock_parser:
            mock_parser.return_value = MagicMock(
                get_format_instructions=lambda: "",
                __or__=lambda self, other: lambda x: shipment_data
            )
            text1 = "Pickup at Warehouse A. Drop at Warehouse B. Carrier: CarrierX. Driver: DriverY. Rate: $1000."
            text2 = "Carrier: CarrierX. Driver: DriverY. Pickup at Warehouse A. Drop at Warehouse B. Rate: $1000."
            result1 = extraction_service.extract(text1)
            result2 = extraction_service.extract(text2)
            assert result1.model_dump() == result2.model_dump()

def test_extraction_service_missing_fields(extraction_service):
    # Simulate missing fields in extraction
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
    with patch.object(extraction_service.llm, "invoke", return_value=shipment_data.model_dump()):
        with patch("streamlit_app.PydanticOutputParser") as mock_parser:
            mock_parser.return_value = MagicMock(
                get_format_instructions=lambda: "",
                __or__=lambda self, other: lambda x: shipment_data
            )
            text = ""
            result = extraction_service.extract(text)
            assert result.model_dump() == shipment_data.model_dump()
            # All fields should be None
            for v in result.model_dump().values():
                assert v is None

def test_document_ingestion_service_error_handling_pdf(ingestion_service, sample_filename_pdf):
    # Simulate pymupdf.open raising an exception
    with patch("streamlit_app.pymupdf.open", side_effect=Exception("PDF error")):
        try:
            ingestion_service.process_file(b"bad pdf", sample_filename_pdf)
        except Exception as e:
            assert str(e) == "PDF error"

def test_document_ingestion_service_error_handling_docx(ingestion_service, sample_filename_docx):
    # Simulate docx.Document raising an exception
    with patch("streamlit_app.docx.Document", side_effect=Exception("DOCX error")):
        try:
            ingestion_service.process_file(b"bad docx", sample_filename_docx)
        except Exception as e:
            assert str(e) == "DOCX error"
