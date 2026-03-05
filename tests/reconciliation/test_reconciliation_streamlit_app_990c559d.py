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

import pytest
from unittest.mock import patch, MagicMock, call
import io

import streamlit_app
from streamlit_app import (
    DocumentIngestionService,
    VectorStoreService,
    RAGService,
    ExtractionService,
    Chunk,
    DocumentMetadata,
    SourcedAnswer,
    ShipmentData,
)

@pytest.fixture
def sample_pdf_bytes():
    return b"%PDF-1.4 sample pdf content"

@pytest.fixture
def sample_docx_bytes():
    # Simulate a docx file in memory
    return b"PK\x03\x04 sample docx content"

@pytest.fixture
def sample_txt_bytes():
    return b"Carrier Details: ABC\nRate Breakdown: $1000\nPickup: NYC\nDrop: LA\nCommodity: Widgets\nSpecial Instructions: None"

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

def test_process_file_pdf_and_docx_and_txt_equivalence(
    ingestion_service, sample_pdf_bytes, sample_docx_bytes, sample_txt_bytes,
    sample_filename_pdf, sample_filename_docx, sample_filename_txt
):
    # Mock pymupdf and docx.Document for deterministic output
    with patch("streamlit_app.pymupdf.open") as mock_pdf_open, \
         patch("streamlit_app.docx.Document") as mock_docx_doc:

        # PDF: simulate 2 pages
        mock_pdf = MagicMock()
        mock_pdf.__iter__.return_value = [
            MagicMock(get_text=MagicMock(return_value="Carrier Details: ABC\nRate Breakdown: $1000")),
            MagicMock(get_text=MagicMock(return_value="Pickup: NYC\nDrop: LA\nCommodity: Widgets\nSpecial Instructions: None"))
        ]
        mock_pdf_open.return_value = mock_pdf

        # DOCX: simulate paragraphs
        mock_docx = MagicMock()
        mock_docx.paragraphs = [
            MagicMock(text="Carrier Details: ABC"),
            MagicMock(text="Rate Breakdown: $1000"),
            MagicMock(text="Pickup: NYC"),
            MagicMock(text="Drop: LA"),
            MagicMock(text="Commodity: Widgets"),
            MagicMock(text="Special Instructions: None"),
        ]
        mock_docx_doc.return_value = mock_docx

        # TXT: already in correct format

        pdf_chunks = ingestion_service.process_file(sample_pdf_bytes, sample_filename_pdf)
        docx_chunks = ingestion_service.process_file(sample_docx_bytes, sample_filename_docx)
        txt_chunks = ingestion_service.process_file(sample_txt_bytes, sample_filename_txt)

        # Reconciliation: All three should produce similar chunked text content
        pdf_texts = [c.text for c in pdf_chunks]
        docx_texts = [c.text for c in docx_chunks]
        txt_texts = [c.text for c in txt_chunks]

        # All should contain the same semantic markers and sections
        for section in ["Carrier Details", "Rate Breakdown", "Pickup", "Drop", "Commodity", "Special Instructions"]:
            assert any(section in t for t in pdf_texts)
            assert any(section in t for t in docx_texts)
            assert any(section in t for t in txt_texts)

        # The number of chunks may differ due to chunking logic, but the content should be reconcilable
        assert set([s for t in pdf_texts for s in t.splitlines() if s.strip()]) == \
               set([s for t in docx_texts for s in t.splitlines() if s.strip()]) == \
               set([s for t in txt_texts for s in t.splitlines() if s.strip()])

def test_process_file_empty_and_invalid_extension(ingestion_service):
    # Edge: empty file
    empty_bytes = b""
    chunks = ingestion_service.process_file(empty_bytes, "empty.txt")
    assert len(chunks) == 1
    assert chunks[0].text == ""

    # Edge: unsupported extension
    chunks = ingestion_service.process_file(b"irrelevant", "file.xyz")
    assert len(chunks) == 1
    assert chunks[0].text == ""

def test_vector_store_add_documents_equivalence(vector_store_service):
    # Prepare two equivalent sets of chunks
    meta1 = DocumentMetadata(filename="a.txt", chunk_id=0, source="a.txt - Part 1")
    meta2 = DocumentMetadata(filename="a.txt", chunk_id=1, source="a.txt - Part 2")
    chunk1 = Chunk(text="Carrier Details: ABC", metadata=meta1)
    chunk2 = Chunk(text="Rate Breakdown: $1000", metadata=meta2)
    chunks_a = [chunk1, chunk2]
    chunks_b = [chunk1, chunk2]

    with patch("streamlit_app.FAISS") as mock_faiss:
        mock_vs = MagicMock()
        mock_faiss.from_documents.return_value = mock_vs

        vs1 = VectorStoreService()
        vs1.embeddings = MagicMock()
        vs1.vector_store = None
        vs1.add_documents(chunks_a)

        vs2 = VectorStoreService()
        vs2.embeddings = MagicMock()
        vs2.vector_store = None
        vs2.add_documents(chunks_b)

        # Reconciliation: Both vector stores should be initialized via FAISS.from_documents with equivalent docs
        assert mock_faiss.from_documents.call_count == 2
        docs1 = mock_faiss.from_documents.call_args_list[0][0][0]
        docs2 = mock_faiss.from_documents.call_args_list[1][0][0]
        assert [d.page_content for d in docs1] == [d.page_content for d in docs2]
        assert [d.metadata for d in docs1] == [d.metadata for d in docs2]

def test_rag_service_answer_equivalence(rag_service):
    # Mock vector_store and retriever
    mock_retriever = MagicMock()
    mock_doc1 = MagicMock(page_content="Carrier Details: ABC", metadata={"filename": "a.txt", "chunk_id": 0, "source": "a.txt - Part 1", "chunk_type": "text"})
    mock_doc2 = MagicMock(page_content="Rate Breakdown: $1000", metadata={"filename": "a.txt", "chunk_id": 1, "source": "a.txt - Part 2", "chunk_type": "text"})
    mock_retriever.invoke.return_value = [mock_doc1, mock_doc2]

    rag_service.vs.vector_store = MagicMock()
    rag_service.vs.vector_store.as_retriever.return_value = mock_retriever

    # Patch LLM chain to return a deterministic answer
    with patch.object(rag_service.llm, "invoke", return_value="The carrier is ABC. The rate is $1000."), \
         patch("streamlit_app.StrOutputParser") as mock_parser:
        mock_parser.return_value.invoke.return_value = "The carrier is ABC. The rate is $1000."

        answer1 = rag_service.answer("Who is the carrier?")
        answer2 = rag_service.answer("What is the rate?")

        # Reconciliation: Both answers should have the same answer text and sources
        assert answer1.answer == answer2.answer
        assert [s.text for s in answer1.sources] == [s.text for s in answer2.sources]
        assert [s.metadata.model_dump() for s in answer1.sources] == [s.metadata.model_dump() for s in answer2.sources]

def test_rag_service_answer_not_found_confidence(rag_service):
    # Mock retriever returns empty
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []

    rag_service.vs.vector_store = MagicMock()
    rag_service.vs.vector_store.as_retriever.return_value = mock_retriever

    with patch.object(rag_service.llm, "invoke", return_value="I cannot find the answer in the provided documents."), \
         patch("streamlit_app.StrOutputParser") as mock_parser:
        mock_parser.return_value.invoke.return_value = "I cannot find the answer in the provided documents."

        answer = rag_service.answer("Unknown question?")
        assert answer.confidence_score == 0.1
        assert answer.answer.lower().startswith("i cannot find")
        assert answer.sources == []

def test_extraction_service_extract_equivalence(extraction_service):
    # Patch LLM and parser to return deterministic ShipmentData
    shipment_data = ShipmentData(reference_id="REF123", shipper="ABC", consignee="XYZ")
    mock_parser = MagicMock()
    mock_parser.get_format_instructions.return_value = "FORMAT"
    mock_parser.invoke.return_value = shipment_data

    with patch.object(extraction_service, "parser", mock_parser):
        text1 = "Reference ID: REF123\nShipper: ABC\nConsignee: XYZ"
        text2 = "Consignee: XYZ\nReference ID: REF123\nShipper: ABC"
        result1 = extraction_service.extract(text1)
        result2 = extraction_service.extract(text2)
        # Reconciliation: Extraction should yield the same ShipmentData for equivalent info
        assert result1 == result2
        assert result1.reference_id == "REF123"
        assert result1.shipper == "ABC"
        assert result1.consignee == "XYZ"

def test_extraction_service_extract_missing_fields(extraction_service):
    # Patch parser to return ShipmentData with None fields
    shipment_data = ShipmentData(reference_id=None, shipper=None, consignee=None)
    mock_parser = MagicMock()
    mock_parser.get_format_instructions.return_value = "FORMAT"
    mock_parser.invoke.return_value = shipment_data

    with patch.object(extraction_service, "parser", mock_parser):
        text = "No relevant data"
        result = extraction_service.extract(text)
        assert result.reference_id is None
        assert result.shipper is None
        assert result.consignee is None

def test_document_ingestion_service_chunking_boundary(ingestion_service):
    # Test chunking at boundary: text exactly at chunk_size*2
    text = "A" * (streamlit_app.CHUNK_SIZE * 2)
    chunks = ingestion_service.text_splitter.split_text(text)
    # Should produce one chunk if no overlap
    assert len(chunks) == 1 or len(chunks) == 2  # Depending on overlap config

def test_document_ingestion_service_section_markers(ingestion_service):
    # Test that section markers are inserted
    text = "Carrier Details\nRate Breakdown\nPickup\nDrop\nCommodity\nSpecial Instructions"
    file_bytes = text.encode("utf-8")
    chunks = ingestion_service.process_file(file_bytes, "test.txt")
    # Each section should be marked with \n## Section Name\n
    for section in ["Carrier Details", "Rate Breakdown", "Pickup", "Drop", "Commodity", "Special Instructions"]:
        assert any(f"## {section}" in c.text for c in chunks)
