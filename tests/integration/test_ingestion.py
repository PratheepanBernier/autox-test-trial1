# source_hash: 7de298d97adb91e8
import io
import pytest
from unittest.mock import patch, MagicMock

from backend.src.services.ingestion import DocumentIngestionService
from models.schemas import Chunk, DocumentMetadata

@pytest.fixture
def service():
    return DocumentIngestionService()

@pytest.fixture
def dummy_settings(monkeypatch):
    class DummySettings:
        CHUNK_SIZE = 100
        CHUNK_OVERLAP = 10
    monkeypatch.setattr("core.config.settings", DummySettings())

def make_pdf_bytes_with_text(pages):
    # Simulate PDF bytes for N pages with given text per page
    # We'll mock pymupdf.open, so content doesn't matter
    return b"%PDF-1.4 dummy content"

def make_docx_bytes_with_text(paragraphs):
    # We'll mock docx.Document, so content doesn't matter
    return b"PK\x03\x04 dummy docx content"

def make_txt_bytes_with_text(text):
    return text.encode("utf-8")

def chunk_texts(chunks):
    return [c.text for c in chunks]

def chunk_sources(chunks):
    return [c.metadata.source for c in chunks]

def chunk_ids(chunks):
    return [c.metadata.chunk_id for c in chunks]

def chunk_types(chunks):
    return [c.metadata.chunk_type for c in chunks]

def chunk_filenames(chunks):
    return [c.metadata.filename for c in chunks]

def chunk_section_names(chunks):
    return [c.metadata.source.split(" - ")[-1] for c in chunks]

# --- Happy Path Tests ---

@patch("backend.src.services.ingestion.pymupdf.open")
def test_process_pdf_happy_path(mock_pdf_open, service):
    # Setup: PDF with 2 pages, each with text
    mock_doc = MagicMock()
    mock_page1 = MagicMock()
    mock_page2 = MagicMock()
    mock_page1.get_text.return_value = "Carrier Details\nSome info\n"
    mock_page2.get_text.return_value = "Rate Breakdown\n$1000\n"
    mock_doc.__iter__.return_value = iter([mock_page1, mock_page2])
    mock_pdf_open.return_value = mock_doc

    pdf_bytes = make_pdf_bytes_with_text(2)
    filename = "test.pdf"
    chunks = service.process_file(pdf_bytes, filename)

    # Should extract both sections, add semantic structure, and chunk accordingly
    assert len(chunks) >= 2
    texts = chunk_texts(chunks)
    assert any("Carrier Details" in t for t in texts)
    assert any("Rate Breakdown" in t for t in texts)
    sources = chunk_sources(chunks)
    assert all(filename in s for s in sources)
    assert all(c.metadata.chunk_type == "text" for c in chunks)

@patch("backend.src.services.ingestion.docx.Document")
def test_process_docx_happy_path(mock_docx_doc, service):
    # Setup: DOCX with 2 paragraphs
    mock_doc = MagicMock()
    para1 = MagicMock()
    para2 = MagicMock()
    para1.text = "Pickup\nLocation A"
    para2.text = "Drop\nLocation B"
    mock_doc.paragraphs = [para1, para2]
    mock_docx_doc.return_value = mock_doc

    docx_bytes = make_docx_bytes_with_text(["Pickup\nLocation A", "Drop\nLocation B"])
    filename = "test.docx"
    chunks = service.process_file(docx_bytes, filename)

    assert len(chunks) >= 2
    texts = chunk_texts(chunks)
    assert any("Pickup" in t for t in texts)
    assert any("Drop" in t for t in texts)
    assert all(filename in c.metadata.source for c in chunks)

def test_process_txt_happy_path(service):
    txt = "Standing Instructions\nDo not stack\n\nSpecial Instructions\nFragile"
    txt_bytes = make_txt_bytes_with_text(txt)
    filename = "test.txt"
    chunks = service.process_file(txt_bytes, filename)
    assert len(chunks) >= 2
    texts = chunk_texts(chunks)
    assert any("Standing Instructions" in t for t in texts)
    assert any("Special Instructions" in t for t in texts)
    assert all(filename in c.metadata.source for c in chunks)

# --- Edge Cases and Boundary Conditions ---

def test_process_file_empty_txt(service):
    txt_bytes = make_txt_bytes_with_text("")
    filename = "empty.txt"
    chunks = service.process_file(txt_bytes, filename)
    assert chunks == []

@patch("backend.src.services.ingestion.pymupdf.open")
def test_process_pdf_empty_pages(mock_pdf_open, service):
    # PDF with 2 pages, both empty
    mock_doc = MagicMock()
    mock_page1 = MagicMock()
    mock_page2 = MagicMock()
    mock_page1.get_text.return_value = ""
    mock_page2.get_text.return_value = ""
    mock_doc.__iter__.return_value = iter([mock_page1, mock_page2])
    mock_pdf_open.return_value = mock_doc

    pdf_bytes = make_pdf_bytes_with_text(2)
    filename = "empty.pdf"
    chunks = service.process_file(pdf_bytes, filename)
    # Should not produce any non-empty chunks
    assert all(not c.text for c in chunks) or chunks == []

@patch("backend.src.services.ingestion.docx.Document")
def test_process_docx_no_paragraphs(mock_docx_doc, service):
    mock_doc = MagicMock()
    mock_doc.paragraphs = []
    mock_docx_doc.return_value = mock_doc

    docx_bytes = make_docx_bytes_with_text([])
    filename = "empty.docx"
    chunks = service.process_file(docx_bytes, filename)
    assert chunks == []

def test_process_txt_only_whitespace(service):
    txt_bytes = make_txt_bytes_with_text("   \n\n   ")
    filename = "whitespace.txt"
    chunks = service.process_file(txt_bytes, filename)
    assert chunks == []

def test_process_file_boundary_chunk_size(service):
    # Text exactly at chunk size boundary
    text = "Carrier Details\n" + "A" * (service.text_splitter.chunk_size)
    txt_bytes = make_txt_bytes_with_text(text)
    filename = "boundary.txt"
    chunks = service.process_file(txt_bytes, filename)
    # Should produce at least one chunk, not more than two
    assert 1 <= len(chunks) <= 2
    assert all(len(c.text) <= service.text_splitter.chunk_size * 2 for c in chunks)

# --- Error Handling ---

def test_process_file_unsupported_type(service):
    file_bytes = b"dummy"
    filename = "test.xlsx"
    chunks = service.process_file(file_bytes, filename)
    assert chunks == []

@patch("backend.src.services.ingestion.pymupdf.open")
def test_process_pdf_exception_handling(mock_pdf_open, service):
    mock_pdf_open.side_effect = Exception("PDF read error")
    pdf_bytes = make_pdf_bytes_with_text(1)
    filename = "bad.pdf"
    chunks = service.process_file(pdf_bytes, filename)
    assert chunks == []

@patch("backend.src.services.ingestion.docx.Document")
def test_process_docx_exception_handling(mock_docx_doc, service):
    mock_docx_doc.side_effect = Exception("DOCX read error")
    docx_bytes = make_docx_bytes_with_text(["foo"])
    filename = "bad.docx"
    chunks = service.process_file(docx_bytes, filename)
    assert chunks == []

def test_process_txt_decode_error(service):
    # Invalid UTF-8 bytes
    txt_bytes = b"\xff\xfe\xfd"
    filename = "bad.txt"
    chunks = service.process_file(txt_bytes, filename)
    assert chunks == []

# --- Reconciliation/Regression: Equivalent Path Consistency ---

@patch("backend.src.services.ingestion.pymupdf.open")
@patch("backend.src.services.ingestion.docx.Document")
def test_equivalent_content_pdf_docx_txt(mock_docx_doc, mock_pdf_open, service):
    # All three files contain "Carrier Details\nJohn Doe\nRate Breakdown\n$1000"
    # PDF
    mock_pdf_doc = MagicMock()
    mock_pdf_page = MagicMock()
    mock_pdf_page.get_text.return_value = "Carrier Details\nJohn Doe\nRate Breakdown\n$1000"
    mock_pdf_doc.__iter__.return_value = iter([mock_pdf_page])
    mock_pdf_open.return_value = mock_pdf_doc
    pdf_bytes = make_pdf_bytes_with_text(1)
    pdf_chunks = service.process_file(pdf_bytes, "eq.pdf")

    # DOCX
    mock_doc = MagicMock()
    para1 = MagicMock()
    para2 = MagicMock()
    para1.text = "Carrier Details\nJohn Doe"
    para2.text = "Rate Breakdown\n$1000"
    mock_doc.paragraphs = [para1, para2]
    mock_docx_doc.return_value = mock_doc
    docx_bytes = make_docx_bytes_with_text(["Carrier Details\nJohn Doe", "Rate Breakdown\n$1000"])
    docx_chunks = service.process_file(docx_bytes, "eq.docx")

    # TXT
    txt = "Carrier Details\nJohn Doe\n\nRate Breakdown\n$1000"
    txt_bytes = make_txt_bytes_with_text(txt)
    txt_chunks = service.process_file(txt_bytes, "eq.txt")

    # All should produce at least two chunks, and the section names should match
    pdf_sections = set(chunk_section_names(pdf_chunks))
    docx_sections = set(chunk_section_names(docx_chunks))
    txt_sections = set(chunk_section_names(txt_chunks))
    assert {"Carrier Details", "Rate Breakdown"} <= pdf_sections
    assert pdf_sections == docx_sections == txt_sections

# --- Internal Method: _add_semantic_structure ---

def test_add_semantic_structure_adds_headers(service):
    text = "Carrier Details\nSome info\nRate Breakdown\n$1000"
    structured = service._add_semantic_structure(text)
    # Should add markdown headers
    assert "\n## Carrier Details\n" in structured
    assert "\n## Rate Breakdown\n" in structured

def test_add_semantic_structure_cleans_whitespace(service):
    text = "\n\n\nCarrier Details\n\n\nSome info\n\n\n"
    structured = service._add_semantic_structure(text)
    # Should not have more than two consecutive newlines
    assert "\n\n\n" not in structured

# --- Internal Method: _clean_chunk ---

def test_clean_chunk_removes_excessive_whitespace_and_page_markers(service):
    chunk = "\n\n\n### Page 1\n\n\nCarrier Details\n\n\nSome info\n\n\n"
    cleaned = service._clean_chunk(chunk)
    assert "### Page 1" not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned.startswith("Carrier Details")

# --- Internal Method: _extract_section_name ---

def test_extract_section_name_with_header(service):
    chunk = "## Pickup\nLocation A"
    section = service._extract_section_name(chunk)
    assert section == "Pickup"

def test_extract_section_name_without_header(service):
    chunk = "No header here"
    section = service._extract_section_name(chunk)
    assert section == "General"
