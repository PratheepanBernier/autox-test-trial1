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

def make_pdf_bytes(pages):
    # Simulate PDF bytes; actual content is not parsed due to mocking
    return b"%PDF-1.4 dummy content"

def make_docx_bytes(paragraphs):
    # Simulate DOCX bytes; actual content is not parsed due to mocking
    return b"PK\x03\x04 dummy docx content"

def make_txt_bytes(text):
    return text.encode("utf-8")

def mock_pymupdf_open(stream, filetype):
    # Simulate a PDF with two pages
    class DummyPage:
        def __init__(self, text):
            self._text = text
        def get_text(self):
            return self._text
    class DummyDoc:
        def __iter__(self):
            return iter([DummyPage("Carrier Details\nSome info."), DummyPage("Rate Breakdown\nMore info.")])
    return DummyDoc()

def mock_docx_document(file_like):
    class DummyParagraph:
        def __init__(self, text):
            self.text = text
    class DummyDoc:
        paragraphs = [DummyParagraph("Carrier Details"), DummyParagraph("Some info."), DummyParagraph("Rate Breakdown"), DummyParagraph("More info.")]
    return DummyDoc()

def test_process_pdf_happy_path(service, monkeypatch):
    monkeypatch.setattr("pymupdf.open", mock_pymupdf_open)
    file_content = make_pdf_bytes(2)
    filename = "test.pdf"
    chunks = service.process_file(file_content, filename)
    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert any("Carrier Details" in c.text for c in chunks)
    assert any("Rate Breakdown" in c.text for c in chunks)
    # Should have at least two chunks (one per section)
    assert len(chunks) >= 2

def test_process_docx_happy_path(service, monkeypatch):
    monkeypatch.setattr("docx.Document", mock_docx_document)
    file_content = make_docx_bytes(4)
    filename = "test.docx"
    chunks = service.process_file(file_content, filename)
    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert any("Carrier Details" in c.text for c in chunks)
    assert any("Rate Breakdown" in c.text for c in chunks)
    assert len(chunks) >= 2

def test_process_txt_happy_path(service):
    text = "Carrier Details\nSome info.\n\nRate Breakdown\nMore info."
    file_content = make_txt_bytes(text)
    filename = "test.txt"
    chunks = service.process_file(file_content, filename)
    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert any("Carrier Details" in c.text for c in chunks)
    assert any("Rate Breakdown" in c.text for c in chunks)
    assert len(chunks) >= 2

def test_process_file_unsupported_type_returns_empty(service):
    file_content = b"random content"
    filename = "test.xlsx"
    chunks = service.process_file(file_content, filename)
    assert chunks == []

def test_process_file_pdf_extraction_error_returns_empty(service, monkeypatch):
    def raise_error(*args, **kwargs):
        raise RuntimeError("PDF error")
    monkeypatch.setattr("pymupdf.open", raise_error)
    file_content = make_pdf_bytes(1)
    filename = "test.pdf"
    chunks = service.process_file(file_content, filename)
    assert chunks == []

def test_process_file_docx_extraction_error_returns_empty(service, monkeypatch):
    def raise_error(*args, **kwargs):
        raise RuntimeError("DOCX error")
    monkeypatch.setattr("docx.Document", raise_error)
    file_content = make_docx_bytes(1)
    filename = "test.docx"
    chunks = service.process_file(file_content, filename)
    assert chunks == []

def test_process_file_txt_decoding_error_returns_empty(service, monkeypatch):
    # Simulate bytes that can't be decoded as utf-8
    file_content = b"\xff\xfe\xfd"
    filename = "test.txt"
    chunks = service.process_file(file_content, filename)
    assert chunks == []

def test_add_semantic_structure_adds_headers(service):
    text = "Carrier Details\nSome info.\nRate Breakdown\nMore info."
    structured = service._add_semantic_structure(text)
    assert "## Carrier Details" in structured
    assert "## Rate Breakdown" in structured

def test_chunk_text_with_title_grouping_groups_sections(service):
    text = "\n## Carrier Details\nCarrier info.\n## Driver Details\nDriver info.\n## Rate Breakdown\nRate info."
    filename = "test.txt"
    chunks = service._chunk_text_with_title_grouping(text, filename)
    # Carrier Details and Driver Details should be grouped
    carrier_chunks = [c for c in chunks if "Carrier Details" in c.metadata.source or "Driver Details" in c.metadata.source]
    assert carrier_chunks
    # Rate Breakdown should be a separate chunk
    rate_chunks = [c for c in chunks if "Rate Breakdown" in c.metadata.source]
    assert rate_chunks

def test_chunk_text_with_title_grouping_handles_empty_sections(service):
    text = "\n## Carrier Details\n\n## Rate Breakdown\n"
    filename = "test.txt"
    chunks = service._chunk_text_with_title_grouping(text, filename)
    assert len(chunks) >= 1  # At least one chunk, even if empty sections

def test_clean_chunk_removes_excess_whitespace_and_page_markers(service):
    chunk_text = "\n\n### Page 1\n\nSome text.\n\n\n### Page 2\n\n"
    cleaned = service._clean_chunk(chunk_text)
    assert "Page 1" not in cleaned
    assert "Page 2" not in cleaned
    assert "Some text." in cleaned

def test_extract_section_name_returns_section(service):
    chunk_text = "## Carrier Details\nSome info."
    section = service._extract_section_name(chunk_text)
    assert section == "Carrier Details"

def test_extract_section_name_returns_general_on_no_match(service):
    chunk_text = "No section header here."
    section = service._extract_section_name(chunk_text)
    assert section == "General"

def test_process_file_boundary_conditions_empty_file(service):
    file_content = b""
    filename = "test.txt"
    chunks = service.process_file(file_content, filename)
    assert chunks == []

def test_process_file_boundary_conditions_large_file(service, monkeypatch):
    # Simulate a large text file
    text = "Carrier Details\n" + ("A" * 10000) + "\nRate Breakdown\n" + ("B" * 10000)
    file_content = make_txt_bytes(text)
    filename = "large.txt"
    chunks = service.process_file(file_content, filename)
    assert isinstance(chunks, list)
    assert len(chunks) > 1
    assert any("Carrier Details" in c.text for c in chunks)
    assert any("Rate Breakdown" in c.text for c in chunks)

def test_reconciliation_txt_vs_docx_equivalent_content(service, monkeypatch):
    # TXT and DOCX with same content should produce similar chunks
    txt_content = "Carrier Details\nSome info.\nRate Breakdown\nMore info."
    txt_bytes = make_txt_bytes(txt_content)
    docx_bytes = make_docx_bytes(4)
    # Patch docx.Document to return paragraphs matching txt_content
    class DummyParagraph:
        def __init__(self, text):
            self.text = text
    class DummyDoc:
        paragraphs = [DummyParagraph("Carrier Details"), DummyParagraph("Some info."), DummyParagraph("Rate Breakdown"), DummyParagraph("More info.")]
    monkeypatch.setattr("docx.Document", lambda file_like: DummyDoc())
    txt_chunks = service.process_file(txt_bytes, "test.txt")
    docx_chunks = service.process_file(docx_bytes, "test.docx")
    # Compare chunk texts (ignoring metadata)
    txt_texts = sorted([c.text for c in txt_chunks])
    docx_texts = sorted([c.text for c in docx_chunks])
    assert txt_texts == docx_texts

def test_process_file_handles_section_headers_case_insensitive(service):
    text = "carrier details\nSome info.\nRATE BREAKDOWN\nMore info."
    file_content = make_txt_bytes(text)
    filename = "test.txt"
    chunks = service.process_file(file_content, filename)
    assert any("Carrier Details" in c.text or "carrier details" in c.text for c in chunks)
    assert any("Rate Breakdown" in c.text or "RATE BREAKDOWN" in c.text for c in chunks)
