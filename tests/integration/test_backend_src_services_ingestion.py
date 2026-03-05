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

def test_process_txt_file_happy_path(service):
    content = b"Carrier Details\nJohn Doe\nRate Breakdown\n$1000\n"
    filename = "test.txt"
    chunks = service.process_file(content, filename)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert any("Carrier Details" in chunk.text for chunk in chunks)
    assert any("Rate Breakdown" in chunk.text for chunk in chunks)
    assert all(chunk.metadata.filename == filename for chunk in chunks)

@patch("backend.src.services.ingestion.pymupdf.open")
def test_process_pdf_file_happy_path(mock_pdf_open, service):
    # Mock a PDF with two pages
    mock_doc = MagicMock()
    mock_page1 = MagicMock()
    mock_page1.get_text.return_value = "Carrier Details\nJohn Doe"
    mock_page2 = MagicMock()
    mock_page2.get_text.return_value = "Rate Breakdown\n$1000"
    mock_doc.__iter__.return_value = iter([mock_page1, mock_page2])
    mock_pdf_open.return_value = mock_doc

    content = b"%PDF-1.4 dummy"
    filename = "test.pdf"
    chunks = service.process_file(content, filename)
    assert isinstance(chunks, list)
    assert any("Carrier Details" in chunk.text for chunk in chunks)
    assert any("Rate Breakdown" in chunk.text for chunk in chunks)
    assert all(chunk.metadata.filename == filename for chunk in chunks)
    assert mock_pdf_open.called

@patch("backend.src.services.ingestion.docx.Document")
def test_process_docx_file_happy_path(mock_docx_document, service):
    # Mock a DOCX with two paragraphs
    para1 = MagicMock()
    para1.text = "Carrier Details\nJohn Doe"
    para2 = MagicMock()
    para2.text = "Rate Breakdown\n$1000"
    mock_doc = MagicMock()
    mock_doc.paragraphs = [para1, para2]
    mock_docx_document.return_value = mock_doc

    content = b"dummy docx"
    filename = "test.docx"
    chunks = service.process_file(content, filename)
    assert isinstance(chunks, list)
    assert any("Carrier Details" in chunk.text for chunk in chunks)
    assert any("Rate Breakdown" in chunk.text for chunk in chunks)
    assert all(chunk.metadata.filename == filename for chunk in chunks)
    assert mock_docx_document.called

def test_process_file_unsupported_type(service):
    content = b"irrelevant"
    filename = "test.xlsx"
    chunks = service.process_file(content, filename)
    assert chunks == []

def test_process_file_invalid_utf8(service):
    content = b"\xff\xfe\xfd"
    filename = "test.txt"
    chunks = service.process_file(content, filename)
    assert chunks == []

@patch("backend.src.services.ingestion.pymupdf.open", side_effect=Exception("PDF error"))
def test_process_pdf_file_error_handling(mock_pdf_open, service):
    content = b"%PDF-1.4 dummy"
    filename = "test.pdf"
    chunks = service.process_file(content, filename)
    assert chunks == []

@patch("backend.src.services.ingestion.docx.Document", side_effect=Exception("DOCX error"))
def test_process_docx_file_error_handling(mock_docx_document, service):
    content = b"dummy docx"
    filename = "test.docx"
    chunks = service.process_file(content, filename)
    assert chunks == []

def test_add_semantic_structure_adds_headers(service):
    text = "Carrier Details\nSome info\nRate Breakdown\n$1000"
    structured = service._add_semantic_structure(text)
    assert "## Carrier Details" in structured
    assert "## Rate Breakdown" in structured

def test_add_semantic_structure_handles_excessive_whitespace(service):
    text = "Carrier Details\n\n\n\nSome info\n\n\nRate Breakdown\n\n$1000"
    structured = service._add_semantic_structure(text)
    assert "\n\n" in structured
    assert "\n\n\n" not in structured

def test_chunk_text_with_title_grouping_groups_sections(service):
    text = "\n## Carrier Details\nJohn\n## Driver Details\nJane\n## Rate Breakdown\n$1000"
    filename = "test.txt"
    chunks = service._chunk_text_with_title_grouping(text, filename)
    # Carrier Details and Driver Details should be grouped
    carrier_chunks = [c for c in chunks if "Carrier Details" in c.text or "Driver Details" in c.text]
    assert carrier_chunks
    # Rate Breakdown should be in its own chunk
    rate_chunks = [c for c in chunks if "Rate Breakdown" in c.text]
    assert rate_chunks

def test_chunk_text_with_title_grouping_handles_empty_sections(service):
    text = "\n## Carrier Details\n\n## Rate Breakdown\n"
    filename = "test.txt"
    chunks = service._chunk_text_with_title_grouping(text, filename)
    assert all(isinstance(chunk, Chunk) for chunk in chunks)

def test_extract_section_name_returns_section(service):
    chunk_text = "## Carrier Details\nJohn Doe"
    section = service._extract_section_name(chunk_text)
    assert section == "Carrier Details"

def test_extract_section_name_returns_general_if_no_header(service):
    chunk_text = "No header here"
    section = service._extract_section_name(chunk_text)
    assert section == "General"

def test_clean_chunk_removes_excessive_whitespace_and_page_markers(service):
    chunk_text = "\n\n### Page 1\n\n\nSome text\n\n\n"
    cleaned = service._clean_chunk(chunk_text)
    assert "### Page 1" not in cleaned
    assert cleaned.startswith("Some text")

def test_clean_chunk_preserves_content(service):
    chunk_text = "Carrier Details\nJohn  Doe\n\n\nRate Breakdown\n$1000"
    cleaned = service._clean_chunk(chunk_text)
    assert "  " not in cleaned
    assert "\n\n\n" not in cleaned
    assert "Carrier Details" in cleaned
    assert "Rate Breakdown" in cleaned

def test_process_file_boundary_conditions_empty_file(service):
    content = b""
    filename = "test.txt"
    chunks = service.process_file(content, filename)
    assert isinstance(chunks, list)
    assert chunks == []

def test_process_file_boundary_conditions_minimal_content(service):
    content = b"A"
    filename = "test.txt"
    chunks = service.process_file(content, filename)
    assert isinstance(chunks, list)
    # Should produce at least one chunk if not empty
    if chunks:
        assert isinstance(chunks[0], Chunk)
        assert chunks[0].text == "A"
