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

def test_process_file_txt_happy_path(service):
    content = b"Carrier Details\nJohn Doe\nRate Breakdown\n$1000\n"
    filename = "test.txt"
    chunks = service.process_file(content, filename)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert any("Carrier Details" in chunk.text for chunk in chunks)
    assert any("Rate Breakdown" in chunk.text for chunk in chunks)
    assert all(chunk.metadata.filename == filename for chunk in chunks)

@patch("backend.src.services.ingestion.pymupdf.open")
def test_process_file_pdf_happy_path(mock_pdf_open, service):
    # Mock PyMuPDF document and pages
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
def test_process_file_docx_happy_path(mock_docx_document, service):
    # Mock docx.Document and paragraphs
    mock_doc = MagicMock()
    mock_para1 = MagicMock()
    mock_para1.text = "Carrier Details"
    mock_para2 = MagicMock()
    mock_para2.text = "John Doe"
    mock_para3 = MagicMock()
    mock_para3.text = "Rate Breakdown"
    mock_para4 = MagicMock()
    mock_para4.text = "$1000"
    mock_doc.paragraphs = [mock_para1, mock_para2, mock_para3, mock_para4]
    mock_docx_document.return_value = mock_doc

    content = b"dummy docx"
    filename = "test.docx"
    chunks = service.process_file(content, filename)
    assert isinstance(chunks, list)
    assert any("Carrier Details" in chunk.text for chunk in chunks)
    assert any("Rate Breakdown" in chunk.text for chunk in chunks)
    assert all(chunk.metadata.filename == filename for chunk in chunks)
    assert mock_docx_document.called

def test_process_file_unsupported_extension(service):
    content = b"irrelevant"
    filename = "test.xls"
    chunks = service.process_file(content, filename)
    assert chunks == []

def test_process_file_txt_empty_content(service):
    content = b""
    filename = "empty.txt"
    chunks = service.process_file(content, filename)
    assert isinstance(chunks, list)
    assert chunks == []

def test_add_semantic_structure_adds_headers(service):
    text = "Carrier Details\nSome info\nRate Breakdown\n$1000"
    structured = service._add_semantic_structure(text)
    assert "\n## Carrier Details\n" in structured
    assert "\n## Rate Breakdown\n" in structured

def test_add_semantic_structure_handles_multiple_whitespace(service):
    text = "Carrier Details\n\n\n\nSome info\n\n\nRate Breakdown"
    structured = service._add_semantic_structure(text)
    assert "\n\n" in structured
    assert "\n## Carrier Details\n" in structured

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

def test_chunk_text_with_title_grouping_handles_no_headers(service):
    text = "Just some random text without headers."
    filename = "test.txt"
    chunks = service._chunk_text_with_title_grouping(text, filename)
    assert len(chunks) == 1
    assert "random text" in chunks[0].text

def test_extract_section_name_with_header(service):
    chunk_text = "## Carrier Details\nJohn Doe"
    section = service._extract_section_name(chunk_text)
    assert section == "Carrier Details"

def test_extract_section_name_without_header(service):
    chunk_text = "No header here"
    section = service._extract_section_name(chunk_text)
    assert section == "General"

def test_clean_chunk_removes_excessive_whitespace_and_page_markers(service):
    chunk_text = "\n\n\n### Page 1\n\n\nCarrier Details\n\n\nJohn Doe\n\n\n"
    cleaned = service._clean_chunk(chunk_text)
    assert "### Page 1" not in cleaned
    assert cleaned.startswith("Carrier Details")
    assert cleaned.endswith("John Doe")

def test_clean_chunk_preserves_content(service):
    chunk_text = "Carrier Details\nJohn  Doe"
    cleaned = service._clean_chunk(chunk_text)
    assert "  " not in cleaned
    assert "Carrier Details" in cleaned
    assert "John Doe" in cleaned

def test_process_file_handles_exception_in_extraction(service):
    # Patch _extract_text_from_pdf to raise
    with patch.object(service, "_extract_text_from_pdf", side_effect=Exception("fail")):
        content = b"dummy"
        filename = "fail.pdf"
        chunks = service.process_file(content, filename)
        assert chunks == []

def test_process_file_handles_unicode_decode_error(service):
    # Simulate decode error for txt
    content = b"\xff\xfe\x00\x00"
    filename = "bad.txt"
    chunks = service.process_file(content, filename)
    assert chunks == []

def test_chunk_text_with_title_grouping_empty_text(service):
    text = ""
    filename = "empty.txt"
    chunks = service._chunk_text_with_title_grouping(text, filename)
    assert chunks == []

def test_chunk_text_with_title_grouping_boundary_chunk_size(service):
    # Simulate a section just at the chunk size boundary
    service.text_splitter.chunk_size = 10
    text = "\n## Carrier Details\n" + "A" * 10
    filename = "test.txt"
    chunks = service._chunk_text_with_title_grouping(text, filename)
    assert all(len(chunk.text) <= 10 + len("\n## Carrier Details\n") for chunk in chunks)
    assert any("Carrier Details" in chunk.text for chunk in chunks)
