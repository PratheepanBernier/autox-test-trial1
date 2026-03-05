# source_hash: 7de298d97adb91e8
# import_target: backend.src.services.ingestion
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import pytest
from unittest.mock import patch, MagicMock
import io

from backend.src.services.ingestion import DocumentIngestionService, ingestion_service
from models.schemas import Chunk, DocumentMetadata

@pytest.fixture
def service():
    return DocumentIngestionService()

def make_chunk(text, filename, chunk_id, section_name="General"):
    return Chunk(
        text=text,
        metadata=DocumentMetadata(
            filename=filename,
            chunk_id=chunk_id,
            source=f"{filename} - {section_name}",
            chunk_type="text"
        )
    )

def test_process_file_pdf_happy_path(service):
    fake_pdf_bytes = b"%PDF-1.4 fake pdf content"
    filename = "test.pdf"
    fake_text = "\n\n### Page 1\n\nCarrier Details: ABC\n\n### Page 2\n\nRate Breakdown: $1000"
    with patch.object(service, "_extract_text_from_pdf", return_value=fake_text) as mock_pdf, \
         patch.object(service, "_add_semantic_structure", wraps=service._add_semantic_structure) as mock_sem, \
         patch.object(service, "_chunk_text_with_title_grouping", wraps=service._chunk_text_with_title_grouping) as mock_chunk:
        chunks = service.process_file(fake_pdf_bytes, filename)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)
        assert mock_pdf.called
        assert mock_sem.called
        assert mock_chunk.called
        assert any("Carrier Details" in c.text or "Rate Breakdown" in c.text for c in chunks)

def test_process_file_docx_happy_path(service):
    filename = "test.docx"
    fake_docx_bytes = b"fake docx bytes"
    fake_text = "Carrier Details\nSome info\nRate Breakdown\n$1000"
    with patch("backend.src.services.ingestion.docx.Document") as mock_docx_doc, \
         patch.object(service, "_add_semantic_structure", wraps=service._add_semantic_structure), \
         patch.object(service, "_chunk_text_with_title_grouping", wraps=service._chunk_text_with_title_grouping):
        mock_doc = MagicMock()
        mock_doc.paragraphs = [MagicMock(text="Carrier Details"), MagicMock(text="Some info"),
                               MagicMock(text="Rate Breakdown"), MagicMock(text="$1000")]
        mock_docx_doc.return_value = mock_doc
        chunks = service.process_file(fake_docx_bytes, filename)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)
        assert any("Carrier Details" in c.text or "Rate Breakdown" in c.text for c in chunks)

def test_process_file_txt_happy_path(service):
    filename = "test.txt"
    content = "Carrier Details\nSome info\nRate Breakdown\n$1000"
    with patch.object(service, "_add_semantic_structure", wraps=service._add_semantic_structure), \
         patch.object(service, "_chunk_text_with_title_grouping", wraps=service._chunk_text_with_title_grouping):
        chunks = service.process_file(content.encode("utf-8"), filename)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)
        assert any("Carrier Details" in c.text or "Rate Breakdown" in c.text for c in chunks)

def test_process_file_unsupported_type_returns_empty(service):
    filename = "test.xlsx"
    content = b"fake"
    with patch("backend.src.services.ingestion.logger") as mock_logger:
        chunks = service.process_file(content, filename)
        assert chunks == []
        assert mock_logger.error.called
        assert "Unsupported file type" in str(mock_logger.error.call_args[0][0])

def test_process_file_pdf_extract_error_returns_empty(service):
    filename = "test.pdf"
    content = b"bad pdf"
    with patch.object(service, "_extract_text_from_pdf", side_effect=Exception("pdf error")), \
         patch("backend.src.services.ingestion.logger") as mock_logger:
        chunks = service.process_file(content, filename)
        assert chunks == []
        assert mock_logger.error.called
        assert "pdf error" in str(mock_logger.error.call_args[0][0])

def test_extract_text_from_pdf_calls_pymupdf_open(service):
    fake_pdf_bytes = b"%PDF-1.4 fake"
    with patch("backend.src.services.ingestion.pymupdf.open") as mock_open:
        mock_doc = [MagicMock(get_text=MagicMock(return_value="Page1Text")),
                    MagicMock(get_text=MagicMock(return_value="Page2Text"))]
        mock_open.return_value = mock_doc
        text = service._extract_text_from_pdf(fake_pdf_bytes)
        assert "### Page 1" in text
        assert "Page1Text" in text
        assert "### Page 2" in text
        assert "Page2Text" in text
        assert mock_open.called

def test_extract_text_from_docx_reads_paragraphs(service):
    fake_docx_bytes = b"fake"
    with patch("backend.src.services.ingestion.docx.Document") as mock_docx_doc:
        mock_doc = MagicMock()
        mock_doc.paragraphs = [MagicMock(text="Para1"), MagicMock(text="Para2")]
        mock_docx_doc.return_value = mock_doc
        text = DocumentIngestionService()._extract_text_from_docx(fake_docx_bytes)
        assert "Para1" in text and "Para2" in text

def test_add_semantic_structure_adds_headers_and_cleans_whitespace(service):
    text = "Carrier Details\n\n\nRate Breakdown\n\n\n\nPickup"
    result = service._add_semantic_structure(text)
    assert "\n## Carrier Details\n" in result
    assert "\n## Rate Breakdown\n" in result
    assert "\n## Pickup\n" in result
    assert "\n\n\n" not in result

def test_chunk_text_with_title_grouping_groups_sections_and_chunks(service):
    text = "\n## Carrier Details\nCarrier info\n## Rate Breakdown\n$1000\n## Pickup\nLocation"
    filename = "file.pdf"
    with patch.object(service.text_splitter, "split_text", side_effect=lambda t: [t]):
        chunks = service._chunk_text_with_title_grouping(text, filename)
        assert len(chunks) == 3
        assert all(isinstance(c, Chunk) for c in chunks)
        assert chunks[0].metadata.source == f"{filename} - Carrier Details"
        assert chunks[1].metadata.source == f"{filename} - Rate Breakdown"
        assert chunks[2].metadata.source == f"{filename} - Pickup"

def test_chunk_text_with_title_grouping_empty_section_skipped(service):
    text = "\n## Carrier Details\n\n## Rate Breakdown\n$1000"
    filename = "file.pdf"
    with patch.object(service.text_splitter, "split_text", side_effect=lambda t: [t]):
        chunks = service._chunk_text_with_title_grouping(text, filename)
        assert any("Rate Breakdown" in c.text for c in chunks)
        assert not any(c.text.strip() == "" for c in chunks)

def test_extract_section_name_with_header(service):
    text = "## Carrier Details\nSome info"
    section = DocumentIngestionService()._extract_section_name(text)
    assert section == "Carrier Details"

def test_extract_section_name_without_header(service):
    text = "Some info"
    section = DocumentIngestionService()._extract_section_name(text)
    assert section == "General"

def test_clean_chunk_removes_excess_whitespace_and_page_markers(service):
    text = "\n\n\n### Page 1\n\n\nSome text\n\n\n"
    cleaned = service._clean_chunk(text)
    assert "### Page 1" not in cleaned
    assert cleaned.startswith("Some text")
    assert "\n\n\n" not in cleaned

def test_clean_chunk_preserves_content_with_page_marker(service):
    text = "### Page 1\nContent here"
    cleaned = service._clean_chunk(text)
    assert "Content here" in cleaned

def test_process_file_empty_txt_returns_empty_chunk(service):
    filename = "empty.txt"
    content = b""
    with patch.object(service, "_add_semantic_structure", wraps=service._add_semantic_structure), \
         patch.object(service, "_chunk_text_with_title_grouping", wraps=service._chunk_text_with_title_grouping):
        chunks = service.process_file(content, filename)
        assert isinstance(chunks, list)
        # Should not produce any non-empty chunks
        assert all(c.text.strip() == "" for c in chunks) or len(chunks) == 0

def test_process_file_boundary_chunk_size(service):
    filename = "boundary.txt"
    # Simulate text just at the chunk size boundary
    from core.config import settings
    chunk_size = settings.CHUNK_SIZE * 2
    text = "A" * chunk_size
    with patch.object(service, "_add_semantic_structure", wraps=service._add_semantic_structure), \
         patch.object(service, "_chunk_text_with_title_grouping", wraps=service._chunk_text_with_title_grouping), \
         patch.object(service.text_splitter, "split_text", side_effect=lambda t: [t]):
        chunks = service.process_file(text.encode("utf-8"), filename)
        assert len(chunks) == 1
        assert chunks[0].text == text

def test_chunk_text_with_title_grouping_large_group_is_split(service):
    filename = "file.pdf"
    text = "\n## Carrier Details\n" + ("A" * 5000)
    # Simulate text_splitter splitting into two chunks
    with patch.object(service.text_splitter, "split_text", return_value=["A"*2500, "A"*2500]):
        chunks = service._chunk_text_with_title_grouping(text, filename)
        assert len(chunks) == 2
        assert all(len(c.text) == 2500 for c in chunks)
        assert all(c.metadata.source == f"{filename} - Carrier Details" for c in chunks)

def test_chunk_text_with_title_grouping_no_headers(service):
    filename = "file.pdf"
    text = "Just some text with no headers"
    with patch.object(service.text_splitter, "split_text", return_value=[text]):
        chunks = service._chunk_text_with_title_grouping(text, filename)
        assert len(chunks) == 1
        assert chunks[0].metadata.source == f"{filename} - General"
        assert "Just some text" in chunks[0].text

def test_add_semantic_structure_handles_no_section_headers(service):
    text = "This is a document with no known headers."
    result = DocumentIngestionService()._add_semantic_structure(text)
    assert result == text

def test_process_file_handles_unicode_txt(service):
    filename = "unicode.txt"
    content = "Carrier Details\nΔοκιμή\nRate Breakdown\n测试".encode("utf-8")
    with patch.object(service, "_add_semantic_structure", wraps=service._add_semantic_structure), \
         patch.object(service, "_chunk_text_with_title_grouping", wraps=service._chunk_text_with_title_grouping):
        chunks = service.process_file(content, filename)
        assert any("Δοκιμή" in c.text or "测试" in c.text for c in chunks)
