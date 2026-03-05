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

import re

from backend.src.services.ingestion import DocumentIngestionService, ingestion_service

@pytest.fixture
def service():
    return DocumentIngestionService()

def test_process_file_pdf_happy_path(service):
    fake_pdf_bytes = b"%PDF-1.4 fake pdf content"
    filename = "test.pdf"
    fake_text = "\n\n### Page 1\n\nCarrier Details\nSome content\n"
    fake_chunks = [MagicMock()]
    with patch.object(service, "_extract_text_from_pdf", return_value=fake_text) as mock_pdf, \
         patch.object(service, "_add_semantic_structure", side_effect=lambda x: x) as mock_sem, \
         patch.object(service, "_chunk_text_with_title_grouping", return_value=fake_chunks) as mock_chunk:
        result = service.process_file(fake_pdf_bytes, filename)
        mock_pdf.assert_called_once_with(fake_pdf_bytes)
        mock_sem.assert_called_once_with(fake_text)
        mock_chunk.assert_called_once()
        assert result == fake_chunks

def test_process_file_docx_happy_path(service):
    fake_docx_bytes = b"PK\x03\x04 fake docx content"
    filename = "test.docx"
    fake_text = "Carrier Details\nSome content\n"
    fake_chunks = [MagicMock()]
    with patch.object(service, "_extract_text_from_docx", return_value=fake_text) as mock_docx, \
         patch.object(service, "_add_semantic_structure", side_effect=lambda x: x) as mock_sem, \
         patch.object(service, "_chunk_text_with_title_grouping", return_value=fake_chunks) as mock_chunk:
        result = service.process_file(fake_docx_bytes, filename)
        mock_docx.assert_called_once_with(fake_docx_bytes)
        mock_sem.assert_called_once_with(fake_text)
        mock_chunk.assert_called_once()
        assert result == fake_chunks

def test_process_file_txt_happy_path(service):
    fake_txt_bytes = b"Carrier Details\nSome content\n"
    filename = "test.txt"
    fake_text = fake_txt_bytes.decode("utf-8")
    fake_chunks = [MagicMock()]
    with patch.object(service, "_add_semantic_structure", side_effect=lambda x: x) as mock_sem, \
         patch.object(service, "_chunk_text_with_title_grouping", return_value=fake_chunks) as mock_chunk:
        result = service.process_file(fake_txt_bytes, filename)
        mock_sem.assert_called_once_with(fake_text)
        mock_chunk.assert_called_once()
        assert result == fake_chunks

def test_process_file_unsupported_extension_returns_empty(service):
    result = service.process_file(b"irrelevant", "file.unsupported")
    assert result == []

def test_process_file_error_in_extraction_returns_empty(service):
    with patch.object(service, "_extract_text_from_pdf", side_effect=Exception("fail")):
        result = service.process_file(b"irrelevant", "file.pdf")
        assert result == []

def test_extract_text_from_pdf_adds_page_markers(service):
    fake_page1 = MagicMock()
    fake_page1.get_text.return_value = "Carrier Details\n"
    fake_page2 = MagicMock()
    fake_page2.get_text.return_value = "Rate Breakdown\n"
    fake_doc = [fake_page1, fake_page2]
    with patch("backend.src.services.ingestion.pymupdf.open", return_value=fake_doc) as mock_open:
        result = service._extract_text_from_pdf(b"fake",)
        assert "### Page 1" in result
        assert "Carrier Details" in result
        assert "### Page 2" in result
        assert "Rate Breakdown" in result
        mock_open.assert_called_once()

def test_extract_text_from_docx_reads_paragraphs(service):
    fake_docx_bytes = b"fake"
    fake_doc = MagicMock()
    fake_doc.paragraphs = [MagicMock(text="Carrier Details"), MagicMock(text="Some content")]
    with patch("backend.src.services.ingestion.docx.Document", return_value=fake_doc):
        result = service._extract_text_from_docx(fake_docx_bytes)
        assert "Carrier Details" in result
        assert "Some content" in result

def test_add_semantic_structure_adds_headers_and_cleans_whitespace(service):
    text = "Carrier Details\n\n\nRate Breakdown\n\n\n\nPickup"
    result = DocumentIngestionService()._add_semantic_structure(text)
    assert "\n## Carrier Details\n" in result
    assert "\n## Rate Breakdown\n" in result
    assert "\n## Pickup\n" in result
    assert "\n\n" in result
    assert "\n\n\n" not in result

def test_chunk_text_with_title_grouping_groups_and_chunks(service):
    # Simulate a text with two sections, both matching section_groups
    text = "\n## Carrier Details\nCarrier X\n## Driver Details\nDriver Y\n## Rate Breakdown\n$1000\n"
    filename = "test.txt"
    # Patch text_splitter to split into one chunk per group
    fake_splitter = MagicMock()
    fake_splitter.split_text.side_effect = lambda x: [x]
    service = DocumentIngestionService()
    service.text_splitter = fake_splitter
    # Patch DocumentMetadata and Chunk to simple dicts for test
    with patch("backend.src.services.ingestion.DocumentMetadata", side_effect=lambda **kwargs: kwargs), \
         patch("backend.src.services.ingestion.Chunk", side_effect=lambda text, metadata: {'text': text, 'metadata': metadata}):
        chunks = service._chunk_text_with_title_grouping(text, filename)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert chunk["metadata"]["filename"] == filename

def test_chunk_text_with_title_grouping_handles_empty_and_general(service):
    text = ""
    filename = "empty.txt"
    service = DocumentIngestionService()
    service.text_splitter = MagicMock()
    service.text_splitter.split_text.return_value = []
    with patch("backend.src.services.ingestion.DocumentMetadata", side_effect=lambda **kwargs: kwargs), \
         patch("backend.src.services.ingestion.Chunk", side_effect=lambda text, metadata: {'text': text, 'metadata': metadata}):
        chunks = service._chunk_text_with_title_grouping(text, filename)
        assert chunks == []

def test_extract_section_name_with_header(service):
    chunk_text = "## Carrier Details\nSome content"
    section = service._extract_section_name(chunk_text)
    assert section == "Carrier Details"

def test_extract_section_name_without_header(service):
    chunk_text = "Some content"
    section = service._extract_section_name(chunk_text)
    assert section == "General"

def test_clean_chunk_removes_excessive_whitespace_and_page_markers(service):
    chunk_text = "\n\n\n### Page 1\n\n\nSome text\n\n\n"
    cleaned = service._clean_chunk(chunk_text)
    assert cleaned.startswith("Some text") or "Some text" in cleaned
    assert "### Page 1" not in cleaned

def test_clean_chunk_preserves_content_within_page_marker(service):
    chunk_text = "### Page 1\nSome text"
    cleaned = service._clean_chunk(chunk_text)
    assert "Some text" in cleaned

def test_ingestion_service_singleton_is_instance():
    from backend.src.services.ingestion import ingestion_service, DocumentIngestionService
    assert isinstance(ingestion_service, DocumentIngestionService)
