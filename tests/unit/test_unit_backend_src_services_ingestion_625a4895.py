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

from backend.src.services.ingestion import DocumentIngestionService, ingestion_service

import io

@pytest.fixture
def service():
    return DocumentIngestionService()

def test_process_file_pdf_happy_path(service):
    fake_pdf_bytes = b"%PDF-1.4 fake content"
    filename = "test.pdf"

    fake_page = MagicMock()
    fake_page.get_text.return_value = "Carrier Details\nSome carrier info.\nRate Breakdown\n$1000"
    fake_doc = [fake_page, fake_page]

    with patch("backend.src.services.ingestion.pymupdf.open", return_value=fake_doc):
        with patch.object(service, "_add_semantic_structure", wraps=service._add_semantic_structure) as add_semantic:
            with patch.object(service, "_chunk_text_with_title_grouping", wraps=service._chunk_text_with_title_grouping) as chunk_group:
                chunks = service.process_file(fake_pdf_bytes, filename)
                assert isinstance(chunks, list)
                assert all(hasattr(chunk, "text") and hasattr(chunk, "metadata") for chunk in chunks)
                assert add_semantic.called
                assert chunk_group.called
                assert len(chunks) > 0
                for chunk in chunks:
                    assert filename in chunk.metadata.filename

def test_process_file_docx_happy_path(service):
    fake_docx_bytes = b"fake docx"
    filename = "test.docx"
    fake_doc = MagicMock()
    fake_doc.paragraphs = [MagicMock(text="Carrier Details"), MagicMock(text="Some carrier info."), MagicMock(text="Rate Breakdown"), MagicMock(text="$1000")]

    with patch("backend.src.services.ingestion.docx.Document", return_value=fake_doc):
        with patch.object(service, "_add_semantic_structure", wraps=service._add_semantic_structure):
            with patch.object(service, "_chunk_text_with_title_grouping", wraps=service._chunk_text_with_title_grouping):
                chunks = service.process_file(fake_docx_bytes, filename)
                assert isinstance(chunks, list)
                assert len(chunks) > 0
                for chunk in chunks:
                    assert filename in chunk.metadata.filename

def test_process_file_txt_happy_path(service):
    content = "Carrier Details\nSome carrier info.\nRate Breakdown\n$1000"
    filename = "test.txt"
    with patch.object(service, "_add_semantic_structure", wraps=service._add_semantic_structure):
        with patch.object(service, "_chunk_text_with_title_grouping", wraps=service._chunk_text_with_title_grouping):
            chunks = service.process_file(content.encode("utf-8"), filename)
            assert isinstance(chunks, list)
            assert len(chunks) > 0
            for chunk in chunks:
                assert filename in chunk.metadata.filename

def test_process_file_unsupported_type_returns_empty(service):
    filename = "test.xlsx"
    result = service.process_file(b"irrelevant", filename)
    assert result == []

def test_process_file_pdf_extract_text_error_returns_empty(service):
    filename = "test.pdf"
    with patch("backend.src.services.ingestion.pymupdf.open", side_effect=Exception("fail")):
        result = service.process_file(b"irrelevant", filename)
        assert result == []

def test_extract_text_from_pdf_adds_page_markers(service):
    fake_pdf_bytes = b"%PDF-1.4 fake content"
    fake_page1 = MagicMock()
    fake_page1.get_text.return_value = "Carrier Details"
    fake_page2 = MagicMock()
    fake_page2.get_text.return_value = "Rate Breakdown"
    fake_doc = [fake_page1, fake_page2]
    with patch("backend.src.services.ingestion.pymupdf.open", return_value=fake_doc):
        text = service._extract_text_from_pdf(fake_pdf_bytes)
        assert "### Page 1" in text
        assert "### Page 2" in text
        assert "Carrier Details" in text
        assert "Rate Breakdown" in text

def test_extract_text_from_docx_reads_paragraphs(service):
    fake_docx_bytes = b"fake docx"
    fake_doc = MagicMock()
    fake_doc.paragraphs = [MagicMock(text="Carrier Details"), MagicMock(text="Rate Breakdown")]
    with patch("backend.src.services.ingestion.docx.Document", return_value=fake_doc):
        text = service._extract_text_from_docx(fake_docx_bytes)
        assert "Carrier Details" in text
        assert "Rate Breakdown" in text

def test_add_semantic_structure_adds_headers_and_cleans_whitespace(service):
    text = "Carrier Details\n\n\nSome info\n\n\nRate Breakdown\n\n$1000"
    result = service._add_semantic_structure(text)
    assert "\n## Carrier Details\n" in result
    assert "\n## Rate Breakdown\n" in result
    assert "\n\n" in result
    assert "Some info" in result
    assert "$1000" in result

def test_chunk_text_with_title_grouping_groups_sections_and_chunks(service):
    text = "\n## Carrier Details\nCarrier info\n## Rate Breakdown\n$1000\n## Pickup\nLocation"
    filename = "test.txt"
    # Patch text_splitter to split only on "\n## " for deterministic output
    with patch.object(service, "text_splitter") as mock_splitter:
        mock_splitter.split_text.side_effect = lambda x: [x]
        chunks = service._chunk_text_with_title_grouping(text, filename)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert hasattr(chunk, "text")
            assert hasattr(chunk, "metadata")
            assert filename in chunk.metadata.filename

def test_chunk_text_with_title_grouping_handles_empty_sections(service):
    text = ""
    filename = "test.txt"
    with patch.object(service, "text_splitter") as mock_splitter:
        mock_splitter.split_text.return_value = []
        chunks = service._chunk_text_with_title_grouping(text, filename)
        assert chunks == []

def test_extract_section_name_returns_section(service):
    chunk_text = "## Carrier Details\nSome info"
    section = DocumentIngestionService()._extract_section_name(chunk_text)
    assert section == "Carrier Details"

def test_extract_section_name_returns_general_when_no_header(service):
    chunk_text = "No header here"
    section = service._extract_section_name(chunk_text)
    assert section == "General"

def test_clean_chunk_removes_excess_whitespace_and_page_markers(service):
    chunk_text = "\n\n\n### Page 1\n\n\nSome info\n\n\n"
    cleaned = service._clean_chunk(chunk_text)
    assert "### Page 1" not in cleaned
    assert cleaned.startswith("Some info")

def test_clean_chunk_preserves_content(service):
    chunk_text = "Carrier Details\n\nSome info"
    cleaned = service._clean_chunk(chunk_text)
    assert "Carrier Details" in cleaned
    assert "Some info" in cleaned

def test_ingestion_service_singleton_is_instance():
    from backend.src.services import ingestion
    assert isinstance(ingestion.ingestion_service, DocumentIngestionService)
