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
    monkeypatch.setattr("backend.src.services.ingestion.settings", DummySettings())

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

def test_process_file_happy_path_pdf(service, dummy_settings):
    fake_pdf_bytes = b"%PDF-1.4 fake content"
    filename = "test.pdf"
    fake_text = "\n\n### Page 1\n\nCarrier Details\nSome info\n"
    with patch("backend.src.services.ingestion.DocumentIngestionService._extract_text_from_pdf", return_value=fake_text) as m_pdf, \
         patch("backend.src.services.ingestion.DocumentIngestionService._chunk_text_with_title_grouping") as m_chunk:
        m_chunk.return_value = [make_chunk("chunk1", filename, 0, "Carrier Details")]
        result = service.process_file(fake_pdf_bytes, filename)
        assert len(result) == 1
        assert result[0].text == "chunk1"
        assert result[0].metadata.filename == filename
        m_pdf.assert_called_once_with(fake_pdf_bytes)
        m_chunk.assert_called()
        
def test_process_file_happy_path_docx(service, dummy_settings):
    fake_docx_bytes = b"PK\x03\x04 fake docx"
    filename = "test.docx"
    fake_text = "Carrier Details\nSome info\n"
    with patch("backend.src.services.ingestion.DocumentIngestionService._extract_text_from_docx", return_value=fake_text) as m_docx, \
         patch("backend.src.services.ingestion.DocumentIngestionService._chunk_text_with_title_grouping") as m_chunk:
        m_chunk.return_value = [make_chunk("chunk2", filename, 0, "Carrier Details")]
        result = service.process_file(fake_docx_bytes, filename)
        assert len(result) == 1
        assert result[0].text == "chunk2"
        assert result[0].metadata.filename == filename
        m_docx.assert_called_once_with(fake_docx_bytes)
        m_chunk.assert_called()

def test_process_file_happy_path_txt(service, dummy_settings):
    fake_txt_bytes = b"Carrier Details\nSome info\n"
    filename = "test.txt"
    with patch("backend.src.services.ingestion.DocumentIngestionService._chunk_text_with_title_grouping") as m_chunk:
        m_chunk.return_value = [make_chunk("chunk3", filename, 0, "Carrier Details")]
        result = service.process_file(fake_txt_bytes, filename)
        assert len(result) == 1
        assert result[0].text == "chunk3"
        assert result[0].metadata.filename == filename
        m_chunk.assert_called()

def test_process_file_unsupported_extension_returns_empty(service, dummy_settings):
    fake_bytes = b"data"
    filename = "test.xlsx"
    result = service.process_file(fake_bytes, filename)
    assert result == []

def test_process_file_error_in_extraction_returns_empty(service, dummy_settings):
    filename = "test.pdf"
    with patch("backend.src.services.ingestion.DocumentIngestionService._extract_text_from_pdf", side_effect=Exception("fail")):
        result = service.process_file(b"bad", filename)
        assert result == []

def test_add_semantic_structure_adds_headers_and_cleans(service):
    text = "Carrier Details\n\n\nSome info\n\n\nRate Breakdown\n\n"
    structured = service._add_semantic_structure(text)
    assert "## Carrier Details" in structured
    assert "## Rate Breakdown" in structured
    assert "\n\n\n" not in structured

def test_chunk_text_with_title_grouping_groups_sections(service, dummy_settings):
    text = "\n## Carrier Details\nCarrier info\n## Driver Details\nDriver info\n## Rate Breakdown\nRate info\n"
    filename = "file.txt"
    # Patch text_splitter to return the group as a single chunk
    with patch.object(service, "text_splitter") as m_splitter:
        m_splitter.split_text.side_effect = lambda x: [x]
        chunks = service._chunk_text_with_title_grouping(text, filename)
        assert len(chunks) == 2  # carrier_info group, rate_info group
        assert "Carrier info" in chunks[0].text
        assert "Driver info" in chunks[0].text
        assert "Rate info" in chunks[1].text
        assert chunks[0].metadata.source == f"{filename} - Carrier Details"
        assert chunks[1].metadata.source == f"{filename} - Rate Breakdown"

def test_chunk_text_with_title_grouping_handles_no_headers(service, dummy_settings):
    text = "Just some text without headers."
    filename = "plain.txt"
    with patch.object(service, "text_splitter") as m_splitter:
        m_splitter.split_text.side_effect = lambda x: [x]
        chunks = service._chunk_text_with_title_grouping(text, filename)
        assert len(chunks) == 1
        assert "Just some text" in chunks[0].text
        assert chunks[0].metadata.source == f"{filename} - General"

def test_chunk_text_with_title_grouping_empty_text(service, dummy_settings):
    text = ""
    filename = "empty.txt"
    with patch.object(service, "text_splitter") as m_splitter:
        m_splitter.split_text.side_effect = lambda x: [x]
        chunks = service._chunk_text_with_title_grouping(text, filename)
        assert chunks == []

def test_extract_section_name_finds_header(service):
    chunk = "## Carrier Details\nSome info"
    section = service._extract_section_name(chunk)
    assert section == "Carrier Details"

def test_extract_section_name_returns_general_if_no_header(service):
    chunk = "No header here"
    section = service._extract_section_name(chunk)
    assert section == "General"

def test_clean_chunk_removes_excess_whitespace_and_page_markers(service):
    chunk = "\n\n\n### Page 1\n\nSome text\n\n\n"
    cleaned = service._clean_chunk(chunk)
    assert "Page 1" not in cleaned
    assert cleaned.startswith("Some text")
    assert "\n\n\n" not in cleaned

def test_pdf_and_docx_extraction_paths_equivalent_for_same_text(service, dummy_settings):
    # Reconciliation: If both extract same text, chunking should be equivalent
    filename_pdf = "file.pdf"
    filename_docx = "file.docx"
    fake_text = "\n## Carrier Details\nCarrier info\n"
    with patch("backend.src.services.ingestion.DocumentIngestionService._extract_text_from_pdf", return_value=fake_text), \
         patch("backend.src.services.ingestion.DocumentIngestionService._extract_text_from_docx", return_value=fake_text), \
         patch.object(service, "text_splitter") as m_splitter:
        m_splitter.split_text.side_effect = lambda x: [x]
        chunks_pdf = service.process_file(b"pdf", filename_pdf)
        chunks_docx = service.process_file(b"docx", filename_docx)
        # The chunk text should be the same except for filename in metadata
        assert len(chunks_pdf) == len(chunks_docx)
        for c_pdf, c_docx in zip(chunks_pdf, chunks_docx):
            assert c_pdf.text == c_docx.text
            assert c_pdf.metadata.chunk_id == c_docx.metadata.chunk_id
            assert c_pdf.metadata.source.replace(filename_pdf, "") == c_docx.metadata.source.replace(filename_docx, "")

def test_reconciliation_txt_and_docx_equivalent_when_content_same(service, dummy_settings):
    filename_txt = "file.txt"
    filename_docx = "file.docx"
    fake_text = "\n## Rate Breakdown\nRate info\n"
    with patch("backend.src.services.ingestion.DocumentIngestionService._extract_text_from_docx", return_value=fake_text), \
         patch.object(service, "text_splitter") as m_splitter:
        m_splitter.split_text.side_effect = lambda x: [x]
        # TXT path
        chunks_txt = service.process_file(fake_text.encode("utf-8"), filename_txt)
        # DOCX path
        chunks_docx = service.process_file(b"docx", filename_docx)
        assert len(chunks_txt) == len(chunks_docx)
        for c_txt, c_docx in zip(chunks_txt, chunks_docx):
            assert c_txt.text == c_docx.text
            assert c_txt.metadata.chunk_id == c_docx.metadata.chunk_id
            assert c_txt.metadata.source.replace(filename_txt, "") == c_docx.metadata.source.replace(filename_docx, "")

def test_chunk_text_with_title_grouping_boundary_conditions(service, dummy_settings):
    # Section at start and end, and empty section
    text = "## Carrier Details\nA\n## EmptySection\n\n## Rate Breakdown\nB"
    filename = "file.txt"
    with patch.object(service, "text_splitter") as m_splitter:
        m_splitter.split_text.side_effect = lambda x: [x]
        chunks = service._chunk_text_with_title_grouping(text, filename)
        assert len(chunks) == 3
        assert chunks[0].text.startswith("## Carrier Details")
        assert chunks[1].text.startswith("## EmptySection")
        assert chunks[2].text.startswith("## Rate Breakdown")
        assert chunks[1].text.strip() == "## EmptySection"

def test_process_file_handles_unicode_txt(service, dummy_settings):
    unicode_text = "Carrier Details\nÜñîçødë info\n"
    filename = "unicode.txt"
    with patch("backend.src.services.ingestion.DocumentIngestionService._chunk_text_with_title_grouping") as m_chunk:
        m_chunk.return_value = [make_chunk("unicode chunk", filename, 0, "Carrier Details")]
        result = service.process_file(unicode_text.encode("utf-8"), filename)
        assert len(result) == 1
        assert result[0].text == "unicode chunk"
        assert result[0].metadata.filename == filename

def test_process_file_handles_empty_txt(service, dummy_settings):
    filename = "empty.txt"
    with patch("backend.src.services.ingestion.DocumentIngestionService._chunk_text_with_title_grouping") as m_chunk:
        m_chunk.return_value = []
        result = service.process_file(b"", filename)
        assert result == []
