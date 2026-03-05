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

import io
import pytest
from unittest.mock import patch, MagicMock

from backend.src.services.ingestion import DocumentIngestionService, ingestion_service
from backend.src.services.ingestion import Chunk, DocumentMetadata

@pytest.fixture
def service():
    return DocumentIngestionService()

@pytest.fixture
def dummy_settings(monkeypatch):
    class DummySettings:
        CHUNK_SIZE = 100
        CHUNK_OVERLAP = 10
    monkeypatch.setattr("backend.src.services.ingestion.settings", DummySettings())

def test_process_file_pdf_happy_path(service, dummy_settings):
    fake_pdf_bytes = b"%PDF-1.4 fake pdf content"
    filename = "test.pdf"
    mock_doc = MagicMock()
    mock_page1 = MagicMock()
    mock_page1.get_text.return_value = "Carrier Details\nSome carrier info."
    mock_page2 = MagicMock()
    mock_page2.get_text.return_value = "Rate Breakdown\nSome rate info."
    mock_doc.__iter__.return_value = iter([mock_page1, mock_page2])
    with patch("backend.src.services.ingestion.pymupdf.open", return_value=mock_doc):
        chunks = service.process_file(fake_pdf_bytes, filename)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert any("Carrier Details" in chunk.text for chunk in chunks)
    assert any("Rate Breakdown" in chunk.text for chunk in chunks)
    assert all(chunk.metadata.filename == filename for chunk in chunks)
    assert all(chunk.metadata.chunk_type == "text" for chunk in chunks)

def test_process_file_docx_happy_path(service, dummy_settings):
    filename = "test.docx"
    fake_docx_bytes = b"Fake docx bytes"
    mock_doc = MagicMock()
    mock_doc.paragraphs = [MagicMock(text="Pickup\nSome pickup info."), MagicMock(text="Drop\nSome drop info.")]
    with patch("backend.src.services.ingestion.docx.Document", return_value=mock_doc):
        chunks = service.process_file(fake_docx_bytes, filename)
    assert isinstance(chunks, list)
    assert any("Pickup" in chunk.text for chunk in chunks)
    assert any("Drop" in chunk.text for chunk in chunks)
    assert all(chunk.metadata.filename == filename for chunk in chunks)

def test_process_file_txt_happy_path(service, dummy_settings):
    filename = "test.txt"
    content = "Standing Instructions\nDo not stack.\nSpecial Instructions\nHandle with care."
    chunks = service.process_file(content.encode("utf-8"), filename)
    assert isinstance(chunks, list)
    assert any("Standing Instructions" in chunk.text for chunk in chunks)
    assert any("Special Instructions" in chunk.text for chunk in chunks)
    assert all(chunk.metadata.filename == filename for chunk in chunks)

def test_process_file_unsupported_type_returns_empty(service, dummy_settings):
    filename = "test.xlsx"
    content = b"Fake excel bytes"
    chunks = service.process_file(content, filename)
    assert chunks == []

def test_process_file_pdf_extract_text_error_returns_empty(service, dummy_settings):
    filename = "test.pdf"
    fake_pdf_bytes = b"%PDF-1.4 fake pdf content"
    with patch("backend.src.services.ingestion.pymupdf.open", side_effect=Exception("PDF error")):
        chunks = service.process_file(fake_pdf_bytes, filename)
    assert chunks == []

def test_process_file_docx_extract_text_error_returns_empty(service, dummy_settings):
    filename = "test.docx"
    fake_docx_bytes = b"Fake docx bytes"
    with patch("backend.src.services.ingestion.docx.Document", side_effect=Exception("DOCX error")):
        chunks = service.process_file(fake_docx_bytes, filename)
    assert chunks == []

def test_process_file_txt_decoding_error_returns_empty(service, dummy_settings):
    filename = "test.txt"
    # Invalid utf-8 bytes
    content = b"\xff\xfe\xfd"
    chunks = service.process_file(content, filename)
    assert chunks == []

def test_add_semantic_structure_adds_headers(service):
    text = "Carrier Details\nSome info.\nRate Breakdown\nMore info."
    structured = service._add_semantic_structure(text)
    assert "## Carrier Details" in structured
    assert "## Rate Breakdown" in structured

def test_chunk_text_with_title_grouping_groups_sections(service, dummy_settings):
    text = "\n## Carrier Details\nCarrier: X\n## Driver Details\nDriver: Y\n## Rate Breakdown\n$1000"
    filename = "file.txt"
    chunks = service._chunk_text_with_title_grouping(text, filename)
    # Carrier Details and Driver Details should be grouped
    carrier_driver_chunks = [c for c in chunks if "Carrier Details" in c.text or "Driver Details" in c.text]
    assert carrier_driver_chunks
    # Rate Breakdown should be in its own chunk or group
    rate_chunks = [c for c in chunks if "Rate Breakdown" in c.text]
    assert rate_chunks

def test_chunk_text_with_title_grouping_handles_empty_sections(service, dummy_settings):
    text = "\n## Carrier Details\n\n## Rate Breakdown\n"
    filename = "file.txt"
    chunks = service._chunk_text_with_title_grouping(text, filename)
    assert all(isinstance(chunk, Chunk) for chunk in chunks)

def test_clean_chunk_removes_excess_whitespace_and_page_markers(service):
    chunk_text = "\n\n\n### Page 1\n\nSome text.\n\n\n"
    cleaned = service._clean_chunk(chunk_text)
    assert "Page 1" not in cleaned
    assert cleaned.startswith("Some text.")

def test_extract_section_name_returns_section(service):
    chunk_text = "## Pickup\nSome details"
    section = service._extract_section_name(chunk_text)
    assert section == "Pickup"

def test_extract_section_name_returns_general_if_no_header(service):
    chunk_text = "No header here"
    section = service._extract_section_name(chunk_text)
    assert section == "General"

def test_chunk_text_with_title_grouping_assigns_incrementing_chunk_ids(service, dummy_settings):
    text = "\n## Pickup\nA\n## Drop\nB"
    filename = "file.txt"
    chunks = service._chunk_text_with_title_grouping(text, filename)
    chunk_ids = [c.metadata.chunk_id for c in chunks]
    assert chunk_ids == list(range(len(chunks)))

def test_process_file_pdf_with_multiple_pages_and_section_headers(service, dummy_settings):
    fake_pdf_bytes = b"%PDF-1.4 fake pdf content"
    filename = "multi.pdf"
    mock_doc = MagicMock()
    mock_page1 = MagicMock()
    mock_page1.get_text.return_value = "Carrier Details\nCarrier X"
    mock_page2 = MagicMock()
    mock_page2.get_text.return_value = "Rate Breakdown\n$2000"
    mock_doc.__iter__.return_value = iter([mock_page1, mock_page2])
    with patch("backend.src.services.ingestion.pymupdf.open", return_value=mock_doc):
        chunks = service.process_file(fake_pdf_bytes, filename)
    assert any("Carrier Details" in c.text for c in chunks)
    assert any("Rate Breakdown" in c.text for c in chunks)
    assert all(c.metadata.filename == filename for c in chunks)

def test_process_file_docx_with_multiple_sections(service, dummy_settings):
    filename = "sections.docx"
    fake_docx_bytes = b"Fake docx bytes"
    mock_doc = MagicMock()
    mock_doc.paragraphs = [
        MagicMock(text="Pickup\nLocation A"),
        MagicMock(text="Drop\nLocation B"),
        MagicMock(text="Stops\nStop 1"),
    ]
    with patch("backend.src.services.ingestion.docx.Document", return_value=mock_doc):
        chunks = service.process_file(fake_docx_bytes, filename)
    assert any("Pickup" in c.text for c in chunks)
    assert any("Drop" in c.text for c in chunks)
    assert any("Stops" in c.text for c in chunks)

def test_process_file_txt_with_no_section_headers(service, dummy_settings):
    filename = "plain.txt"
    content = "Just some plain text with no headers."
    chunks = service.process_file(content.encode("utf-8"), filename)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.metadata.filename == filename for chunk in chunks)

def test_process_file_empty_txt(service, dummy_settings):
    filename = "empty.txt"
    content = ""
    chunks = service.process_file(content.encode("utf-8"), filename)
    assert chunks == []

def test_process_file_txt_with_boundary_chunk_size(service, dummy_settings):
    filename = "boundary.txt"
    # Create text just at the chunk size boundary
    content = "A" * (service.text_splitter.chunk_size)
    chunks = service.process_file(content.encode("utf-8"), filename)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(len(chunk.text) <= service.text_splitter.chunk_size for chunk in chunks)
