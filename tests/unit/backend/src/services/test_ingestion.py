import pytest
import re
import logging
from unittest.mock import patch, MagicMock, call
from backend.src.services.ingestion import DocumentIngestionService
from backend.src.models.schemas import Chunk, DocumentMetadata

@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    class DummySettings:
        CHUNK_SIZE = 100
        CHUNK_OVERLAP = 10
    monkeypatch.setattr("backend.src.core.config.settings", DummySettings())

@pytest.fixture
def ingestion_service():
    return DocumentIngestionService()

def test_process_file_happy_path_pdf(ingestion_service):
    fake_pdf_bytes = b"%PDF-1.4 fake pdf content"
    filename = "test.pdf"
    fake_text = "\n\n### Page 1\n\nCarrier Details\nSome content\n"
    with patch.object(ingestion_service, "_extract_text_from_pdf", return_value=fake_text) as mock_pdf, \
         patch.object(ingestion_service, "_chunk_text_with_title_grouping", return_value=[Chunk(text="chunk1", metadata=DocumentMetadata(filename=filename, chunk_id=0, source=f"{filename} - Carrier Details", chunk_type="text"))]) as mock_chunk:
        chunks = ingestion_service.process_file(fake_pdf_bytes, filename)
        assert len(chunks) == 1
        assert chunks[0].text == "chunk1"
        assert chunks[0].metadata.filename == filename
        mock_pdf.assert_called_once_with(fake_pdf_bytes)
        mock_chunk.assert_called_once()
        
def test_process_file_happy_path_docx(ingestion_service):
    fake_docx_bytes = b"PK\x03\x04 fake docx content"
    filename = "test.docx"
    fake_text = "Carrier Details\nSome content\n"
    with patch.object(ingestion_service, "_extract_text_from_docx", return_value=fake_text) as mock_docx, \
         patch.object(ingestion_service, "_chunk_text_with_title_grouping", return_value=[Chunk(text="chunk2", metadata=DocumentMetadata(filename=filename, chunk_id=0, source=f"{filename} - Carrier Details", chunk_type="text"))]) as mock_chunk:
        chunks = ingestion_service.process_file(fake_docx_bytes, filename)
        assert len(chunks) == 1
        assert chunks[0].text == "chunk2"
        assert chunks[0].metadata.filename == filename
        mock_docx.assert_called_once_with(fake_docx_bytes)
        mock_chunk.assert_called_once()

def test_process_file_happy_path_txt(ingestion_service):
    fake_txt_bytes = b"Carrier Details\nSome content\n"
    filename = "test.txt"
    with patch.object(ingestion_service, "_chunk_text_with_title_grouping", return_value=[Chunk(text="chunk3", metadata=DocumentMetadata(filename=filename, chunk_id=0, source=f"{filename} - Carrier Details", chunk_type="text"))]) as mock_chunk:
        chunks = ingestion_service.process_file(fake_txt_bytes, filename)
        assert len(chunks) == 1
        assert chunks[0].text == "chunk3"
        assert chunks[0].metadata.filename == filename
        mock_chunk.assert_called_once()

def test_process_file_unsupported_filetype(ingestion_service):
    fake_bytes = b"random"
    filename = "test.csv"
    with patch("backend.src.services.ingestion.logger") as mock_logger:
        chunks = ingestion_service.process_file(fake_bytes, filename)
        assert chunks == []
        assert mock_logger.error.call_count == 1
        assert "Unsupported file type" in mock_logger.error.call_args[0][0]

def test_process_file_exception_in_extraction(ingestion_service):
    filename = "test.pdf"
    with patch.object(ingestion_service, "_extract_text_from_pdf", side_effect=RuntimeError("fail")), \
         patch("backend.src.services.ingestion.logger") as mock_logger:
        chunks = ingestion_service.process_file(b"irrelevant", filename)
        assert chunks == []
        assert mock_logger.error.call_count == 1
        assert "Error processing file" in mock_logger.error.call_args[0][0]

def test_extract_text_from_pdf_multiple_pages(monkeypatch, ingestion_service):
    fake_page1 = MagicMock()
    fake_page1.get_text.return_value = "Carrier Details\nPage1 content"
    fake_page2 = MagicMock()
    fake_page2.get_text.return_value = "Rate Breakdown\nPage2 content"
    fake_doc = [fake_page1, fake_page2]
    fake_open = MagicMock(return_value=fake_doc)
    monkeypatch.setattr("pymupdf.open", fake_open)
    result = ingestion_service._extract_text_from_pdf(b"fake")
    assert "### Page 1" in result
    assert "Carrier Details" in result
    assert "### Page 2" in result
    assert "Rate Breakdown" in result

def test_extract_text_from_docx(monkeypatch, ingestion_service):
    fake_doc = MagicMock()
    fake_doc.paragraphs = [MagicMock(text="Carrier Details"), MagicMock(text="Some content")]
    fake_docx = MagicMock(return_value=fake_doc)
    monkeypatch.setattr("docx.Document", fake_docx)
    from io import BytesIO
    result = ingestion_service._extract_text_from_docx(b"fake")
    assert "Carrier Details" in result
    assert "Some content" in result
    assert "\n" in result

def test_add_semantic_structure_adds_headers_and_cleans_whitespace(ingestion_service):
    text = "Carrier Details\n\n\n\nRate Breakdown\n\n\nPickup\n"
    result = ingestion_service._add_semantic_structure(text)
    # Should add markdown headers and reduce whitespace
    assert result.count("## Carrier Details") == 1
    assert result.count("## Rate Breakdown") == 1
    assert result.count("## Pickup") == 1
    assert "\n\n\n" not in result

def test_add_semantic_structure_no_section_headers(ingestion_service):
    text = "Random text with no headers"
    result = ingestion_service._add_semantic_structure(text)
    assert result == "Random text with no headers"

def test_chunk_text_with_title_grouping_groups_sections_and_chunks(monkeypatch, ingestion_service):
    # Patch text_splitter to split into two chunks per group
    fake_splitter = MagicMock()
    fake_splitter.split_text.side_effect = lambda group: [group[:10], group[10:]] if len(group) > 10 else [group]
    ingestion_service.text_splitter = fake_splitter
    text = "\n## Carrier Details\nCarrier content\n## Rate Breakdown\nRate content"
    filename = "file.pdf"
    chunks = ingestion_service._chunk_text_with_title_grouping(text, filename)
    # Should produce at least two chunks, one for each section
    assert any("Carrier" in c.text for c in chunks)
    assert any("Rate" in c.text for c in chunks)
    # Metadata should be correct
    for idx, chunk in enumerate(chunks):
        assert chunk.metadata.filename == filename
        assert chunk.metadata.chunk_id == idx
        assert chunk.metadata.chunk_type == "text"
        assert filename in chunk.metadata.source

def test_chunk_text_with_title_grouping_handles_empty_and_general(monkeypatch, ingestion_service):
    fake_splitter = MagicMock()
    fake_splitter.split_text.return_value = ["General content"]
    ingestion_service.text_splitter = fake_splitter
    text = "Some content without headers"
    filename = "file.txt"
    chunks = ingestion_service._chunk_text_with_title_grouping(text, filename)
    assert len(chunks) == 1
    assert chunks[0].metadata.filename == filename
    assert chunks[0].metadata.source == f"{filename} - General"

def test_extract_section_name_with_header(ingestion_service):
    chunk_text = "## Carrier Details\nSome content"
    section = ingestion_service._extract_section_name(chunk_text)
    assert section == "Carrier Details"

def test_extract_section_name_without_header(ingestion_service):
    chunk_text = "No header here"
    section = ingestion_service._extract_section_name(chunk_text)
    assert section == "General"

def test_clean_chunk_removes_excess_whitespace_and_page_markers(ingestion_service):
    chunk_text = "\n\n\n### Page 1\n\n\nSome content\n\n\n"
    cleaned = ingestion_service._clean_chunk(chunk_text)
    assert "### Page 1" not in cleaned
    assert cleaned.startswith("Some content")
    assert "\n\n\n" not in cleaned

def test_clean_chunk_preserves_content_and_structure(ingestion_service):
    chunk_text = "Carrier Details\n\n  Some   content   here\n\n"
    cleaned = ingestion_service._clean_chunk(chunk_text)
    assert "Carrier Details" in cleaned
    assert "  " not in cleaned
    assert cleaned.endswith("here")
