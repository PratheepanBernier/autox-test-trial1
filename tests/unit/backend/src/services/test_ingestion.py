import pytest
from unittest.mock import patch, MagicMock, call
from backend.src.services.ingestion import DocumentIngestionService
from models.schemas import Chunk, DocumentMetadata

class DummySettings:
    CHUNK_SIZE = 100
    CHUNK_OVERLAP = 10

@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    monkeypatch.setattr("core.config.settings", DummySettings())

@pytest.fixture
def ingestion_service():
    return DocumentIngestionService()

def test_process_file_pdf_happy_path(ingestion_service):
    dummy_pdf_bytes = b"%PDF-1.4 dummy"
    dummy_filename = "test.pdf"
    dummy_text = "\n\n### Page 1\n\nCarrier Details\nSome content\n"
    with patch.object(ingestion_service, "_extract_text_from_pdf", return_value=dummy_text) as mock_pdf, \
         patch.object(ingestion_service, "_add_semantic_structure", side_effect=lambda x: x) as mock_sem, \
         patch.object(ingestion_service, "_chunk_text_with_title_grouping", return_value=[Chunk(text="chunk1", metadata=DocumentMetadata(filename="test.pdf", chunk_id=0, source="test.pdf - Carrier Details", chunk_type="text"))]) as mock_chunk:
        result = ingestion_service.process_file(dummy_pdf_bytes, dummy_filename)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].text == "chunk1"
        mock_pdf.assert_called_once_with(dummy_pdf_bytes)
        mock_sem.assert_called_once()
        mock_chunk.assert_called_once()

def test_process_file_docx_happy_path(ingestion_service):
    dummy_docx_bytes = b"PK\x03\x04 dummy"
    dummy_filename = "test.docx"
    dummy_text = "Carrier Details\nSome content\n"
    with patch.object(ingestion_service, "_extract_text_from_docx", return_value=dummy_text) as mock_docx, \
         patch.object(ingestion_service, "_add_semantic_structure", side_effect=lambda x: x) as mock_sem, \
         patch.object(ingestion_service, "_chunk_text_with_title_grouping", return_value=[Chunk(text="chunk2", metadata=DocumentMetadata(filename="test.docx", chunk_id=0, source="test.docx - Carrier Details", chunk_type="text"))]) as mock_chunk:
        result = ingestion_service.process_file(dummy_docx_bytes, dummy_filename)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].text == "chunk2"
        mock_docx.assert_called_once_with(dummy_docx_bytes)
        mock_sem.assert_called_once()
        mock_chunk.assert_called_once()

def test_process_file_txt_happy_path(ingestion_service):
    dummy_txt_bytes = b"Carrier Details\nSome content\n"
    dummy_filename = "test.txt"
    with patch.object(ingestion_service, "_add_semantic_structure", side_effect=lambda x: x) as mock_sem, \
         patch.object(ingestion_service, "_chunk_text_with_title_grouping", return_value=[Chunk(text="chunk3", metadata=DocumentMetadata(filename="test.txt", chunk_id=0, source="test.txt - Carrier Details", chunk_type="text"))]) as mock_chunk:
        result = ingestion_service.process_file(dummy_txt_bytes, dummy_filename)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].text == "chunk3"
        mock_sem.assert_called_once()
        mock_chunk.assert_called_once()

def test_process_file_unsupported_extension_returns_empty(ingestion_service):
    dummy_bytes = b"irrelevant"
    dummy_filename = "test.xyz"
    result = ingestion_service.process_file(dummy_bytes, dummy_filename)
    assert result == []

def test_process_file_error_handling_returns_empty(ingestion_service):
    dummy_bytes = b"irrelevant"
    dummy_filename = "test.pdf"
    with patch.object(ingestion_service, "_extract_text_from_pdf", side_effect=Exception("fail")):
        result = ingestion_service.process_file(dummy_bytes, dummy_filename)
        assert result == []

def test_extract_text_from_pdf_calls_pymupdf(monkeypatch, ingestion_service):
    dummy_bytes = b"dummy"
    dummy_page1 = MagicMock()
    dummy_page1.get_text.return_value = "Carrier Details\n"
    dummy_page2 = MagicMock()
    dummy_page2.get_text.return_value = "Rate Breakdown\n"
    dummy_doc = [dummy_page1, dummy_page2]
    open_mock = MagicMock(return_value=dummy_doc)
    monkeypatch.setattr("pymupdf.open", open_mock)
    result = ingestion_service._extract_text_from_pdf(dummy_bytes)
    assert "### Page 1" in result
    assert "Carrier Details" in result
    assert "### Page 2" in result
    assert "Rate Breakdown" in result
    open_mock.assert_called_once_with(stream=dummy_bytes, filetype="pdf")

def test_extract_text_from_docx_reads_paragraphs(monkeypatch, ingestion_service):
    dummy_bytes = b"dummy"
    dummy_doc = MagicMock()
    dummy_doc.paragraphs = [MagicMock(text="Carrier Details"), MagicMock(text="Some content")]
    docx_document_mock = MagicMock(return_value=dummy_doc)
    monkeypatch.setattr("docx.Document", docx_document_mock)
    result = ingestion_service._extract_text_from_docx(dummy_bytes)
    assert "Carrier Details" in result
    assert "Some content" in result
    docx_document_mock.assert_called_once()

def test_add_semantic_structure_adds_headers(ingestion_service):
    text = "Carrier Details\nSome content\nRate Breakdown\n"
    result = ingestion_service._add_semantic_structure(text)
    assert "\n## Carrier Details\n" in result
    assert "\n## Rate Breakdown\n" in result

def test_add_semantic_structure_cleans_whitespace(ingestion_service):
    text = "\n\n\nCarrier Details\n\n\n\nSome content\n\n\n"
    result = ingestion_service._add_semantic_structure(text)
    assert result.count("\n\n") == 1 or result.count("\n\n") == 2  # Should not have excessive newlines

def test_chunk_text_with_title_grouping_groups_sections(monkeypatch, ingestion_service):
    # Simulate text with two headers, both in the same group
    text = "\n## Carrier Details\nCarrier info\n## Driver Details\nDriver info\n"
    # Patch text_splitter to just split on double newlines for simplicity
    ingestion_service.text_splitter = MagicMock()
    ingestion_service.text_splitter.split_text.side_effect = lambda x: [x]
    result = ingestion_service._chunk_text_with_title_grouping(text, "file.pdf")
    assert len(result) == 1
    assert "Carrier Details" in result[0].text
    assert "Driver Details" in result[0].text
    assert result[0].metadata.filename == "file.pdf"
    assert result[0].metadata.chunk_id == 0

def test_chunk_text_with_title_grouping_handles_no_headers(monkeypatch, ingestion_service):
    text = "No headers here, just content."
    ingestion_service.text_splitter = MagicMock()
    ingestion_service.text_splitter.split_text.side_effect = lambda x: [x]
    result = ingestion_service._chunk_text_with_title_grouping(text, "plain.txt")
    assert len(result) == 1
    assert result[0].text == "No headers here, just content."
    assert result[0].metadata.filename == "plain.txt"

def test_chunk_text_with_title_grouping_multiple_groups(monkeypatch, ingestion_service):
    # Simulate text with headers from different groups
    text = "\n## Carrier Details\nCarrier info\n## Pickup\nPickup info\n"
    ingestion_service.text_splitter = MagicMock()
    ingestion_service.text_splitter.split_text.side_effect = lambda x: [x]
    result = ingestion_service._chunk_text_with_title_grouping(text, "file.pdf")
    assert len(result) == 2
    assert "Carrier Details" in result[0].text
    assert "Pickup" in result[1].text

def test_extract_section_name_with_header(ingestion_service):
    chunk_text = "## Carrier Details\nSome content"
    section = ingestion_service._extract_section_name(chunk_text)
    assert section == "Carrier Details"

def test_extract_section_name_without_header(ingestion_service):
    chunk_text = "Some content without header"
    section = ingestion_service._extract_section_name(chunk_text)
    assert section == "General"

def test_clean_chunk_removes_excessive_whitespace_and_page_markers(ingestion_service):
    chunk_text = "\n\n\n### Page 1\n\n\nSome content\n\n\n"
    cleaned = ingestion_service._clean_chunk(chunk_text)
    assert "### Page 1" not in cleaned
    assert cleaned.startswith("Some content")
    assert "\n\n\n" not in cleaned

def test_clean_chunk_preserves_content(ingestion_service):
    chunk_text = "## Carrier Details\nSome   content\n"
    cleaned = ingestion_service._clean_chunk(chunk_text)
    assert "Carrier Details" in cleaned
    assert "  " not in cleaned
