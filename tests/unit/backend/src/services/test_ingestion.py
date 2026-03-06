import pytest
import re
from unittest.mock import patch, MagicMock, call
from backend.src.services.ingestion import DocumentIngestionService
from backend.src.models.schemas import Chunk, DocumentMetadata

@pytest.fixture
def service():
    return DocumentIngestionService()

@pytest.fixture
def mock_settings(monkeypatch):
    class DummySettings:
        CHUNK_SIZE = 100
        CHUNK_OVERLAP = 10
    monkeypatch.setattr("backend.src.core.config.settings", DummySettings())

@pytest.fixture
def dummy_filename():
    return "testfile.txt"

@pytest.fixture
def dummy_pdf_filename():
    return "testfile.pdf"

@pytest.fixture
def dummy_docx_filename():
    return "testfile.docx"

@pytest.fixture
def dummy_txt_content():
    return b"Carrier Details\nSome carrier info.\nRate Breakdown\nSome rate info.\n"

@pytest.fixture
def dummy_pdf_content():
    return b"%PDF-1.4 dummy pdf bytes"

@pytest.fixture
def dummy_docx_content():
    return b"PK\x03\x04 dummy docx bytes"

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

def test_process_file_happy_path_txt(service, dummy_txt_content, dummy_filename, mock_settings):
    # Arrange
    # Patch text_splitter to return deterministic chunks
    with patch.object(service, "text_splitter") as mock_splitter:
        mock_splitter.split_text.side_effect = lambda text: [text]
        # Act
        chunks = service.process_file(dummy_txt_content, dummy_filename)
        # Assert
        assert isinstance(chunks, list)
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert len(chunks) == 1
        assert "Carrier Details" in chunks[0].text
        assert chunks[0].metadata.filename == dummy_filename
        assert chunks[0].metadata.chunk_id == 0
        assert chunks[0].metadata.chunk_type == "text"

def test_process_file_happy_path_pdf(service, dummy_pdf_content, dummy_pdf_filename, mock_settings):
    # Arrange
    fake_pdf_text = "\n\n### Page 1\n\nCarrier Details\nSome carrier info.\n"
    with patch("backend.src.services.ingestion.pymupdf.open") as mock_open, \
         patch.object(service, "text_splitter") as mock_splitter:
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Carrier Details\nSome carrier info.\n"
        mock_doc.__iter__.return_value = [mock_page]
        mock_open.return_value = mock_doc
        mock_splitter.split_text.side_effect = lambda text: [text]
        # Act
        chunks = service.process_file(dummy_pdf_content, dummy_pdf_filename)
        # Assert
        assert isinstance(chunks, list)
        assert len(chunks) == 1
        assert "Carrier Details" in chunks[0].text
        assert chunks[0].metadata.filename == dummy_pdf_filename

def test_process_file_happy_path_docx(service, dummy_docx_content, dummy_docx_filename, mock_settings):
    # Arrange
    with patch("backend.src.services.ingestion.docx.Document") as mock_docx, \
         patch.object(service, "text_splitter") as mock_splitter:
        mock_doc = MagicMock()
        mock_doc.paragraphs = [MagicMock(text="Carrier Details"), MagicMock(text="Some carrier info.")]
        mock_docx.return_value = mock_doc
        mock_splitter.split_text.side_effect = lambda text: [text]
        # Act
        chunks = service.process_file(dummy_docx_content, dummy_docx_filename)
        # Assert
        assert isinstance(chunks, list)
        assert len(chunks) == 1
        assert "Carrier Details" in chunks[0].text
        assert chunks[0].metadata.filename == dummy_docx_filename

def test_process_file_unsupported_filetype(service, mock_settings):
    # Arrange
    filename = "file.unsupported"
    content = b"dummy"
    # Act
    chunks = service.process_file(content, filename)
    # Assert
    assert chunks == []

def test_process_file_invalid_txt_content(service, mock_settings):
    # Arrange
    filename = "file.txt"
    content = b"\xff\xfe\xfd"  # Invalid utf-8
    # Act
    chunks = service.process_file(content, filename)
    # Assert
    assert chunks == []

def test_process_file_pdf_extraction_error(service, dummy_pdf_content, dummy_pdf_filename, mock_settings):
    # Arrange
    with patch("backend.src.services.ingestion.pymupdf.open", side_effect=RuntimeError("PDF error")), \
         patch.object(service, "text_splitter"):
        # Act
        chunks = service.process_file(dummy_pdf_content, dummy_pdf_filename)
        # Assert
        assert chunks == []

def test_add_semantic_structure_adds_headers(service):
    # Arrange
    text = "Carrier Details\nSome info.\nRate Breakdown\nMore info.\n"
    # Act
    structured = service._add_semantic_structure(text)
    # Assert
    assert "\n## Carrier Details\n" in structured
    assert "\n## Rate Breakdown\n" in structured
    # Should not have excessive whitespace
    assert not re.search(r"\n{3,}", structured)

def test_add_semantic_structure_no_headers(service):
    # Arrange
    text = "Random text with no headers."
    # Act
    structured = service._add_semantic_structure(text)
    # Assert
    assert structured == "Random text with no headers."

def test_chunk_text_with_title_grouping_groups_sections(service, dummy_filename, mock_settings):
    # Arrange
    text = "\n## Carrier Details\nCarrier info.\n## Rate Breakdown\nRate info.\n"
    with patch.object(service, "text_splitter") as mock_splitter:
        # Simulate splitting returns the group as a single chunk
        mock_splitter.split_text.side_effect = lambda t: [t]
        # Act
        chunks = service._chunk_text_with_title_grouping(text, dummy_filename)
        # Assert
        assert len(chunks) == 2
        assert "Carrier Details" in chunks[0].text
        assert "Rate Breakdown" in chunks[1].text
        assert chunks[0].metadata.chunk_id == 0
        assert chunks[1].metadata.chunk_id == 1

def test_chunk_text_with_title_grouping_empty_input(service, dummy_filename, mock_settings):
    # Arrange
    text = ""
    with patch.object(service, "text_splitter") as mock_splitter:
        mock_splitter.split_text.return_value = []
        # Act
        chunks = service._chunk_text_with_title_grouping(text, dummy_filename)
        # Assert
        assert chunks == []

def test_chunk_text_with_title_grouping_large_group_split(service, dummy_filename, mock_settings):
    # Arrange
    text = "\n## Carrier Details\n" + "A" * 500 + "\n"
    with patch.object(service, "text_splitter") as mock_splitter:
        # Simulate splitting into two chunks
        mock_splitter.split_text.return_value = ["A" * 250, "A" * 250]
        # Act
        chunks = service._chunk_text_with_title_grouping(text, dummy_filename)
        # Assert
        assert len(chunks) == 2
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.metadata.filename == dummy_filename for chunk in chunks)

def test_extract_section_name_with_header(service):
    # Arrange
    chunk_text = "## Carrier Details\nSome info."
    # Act
    section = service._extract_section_name(chunk_text)
    # Assert
    assert section == "Carrier Details"

def test_extract_section_name_without_header(service):
    # Arrange
    chunk_text = "Some info without header."
    # Act
    section = service._extract_section_name(chunk_text)
    # Assert
    assert section == "General"

def test_clean_chunk_removes_excess_whitespace_and_page_markers(service):
    # Arrange
    chunk_text = "\n\n\n### Page 1\n\n\nSome text.  \n\n\n"
    # Act
    cleaned = service._clean_chunk(chunk_text)
    # Assert
    assert "### Page 1" not in cleaned
    assert cleaned.startswith("Some text.")
    assert "\n\n\n" not in cleaned

def test_clean_chunk_preserves_content(service):
    # Arrange
    chunk_text = "## Carrier Details\nSome info.\n\n"
    # Act
    cleaned = service._clean_chunk(chunk_text)
    # Assert
    assert cleaned == "## Carrier Details\nSome info."

def test_process_file_logs_error_on_exception(service, mock_settings, caplog):
    # Arrange
    filename = "file.txt"
    content = b"bad"
    with patch.object(service, "_add_semantic_structure", side_effect=Exception("fail")), \
         patch.object(service, "text_splitter"):
        # Act
        with caplog.at_level("ERROR"):
            chunks = service.process_file(content, filename)
        # Assert
        assert chunks == []
        assert any("Error processing file" in r.message for r in caplog.records)
