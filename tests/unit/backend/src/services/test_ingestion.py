import pytest
import re
from unittest.mock import patch, MagicMock, call
from backend.src.services.ingestion import DocumentIngestionService
from backend.src.models.schemas import Chunk, DocumentMetadata

@pytest.fixture
def service():
    return DocumentIngestionService()

@pytest.fixture
def fake_settings(monkeypatch):
    class FakeSettings:
        CHUNK_SIZE = 100
        CHUNK_OVERLAP = 10
    monkeypatch.setattr("backend.src.core.config.settings", FakeSettings())

@pytest.fixture
def minimal_pdf_bytes():
    # Not a real PDF, but enough for the mock to work
    return b"%PDF-1.4\n%Fake PDF content"

@pytest.fixture
def minimal_docx_bytes():
    # Not a real DOCX, but enough for the mock to work
    return b"PK\x03\x04Fake DOCX content"

@pytest.fixture
def simple_txt_bytes():
    return b"Carrier Details\nJohn Doe\nRate Breakdown\n$1000\n"

def test_process_file_happy_path_pdf(service, minimal_pdf_bytes):
    # Arrange
    filename = "test.pdf"
    fake_text = "\n\n### Page 1\n\nCarrier Details\nJohn Doe\nRate Breakdown\n$1000\n"
    with patch.object(service, "_extract_text_from_pdf", return_value=fake_text) as mock_pdf, \
         patch.object(service, "_add_semantic_structure", side_effect=lambda x: x) as mock_sem, \
         patch.object(service, "_chunk_text_with_title_grouping", return_value=[Chunk(text="chunk1", metadata=DocumentMetadata(filename=filename, chunk_id=0, source="test.pdf - Carrier Details", chunk_type="text"))]) as mock_chunk:
        # Act
        result = service.process_file(minimal_pdf_bytes, filename)
        # Assert
        mock_pdf.assert_called_once_with(minimal_pdf_bytes)
        mock_sem.assert_called_once_with(fake_text)
        mock_chunk.assert_called_once()
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Chunk)
        assert result[0].text == "chunk1"

def test_process_file_happy_path_docx(service, minimal_docx_bytes):
    filename = "test.docx"
    fake_text = "Carrier Details\nJohn Doe\nRate Breakdown\n$1000\n"
    with patch.object(service, "_extract_text_from_docx", return_value=fake_text) as mock_docx, \
         patch.object(service, "_add_semantic_structure", side_effect=lambda x: x) as mock_sem, \
         patch.object(service, "_chunk_text_with_title_grouping", return_value=[Chunk(text="chunk2", metadata=DocumentMetadata(filename=filename, chunk_id=0, source="test.docx - Carrier Details", chunk_type="text"))]) as mock_chunk:
        result = service.process_file(minimal_docx_bytes, filename)
        mock_docx.assert_called_once_with(minimal_docx_bytes)
        mock_sem.assert_called_once_with(fake_text)
        mock_chunk.assert_called_once()
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].text == "chunk2"

def test_process_file_happy_path_txt(service, simple_txt_bytes):
    filename = "test.txt"
    with patch.object(service, "_add_semantic_structure", side_effect=lambda x: x) as mock_sem, \
         patch.object(service, "_chunk_text_with_title_grouping", return_value=[Chunk(text="chunk3", metadata=DocumentMetadata(filename=filename, chunk_id=0, source="test.txt - Carrier Details", chunk_type="text"))]) as mock_chunk:
        result = service.process_file(simple_txt_bytes, filename)
        mock_sem.assert_called_once_with(simple_txt_bytes.decode("utf-8"))
        mock_chunk.assert_called_once()
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].text == "chunk3"

def test_process_file_unsupported_filetype(service):
    filename = "test.csv"
    file_content = b"some,data,here"
    result = service.process_file(file_content, filename)
    assert result == []

def test_process_file_exception_handling(service):
    filename = "test.pdf"
    with patch.object(service, "_extract_text_from_pdf", side_effect=Exception("fail")):
        result = service.process_file(b"irrelevant", filename)
        assert result == []

def test_add_semantic_structure_adds_headers(service):
    text = "Carrier Details\nSome info\nRate Breakdown\n$1000\n"
    result = service._add_semantic_structure(text)
    # Should add markdown headers
    assert "\n## Carrier Details\n" in result
    assert "\n## Rate Breakdown\n" in result
    # Should not have excessive whitespace
    assert not re.search(r"\n{3,}", result)

def test_add_semantic_structure_preserves_structure(service):
    text = "Carrier Details\n\n\n\nRate Breakdown\n"
    result = service._add_semantic_structure(text)
    # Should collapse excessive newlines
    assert "\n\n\n" not in result

def test_chunk_text_with_title_grouping_basic(service):
    # Arrange
    text = "\n## Carrier Details\nJohn Doe\n## Rate Breakdown\n$1000\n"
    filename = "doc.txt"
    # Patch text_splitter
    service.text_splitter = MagicMock()
    service.text_splitter.split_text.side_effect = lambda x: [x]
    # Act
    chunks = service._chunk_text_with_title_grouping(text, filename)
    # Assert
    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert any("Carrier Details" in c.metadata.source for c in chunks)
    assert any("Rate Breakdown" in c.metadata.source for c in chunks)

def test_chunk_text_with_title_grouping_empty_section(service):
    text = "\n## Carrier Details\n\n## Rate Breakdown\n"
    filename = "doc.txt"
    service.text_splitter = MagicMock()
    service.text_splitter.split_text.side_effect = lambda x: [x]
    chunks = service._chunk_text_with_title_grouping(text, filename)
    # Should still produce chunks, even if section is empty
    assert len(chunks) == 2

def test_chunk_text_with_title_grouping_large_section_split(service):
    text = "\n## Carrier Details\n" + ("A" * 300)
    filename = "doc.txt"
    # Simulate splitter splitting into 3 chunks
    service.text_splitter = MagicMock()
    service.text_splitter.split_text.side_effect = lambda x: [x[:100], x[100:200], x[200:]]
    chunks = service._chunk_text_with_title_grouping(text, filename)
    assert len(chunks) == 3
    for i, chunk in enumerate(chunks):
        assert chunk.metadata.chunk_id == i

def test_extract_section_name_found(service):
    text = "## Carrier Details\nSome info"
    section = service._extract_section_name(text)
    assert section == "Carrier Details"

def test_extract_section_name_not_found(service):
    text = "No header here"
    section = service._extract_section_name(text)
    assert section == "General"

def test_clean_chunk_removes_excess_whitespace_and_page_markers(service):
    text = "\n\n\n### Page 1\n\nSome text\n\n\n"
    cleaned = service._clean_chunk(text)
    assert "### Page 1" not in cleaned
    assert cleaned.startswith("Some text")
    assert "\n\n\n" not in cleaned

def test_clean_chunk_preserves_content(service):
    text = "Some text  with   extra   spaces"
    cleaned = service._clean_chunk(text)
    assert "  " not in cleaned
    assert cleaned == "Some text with extra spaces"

def test_extract_text_from_pdf_calls_pymupdf(monkeypatch, service, minimal_pdf_bytes):
    # Arrange
    fake_page = MagicMock()
    fake_page.get_text.return_value = "Page content"
    fake_doc = [fake_page, fake_page]
    fake_open = MagicMock(return_value=fake_doc)
    monkeypatch.setitem(__import__("sys").modules, "pymupdf", MagicMock(open=fake_open))
    # Act
    result = service._extract_text_from_pdf(minimal_pdf_bytes)
    # Assert
    assert "### Page 1" in result
    assert "Page content" in result
    assert "### Page 2" in result

def test_extract_text_from_docx_calls_docx(monkeypatch, service, minimal_docx_bytes):
    # Arrange
    fake_para1 = MagicMock()
    fake_para1.text = "First paragraph"
    fake_para2 = MagicMock()
    fake_para2.text = "Second paragraph"
    fake_doc = MagicMock()
    fake_doc.paragraphs = [fake_para1, fake_para2]
    fake_docx = MagicMock()
    fake_docx.Document.return_value = fake_doc
    monkeypatch.setitem(__import__("sys").modules, "docx", fake_docx)
    # Act
    result = service._extract_text_from_docx(minimal_docx_bytes)
    # Assert
    assert "First paragraph" in result
    assert "Second paragraph" in result
    assert "\n" in result

def test_process_file_boundary_chunking(service):
    # Simulate a file that is exactly at the chunk boundary
    filename = "test.txt"
    text = "A" * (service.text_splitter.chunk_size * 2)
    with patch.object(service, "_add_semantic_structure", side_effect=lambda x: x), \
         patch.object(service, "_chunk_text_with_title_grouping", return_value=[Chunk(text=text, metadata=DocumentMetadata(filename=filename, chunk_id=0, source="test.txt - General", chunk_type="text"))]):
        result = service.process_file(text.encode("utf-8"), filename)
        assert len(result) == 1
        assert result[0].text == text

def test_chunk_text_with_title_grouping_no_headers(service):
    text = "Just some text without headers."
    filename = "plain.txt"
    service.text_splitter = MagicMock()
    service.text_splitter.split_text.side_effect = lambda x: [x]
    chunks = service._chunk_text_with_title_grouping(text, filename)
    assert len(chunks) == 1
    assert chunks[0].metadata.source == "plain.txt - General"
    assert "Just some text" in chunks[0].text
