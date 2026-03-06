import pytest
from unittest.mock import patch, MagicMock, call
from backend.src.services.ingestion import DocumentIngestionService
from backend.src.models.schemas import Chunk, DocumentMetadata

@pytest.fixture
def service():
    return DocumentIngestionService()

@pytest.fixture
def dummy_settings(monkeypatch):
    class DummySettings:
        CHUNK_SIZE = 100
        CHUNK_OVERLAP = 10
    monkeypatch.setattr("backend.src.core.config.settings", DummySettings())

def test_process_file_txt_happy_path(service):
    # Simple text file with two sections
    content = b"Carrier Details\nJohn Doe\nRate Breakdown\n$1000"
    filename = "test.txt"
    with patch.object(service, "_add_semantic_structure", wraps=service._add_semantic_structure) as add_semantic, \
         patch.object(service, "_chunk_text_with_title_grouping", wraps=service._chunk_text_with_title_grouping) as chunk_group:
        chunks = service.process_file(content, filename)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)
        assert add_semantic.called
        assert chunk_group.called
        assert any("Carrier Details" in c.text or "Rate Breakdown" in c.text for c in chunks)

def test_process_file_pdf_calls_pdf_extractor(service):
    content = b"%PDF-1.4 dummy"
    filename = "file.pdf"
    with patch.object(service, "_extract_text_from_pdf", return_value="Carrier Details\nJohn Doe") as pdf_extract, \
         patch.object(service, "_add_semantic_structure", wraps=service._add_semantic_structure), \
         patch.object(service, "_chunk_text_with_title_grouping", wraps=service._chunk_text_with_title_grouping):
        service.process_file(content, filename)
        pdf_extract.assert_called_once_with(content)

def test_process_file_docx_calls_docx_extractor(service):
    content = b"PK\x03\x04 dummy"
    filename = "file.docx"
    with patch.object(service, "_extract_text_from_docx", return_value="Carrier Details\nJohn Doe") as docx_extract, \
         patch.object(service, "_add_semantic_structure", wraps=service._add_semantic_structure), \
         patch.object(service, "_chunk_text_with_title_grouping", wraps=service._chunk_text_with_title_grouping):
        service.process_file(content, filename)
        docx_extract.assert_called_once_with(content)

def test_process_file_unsupported_extension_returns_empty(service):
    content = b"dummy"
    filename = "file.xls"
    with patch("backend.src.services.ingestion.logger") as mock_logger:
        result = DocumentIngestionService().process_file(content, filename)
        assert result == []
        assert mock_logger.error.called
        assert "Unsupported file type" in str(mock_logger.error.call_args)

def test_process_file_exception_returns_empty(service):
    content = b"dummy"
    filename = "file.txt"
    with patch.object(service, "_add_semantic_structure", side_effect=Exception("fail")), \
         patch("backend.src.services.ingestion.logger") as mock_logger:
        result = service.process_file(content, filename)
        assert result == []
        assert mock_logger.error.called
        assert "fail" in str(mock_logger.error.call_args)

def test_add_semantic_structure_adds_headers(service):
    text = "Carrier Details\nSome info\nRate Breakdown\n$1000"
    result = service._add_semantic_structure(text)
    assert "\n## Carrier Details\n" in result
    assert "\n## Rate Breakdown\n" in result

def test_add_semantic_structure_cleans_whitespace(service):
    text = "Carrier Details\n\n\n\nSome info"
    result = service._add_semantic_structure(text)
    assert "\n\n" in result
    assert "\n\n\n" not in result

def test_chunk_text_with_title_grouping_groups_sections(service):
    # Simulate text with two headers, both in the same group
    text = "\n## Carrier Details\nJohn\n## Driver Details\nJane"
    filename = "doc.txt"
    # Patch text_splitter to just split on headers for predictability
    service.text_splitter = MagicMock()
    service.text_splitter.split_text.side_effect = lambda t: [t]
    chunks = service._chunk_text_with_title_grouping(text, filename)
    assert len(chunks) == 1 or len(chunks) == 2  # Depending on grouping logic
    assert all(isinstance(c, Chunk) for c in chunks)
    assert any("Carrier Details" in c.text for c in chunks)
    assert any("Driver Details" in c.text for c in chunks)

def test_chunk_text_with_title_grouping_handles_no_headers(service):
    text = "No headers here, just text."
    filename = "plain.txt"
    service.text_splitter = MagicMock()
    service.text_splitter.split_text.side_effect = lambda t: [t]
    chunks = service._chunk_text_with_title_grouping(text, filename)
    assert len(chunks) == 1
    assert chunks[0].metadata.filename == filename
    assert "No headers" in chunks[0].text

def test_chunk_text_with_title_grouping_empty_text(service):
    text = ""
    filename = "empty.txt"
    service.text_splitter = MagicMock()
    service.text_splitter.split_text.side_effect = lambda t: [t]
    chunks = service._chunk_text_with_title_grouping(text, filename)
    assert chunks == []

def test_extract_section_name_with_header(service):
    text = "## Carrier Details\nSome info"
    section = service._extract_section_name(text)
    assert section == "Carrier Details"

def test_extract_section_name_without_header(service):
    text = "Some info"
    section = service._extract_section_name(text)
    assert section == "General"

def test_clean_chunk_removes_excessive_whitespace_and_page_markers(service):
    text = "\n\n\n### Page 1\n\n\nSome text\n\n\n"
    cleaned = service._clean_chunk(text)
    assert "Page 1" not in cleaned
    assert cleaned.startswith("Some text")
    assert "\n\n\n" not in cleaned

def test_clean_chunk_preserves_content_within_page_marker(service):
    text = "### Page 2\nContent"
    cleaned = service._clean_chunk(text)
    assert "Page 2" in cleaned or "Content" in cleaned

def test_pdf_extractor_calls_pymupdf(monkeypatch, service):
    fake_page = MagicMock()
    fake_page.get_text.return_value = "Page text"
    fake_doc = [fake_page, fake_page]
    open_mock = MagicMock(return_value=fake_doc)
    monkeypatch.setitem(__import__("sys").modules, "pymupdf", MagicMock(open=open_mock))
    result = service._extract_text_from_pdf(b"dummy")
    assert "Page 1" in result and "Page 2" in result
    assert "Page text" in result

def test_docx_extractor_reads_docx(monkeypatch, service):
    class DummyPara:
        def __init__(self, text): self.text = text
    class DummyDoc:
        paragraphs = [DummyPara("A"), DummyPara("B")]
    docx_mock = MagicMock()
    docx_mock.Document.return_value = DummyDoc()
    monkeypatch.setitem(__import__("sys").modules, "docx", docx_mock)
    result = service._extract_text_from_docx(b"dummy")
    assert "A" in result and "B" in result
    assert "\n" in result

def test_chunk_text_with_title_grouping_boundary_conditions(service):
    # Section at the very start and end
    text = "## Carrier Details\nA\n## Rate Breakdown\nB"
    filename = "boundary.txt"
    service.text_splitter = MagicMock()
    service.text_splitter.split_text.side_effect = lambda t: [t]
    chunks = service._chunk_text_with_title_grouping(text, filename)
    assert len(chunks) >= 1
    assert any("Carrier Details" in c.text for c in chunks)
    assert any("Rate Breakdown" in c.text for c in chunks)

def test_chunk_text_with_title_grouping_multiple_groups(service):
    # Sections from different groups
    text = "## Carrier Details\nA\n## Pickup\nB\n## Rate Breakdown\nC"
    filename = "multi.txt"
    service.text_splitter = MagicMock()
    service.text_splitter.split_text.side_effect = lambda t: [t]
    chunks = service._chunk_text_with_title_grouping(text, filename)
    assert any("Carrier Details" in c.text for c in chunks)
    assert any("Pickup" in c.text for c in chunks)
    assert any("Rate Breakdown" in c.text for c in chunks)
    # Ensure chunk_id increments
    chunk_ids = [c.metadata.chunk_id for c in chunks]
    assert chunk_ids == list(range(len(chunks)))
