# source_hash: 7de298d97adb91e8
import pytest
from unittest.mock import patch, MagicMock, call
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
    monkeypatch.setattr("core.config.settings", DummySettings())

def test_process_file_txt_happy_path(service):
    text = "Carrier Details\nJohn Doe\nRate Breakdown\n$1000"
    filename = "test.txt"
    with patch.object(service, "_add_semantic_structure", wraps=service._add_semantic_structure) as add_semantic, \
         patch.object(service, "_chunk_text_with_title_grouping", wraps=service._chunk_text_with_title_grouping) as chunk_group:
        result = service.process_file(text.encode("utf-8"), filename)
        assert isinstance(result, list)
        assert all(isinstance(chunk, Chunk) for chunk in result)
        add_semantic.assert_called_once()
        chunk_group.assert_called_once()
        assert any("Carrier Details" in c.text for c in result)
        assert any("Rate Breakdown" in c.text for c in result)

def test_process_file_pdf_calls_pdf_extraction(service):
    filename = "doc.pdf"
    fake_bytes = b"%PDF-1.4"
    with patch.object(service, "_extract_text_from_pdf", return_value="Carrier Details\nPage 1") as pdf_extract, \
         patch.object(service, "_add_semantic_structure", wraps=service._add_semantic_structure), \
         patch.object(service, "_chunk_text_with_title_grouping", wraps=service._chunk_text_with_title_grouping):
        service.process_file(fake_bytes, filename)
        pdf_extract.assert_called_once_with(fake_bytes)

def test_process_file_docx_calls_docx_extraction(service):
    filename = "doc.docx"
    fake_bytes = b"PK\x03\x04"
    with patch.object(service, "_extract_text_from_docx", return_value="Carrier Details\nDocx") as docx_extract, \
         patch.object(service, "_add_semantic_structure", wraps=service._add_semantic_structure), \
         patch.object(service, "_chunk_text_with_title_grouping", wraps=service._chunk_text_with_title_grouping):
        service.process_file(fake_bytes, filename)
        docx_extract.assert_called_once_with(fake_bytes)

def test_process_file_unsupported_extension_returns_empty(service):
    filename = "doc.xls"
    fake_bytes = b""
    result = service.process_file(fake_bytes, filename)
    assert result == []

def test_process_file_error_handling_returns_empty(service):
    filename = "doc.txt"
    with patch.object(service, "_add_semantic_structure", side_effect=Exception("fail")):
        result = service.process_file(b"abc", filename)
        assert result == []

def test_extract_text_from_pdf_adds_page_markers(service):
    fake_bytes = b"pdf"
    mock_doc = [MagicMock(), MagicMock()]
    mock_doc[0].get_text.return_value = "Page1Text"
    mock_doc[1].get_text.return_value = "Page2Text"
    with patch("pymupdf.open", return_value=mock_doc):
        text = service._extract_text_from_pdf(fake_bytes)
        assert "### Page 1" in text
        assert "Page1Text" in text
        assert "### Page 2" in text
        assert "Page2Text" in text

def test_extract_text_from_docx_reads_paragraphs(service):
    fake_bytes = b"docx"
    paragraphs = [MagicMock(text="Para1"), MagicMock(text="Para2")]
    mock_doc = MagicMock()
    mock_doc.paragraphs = paragraphs
    with patch("docx.Document", return_value=mock_doc):
        text = service._extract_text_from_docx(fake_bytes)
        assert "Para1" in text
        assert "Para2" in text
        assert "\n" in text

def test_add_semantic_structure_adds_headers_and_cleans_whitespace(service):
    raw = "Carrier Details\n\n\nRate Breakdown\n\n\n\nPickup"
    result = DocumentIngestionService()._add_semantic_structure(raw)
    assert result.count("## Carrier Details") == 1
    assert result.count("## Rate Breakdown") == 1
    assert result.count("## Pickup") == 1
    assert "\n\n\n" not in result

def test_chunk_text_with_title_grouping_groups_sections(service):
    service = DocumentIngestionService()
    text = "\n## Carrier Details\nJohn\n## Driver Details\nJane\n## Rate Breakdown\n$1000"
    filename = "file.txt"
    # Patch text_splitter to avoid actual splitting
    service.text_splitter = MagicMock()
    service.text_splitter.split_text.side_effect = lambda x: [x]
    chunks = service._chunk_text_with_title_grouping(text, filename)
    # Carrier Details and Driver Details should be grouped
    assert any("Carrier Details" in c.text and "Driver Details" in c.text for c in chunks)
    # Rate Breakdown should be its own group
    assert any("Rate Breakdown" in c.text for c in chunks)
    # Metadata should be correct
    for idx, chunk in enumerate(chunks):
        assert chunk.metadata.filename == filename
        assert chunk.metadata.chunk_id == idx
        assert chunk.metadata.chunk_type == "text"

def test_chunk_text_with_title_grouping_handles_empty_and_general(service):
    service = DocumentIngestionService()
    text = "No headers here, just text."
    filename = "plain.txt"
    service.text_splitter = MagicMock()
    service.text_splitter.split_text.side_effect = lambda x: [x]
    chunks = service._chunk_text_with_title_grouping(text, filename)
    assert len(chunks) == 1
    assert chunks[0].metadata.source == f"{filename} - General"

def test_extract_section_name_with_header(service):
    text = "## Rate Breakdown\nDetails"
    section = service._extract_section_name(text)
    assert section == "Rate Breakdown"

def test_extract_section_name_without_header(service):
    text = "No header"
    section = service._extract_section_name(text)
    assert section == "General"

@pytest.mark.parametrize("chunk_text,expected", [
    ("Text\n\n\nMore", "Text\n\nMore"),
    ("   Text   ", "Text"),
    ("### Page 1\n", ""),
    ("### Page 2\nContent", "### Page 2\nContent"),
])
def test_clean_chunk_removes_excess_whitespace_and_page_markers(service, chunk_text, expected):
    assert service._clean_chunk(chunk_text) == expected

def test_chunk_text_with_title_grouping_empty_input(service):
    service.text_splitter = MagicMock()
    service.text_splitter.split_text.return_value = []
    chunks = service._chunk_text_with_title_grouping("", "file.txt")
    assert chunks == []

def test_chunk_text_with_title_grouping_boundary_conditions(service):
    service.text_splitter = MagicMock()
    # Simulate splitting into multiple small chunks
    service.text_splitter.split_text.side_effect = lambda x: [x[i:i+5] for i in range(0, len(x), 5)]
    text = "\n## Carrier Details\n" + "A" * 12
    chunks = service._chunk_text_with_title_grouping(text, "file.txt")
    assert all(len(c.text) <= 7 for c in chunks)  # 5 chars + possible header

def test_process_file_txt_with_unicode(service):
    text = "Carrier Details\nJöhn Döe\nRate Breakdown\n€1000"
    filename = "unicode.txt"
    with patch.object(service, "_add_semantic_structure", wraps=service._add_semantic_structure), \
         patch.object(service, "_chunk_text_with_title_grouping", wraps=service._chunk_text_with_title_grouping):
        result = service.process_file(text.encode("utf-8"), filename)
        assert any("Jöhn Döe" in c.text for c in result)
        assert any("€1000" in c.text for c in result)

def test_process_file_txt_with_empty_content(service):
    filename = "empty.txt"
    result = service.process_file(b"", filename)
    assert isinstance(result, list)
    assert result == []

def test_chunk_text_with_title_grouping_reconciliation(service):
    service = DocumentIngestionService()
    text = "\n## Carrier Details\nJohn\n## Driver Details\nJane"
    filename = "file.txt"
    # Patch text_splitter to split into two chunks
    service.text_splitter = MagicMock()
    service.text_splitter.split_text.side_effect = lambda x: [x[:10], x[10:]]
    chunks1 = service._chunk_text_with_title_grouping(text, filename)
    # Now, simulate splitting all as one chunk
    service.text_splitter.split_text.side_effect = lambda x: [x]
    chunks2 = service._chunk_text_with_title_grouping(text, filename)
    # The concatenation of all chunk texts should be the same
    assert "".join(c.text for c in chunks1) == "".join(c.text for c in chunks2)
    # The number of chunks should differ
    assert len(chunks1) != len(chunks2)
