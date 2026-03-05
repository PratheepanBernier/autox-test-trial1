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
        CHUNK_SIZE = 10
        CHUNK_OVERLAP = 2
    monkeypatch.setattr("core.config.settings", DummySettings())

def test_process_file_pdf_happy_path(service):
    fake_pdf_bytes = b"%PDF-1.4 fake"
    filename = "test.pdf"
    fake_text = "Carrier Details\nSome info\nRate Breakdown\nMore info"
    # Patch pymupdf.open and page.get_text
    fake_doc = [MagicMock(), MagicMock()]
    fake_doc[0].get_text.return_value = "Carrier Details\nSome info"
    fake_doc[1].get_text.return_value = "Rate Breakdown\nMore info"
    with patch("pymupdf.open", return_value=fake_doc):
        with patch.object(service, "_add_semantic_structure", wraps=service._add_semantic_structure) as add_semantic:
            with patch.object(service, "_chunk_text_with_title_grouping", wraps=service._chunk_text_with_title_grouping) as chunker:
                result = service.process_file(fake_pdf_bytes, filename)
                assert isinstance(result, list)
                assert all(isinstance(c, Chunk) for c in result)
                assert add_semantic.called
                assert chunker.called
                # Should have at least one chunk with Carrier Details and one with Rate Breakdown
                texts = [c.text for c in result]
                assert any("Carrier Details" in t for t in texts)
                assert any("Rate Breakdown" in t for t in texts)

def test_process_file_docx_happy_path(service):
    fake_docx_bytes = b"PK\x03\x04 fake docx"
    filename = "test.docx"
    fake_doc = MagicMock()
    fake_doc.paragraphs = [MagicMock(text="Carrier Details"), MagicMock(text="Some info")]
    with patch("docx.Document", return_value=fake_doc):
        with patch.object(service, "_add_semantic_structure", wraps=service._add_semantic_structure) as add_semantic:
            with patch.object(service, "_chunk_text_with_title_grouping", wraps=service._chunk_text_with_title_grouping) as chunker:
                result = service.process_file(fake_docx_bytes, filename)
                assert isinstance(result, list)
                assert all(isinstance(c, Chunk) for c in result)
                assert add_semantic.called
                assert chunker.called
                texts = [c.text for c in result]
                assert any("Carrier Details" in t for t in texts)

def test_process_file_txt_happy_path(service):
    content = "Carrier Details\nSome info\nRate Breakdown\nMore info"
    filename = "test.txt"
    with patch.object(service, "_add_semantic_structure", wraps=service._add_semantic_structure) as add_semantic:
        with patch.object(service, "_chunk_text_with_title_grouping", wraps=service._chunk_text_with_title_grouping) as chunker:
            result = service.process_file(content.encode("utf-8"), filename)
            assert isinstance(result, list)
            assert all(isinstance(c, Chunk) for c in result)
            assert add_semantic.called
            assert chunker.called
            texts = [c.text for c in result]
            assert any("Carrier Details" in t for t in texts)

def test_process_file_unsupported_type_returns_empty(service):
    filename = "test.xlsx"
    result = service.process_file(b"irrelevant", filename)
    assert result == []

def test_process_file_error_handling_returns_empty(service):
    # Simulate error in _extract_text_from_pdf
    with patch.object(service, "_extract_text_from_pdf", side_effect=Exception("fail")):
        result = service.process_file(b"bad", "test.pdf")
        assert result == []

def test_extract_text_from_pdf_adds_page_markers(service):
    fake_pdf_bytes = b"%PDF-1.4 fake"
    fake_doc = [MagicMock(), MagicMock()]
    fake_doc[0].get_text.return_value = "Page1"
    fake_doc[1].get_text.return_value = "Page2"
    with patch("pymupdf.open", return_value=fake_doc):
        text = service._extract_text_from_pdf(fake_pdf_bytes)
        assert "### Page 1" in text
        assert "Page1" in text
        assert "### Page 2" in text
        assert "Page2" in text

def test_extract_text_from_docx_reads_paragraphs(service):
    fake_docx_bytes = b"PK\x03\x04 fake docx"
    fake_doc = MagicMock()
    fake_doc.paragraphs = [MagicMock(text="Para1"), MagicMock(text="Para2")]
    with patch("docx.Document", return_value=fake_doc):
        text = service._extract_text_from_docx(fake_docx_bytes)
        assert "Para1" in text
        assert "Para2" in text
        assert "\n" in text

def test_add_semantic_structure_adds_headers_and_cleans_whitespace(service):
    text = "Carrier Details\n\n\nRate Breakdown\n\n\n\nPickup"
    result = DocumentIngestionService()._add_semantic_structure(text)
    # Should add markdown headers and reduce whitespace
    assert "\n## Carrier Details\n" in result
    assert "\n## Rate Breakdown\n" in result
    assert "\n## Pickup\n" in result
    assert "\n\n\n" not in result

def test_chunk_text_with_title_grouping_groups_sections_and_chunks(service):
    # Simulate a text with two groupable sections and one general
    text = "\n## Carrier Details\nCarrier info\n## Driver Details\nDriver info\n## Reference ID\nRef123"
    filename = "file.txt"
    # Patch text_splitter to split only by group
    service = DocumentIngestionService()
    with patch.object(service.text_splitter, "split_text", side_effect=lambda t: [t]):
        chunks = service._chunk_text_with_title_grouping(text, filename)
        assert len(chunks) == 2  # Carrier+Driver grouped, Reference ID separate
        assert any("Carrier info" in c.text and "Driver info" in c.text for c in chunks)
        assert any("Ref123" in c.text for c in chunks)
        # Metadata checks
        for c in chunks:
            assert isinstance(c.metadata, DocumentMetadata)
            assert c.metadata.filename == filename
            assert c.metadata.chunk_type == "text"

def test_chunk_text_with_title_grouping_handles_empty_and_non_header(service):
    text = "No headers here, just text."
    filename = "plain.txt"
    service = DocumentIngestionService()
    with patch.object(service.text_splitter, "split_text", side_effect=lambda t: [t]):
        chunks = service._chunk_text_with_title_grouping(text, filename)
        assert len(chunks) == 1
        assert "No headers here" in chunks[0].text

def test_extract_section_name_extracts_header(service):
    text = "## Rate Breakdown\nDetails"
    section = DocumentIngestionService()._extract_section_name(text)
    assert section == "Rate Breakdown"

def test_extract_section_name_returns_general_if_no_header(service):
    text = "No header here"
    section = DocumentIngestionService()._extract_section_name(text)
    assert section == "General"

def test_clean_chunk_removes_excess_whitespace_and_page_markers(service):
    text = "\n\n\n### Page 1\n\nSome text\n\n\n"
    cleaned = service._clean_chunk(text)
    assert cleaned.startswith("Some text")
    assert "### Page 1" not in cleaned
    assert "\n\n\n" not in cleaned

def test_clean_chunk_preserves_content_within_page_marker(service):
    text = "### Page 2\nContent"
    cleaned = DocumentIngestionService()._clean_chunk(text)
    assert "Content" in cleaned

def test_chunk_text_with_title_grouping_boundary_conditions(service):
    # Only headers, no content
    text = "\n## Carrier Details\n\n## Driver Details\n"
    filename = "file.txt"
    with patch.object(service.text_splitter, "split_text", side_effect=lambda t: [t]):
        chunks = service._chunk_text_with_title_grouping(text, filename)
        # Should still produce chunks for headers even if content is empty
        assert len(chunks) >= 1
        for c in chunks:
            assert c.text.strip() != ""

def test_process_file_empty_txt_returns_empty_chunks(service):
    filename = "empty.txt"
    result = service.process_file(b"", filename)
    # Should not error, but may return empty or one empty chunk
    assert isinstance(result, list)
    # All chunks should be empty or whitespace
    for c in result:
        assert c.text.strip() == ""
