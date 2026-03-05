# source_hash: 7de298d97adb91e8
import pytest
from unittest.mock import patch, MagicMock
from backend.src.services.ingestion import DocumentIngestionService
from models.schemas import Chunk, DocumentMetadata

@pytest.fixture
def ingestion_service():
    return DocumentIngestionService()

@pytest.fixture
def dummy_settings(monkeypatch):
    class DummySettings:
        CHUNK_SIZE = 100
        CHUNK_OVERLAP = 10
    monkeypatch.setattr("core.config.settings", DummySettings())

def test_process_file_txt_happy_path(ingestion_service):
    content = b"Carrier Details\nJohn Doe\nRate Breakdown\n$1000\n"
    filename = "test.txt"
    with patch.object(ingestion_service, "_add_semantic_structure", wraps=ingestion_service._add_semantic_structure) as add_semantic, \
         patch.object(ingestion_service, "_chunk_text_with_title_grouping", wraps=ingestion_service._chunk_text_with_title_grouping) as chunker:
        chunks = ingestion_service.process_file(content, filename)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)
        assert add_semantic.called
        assert chunker.called
        assert any("Carrier Details" in c.text for c in chunks)

def test_process_file_docx_happy_path(ingestion_service):
    # Simulate a docx file with two paragraphs
    docx_bytes = b"dummy"
    filename = "test.docx"
    fake_doc = MagicMock()
    fake_doc.paragraphs = [MagicMock(text="Carrier Details"), MagicMock(text="Rate Breakdown")]
    with patch("docx.Document", return_value=fake_doc):
        with patch.object(ingestion_service, "_add_semantic_structure", wraps=ingestion_service._add_semantic_structure) as add_semantic, \
             patch.object(ingestion_service, "_chunk_text_with_title_grouping", wraps=ingestion_service._chunk_text_with_title_grouping) as chunker:
            chunks = ingestion_service.process_file(docx_bytes, filename)
            assert isinstance(chunks, list)
            assert all(isinstance(c, Chunk) for c in chunks)
            assert add_semantic.called
            assert chunker.called
            assert any("Carrier Details" in c.text for c in chunks)

def test_process_file_pdf_happy_path(ingestion_service):
    pdf_bytes = b"dummy"
    filename = "test.pdf"
    fake_page = MagicMock()
    fake_page.get_text.return_value = "Carrier Details\nJohn Doe"
    fake_doc = [fake_page, fake_page]
    with patch("pymupdf.open", return_value=fake_doc):
        with patch.object(ingestion_service, "_add_semantic_structure", wraps=ingestion_service._add_semantic_structure) as add_semantic, \
             patch.object(ingestion_service, "_chunk_text_with_title_grouping", wraps=ingestion_service._chunk_text_with_title_grouping) as chunker:
            chunks = ingestion_service.process_file(pdf_bytes, filename)
            assert isinstance(chunks, list)
            assert all(isinstance(c, Chunk) for c in chunks)
            assert add_semantic.called
            assert chunker.called
            assert any("Carrier Details" in c.text for c in chunks)

def test_process_file_unsupported_type_returns_empty(ingestion_service):
    content = b"irrelevant"
    filename = "test.xlsx"
    chunks = ingestion_service.process_file(content, filename)
    assert chunks == []

def test_process_file_error_handling_returns_empty(ingestion_service):
    # Simulate error in _extract_text_from_pdf
    filename = "test.pdf"
    with patch.object(ingestion_service, "_extract_text_from_pdf", side_effect=Exception("fail")):
        chunks = ingestion_service.process_file(b"irrelevant", filename)
        assert chunks == []

def test_add_semantic_structure_adds_headers_and_cleans_whitespace(ingestion_service):
    text = "Carrier Details\n\n\nRate Breakdown\n\n\n\n"
    result = ingestion_service._add_semantic_structure(text)
    assert "## Carrier Details" in result
    assert "## Rate Breakdown" in result
    assert "\n\n\n" not in result

def test_chunk_text_with_title_grouping_groups_sections_and_chunks(ingestion_service):
    text = "\n## Carrier Details\nJohn\n## Rate Breakdown\n$1000\n"
    filename = "file.txt"
    # Patch text_splitter to split into 1 chunk per group
    ingestion_service.text_splitter = MagicMock()
    ingestion_service.text_splitter.split_text.side_effect = lambda t: [t]
    chunks = ingestion_service._chunk_text_with_title_grouping(text, filename)
    assert len(chunks) == 2
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].metadata.filename == filename
    assert "Carrier Details" in chunks[0].metadata.source
    assert "Rate Breakdown" in chunks[1].metadata.source

def test_chunk_text_with_title_grouping_handles_empty_and_general_sections(ingestion_service):
    text = "Some intro text\n## Carrier Details\nJohn"
    filename = "file.txt"
    ingestion_service.text_splitter = MagicMock()
    ingestion_service.text_splitter.split_text.side_effect = lambda t: [t]
    chunks = ingestion_service._chunk_text_with_title_grouping(text, filename)
    assert any(c.metadata.source.endswith("Carrier Details") for c in chunks)
    assert any(c.metadata.source.endswith("General") for c in chunks)

def test_extract_section_name_returns_section(ingestion_service):
    chunk_text = "## Carrier Details\nJohn"
    section = ingestion_service._extract_section_name(chunk_text)
    assert section == "Carrier Details"

def test_extract_section_name_returns_general_when_no_header(ingestion_service):
    chunk_text = "No header here"
    section = ingestion_service._extract_section_name(chunk_text)
    assert section == "General"

def test_clean_chunk_removes_excessive_whitespace_and_page_markers(ingestion_service):
    chunk_text = "\n\n\n### Page 1\n\n\nSome text\n\n\n"
    cleaned = ingestion_service._clean_chunk(chunk_text)
    assert "### Page 1" not in cleaned
    assert cleaned.startswith("Some text")

def test_chunk_text_with_title_grouping_boundary_conditions(ingestion_service):
    # Edge: empty text
    filename = "file.txt"
    ingestion_service.text_splitter = MagicMock()
    ingestion_service.text_splitter.split_text.side_effect = lambda t: [t]
    chunks = ingestion_service._chunk_text_with_title_grouping("", filename)
    assert chunks == []

    # Edge: only headers, no content
    text = "\n## Carrier Details\n\n## Rate Breakdown\n"
    chunks = ingestion_service._chunk_text_with_title_grouping(text, filename)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert len(chunks) == 2

def test_process_file_txt_unicode_and_boundary(ingestion_service):
    # Unicode and boundary: very short file
    content = "Carrier Details\nJosé\n".encode("utf-8")
    filename = "unicode.txt"
    with patch.object(ingestion_service, "_add_semantic_structure", wraps=ingestion_service._add_semantic_structure) as add_semantic, \
         patch.object(ingestion_service, "_chunk_text_with_title_grouping", wraps=ingestion_service._chunk_text_with_title_grouping) as chunker:
        chunks = ingestion_service.process_file(content, filename)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)
        assert any("José" in c.text for c in chunks)

def test_chunk_text_with_title_grouping_reconciliation(ingestion_service):
    # Reconciliation: two equivalent paths should yield same output
    text1 = "\n## Carrier Details\nJohn\n## Rate Breakdown\n$1000\n"
    text2 = "\n## Carrier Details\nJohn\n\n## Rate Breakdown\n$1000\n"
    filename = "file.txt"
    ingestion_service.text_splitter = MagicMock()
    ingestion_service.text_splitter.split_text.side_effect = lambda t: [t]
    chunks1 = ingestion_service._chunk_text_with_title_grouping(text1, filename)
    chunks2 = ingestion_service._chunk_text_with_title_grouping(text2, filename)
    assert [c.text for c in chunks1] == [c.text for c in chunks2]
    assert [c.metadata.source for c in chunks1] == [c.metadata.source for c in chunks2]
