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
    monkeypatch.setattr("core.config.settings", DummySettings())

def make_metadata(filename, chunk_id, section):
    return DocumentMetadata(
        filename=filename,
        chunk_id=chunk_id,
        source=f"{filename} - {section}",
        chunk_type="text"
    )

def test_process_file_happy_path_pdf(service):
    # Mock PyMuPDF
    fake_page = MagicMock()
    fake_page.get_text.return_value = "Carrier Details\nSome carrier info.\nRate Breakdown\n$1000"
    fake_doc = [fake_page]
    with patch("pymupdf.open", return_value=fake_doc):
        with patch.object(service, "_chunk_text_with_title_grouping", wraps=service._chunk_text_with_title_grouping) as chunker:
            # Patch text_splitter to avoid actual splitting
            service.text_splitter.split_text = lambda x: [x]
            # Patch _add_semantic_structure to pass through
            service._add_semantic_structure = lambda x: x
            result = service.process_file(b"pdfbytes", "test.pdf")
            assert isinstance(result, list)
            assert all(isinstance(c, Chunk) for c in result)
            assert len(result) == 1
            assert "Carrier Details" in result[0].text

def test_process_file_happy_path_docx(service):
    # Mock docx.Document
    fake_doc = MagicMock()
    fake_doc.paragraphs = [MagicMock(text="Carrier Details"), MagicMock(text="Some carrier info.")]
    with patch("docx.Document", return_value=fake_doc):
        with patch.object(service, "_chunk_text_with_title_grouping", wraps=service._chunk_text_with_title_grouping):
            service.text_splitter.split_text = lambda x: [x]
            service._add_semantic_structure = lambda x: x
            result = service.process_file(b"docxbytes", "test.docx")
            assert isinstance(result, list)
            assert all(isinstance(c, Chunk) for c in result)
            assert len(result) == 1
            assert "Carrier Details" in result[0].text

def test_process_file_happy_path_txt(service):
    service.text_splitter.split_text = lambda x: [x]
    service._add_semantic_structure = lambda x: x
    content = "Carrier Details\nSome carrier info.\nRate Breakdown\n$1000"
    result = service.process_file(content.encode("utf-8"), "test.txt")
    assert isinstance(result, list)
    assert all(isinstance(c, Chunk) for c in result)
    assert len(result) == 1
    assert "Carrier Details" in result[0].text

def test_process_file_unsupported_type_returns_empty(service):
    result = service.process_file(b"irrelevant", "test.xlsx")
    assert result == []

def test_process_file_pdf_extraction_error_returns_empty(service):
    with patch("pymupdf.open", side_effect=Exception("bad pdf")):
        result = service.process_file(b"badbytes", "bad.pdf")
        assert result == []

def test_process_file_docx_extraction_error_returns_empty(service):
    with patch("docx.Document", side_effect=Exception("bad docx")):
        result = service.process_file(b"badbytes", "bad.docx")
        assert result == []

def test_add_semantic_structure_adds_headers(service):
    text = "Carrier Details\nSome info\nRate Breakdown\n$1000"
    structured = service._add_semantic_structure(text)
    assert "\n## Carrier Details\n" in structured
    assert "\n## Rate Breakdown\n" in structured

def test_add_semantic_structure_handles_excessive_whitespace(service):
    text = "Carrier Details\n\n\n\nSome info\n\n\nRate Breakdown"
    structured = service._add_semantic_structure(text)
    assert "\n\n" in structured
    assert "\n\n\n" not in structured

def test_chunk_text_with_title_grouping_groups_sections(service):
    # Simulate text with two groupable sections and one standalone
    text = "\n## Carrier Details\nCarrier X\n## Driver Details\nDriver Y\n## Reference ID\n123"
    service.text_splitter.split_text = lambda x: [x]
    chunks = service._chunk_text_with_title_grouping(text, "file.txt")
    assert len(chunks) == 2
    assert "Carrier Details" in chunks[0].text
    assert "Driver Details" in chunks[0].text
    assert "Reference ID" in chunks[1].text

def test_chunk_text_with_title_grouping_handles_empty_sections(service):
    text = "\n## Carrier Details\n\n## Driver Details\n\n"
    service.text_splitter.split_text = lambda x: [x]
    chunks = service._chunk_text_with_title_grouping(text, "file.txt")
    assert len(chunks) == 1
    assert "Carrier Details" in chunks[0].text

def test_chunk_text_with_title_grouping_large_section_splits(service):
    # Simulate splitting into multiple chunks
    text = "\n## Carrier Details\n" + ("A" * 300)
    service.text_splitter.split_text = lambda x: [x[:100], x[100:200], x[200:]]
    chunks = service._chunk_text_with_title_grouping(text, "file.txt")
    assert len(chunks) == 3
    for i, chunk in enumerate(chunks):
        assert chunk.metadata.chunk_id == i

def test_extract_section_name_with_header(service):
    text = "## Carrier Details\nSome info"
    section = service._extract_section_name(text)
    assert section == "Carrier Details"

def test_extract_section_name_without_header(service):
    text = "Some info"
    section = service._extract_section_name(text)
    assert section == "General"

def test_clean_chunk_removes_excessive_whitespace_and_page_markers(service):
    text = "\n\n\n### Page 1\n\n\nSome info\n\n\n"
    cleaned = service._clean_chunk(text)
    assert "### Page 1" not in cleaned
    assert cleaned.startswith("Some info")

def test_clean_chunk_preserves_content_within_page_marker(service):
    text = "### Page 1\nSome info"
    cleaned = service._clean_chunk(text)
    assert "Some info" in cleaned

def test_process_file_boundary_empty_txt(service):
    service.text_splitter.split_text = lambda x: [x]
    service._add_semantic_structure = lambda x: x
    result = service.process_file(b"", "empty.txt")
    assert isinstance(result, list)
    assert result == []

def test_process_file_boundary_empty_docx(service):
    fake_doc = MagicMock()
    fake_doc.paragraphs = []
    with patch("docx.Document", return_value=fake_doc):
        service.text_splitter.split_text = lambda x: [x]
        service._add_semantic_structure = lambda x: x
        result = service.process_file(b"", "empty.docx")
        assert isinstance(result, list)
        assert result == []

def test_process_file_boundary_empty_pdf(service):
    fake_doc = []
    with patch("pymupdf.open", return_value=fake_doc):
        service.text_splitter.split_text = lambda x: [x]
        service._add_semantic_structure = lambda x: x
        result = service.process_file(b"", "empty.pdf")
        assert isinstance(result, list)
        assert result == []

def test_chunk_text_with_title_grouping_no_headers(service):
    text = "Just some text without headers."
    service.text_splitter.split_text = lambda x: [x]
    chunks = service._chunk_text_with_title_grouping(text, "file.txt")
    assert len(chunks) == 1
    assert "Just some text" in chunks[0].text

def test_chunk_text_with_title_grouping_multiple_unrelated_headers(service):
    text = "\n## Section A\nA\n## Section B\nB\n## Section C\nC"
    service.text_splitter.split_text = lambda x: [x]
    chunks = service._chunk_text_with_title_grouping(text, "file.txt")
    assert len(chunks) == 3
    assert "Section A" in chunks[0].text
    assert "Section B" in chunks[1].text
    assert "Section C" in chunks[2].text

def test_chunk_text_with_title_grouping_group_switching(service):
    # Carrier Details and Driver Details are grouped, then Reference ID is standalone
    text = "\n## Carrier Details\nCarrier X\n## Driver Details\nDriver Y\n## Reference ID\n123"
    service.text_splitter.split_text = lambda x: [x]
    chunks = service._chunk_text_with_title_grouping(text, "file.txt")
    assert len(chunks) == 2
    assert "Carrier Details" in chunks[0].text
    assert "Driver Details" in chunks[0].text
    assert "Reference ID" in chunks[1].text

def test_chunk_text_with_title_grouping_empty_input(service):
    text = ""
    service.text_splitter.split_text = lambda x: [x]
    chunks = service._chunk_text_with_title_grouping(text, "file.txt")
    assert chunks == []
