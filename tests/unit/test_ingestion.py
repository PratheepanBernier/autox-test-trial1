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

def test_process_file_pdf_happy_path(service):
    fake_pdf_bytes = b"%PDF-1.4 fake content"
    filename = "test.pdf"
    fake_text = "Carrier Details\nSome info\nRate Breakdown\nMore info"
    fake_pdf_doc = MagicMock()
    fake_pdf_doc.__iter__.return_value = [MagicMock(get_text=MagicMock(return_value="Page1Text")), MagicMock(get_text=MagicMock(return_value="Page2Text"))]
    with patch("pymupdf.open", return_value=fake_pdf_doc):
        with patch.object(service, "_add_semantic_structure", wraps=service._add_semantic_structure) as add_semantic:
            with patch.object(service, "_chunk_text_with_title_grouping", wraps=service._chunk_text_with_title_grouping) as chunker:
                result = service.process_file(fake_pdf_bytes, filename)
                assert isinstance(result, list)
                add_semantic.assert_called()
                chunker.assert_called()
                # Should call PDF extraction
                assert fake_pdf_doc.__iter__.called

def test_process_file_docx_happy_path(service):
    fake_docx_bytes = b"PK\x03\x04 fake docx"
    filename = "test.docx"
    fake_doc = MagicMock()
    fake_doc.paragraphs = [MagicMock(text="Carrier Details"), MagicMock(text="Some info"), MagicMock(text="Rate Breakdown")]
    with patch("docx.Document", return_value=fake_doc):
        with patch.object(service, "_add_semantic_structure", wraps=service._add_semantic_structure):
            with patch.object(service, "_chunk_text_with_title_grouping", wraps=service._chunk_text_with_title_grouping):
                result = service.process_file(fake_docx_bytes, filename)
                assert isinstance(result, list)
                assert all(isinstance(chunk, Chunk) for chunk in result)

def test_process_file_txt_happy_path(service):
    content = "Carrier Details\nSome info\nRate Breakdown\nMore info"
    filename = "test.txt"
    with patch.object(service, "_add_semantic_structure", wraps=service._add_semantic_structure):
        with patch.object(service, "_chunk_text_with_title_grouping", wraps=service._chunk_text_with_title_grouping):
            result = service.process_file(content.encode("utf-8"), filename)
            assert isinstance(result, list)
            assert all(isinstance(chunk, Chunk) for chunk in result)

def test_process_file_unsupported_type_returns_empty(service):
    filename = "test.xlsx"
    result = service.process_file(b"irrelevant", filename)
    assert result == []

def test_process_file_handles_extraction_exception(service):
    filename = "test.pdf"
    with patch.object(service, "_extract_text_from_pdf", side_effect=Exception("fail")):
        result = service.process_file(b"irrelevant", filename)
        assert result == []

def test_extract_text_from_pdf_adds_page_markers(service):
    fake_pdf_doc = MagicMock()
    fake_pdf_doc.__iter__.return_value = [
        MagicMock(get_text=MagicMock(return_value="Page1Text")),
        MagicMock(get_text=MagicMock(return_value="Page2Text"))
    ]
    with patch("pymupdf.open", return_value=fake_pdf_doc):
        result = service._extract_text_from_pdf(b"fake")
        assert "### Page 1" in result
        assert "Page1Text" in result
        assert "### Page 2" in result
        assert "Page2Text" in result

def test_extract_text_from_docx_reads_paragraphs(service):
    fake_doc = MagicMock()
    fake_doc.paragraphs = [MagicMock(text="A"), MagicMock(text="B")]
    with patch("docx.Document", return_value=fake_doc):
        result = service._extract_text_from_docx(b"irrelevant")
        assert result == "A\nB"

def test_add_semantic_structure_adds_headers_and_cleans_whitespace(service):
    text = "Carrier Details\n\n\nRate Breakdown\n\n\n\nPickup"
    result = DocumentIngestionService()._add_semantic_structure(text)
    # Should add markdown headers and reduce whitespace
    assert "\n## Carrier Details\n" in result
    assert "\n## Rate Breakdown\n" in result
    assert "\n## Pickup\n" in result
    assert "\n\n\n" not in result

def test_chunk_text_with_title_grouping_groups_sections_and_chunks(service):
    # Simulate text with two headers, one group
    text = "\n## Carrier Details\nCarrier info\n## Driver Details\nDriver info\n## Rate Breakdown\nRate info"
    filename = "file.txt"
    # Patch text_splitter to split into 2 chunks per group
    service = DocumentIngestionService()
    service.text_splitter = MagicMock()
    service.text_splitter.split_text.side_effect = lambda group: [group[:10], group[10:]] if len(group) > 10 else [group]
    result = service._chunk_text_with_title_grouping(text, filename)
    # Should produce chunks with correct metadata
    assert all(isinstance(chunk, Chunk) for chunk in result)
    assert result[0].metadata.filename == filename
    assert "Carrier Details" in result[0].metadata.source or "Driver Details" in result[0].metadata.source

def test_chunk_text_with_title_grouping_handles_no_headers(service):
    text = "No headers here, just text."
    filename = "file.txt"
    service = DocumentIngestionService()
    service.text_splitter = MagicMock()
    service.text_splitter.split_text.return_value = [text]
    result = service._chunk_text_with_title_grouping(text, filename)
    assert len(result) == 1
    assert result[0].metadata.source == f"{filename} - General"

def test_extract_section_name_extracts_header(service):
    text = "## Carrier Details\nSome info"
    result = DocumentIngestionService()._extract_section_name(text)
    assert result == "Carrier Details"

def test_extract_section_name_returns_general_if_no_header(service):
    text = "No header here"
    result = DocumentIngestionService()._extract_section_name(text)
    assert result == "General"

@pytest.mark.parametrize("chunk_text,expected", [
    ("Text\n\n\nMore", "Text\n\nMore"),
    ("Text  with   spaces", "Text with spaces"),
    ("### Page 1\n", ""),
    ("### Page 1\nContent", "### Page 1\nContent"),
])
def test_clean_chunk_removes_excess_whitespace_and_page_markers(chunk_text, expected):
    result = DocumentIngestionService()._clean_chunk(chunk_text)
    assert result == expected

def test_chunk_text_with_title_grouping_empty_group_skipped(service):
    text = "\n## Carrier Details\n\n## Rate Breakdown\n"
    filename = "file.txt"
    service.text_splitter = MagicMock()
    service.text_splitter.split_text.return_value = []
    result = service._chunk_text_with_title_grouping(text, filename)
    assert result == []

def test_chunk_text_with_title_grouping_boundary_conditions(service):
    # Test with minimal text and only one header
    text = "\n## Carrier Details\n"
    filename = "file.txt"
    service.text_splitter = MagicMock()
    service.text_splitter.split_text.return_value = ["\n## Carrier Details\n"]
    result = service._chunk_text_with_title_grouping(text, filename)
    assert len(result) == 1
    assert result[0].metadata.source == f"{filename} - Carrier Details"

def test_process_file_empty_txt_returns_empty_chunks(service):
    filename = "empty.txt"
    result = service.process_file(b"", filename)
    assert isinstance(result, list)
    # Should still return at least one chunk (empty string chunk)
    # unless chunking removes it
    # Let's check that all chunks are empty or the list is empty
    assert all(chunk.text == "" for chunk in result) or result == []

def test_process_file_handles_unicode_decode_error(service):
    filename = "bad.txt"
    # bytes that can't be decoded as utf-8
    bad_bytes = b"\xff\xfe\xfd"
    result = DocumentIngestionService().process_file(bad_bytes, filename)
    assert result == []

def test_chunk_text_with_title_grouping_reconciliation(service):
    # If two equivalent texts with different section order, output chunk count should match
    text1 = "\n## Carrier Details\nA\n## Rate Breakdown\nB"
    text2 = "\n## Rate Breakdown\nB\n## Carrier Details\nA"
    filename = "file.txt"
    service1 = DocumentIngestionService()
    service2 = DocumentIngestionService()
    service1.text_splitter = MagicMock()
    service2.text_splitter = MagicMock()
    service1.text_splitter.split_text.return_value = ["chunk1", "chunk2"]
    service2.text_splitter.split_text.return_value = ["chunk1", "chunk2"]
    result1 = service1._chunk_text_with_title_grouping(text1, filename)
    result2 = service2._chunk_text_with_title_grouping(text2, filename)
    assert len(result1) == len(result2)
    # The chunk text should be the same if the content is the same
    assert [c.text for c in result1] == [c.text for c in result2]
