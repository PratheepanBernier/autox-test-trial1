import pytest
from unittest.mock import patch, MagicMock, call
from streamlit_app import DocumentIngestionService, Chunk, DocumentMetadata

@pytest.fixture
def ingestion_service():
    return DocumentIngestionService()

def make_pdf_bytes():
    # Return dummy bytes for a PDF file
    return b"%PDF-1.4\n%Fake PDF content\n"

def make_docx_bytes():
    # Return dummy bytes for a DOCX file
    return b"PK\x03\x04Fake DOCX content"

def make_txt_bytes():
    return b"Carrier Details\nRate Breakdown\nPickup\nDrop\nCommodity\nSpecial Instructions\n"

def test_process_file_pdf_happy_path(ingestion_service):
    fake_pdf_bytes = make_pdf_bytes()
    filename = "test.pdf"
    # Mock pymupdf.open and page.get_text
    with patch("streamlit_app.pymupdf.open") as mock_open:
        mock_doc = MagicMock()
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Carrier Details: ABC\nRate Breakdown: $1000"
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Pickup: Location X\nDrop: Location Y"
        mock_doc.__iter__.return_value = iter([mock_page1, mock_page2])
        mock_open.return_value = mock_doc

        chunks = ingestion_service.process_file(fake_pdf_bytes, filename)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)
        # Should contain semantic markers
        texts = [c.text for c in chunks]
        assert any("## Carrier Details" in t for t in texts)
        assert any("## Rate Breakdown" in t for t in texts)
        assert any("## Pickup" in t for t in texts)
        assert any("## Drop" in t for t in texts)
        # Metadata checks
        for i, c in enumerate(chunks):
            assert c.metadata.filename == filename
            assert c.metadata.chunk_id == i
            assert c.metadata.chunk_type == "text"
            assert filename in c.metadata.source

def test_process_file_docx_happy_path(ingestion_service):
    fake_docx_bytes = make_docx_bytes()
    filename = "test.docx"
    # Mock docx.Document and paragraphs
    with patch("streamlit_app.docx.Document") as mock_docx:
        mock_doc = MagicMock()
        mock_para1 = MagicMock()
        mock_para1.text = "Carrier Details: DEF"
        mock_para2 = MagicMock()
        mock_para2.text = "Pickup: Location Z"
        mock_doc.paragraphs = [mock_para1, mock_para2]
        mock_docx.return_value = mock_doc

        chunks = ingestion_service.process_file(fake_docx_bytes, filename)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)
        texts = [c.text for c in chunks]
        assert any("## Carrier Details" in t for t in texts)
        assert any("## Pickup" in t for t in texts)

def test_process_file_txt_happy_path(ingestion_service):
    fake_txt_bytes = make_txt_bytes()
    filename = "test.txt"
    chunks = ingestion_service.process_file(fake_txt_bytes, filename)
    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks)
    texts = [c.text for c in chunks]
    assert any("## Carrier Details" in t for t in texts)
    assert any("## Rate Breakdown" in t for t in texts)
    assert any("## Pickup" in t for t in texts)
    assert any("## Drop" in t for t in texts)
    assert any("## Commodity" in t for t in texts)
    assert any("## Special Instructions" in t for t in texts)

def test_process_file_empty_txt(ingestion_service):
    filename = "empty.txt"
    chunks = ingestion_service.process_file(b"", filename)
    assert isinstance(chunks, list)
    # Should still return at least one chunk (possibly empty string)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)
    # All chunk texts should be empty or whitespace
    assert all(isinstance(c.text, str) for c in chunks)

def test_process_file_unsupported_extension(ingestion_service):
    filename = "test.csv"
    # Should not raise, but produce empty chunks
    chunks = ingestion_service.process_file(b"irrelevant", filename)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)

def test_process_file_semantic_markers_case_insensitive(ingestion_service):
    # Test that markers are added regardless of case
    content = b"carrier details\nRATE BREAKDOWN\npickup\nDrop\ncommodity\nspecial instructions"
    filename = "test.txt"
    chunks = ingestion_service.process_file(content, filename)
    texts = [c.text for c in chunks]
    # All markers should be present with '##' prefix
    assert any("## Carrier Details" in t for t in texts)
    assert any("## Rate Breakdown" in t for t in texts)
    assert any("## Pickup" in t for t in texts)
    assert any("## Drop" in t for t in texts)
    assert any("## Commodity" in t for t in texts)
    assert any("## Special Instructions" in t for t in texts)

def test_process_file_chunking_boundary(ingestion_service):
    # Test chunking at boundary: text just at chunk_size*2
    # Use a single section marker and a long text
    section = "Carrier Details"
    text = (section + "\n" + "A" * (ingestion_service.text_splitter.chunk_size * 2))
    filename = "boundary.txt"
    content = text.encode("utf-8")
    chunks = ingestion_service.process_file(content, filename)
    # Should split into at least 1 chunk, possibly 2 if overlap
    assert len(chunks) >= 1
    # No chunk should be longer than chunk_size*2
    for c in chunks:
        assert len(c.text) <= ingestion_service.text_splitter.chunk_size * 2

def test_process_file_handles_unicode(ingestion_service):
    # Test with unicode characters
    content = "Carrier Details: 运输公司\nPickup: 北京\nDrop: 上海".encode("utf-8")
    filename = "unicode.txt"
    chunks = ingestion_service.process_file(content, filename)
    texts = [c.text for c in chunks]
    assert any("运输公司" in t for t in texts)
    assert any("北京" in t for t in texts)
    assert any("上海" in t for t in texts)

def test_process_file_handles_large_file(ingestion_service):
    # Simulate a large file by repeating a section
    section = "Carrier Details\n" + "A" * 500
    content = ("\n".join([section] * 20)).encode("utf-8")
    filename = "large.txt"
    chunks = ingestion_service.process_file(content, filename)
    # Should produce multiple chunks
    assert len(chunks) > 1
    # All chunks should have correct metadata
    for i, c in enumerate(chunks):
        assert c.metadata.filename == filename
        assert c.metadata.chunk_id == i
        assert c.metadata.chunk_type == "text"
        assert filename in c.metadata.source

def test_process_file_handles_no_section_markers(ingestion_service):
    # Text with no known section markers
    content = b"This is a logistics document with no known markers."
    filename = "plain.txt"
    chunks = ingestion_service.process_file(content, filename)
    # Should still produce at least one chunk
    assert len(chunks) >= 1
    # No '##' markers should be present
    texts = [c.text for c in chunks]
    assert not any("## " in t for t in texts)
