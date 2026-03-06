import pytest
from unittest.mock import patch, MagicMock, call
from streamlit_app import DocumentIngestionService, Chunk, DocumentMetadata

@pytest.fixture
def ingestion_service():
    return DocumentIngestionService()

@pytest.fixture
def sample_pdf_bytes():
    # Simulate a PDF file as bytes
    return b"%PDF-1.4 sample pdf content"

@pytest.fixture
def sample_docx_bytes():
    # Simulate a DOCX file as bytes
    return b"PK\x03\x04 sample docx content"

@pytest.fixture
def sample_txt_bytes():
    return b"Carrier Details\nPickup\nDrop\nCommodity\nSpecial Instructions\n"

def test_process_file_pdf_happy_path(ingestion_service, sample_pdf_bytes):
    # Mock pymupdf.open and page.get_text
    mock_doc = [MagicMock(), MagicMock()]
    mock_doc[0].get_text.return_value = "Carrier Details: ABC\nPickup: NYC"
    mock_doc[1].get_text.return_value = "Drop: LA\nCommodity: Apples"
    with patch("streamlit_app.pymupdf.open", return_value=mock_doc):
        chunks = ingestion_service.process_file(sample_pdf_bytes, "test.pdf")
    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks)
    # Should contain semantic markers
    joined = "\n".join([c.text for c in chunks])
    assert "## Carrier Details" in joined
    assert "## Pickup" in joined
    assert "## Drop" in joined
    assert "## Commodity" in joined
    assert all(c.metadata.filename == "test.pdf" for c in chunks)

def test_process_file_docx_happy_path(ingestion_service, sample_docx_bytes):
    # Mock docx.Document and paragraphs
    mock_doc = MagicMock()
    mock_doc.paragraphs = [MagicMock(text="Carrier Details: DEF"), MagicMock(text="Pickup: SF")]
    with patch("streamlit_app.docx.Document", return_value=mock_doc):
        chunks = ingestion_service.process_file(sample_docx_bytes, "test.docx")
    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks)
    joined = "\n".join([c.text for c in chunks])
    assert "## Carrier Details" in joined
    assert "## Pickup" in joined
    assert all(c.metadata.filename == "test.docx" for c in chunks)

def test_process_file_txt_happy_path(ingestion_service, sample_txt_bytes):
    chunks = ingestion_service.process_file(sample_txt_bytes, "test.txt")
    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks)
    joined = "\n".join([c.text for c in chunks])
    assert "## Carrier Details" in joined
    assert "## Pickup" in joined
    assert "## Drop" in joined
    assert "## Commodity" in joined
    assert "## Special Instructions" in joined
    assert all(c.metadata.filename == "test.txt" for c in chunks)

def test_process_file_empty_txt(ingestion_service):
    chunks = ingestion_service.process_file(b"", "empty.txt")
    # Should still return at least one chunk (possibly empty string)
    assert isinstance(chunks, list)
    assert len(chunks) == 1
    assert chunks[0].text == ""
    assert chunks[0].metadata.filename == "empty.txt"

def test_process_file_unsupported_extension(ingestion_service):
    # Should not raise, just return empty chunk
    chunks = ingestion_service.process_file(b"irrelevant", "file.xyz")
    assert isinstance(chunks, list)
    assert len(chunks) == 1
    assert chunks[0].text == ""
    assert chunks[0].metadata.filename == "file.xyz"

def test_process_file_pdf_with_multiple_pages_and_section_markers(ingestion_service, sample_pdf_bytes):
    # Simulate a PDF with multiple pages and repeated section headers
    mock_doc = [MagicMock(), MagicMock()]
    mock_doc[0].get_text.return_value = "Carrier Details\nCarrier Details\nPickup"
    mock_doc[1].get_text.return_value = "Drop\nCommodity\nSpecial Instructions"
    with patch("streamlit_app.pymupdf.open", return_value=mock_doc):
        chunks = ingestion_service.process_file(sample_pdf_bytes, "multi.pdf")
    joined = "\n".join([c.text for c in chunks])
    # Each section marker should be present only once per occurrence
    assert joined.count("## Carrier Details") >= 2
    assert "## Special Instructions" in joined

def test_process_file_docx_with_no_paragraphs(ingestion_service, sample_docx_bytes):
    mock_doc = MagicMock()
    mock_doc.paragraphs = []
    with patch("streamlit_app.docx.Document", return_value=mock_doc):
        chunks = ingestion_service.process_file(sample_docx_bytes, "empty.docx")
    assert isinstance(chunks, list)
    assert len(chunks) == 1
    assert chunks[0].text == ""
    assert chunks[0].metadata.filename == "empty.docx"

def test_process_file_txt_with_unicode(ingestion_service):
    # Test with unicode characters
    content = "Carrier Details\nPickup – München\nDrop – 東京\nCommodity – Café\nSpecial Instructions – 🚚".encode("utf-8")
    chunks = ingestion_service.process_file(content, "unicode.txt")
    joined = "\n".join([c.text for c in chunks])
    assert "München" in joined
    assert "東京" in joined
    assert "Café" in joined
    assert "🚚" in joined

def test_process_file_chunk_metadata_fields(ingestion_service, sample_txt_bytes):
    chunks = ingestion_service.process_file(sample_txt_bytes, "meta.txt")
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata
        assert meta.filename == "meta.txt"
        assert meta.chunk_id == i
        assert meta.source == f"meta.txt - Part {i+1}"
        assert meta.chunk_type == "text"
        assert meta.page_number is None

def test_process_file_chunking_boundary(ingestion_service):
    # Test chunking at boundary: text just at chunk_size*2
    text = "Carrier Details\n" + ("A" * (ingestion_service.text_splitter.chunk_size))
    content = text.encode("utf-8")
    chunks = ingestion_service.process_file(content, "boundary.txt")
    # Should not produce more than 2 chunks
    assert 1 <= len(chunks) <= 2
    assert any("Carrier Details" in c.text for c in chunks)
    assert all(isinstance(c, Chunk) for c in chunks)
