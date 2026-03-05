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

@pytest.fixture
def minimal_pdf_bytes():
    # Minimal valid PDF bytes for PyMuPDF
    return b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\nstartxref\n0\n%%EOF"

@pytest.fixture
def minimal_docx_bytes():
    # Minimal valid DOCX bytes (not a real docx, but enough for mocking)
    return b"PK\x03\x04fake-docx-content"

@pytest.fixture
def simple_txt_bytes():
    return b"Carrier Details\nJohn Doe\nRate Breakdown\n$1000\n"

@pytest.fixture
def complex_txt_bytes():
    return (
        b"Carrier Details\nJohn Doe\n\n"
        b"Driver Details\nJane Smith\n\n"
        b"Rate Breakdown\n$1000\n\n"
        b"Pickup\nLocation A\n\n"
        b"Drop\nLocation B\n\n"
        b"Standing Instructions\nCall before delivery.\n"
    )

def make_chunk(text, filename, chunk_id, section):
    return Chunk(
        text=text,
        metadata=DocumentMetadata(
            filename=filename,
            chunk_id=chunk_id,
            source=f"{filename} - {section}",
            chunk_type="text"
        )
    )

def test_process_file_happy_path_txt(monkeypatch, ingestion_service, simple_txt_bytes):
    # Patch text_splitter to return predictable chunks
    monkeypatch.setattr(ingestion_service, "text_splitter", MagicMock())
    ingestion_service.text_splitter.split_text = lambda text: [text]
    filename = "test.txt"
    chunks = ingestion_service.process_file(simple_txt_bytes, filename)
    assert len(chunks) == 1
    assert "Carrier Details" in chunks[0].text
    assert chunks[0].metadata.filename == filename
    assert chunks[0].metadata.chunk_id == 0

def test_process_file_happy_path_docx(monkeypatch, ingestion_service, minimal_docx_bytes):
    # Patch docx.Document and text_splitter
    fake_doc = MagicMock()
    fake_doc.paragraphs = [MagicMock(text="Carrier Details"), MagicMock(text="Rate Breakdown")]
    monkeypatch.setattr("docx.Document", lambda stream: fake_doc)
    monkeypatch.setattr(ingestion_service, "text_splitter", MagicMock())
    ingestion_service.text_splitter.split_text = lambda text: [text]
    filename = "test.docx"
    chunks = ingestion_service.process_file(minimal_docx_bytes, filename)
    assert len(chunks) == 1
    assert "Carrier Details" in chunks[0].text
    assert "Rate Breakdown" in chunks[0].text
    assert chunks[0].metadata.filename == filename

def test_process_file_happy_path_pdf(monkeypatch, ingestion_service, minimal_pdf_bytes):
    # Patch pymupdf.open and text_splitter
    fake_page = MagicMock()
    fake_page.get_text.return_value = "Carrier Details\nJohn Doe"
    fake_doc = [fake_page]
    monkeypatch.setattr("pymupdf.open", lambda stream, filetype: fake_doc)
    monkeypatch.setattr(ingestion_service, "text_splitter", MagicMock())
    ingestion_service.text_splitter.split_text = lambda text: [text]
    filename = "test.pdf"
    chunks = ingestion_service.process_file(minimal_pdf_bytes, filename)
    assert len(chunks) == 1
    assert "Carrier Details" in chunks[0].text
    assert chunks[0].metadata.filename == filename

def test_process_file_unsupported_type(ingestion_service):
    result = ingestion_service.process_file(b"irrelevant", "test.xlsx")
    assert result == []

def test_process_file_txt_decoding_error(monkeypatch, ingestion_service):
    # Simulate decode error
    bad_bytes = b"\xff\xfe\xfd"
    result = ingestion_service.process_file(bad_bytes, "test.txt")
    assert result == []

def test_process_file_pdf_extraction_error(monkeypatch, ingestion_service, minimal_pdf_bytes):
    # Simulate pymupdf.open raising
    monkeypatch.setattr("pymupdf.open", lambda *a, **kw: (_ for _ in ()).throw(Exception("PDF error")))
    result = ingestion_service.process_file(minimal_pdf_bytes, "test.pdf")
    assert result == []

def test_process_file_docx_extraction_error(monkeypatch, ingestion_service, minimal_docx_bytes):
    # Simulate docx.Document raising
    monkeypatch.setattr("docx.Document", lambda *a, **kw: (_ for _ in ()).throw(Exception("DOCX error")))
    result = ingestion_service.process_file(minimal_docx_bytes, "test.docx")
    assert result == []

def test_add_semantic_structure_adds_headers(ingestion_service):
    text = "Carrier Details\nSome info\nRate Breakdown\n$1000"
    structured = ingestion_service._add_semantic_structure(text)
    assert "\n## Carrier Details\n" in structured
    assert "\n## Rate Breakdown\n" in structured

def test_add_semantic_structure_preserves_structure(ingestion_service):
    text = "Carrier Details\n\n\n\nRate Breakdown"
    structured = ingestion_service._add_semantic_structure(text)
    assert "\n## Carrier Details\n" in structured
    assert "\n## Rate Breakdown\n" in structured
    assert "\n\n" in structured
    assert "\n\n\n" not in structured

def test_chunk_text_with_title_grouping_groups_sections(monkeypatch, ingestion_service):
    # Patch text_splitter to split on double newlines
    ingestion_service.text_splitter.split_text = lambda text: [text]
    text = (
        "\n## Carrier Details\nJohn Doe\n"
        "\n## Driver Details\nJane Smith\n"
        "\n## Rate Breakdown\n$1000\n"
        "\n## Pickup\nLocation A\n"
        "\n## Drop\nLocation B\n"
    )
    filename = "test.txt"
    chunks = ingestion_service._chunk_text_with_title_grouping(text, filename)
    # Carrier Details and Driver Details should be grouped
    assert any("Carrier Details" in c.text and "Driver Details" in c.text for c in chunks)
    # Pickup and Drop should be grouped
    assert any("Pickup" in c.text and "Drop" in c.text for c in chunks)
    # Rate Breakdown should be its own group or with Agreed Amount (not present)
    assert any("Rate Breakdown" in c.text for c in chunks)

def test_chunk_text_with_title_grouping_handles_empty(monkeypatch, ingestion_service):
    ingestion_service.text_splitter.split_text = lambda text: [text]
    chunks = ingestion_service._chunk_text_with_title_grouping("", "file.txt")
    assert chunks == []

def test_chunk_text_with_title_grouping_boundary(monkeypatch, ingestion_service):
    # Only one section, should produce one chunk
    ingestion_service.text_splitter.split_text = lambda text: [text]
    text = "\n## Carrier Details\nJohn Doe"
    chunks = ingestion_service._chunk_text_with_title_grouping(text, "file.txt")
    assert len(chunks) == 1
    assert "Carrier Details" in chunks[0].text

def test_clean_chunk_removes_excess_whitespace_and_page_markers(ingestion_service):
    text = "\n\n\n### Page 1\n\n\nSome text\n\n\n"
    cleaned = ingestion_service._clean_chunk(text)
    assert "Page 1" not in cleaned
    assert cleaned.startswith("Some text")
    assert "\n\n\n" not in cleaned

def test_extract_section_name_finds_header(ingestion_service):
    text = "## Carrier Details\nSome info"
    section = ingestion_service._extract_section_name(text)
    assert section == "Carrier Details"

def test_extract_section_name_default(ingestion_service):
    text = "No header here"
    section = ingestion_service._extract_section_name(text)
    assert section == "General"

def test_reconciliation_txt_vs_docx(monkeypatch, ingestion_service):
    # Simulate equivalent content in TXT and DOCX
    txt_bytes = b"Carrier Details\nJohn Doe\nRate Breakdown\n$1000"
    # Patch docx.Document to return same paragraphs
    fake_doc = MagicMock()
    fake_doc.paragraphs = [MagicMock(text="Carrier Details"), MagicMock(text="John Doe"), MagicMock(text="Rate Breakdown"), MagicMock(text="$1000")]
    monkeypatch.setattr("docx.Document", lambda stream: fake_doc)
    # Patch text_splitter to split on all text
    ingestion_service.text_splitter.split_text = lambda text: [text]
    txt_chunks = ingestion_service.process_file(txt_bytes, "file.txt")
    docx_chunks = ingestion_service.process_file(b"irrelevant", "file.docx")
    # The text content of the chunks should be equivalent
    assert len(txt_chunks) == len(docx_chunks)
    for t, d in zip(txt_chunks, docx_chunks):
        assert t.text.replace("\n", "").replace(" ", "") == d.text.replace("\n", "").replace(" ", "")

def test_reconciliation_pdf_vs_txt(monkeypatch, ingestion_service):
    # Simulate equivalent content in PDF and TXT
    txt_bytes = b"Carrier Details\nJohn Doe\nRate Breakdown\n$1000"
    fake_page = MagicMock()
    fake_page.get_text.return_value = "Carrier Details\nJohn Doe\nRate Breakdown\n$1000"
    fake_doc = [fake_page]
    monkeypatch.setattr("pymupdf.open", lambda stream, filetype: fake_doc)
    ingestion_service.text_splitter.split_text = lambda text: [text]
    txt_chunks = ingestion_service.process_file(txt_bytes, "file.txt")
    pdf_chunks = ingestion_service.process_file(b"irrelevant", "file.pdf")
    assert len(txt_chunks) == len(pdf_chunks)
    for t, p in zip(txt_chunks, pdf_chunks):
        assert t.text.replace("\n", "").replace(" ", "") == p.text.replace("\n", "").replace(" ", "")

def test_reconciliation_grouping(monkeypatch, ingestion_service):
    # Test that grouping logic is consistent across different section orderings
    ingestion_service.text_splitter.split_text = lambda text: [text]
    text1 = (
        "\n## Carrier Details\nJohn Doe\n"
        "\n## Driver Details\nJane Smith\n"
        "\n## Rate Breakdown\n$1000\n"
    )
    text2 = (
        "\n## Driver Details\nJane Smith\n"
        "\n## Carrier Details\nJohn Doe\n"
        "\n## Rate Breakdown\n$1000\n"
    )
    chunks1 = ingestion_service._chunk_text_with_title_grouping(text1, "file.txt")
    chunks2 = ingestion_service._chunk_text_with_title_grouping(text2, "file.txt")
    # The grouped chunk texts should be equivalent regardless of order
    sorted1 = sorted([c.text.replace("\n", "").replace(" ", "") for c in chunks1])
    sorted2 = sorted([c.text.replace("\n", "").replace(" ", "") for c in chunks2])
    assert sorted1 == sorted2
