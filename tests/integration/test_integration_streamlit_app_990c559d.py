# source_hash: 5fab573753f93532
# import_target: streamlit_app
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "vs" not in st.session_state:
    st.session_state["vs"] = None

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import io
import types
import pytest
import builtins

import streamlit_app

import streamlit as st

from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def reset_streamlit_session_state():
    st.session_state.clear()
    # Bootstrap required session_state keys
    st.session_state["messages"] = []
    st.session_state["vs"] = None
    yield
    st.session_state.clear()

@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("QA_MODEL", "test-qa-model")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setenv("CHUNK_SIZE", "1000")
    monkeypatch.setenv("CHUNK_OVERLAP", "200")
    monkeypatch.setenv("TOP_K", "4")

@pytest.fixture
def mock_streamlit(monkeypatch):
    # Patch all Streamlit UI functions to no-op or record calls
    for fn in [
        "set_page_config", "title", "info", "error", "stop", "tabs", "header", "file_uploader",
        "button", "spinner", "success", "text_input", "markdown", "metric", "expander", "caption",
        "write", "divider", "warning", "json"
    ]:
        monkeypatch.setattr(st, fn, MagicMock())
    # Patch st.session_state to a dict-like object
    monkeypatch.setattr(st, "session_state", st.session_state)

@pytest.fixture
def mock_pymupdf(monkeypatch):
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Sample PDF page text"
    mock_doc = [mock_page, mock_page]
    mock_open = MagicMock(return_value=mock_doc)
    monkeypatch.setattr("pymupdf.open", mock_open)
    return mock_open

@pytest.fixture
def mock_docx(monkeypatch):
    mock_para = MagicMock()
    mock_para.text = "Sample DOCX paragraph"
    mock_doc = MagicMock()
    mock_doc.paragraphs = [mock_para, mock_para]
    monkeypatch.setattr("docx.Document", MagicMock(return_value=mock_doc))

@pytest.fixture
def mock_vectorstore(monkeypatch):
    # Patch FAISS and HuggingFaceEmbeddings
    mock_faiss = MagicMock()
    mock_faiss.from_documents = MagicMock(return_value=MagicMock())
    monkeypatch.setattr("langchain_community.vectorstores.FAISS", mock_faiss)
    monkeypatch.setattr("langchain_huggingface.HuggingFaceEmbeddings", MagicMock(return_value=MagicMock()))

@pytest.fixture
def mock_llm(monkeypatch):
    # Patch ChatGroq and chain.invoke
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(return_value="Mocked answer")
    monkeypatch.setattr("langchain_groq.ChatGroq", MagicMock(return_value=mock_llm))
    # Patch ChatPromptTemplate
    mock_prompt = MagicMock()
    mock_prompt.from_template = MagicMock(return_value=mock_prompt)
    monkeypatch.setattr("langchain_core.prompts.ChatPromptTemplate", mock_prompt)
    # Patch StrOutputParser
    mock_parser = MagicMock()
    mock_parser.invoke = MagicMock(return_value="Mocked answer")
    monkeypatch.setattr("langchain_core.output_parsers.StrOutputParser", MagicMock(return_value=mock_parser))
    # Patch PydanticOutputParser
    mock_pydantic_parser = MagicMock()
    mock_pydantic_parser.get_format_instructions.return_value = "format"
    mock_pydantic_parser.invoke.return_value = streamlit_app.ShipmentData()
    monkeypatch.setattr("langchain_core.output_parsers.PydanticOutputParser", MagicMock(return_value=mock_pydantic_parser))
    # Patch Document
    monkeypatch.setattr("langchain_core.documents.Document", MagicMock())

@pytest.fixture
def mock_text_splitter(monkeypatch):
    mock_splitter = MagicMock()
    mock_splitter.split_text = MagicMock(return_value=["chunk1", "chunk2"])
    monkeypatch.setattr("langchain_text_splitters.RecursiveCharacterTextSplitter", MagicMock(return_value=mock_splitter))

def test_document_ingestion_pdf_happy_path(mock_env, mock_streamlit, mock_pymupdf, mock_text_splitter):
    ingestion = streamlit_app.DocumentIngestionService()
    pdf_bytes = b"%PDF-1.4 fake pdf content"
    filename = "test.pdf"
    chunks = ingestion.process_file(pdf_bytes, filename)
    assert isinstance(chunks, list)
    assert all(isinstance(c, streamlit_app.Chunk) for c in chunks)
    assert all(c.metadata.filename == filename for c in chunks)
    assert len(chunks) == 2

def test_document_ingestion_docx_happy_path(mock_env, mock_streamlit, mock_docx, mock_text_splitter):
    ingestion = streamlit_app.DocumentIngestionService()
    docx_bytes = b"Fake DOCX content"
    filename = "test.docx"
    chunks = ingestion.process_file(docx_bytes, filename)
    assert isinstance(chunks, list)
    assert all(isinstance(c, streamlit_app.Chunk) for c in chunks)
    assert all(c.metadata.filename == filename for c in chunks)
    assert len(chunks) == 2

def test_document_ingestion_txt_happy_path(mock_env, mock_streamlit, mock_text_splitter):
    ingestion = streamlit_app.DocumentIngestionService()
    txt_bytes = b"Some plain text content"
    filename = "test.txt"
    chunks = ingestion.process_file(txt_bytes, filename)
    assert isinstance(chunks, list)
    assert all(isinstance(c, streamlit_app.Chunk) for c in chunks)
    assert all(c.metadata.filename == filename for c in chunks)
    assert len(chunks) == 2

def test_document_ingestion_empty_file(mock_env, mock_streamlit, mock_text_splitter):
    ingestion = streamlit_app.DocumentIngestionService()
    txt_bytes = b""
    filename = "empty.txt"
    chunks = ingestion.process_file(txt_bytes, filename)
    assert isinstance(chunks, list)
    assert len(chunks) == 2  # Even empty, splitter returns two chunks (from mock)

def test_document_ingestion_unsupported_extension(mock_env, mock_streamlit, mock_text_splitter):
    ingestion = streamlit_app.DocumentIngestionService()
    data_bytes = b"Some data"
    filename = "file.xyz"
    chunks = ingestion.process_file(data_bytes, filename)
    assert isinstance(chunks, list)
    assert len(chunks) == 2

def test_vectorstore_add_documents_creates_store(mock_env, mock_streamlit, mock_vectorstore):
    vs = streamlit_app.VectorStoreService()
    chunk = streamlit_app.Chunk(text="abc", metadata=streamlit_app.DocumentMetadata(filename="f", chunk_id=0, source="f", chunk_type="text"))
    vs.add_documents([chunk])
    assert vs.vector_store is not None

def test_vectorstore_add_documents_appends_to_existing(mock_env, mock_streamlit, mock_vectorstore):
    vs = streamlit_app.VectorStoreService()
    vs.vector_store = MagicMock()
    chunk = streamlit_app.Chunk(text="abc", metadata=streamlit_app.DocumentMetadata(filename="f", chunk_id=0, source="f", chunk_type="text"))
    vs.add_documents([chunk])
    vs.vector_store.add_documents.assert_called()

def test_ragservice_answer_returns_sourced_answer(mock_env, mock_streamlit, mock_vectorstore, mock_llm):
    vs = streamlit_app.VectorStoreService()
    vs.vector_store = MagicMock()
    retriever = MagicMock()
    doc = MagicMock()
    doc.page_content = "context"
    doc.metadata = {"filename": "f", "chunk_id": 0, "source": "f", "chunk_type": "text"}
    retriever.invoke.return_value = [doc]
    vs.vector_store.as_retriever.return_value = retriever
    rag = streamlit_app.RAGService(vs)
    answer = rag.answer("What is the carrier?")
    assert isinstance(answer, streamlit_app.SourcedAnswer)
    assert answer.answer == "Mocked answer"
    assert answer.confidence_score in (0.9, 0.1)
    assert isinstance(answer.sources, list)
    assert isinstance(answer.sources[0], streamlit_app.Chunk)

def test_ragservice_answer_low_confidence_on_not_found(mock_env, mock_streamlit, mock_vectorstore, mock_llm):
    vs = streamlit_app.VectorStoreService()
    vs.vector_store = MagicMock()
    retriever = MagicMock()
    doc = MagicMock()
    doc.page_content = "context"
    doc.metadata = {"filename": "f", "chunk_id": 0, "source": "f", "chunk_type": "text"}
    retriever.invoke.return_value = [doc]
    vs.vector_store.as_retriever.return_value = retriever
    with patch("langchain_core.output_parsers.StrOutputParser") as sop:
        parser = MagicMock()
        parser.invoke.return_value = "I cannot find the answer in the provided documents."
        sop.return_value = parser
        rag = streamlit_app.RAGService(vs)
        answer = rag.answer("What is the carrier?")
        assert answer.confidence_score == 0.1

def test_extraction_service_extract_returns_shipment_data(mock_env, mock_streamlit, mock_llm):
    extractor = streamlit_app.ExtractionService()
    result = extractor.extract("Some text to extract")
    assert isinstance(result, streamlit_app.ShipmentData)

def test_extraction_service_extract_handles_empty_text(mock_env, mock_streamlit, mock_llm):
    extractor = streamlit_app.ExtractionService()
    result = extractor.extract("")
    assert isinstance(result, streamlit_app.ShipmentData)

def test_document_metadata_boundary_conditions():
    meta = streamlit_app.DocumentMetadata(filename="f", chunk_id=0, source="s", chunk_type="text")
    assert meta.filename == "f"
    assert meta.chunk_id == 0
    assert meta.source == "s"
    assert meta.chunk_type == "text"
    assert meta.page_number is None

def test_chunk_model_boundary_conditions():
    meta = streamlit_app.DocumentMetadata(filename="f", chunk_id=1, source="s", chunk_type="text")
    chunk = streamlit_app.Chunk(text="", metadata=meta)
    assert chunk.text == ""
    assert chunk.metadata == meta

def test_sourced_answer_model_boundary_conditions():
    meta = streamlit_app.DocumentMetadata(filename="f", chunk_id=2, source="s", chunk_type="text")
    chunk = streamlit_app.Chunk(text="abc", metadata=meta)
    answer = streamlit_app.SourcedAnswer(answer="", confidence_score=0.0, sources=[chunk])
    assert answer.answer == ""
    assert answer.confidence_score == 0.0
    assert answer.sources == [chunk]

def test_shipment_data_model_all_fields():
    carrier = streamlit_app.CarrierInfo(carrier_name="Carrier", mc_number="123", phone="555-5555")
    driver = streamlit_app.DriverInfo(driver_name="Driver", cell_number="555-0000", truck_number="T1")
    pickup = streamlit_app.Location(name="Pickup", address="123 St", city="City", state="ST", zip_code="00000", appointment_time="10:00")
    drop = streamlit_app.Location(name="Drop", address="456 Ave", city="Town", state="TS", zip_code="11111", appointment_time="12:00")
    rate_info = streamlit_app.RateInfo(total_rate=1000.0, currency="USD", rate_breakdown={"linehaul": 900, "fuel": 100})
    data = streamlit_app.ShipmentData(
        reference_id="REF1",
        shipper="Shipper",
        consignee="Consignee",
        carrier=carrier,
        driver=driver,
        pickup=pickup,
        drop=drop,
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Van",
        rate_info=rate_info,
        special_instructions="None"
    )
    assert data.reference_id == "REF1"
    assert data.carrier == carrier
    assert data.driver == driver
    assert data.pickup == pickup
    assert data.drop == drop
    assert data.rate_info == rate_info

def test_location_model_partial_fields():
    loc = streamlit_app.Location(name="Loc", city="City")
    assert loc.name == "Loc"
    assert loc.city == "City"
    assert loc.address is None

def test_commodity_item_model_partial_fields():
    item = streamlit_app.CommodityItem(commodity_name="Widgets")
    assert item.commodity_name == "Widgets"
    assert item.weight is None

def test_carrier_info_model_partial_fields():
    carrier = streamlit_app.CarrierInfo(carrier_name="Carrier")
    assert carrier.carrier_name == "Carrier"
    assert carrier.mc_number is None

def test_driver_info_model_partial_fields():
    driver = streamlit_app.DriverInfo(driver_name="Driver")
    assert driver.driver_name == "Driver"
    assert driver.cell_number is None

def test_rate_info_model_partial_fields():
    rate = streamlit_app.RateInfo(total_rate=123.45)
    assert rate.total_rate == 123.45
    assert rate.currency is None

def test_structured_extraction_equivalent_paths(mock_env, mock_streamlit, mock_llm):
    extractor = streamlit_app.ExtractionService()
    text1 = "Extract this shipment data"
    text2 = "Extract this shipment data"
    result1 = extractor.extract(text1)
    result2 = extractor.extract(text2)
    assert result1 == result2

def test_ragservice_equivalent_paths(mock_env, mock_streamlit, mock_vectorstore, mock_llm):
    vs = streamlit_app.VectorStoreService()
    vs.vector_store = MagicMock()
    retriever = MagicMock()
    doc = MagicMock()
    doc.page_content = "context"
    doc.metadata = {"filename": "f", "chunk_id": 0, "source": "f", "chunk_type": "text"}
    retriever.invoke.return_value = [doc]
    vs.vector_store.as_retriever.return_value = retriever
    rag = streamlit_app.RAGService(vs)
    answer1 = rag.answer("What is the carrier?")
    answer2 = rag.answer("What is the carrier?")
    assert answer1.answer == answer2.answer
    assert answer1.confidence_score == answer2.confidence_score
    assert answer1.sources == answer2.sources

def test_missing_groq_api_key_stops_app(monkeypatch, mock_streamlit):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    # Patch st.stop to raise SystemExit so we can catch it
    st.stop.side_effect = SystemExit
    with pytest.raises(SystemExit):
        # Re-import to trigger the Streamlit app logic
        import importlib
        importlib.reload(streamlit_app)
