import io
import os
import sys
import types
import pytest
import builtins

import streamlit_app

from unittest.mock import patch, MagicMock

import numpy as np

from pydantic import ValidationError

# --- Helper Mocks ---

class DummyPage:
    def __init__(self, text):
        self._text = text
    def get_text(self):
        return self._text

class DummyPDF:
    def __init__(self, texts):
        self._texts = texts
    def __iter__(self):
        for t in self._texts:
            yield DummyPage(t)

class DummyDocx:
    def __init__(self, paragraphs):
        self.paragraphs = [type("Para", (), {"text": p})() for p in paragraphs]

class DummyRetriever:
    def __init__(self, docs):
        self._docs = docs
    def invoke(self, question):
        return self._docs

class DummyChain:
    def __init__(self, output):
        self._output = output
    def invoke(self, args):
        return self._output

class DummyPrompt:
    def __init__(self, template):
        self.template = template
    def __or__(self, other):
        return DummyChain("dummy answer")
    @classmethod
    def from_template(cls, template):
        return DummyPrompt(template)

class DummyEmbeddings:
    def __init__(self, model_name=None):
        self.model_name = model_name

class DummyFAISS:
    def __init__(self):
        self.docs = []
    @classmethod
    def from_documents(cls, docs, embeddings):
        inst = cls()
        inst.docs.extend(docs)
        return inst
    def add_documents(self, docs):
        self.docs.extend(docs)
    def as_retriever(self, search_kwargs=None):
        return DummyRetriever(self.docs)

class DummyLLM:
    def __init__(self, temperature=0, model_name=None, api_key=None):
        self.temperature = temperature
        self.model_name = model_name
        self.api_key = api_key
    def __or__(self, other):
        return DummyChain("dummy answer")
    def invoke(self, args):
        return "dummy answer"

class DummyOutputParser:
    def __init__(self, pydantic_object=None):
        self.pydantic_object = pydantic_object
    def __or__(self, other):
        return DummyChain(self.pydantic_object(**{}))
    def get_format_instructions(self):
        return "format instructions"
    def invoke(self, args):
        return self.pydantic_object(**{})

# --- Patch LangChain and External Dependencies ---

@pytest.fixture(autouse=True)
def patch_external_deps(monkeypatch):
    # Patch pymupdf.open
    monkeypatch.setattr(streamlit_app.pymupdf, "open", lambda stream, filetype: DummyPDF(["Carrier Details: ACME", "Rate Breakdown: $1000"]))
    # Patch docx.Document
    monkeypatch.setattr(streamlit_app.docx, "Document", lambda file: DummyDocx(["Carrier Details: ACME", "Rate Breakdown: $1000"]))
    # Patch HuggingFaceEmbeddings
    monkeypatch.setattr(streamlit_app, "HuggingFaceEmbeddings", DummyEmbeddings)
    # Patch FAISS
    monkeypatch.setattr(streamlit_app, "FAISS", DummyFAISS)
    # Patch ChatGroq
    monkeypatch.setattr(streamlit_app, "ChatGroq", DummyLLM)
    # Patch ChatPromptTemplate
    monkeypatch.setattr(streamlit_app, "ChatPromptTemplate", DummyPrompt)
    # Patch StrOutputParser
    monkeypatch.setattr(streamlit_app, "StrOutputParser", lambda: DummyChain("dummy answer"))
    # Patch PydanticOutputParser
    monkeypatch.setattr(streamlit_app, "PydanticOutputParser", lambda pydantic_object: DummyOutputParser(pydantic_object))
    # Patch Document
    monkeypatch.setattr(streamlit_app, "Document", lambda page_content, metadata: type("Doc", (), {"page_content": page_content, "metadata": metadata})())
    # Patch dotenv.load_dotenv to do nothing
    monkeypatch.setattr(streamlit_app, "load_dotenv", lambda: None)
    # Patch st to dummy object to avoid UI code execution
    dummy_st = types.SimpleNamespace()
    dummy_st.set_page_config = lambda **kwargs: None
    dummy_st.session_state = {}
    dummy_st.title = lambda *a, **k: None
    dummy_st.info = lambda *a, **k: None
    dummy_st.error = lambda *a, **k: None
    dummy_st.stop = lambda: None
    dummy_st.tabs = lambda x: [types.SimpleNamespace() for _ in x]
    dummy_st.header = lambda *a, **k: None
    dummy_st.file_uploader = lambda *a, **k: []
    dummy_st.button = lambda *a, **k: False
    dummy_st.spinner = lambda *a, **k: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None)
    dummy_st.success = lambda *a, **k: None
    dummy_st.text_input = lambda *a, **k: ""
    dummy_st.markdown = lambda *a, **k: None
    dummy_st.metric = lambda *a, **k: None
    dummy_st.expander = lambda *a, **k: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None)
    dummy_st.caption = lambda *a, **k: None
    dummy_st.write = lambda *a, **k: None
    dummy_st.divider = lambda *a, **k: None
    dummy_st.warning = lambda *a, **k: None
    dummy_st.json = lambda *a, **k: None
    monkeypatch.setattr(streamlit_app, "st", dummy_st)
    yield

# --- Tests ---

def test_document_ingestion_pdf_happy_path():
    service = streamlit_app.DocumentIngestionService()
    # Simulate PDF file
    file_content = b"dummy pdf bytes"
    filename = "test.pdf"
    chunks = service.process_file(file_content, filename)
    assert isinstance(chunks, list)
    assert all(isinstance(c, streamlit_app.Chunk) for c in chunks)
    assert any("Carrier Details" in c.text for c in chunks)
    assert any("Rate Breakdown" in c.text for c in chunks)
    # Check metadata
    for c in chunks:
        assert c.metadata.filename == filename
        assert c.metadata.chunk_type == "text"

def test_document_ingestion_docx_happy_path():
    service = streamlit_app.DocumentIngestionService()
    file_content = b"dummy docx bytes"
    filename = "test.docx"
    chunks = service.process_file(file_content, filename)
    assert isinstance(chunks, list)
    assert all(isinstance(c, streamlit_app.Chunk) for c in chunks)
    assert any("Carrier Details" in c.text for c in chunks)
    assert any("Rate Breakdown" in c.text for c in chunks)

def test_document_ingestion_txt_happy_path():
    service = streamlit_app.DocumentIngestionService()
    file_content = b"Carrier Details: ACME\nRate Breakdown: $1000"
    filename = "test.txt"
    chunks = service.process_file(file_content, filename)
    assert isinstance(chunks, list)
    assert all(isinstance(c, streamlit_app.Chunk) for c in chunks)
    assert any("Carrier Details" in c.text for c in chunks)
    assert any("Rate Breakdown" in c.text for c in chunks)

def test_document_ingestion_empty_file():
    service = streamlit_app.DocumentIngestionService()
    file_content = b""
    filename = "empty.txt"
    chunks = service.process_file(file_content, filename)
    assert isinstance(chunks, list)
    assert len(chunks) == 1
    assert chunks[0].text == ""

def test_document_ingestion_unknown_extension():
    service = streamlit_app.DocumentIngestionService()
    file_content = b"Some content"
    filename = "file.unknown"
    chunks = service.process_file(file_content, filename)
    assert isinstance(chunks, list)
    assert len(chunks) == 1
    assert chunks[0].text == ""

def test_vector_store_add_documents_and_retrieve():
    vs = streamlit_app.VectorStoreService()
    # Add two chunks
    chunks = [
        streamlit_app.Chunk(text="Carrier Details: ACME", metadata=streamlit_app.DocumentMetadata(filename="a.txt", chunk_id=0, source="a.txt - Part 1")),
        streamlit_app.Chunk(text="Rate Breakdown: $1000", metadata=streamlit_app.DocumentMetadata(filename="a.txt", chunk_id=1, source="a.txt - Part 2")),
    ]
    vs.add_documents(chunks)
    assert vs.vector_store is not None
    # Add more documents
    more_chunks = [
        streamlit_app.Chunk(text="Pickup: NYC", metadata=streamlit_app.DocumentMetadata(filename="b.txt", chunk_id=0, source="b.txt - Part 1")),
    ]
    vs.add_documents(more_chunks)
    assert len(vs.vector_store.docs) == 3

def test_rag_service_answer_returns_sourced_answer():
    vs = streamlit_app.VectorStoreService()
    # Add a chunk
    chunk = streamlit_app.Chunk(text="Carrier Details: ACME", metadata=streamlit_app.DocumentMetadata(filename="a.txt", chunk_id=0, source="a.txt - Part 1"))
    vs.add_documents([chunk])
    rag = streamlit_app.RAGService(vs)
    answer = rag.answer("Who is the carrier?")
    assert isinstance(answer, streamlit_app.SourcedAnswer)
    assert isinstance(answer.answer, str)
    assert isinstance(answer.confidence_score, float)
    assert isinstance(answer.sources, list)
    assert any("Carrier Details" in s.text for s in answer.sources)

def test_rag_service_low_confidence_when_no_answer(monkeypatch):
    vs = streamlit_app.VectorStoreService()
    chunk = streamlit_app.Chunk(text="Random unrelated text", metadata=streamlit_app.DocumentMetadata(filename="a.txt", chunk_id=0, source="a.txt - Part 1"))
    vs.add_documents([chunk])
    rag = streamlit_app.RAGService(vs)
    # Patch chain.invoke to return "I cannot find the answer in the provided documents."
    class DummyChainNoAnswer:
        def invoke(self, args):
            return "I cannot find the answer in the provided documents."
    monkeypatch.setattr(rag, "prompt", DummyPrompt("dummy"))
    monkeypatch.setattr(rag, "llm", DummyLLM())
    monkeypatch.setattr(streamlit_app, "StrOutputParser", lambda: DummyChainNoAnswer())
    answer = rag.answer("Unanswerable question?")
    assert answer.confidence_score == 0.1

def test_extraction_service_extract_returns_shipment_data():
    extractor = streamlit_app.ExtractionService()
    text = "Carrier Details: ACME\nPickup: NYC"
    data = extractor.extract(text)
    assert isinstance(data, streamlit_app.ShipmentData)

def test_extraction_service_extract_handles_empty(monkeypatch):
    extractor = streamlit_app.ExtractionService()
    # Patch chain.invoke to return empty ShipmentData
    class DummyChainEmpty:
        def invoke(self, args):
            return streamlit_app.ShipmentData()
    monkeypatch.setattr(extractor, "prompt", DummyPrompt("dummy"))
    monkeypatch.setattr(extractor, "llm", DummyLLM())
    monkeypatch.setattr(extractor, "parser", DummyOutputParser(streamlit_app.ShipmentData))
    data = extractor.extract("")
    assert isinstance(data, streamlit_app.ShipmentData)

def test_document_metadata_validation():
    # Valid
    meta = streamlit_app.DocumentMetadata(filename="a.txt", chunk_id=1, source="a.txt - Part 2")
    assert meta.filename == "a.txt"
    # Missing required
    with pytest.raises(ValidationError):
        streamlit_app.DocumentMetadata(filename="a.txt", source="a.txt - Part 2")

def test_chunk_validation():
    meta = streamlit_app.DocumentMetadata(filename="a.txt", chunk_id=1, source="a.txt - Part 2")
    chunk = streamlit_app.Chunk(text="Some text", metadata=meta)
    assert chunk.text == "Some text"
    assert chunk.metadata.filename == "a.txt"

def test_sourced_answer_validation():
    meta = streamlit_app.DocumentMetadata(filename="a.txt", chunk_id=1, source="a.txt - Part 2")
    chunk = streamlit_app.Chunk(text="Some text", metadata=meta)
    answer = streamlit_app.SourcedAnswer(answer="42", confidence_score=0.8, sources=[chunk])
    assert answer.answer == "42"
    assert answer.confidence_score == 0.8
    assert answer.sources[0].text == "Some text"

def test_shipment_data_model_fields():
    data = streamlit_app.ShipmentData(reference_id="REF123", shipper="ACME", consignee="XYZ")
    assert data.reference_id == "REF123"
    assert data.shipper == "ACME"
    assert data.consignee == "XYZ"

def test_location_model_fields():
    loc = streamlit_app.Location(name="Warehouse", address="123 Main", city="NYC", state="NY", zip_code="10001", appointment_time="10:00")
    assert loc.name == "Warehouse"
    assert loc.address == "123 Main"
    assert loc.city == "NYC"
    assert loc.state == "NY"
    assert loc.zip_code == "10001"
    assert loc.appointment_time == "10:00"

def test_commodity_item_model_fields():
    item = streamlit_app.CommodityItem(commodity_name="Widgets", weight="100kg", quantity="10")
    assert item.commodity_name == "Widgets"
    assert item.weight == "100kg"
    assert item.quantity == "10"

def test_carrier_info_model_fields():
    carrier = streamlit_app.CarrierInfo(carrier_name="ACME", mc_number="123456", phone="555-1234")
    assert carrier.carrier_name == "ACME"
    assert carrier.mc_number == "123456"
    assert carrier.phone == "555-1234"

def test_driver_info_model_fields():
    driver = streamlit_app.DriverInfo(driver_name="John Doe", cell_number="555-5678", truck_number="TRK123")
    assert driver.driver_name == "John Doe"
    assert driver.cell_number == "555-5678"
    assert driver.truck_number == "TRK123"

def test_rate_info_model_fields():
    rate = streamlit_app.RateInfo(total_rate=1000.0, currency="USD", rate_breakdown={"linehaul": 800, "fuel": 200})
    assert rate.total_rate == 1000.0
    assert rate.currency == "USD"
    assert rate.rate_breakdown["linehaul"] == 800
    assert rate.rate_breakdown["fuel"] == 200

def test_shipment_data_full_model():
    carrier = streamlit_app.CarrierInfo(carrier_name="ACME", mc_number="123456", phone="555-1234")
    driver = streamlit_app.DriverInfo(driver_name="John Doe", cell_number="555-5678", truck_number="TRK123")
    pickup = streamlit_app.Location(name="Warehouse", address="123 Main", city="NYC", state="NY", zip_code="10001", appointment_time="10:00")
    drop = streamlit_app.Location(name="Store", address="456 Elm", city="Boston", state="MA", zip_code="02110", appointment_time="16:00")
    rate = streamlit_app.RateInfo(total_rate=1000.0, currency="USD", rate_breakdown={"linehaul": 800, "fuel": 200})
    data = streamlit_app.ShipmentData(
        reference_id="REF123",
        shipper="ACME",
        consignee="XYZ",
        carrier=carrier,
        driver=driver,
        pickup=pickup,
        drop=drop,
        shipping_date="2024-01-01",
        delivery_date="2024-01-02",
        equipment_type="Van",
        rate_info=rate,
        special_instructions="Handle with care"
    )
    assert data.reference_id == "REF123"
    assert data.carrier.carrier_name == "ACME"
    assert data.driver.driver_name == "John Doe"
    assert data.pickup.city == "NYC"
    assert data.drop.city == "Boston"
    assert data.rate_info.total_rate == 1000.0
    assert data.special_instructions == "Handle with care"

def test_document_ingestion_section_markers():
    service = streamlit_app.DocumentIngestionService()
    file_content = b"Carrier Details\nRate Breakdown\nPickup\nDrop\nCommodity\nSpecial Instructions"
    filename = "test.txt"
    chunks = service.process_file(file_content, filename)
    # Should have section markers inserted
    found_sections = [s for c in chunks for s in ["Carrier Details", "Rate Breakdown", "Pickup", "Drop", "Commodity", "Special Instructions"] if s in c.text]
    assert set(found_sections) == set(["Carrier Details", "Rate Breakdown", "Pickup", "Drop", "Commodity", "Special Instructions"])

def test_vector_store_add_documents_idempotent():
    vs = streamlit_app.VectorStoreService()
    chunks = [
        streamlit_app.Chunk(text="Carrier Details: ACME", metadata=streamlit_app.DocumentMetadata(filename="a.txt", chunk_id=0, source="a.txt - Part 1")),
    ]
    vs.add_documents(chunks)
    count1 = len(vs.vector_store.docs)
    vs.add_documents(chunks)
    count2 = len(vs.vector_store.docs)
    assert count2 == count1 * 2

def test_rag_service_answer_equivalent_paths(monkeypatch):
    vs = streamlit_app.VectorStoreService()
    chunk = streamlit_app.Chunk(text="Carrier Details: ACME", metadata=streamlit_app.DocumentMetadata(filename="a.txt", chunk_id=0, source="a.txt - Part 1"))
    vs.add_documents([chunk])
    rag = streamlit_app.RAGService(vs)
    # Patch chain.invoke to always return "ACME"
    class DummyChainACME:
        def invoke(self, args):
            return "ACME"
    monkeypatch.setattr(rag, "prompt", DummyPrompt("dummy"))
    monkeypatch.setattr(rag, "llm", DummyLLM())
    monkeypatch.setattr(streamlit_app, "StrOutputParser", lambda: DummyChainACME())
    answer1 = rag.answer("Who is the carrier?")
    answer2 = rag.answer("Who is the carrier?")
    assert answer1.answer == answer2.answer == "ACME"
    assert answer1.confidence_score == answer2.confidence_score

def test_extraction_service_extract_equivalent_paths(monkeypatch):
    extractor = streamlit_app.ExtractionService()
    # Patch chain.invoke to return ShipmentData with reference_id
    class DummyChainRef:
        def invoke(self, args):
            return streamlit_app.ShipmentData(reference_id="REF123")
    monkeypatch.setattr(extractor, "prompt", DummyPrompt("dummy"))
    monkeypatch.setattr(extractor, "llm", DummyLLM())
    monkeypatch.setattr(extractor, "parser", DummyOutputParser(streamlit_app.ShipmentData))
    data1 = extractor.extract("Carrier Details: ACME")
    data2 = extractor.extract("Carrier Details: ACME")
    assert data1.reference_id == data2.reference_id == "REF123"
