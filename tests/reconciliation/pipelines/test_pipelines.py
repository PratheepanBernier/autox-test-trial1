import pytest
from backend.src.services.ingestion import ingestion_service
from backend.src.services.vector_store import vector_store_service
from backend.src.services.extraction import extraction_service
from backend.src.models.schemas import Chunk, DocumentMetadata
from backend.src.models.extraction_schema import ShipmentData, ExtractionResponse
import copy

@pytest.fixture
def sample_txt_file():
    # Simulate a logistics document with clear sections and values
    content = (
        "Reference ID: ABC123\n"
        "Carrier Details\n"
        "Carrier Name: FastTrans\n"
        "MC Number: 987654\n"
        "Carrier Phone: 555-1234\n"
        "Driver Details\n"
        "Driver Name: John Doe\n"
        "Driver Phone: 555-5678\n"
        "Truck Number: TRK-001\n"
        "Pickup\n"
        "Warehouse A\n"
        "123 Main St\n"
        "Cityville, ST 12345\n"
        "Appointment: 2024-05-01 09:00\n"
        "Drop\n"
        "Warehouse B\n"
        "456 Elm St\n"
        "Townburg, ST 54321\n"
        "Rate Breakdown\n"
        "Total Rate: 1500.00 USD\n"
        "Commodity\n"
        "Widgets\n"
        "Weight: 1000 lbs\n"
        "Quantity: 50\n"
        "Special Instructions\n"
        "Handle with care.\n"
    )
    filename = "shipment1.txt"
    return content.encode("utf-8"), filename

@pytest.fixture
def sample_txt_file_duplicate():
    # Same as sample_txt_file but with a different filename to test duplicate detection
    content = (
        "Reference ID: ABC123\n"
        "Carrier Details\n"
        "Carrier Name: FastTrans\n"
        "MC Number: 987654\n"
        "Carrier Phone: 555-1234\n"
        "Driver Details\n"
        "Driver Name: John Doe\n"
        "Driver Phone: 555-5678\n"
        "Truck Number: TRK-001\n"
        "Pickup\n"
        "Warehouse A\n"
        "123 Main St\n"
        "Cityville, ST 12345\n"
        "Appointment: 2024-05-01 09:00\n"
        "Drop\n"
        "Warehouse B\n"
        "456 Elm St\n"
        "Townburg, ST 54321\n"
        "Rate Breakdown\n"
        "Total Rate: 1500.00 USD\n"
        "Commodity\n"
        "Widgets\n"
        "Weight: 1000 lbs\n"
        "Quantity: 50\n"
        "Special Instructions\n"
        "Handle with care.\n"
    )
    filename = "shipment1_copy.txt"
    return content.encode("utf-8"), filename

@pytest.fixture
def sample_txt_file_missing():
    # Simulate a document missing some fields
    content = (
        "Reference ID: XYZ789\n"
        "Carrier Details\n"
        "Carrier Name: SlowShip\n"
        "Pickup\n"
        "Warehouse C\n"
        "789 Oak St\n"
        "Othercity, ST 99999\n"
        "Drop\n"
        "Warehouse D\n"
        "321 Pine St\n"
        "Anothertown, ST 88888\n"
        "Commodity\n"
        "Gadgets\n"
        "Quantity: 20\n"
    )
    filename = "shipment2.txt"
    return content.encode("utf-8"), filename

def test_ingestion_and_vector_store_consistency(sample_txt_file):
    file_content, filename = sample_txt_file
    # Ingest file and chunk
    chunks = ingestion_service.process_file(file_content, filename)
    assert chunks, "No chunks produced from ingestion"
    # Add to vector store
    vector_store_service.vector_store = None  # Reset for test isolation
    vector_store_service.add_documents(chunks)
    # Check vector store internal state
    vs = vector_store_service.vector_store
    assert vs is not None, "Vector store not initialized"
    # Check record count matches
    assert len(vs.docstore._dict) == len(chunks), "Vector store document count mismatch with chunks"
    # Check aggregate: all chunk texts are present in vector store
    stored_texts = set(doc.page_content for doc in vs.docstore._dict.values())
    chunk_texts = set(c.text for c in chunks)
    assert stored_texts == chunk_texts, "Mismatch between ingested chunk texts and vector store contents"

def test_duplicate_detection(sample_txt_file, sample_txt_file_duplicate):
    # Ingest two files with identical content but different filenames
    file_content1, filename1 = sample_txt_file
    file_content2, filename2 = sample_txt_file_duplicate
    chunks1 = ingestion_service.process_file(file_content1, filename1)
    chunks2 = ingestion_service.process_file(file_content2, filename2)
    # Add both to vector store
    vector_store_service.vector_store = None
    vector_store_service.add_documents(chunks1)
    vector_store_service.add_documents(chunks2)
    vs = vector_store_service.vector_store
    # Check for duplicate chunk texts (should be present, but with different metadata)
    texts1 = set(c.text for c in chunks1)
    texts2 = set(c.text for c in chunks2)
    assert texts1 == texts2, "Duplicate file content mismatch"
    # Check that all chunks are present (count should double)
    assert len(vs.docstore._dict) == len(chunks1) + len(chunks2), "Vector store count does not reflect duplicates"
    # Check that metadata distinguishes the two sets
    filenames = set(doc.metadata["filename"] for doc in vs.docstore._dict.values())
    assert filename1 in filenames and filename2 in filenames, "Filenames not preserved in metadata"

def test_missing_record_detection(sample_txt_file, sample_txt_file_missing):
    # Ingest two files, one with missing fields
    file_content1, filename1 = sample_txt_file
    file_content2, filename2 = sample_txt_file_missing
    chunks1 = ingestion_service.process_file(file_content1, filename1)
    chunks2 = ingestion_service.process_file(file_content2, filename2)
    # Add both to vector store
    vector_store_service.vector_store = None
    vector_store_service.add_documents(chunks1)
    vector_store_service.add_documents(chunks2)
    vs = vector_store_service.vector_store
    # Check that all chunks from both files are present
    assert len(vs.docstore._dict) == len(chunks1) + len(chunks2), "Vector store missing records"
    # Check that at least one chunk from the missing-fields file contains 'Gadgets'
    found = any("Gadgets" in doc.page_content for doc in vs.docstore._dict.values())
    assert found, "Missing-fields document content not found in vector store"

def test_extraction_transformation_correctness(sample_txt_file):
    file_content, filename = sample_txt_file
    # Ingest and extract
    chunks = ingestion_service.process_file(file_content, filename)
    full_text = "\n".join([c.text for c in chunks])
    extraction = extraction_service.extract_data(full_text, filename)
    assert isinstance(extraction, ExtractionResponse)
    data = extraction.data
    # Validate transformation: key fields must match deterministic values
    assert data.reference_id == "ABC123"
    assert data.carrier is not None
    assert data.carrier.carrier_name == "FastTrans"
    assert data.carrier.mc_number == "987654"
    assert data.driver is not None
    assert data.driver.driver_name == "John Doe"
    assert data.pickup is not None
    assert data.pickup.name == "Warehouse A"
    assert data.drop is not None
    assert data.drop.name == "Warehouse B"
    assert data.rate_info is not None
    assert data.rate_info.total_rate == 1500.00
    assert data.rate_info.currency == "USD"
    assert data.special_instructions == "Handle with care."
    # Commodity
    if hasattr(data, "commodities") and data.commodities:
        commodity_names = [c.commodity_name for c in data.commodities if c.commodity_name]
        assert "Widgets" in commodity_names

def test_extraction_missing_fields(sample_txt_file_missing):
    file_content, filename = sample_txt_file_missing
    chunks = ingestion_service.process_file(file_content, filename)
    full_text = "\n".join([c.text for c in chunks])
    extraction = extraction_service.extract_data(full_text, filename)
    data = extraction.data
    # Validate that missing fields are None
    assert data.reference_id == "XYZ789"
    assert data.carrier is not None
    assert data.carrier.carrier_name == "SlowShip"
    assert data.driver is None or (not getattr(data.driver, "driver_name", None))
    assert data.rate_info is None or (not getattr(data.rate_info, "total_rate", None))
    # Commodity
    if hasattr(data, "commodities") and data.commodities:
        commodity_names = [c.commodity_name for c in data.commodities if c.commodity_name]
        assert "Gadgets" in commodity_names
    # Check that missing fields are actually None
    assert data.special_instructions is None

def test_structured_chunk_consistency(sample_txt_file):
    file_content, filename = sample_txt_file
    chunks = ingestion_service.process_file(file_content, filename)
    full_text = "\n".join([c.text for c in chunks])
    extraction = extraction_service.extract_data(full_text, filename)
    structured_chunk = extraction_service.create_structured_chunk(extraction, filename)
    # Add both text chunks and structured chunk to vector store
    vector_store_service.vector_store = None
    vector_store_service.add_documents(chunks)
    vector_store_service.add_documents([structured_chunk])
    vs = vector_store_service.vector_store
    # Check that the structured chunk is present and distinguishable
    found_structured = False
    for doc in vs.docstore._dict.values():
        if doc.metadata.get("chunk_type") == "structured_data":
            found_structured = True
            assert "EXTRACTED STRUCTURED DATA" in doc.page_content
            assert doc.metadata["filename"] == filename
    assert found_structured, "Structured data chunk not found in vector store"

def test_record_level_equivalence_between_ingestion_and_extraction(sample_txt_file):
    file_content, filename = sample_txt_file
    chunks = ingestion_service.process_file(file_content, filename)
    full_text = "\n".join([c.text for c in chunks])
    extraction = extraction_service.extract_data(full_text, filename)
    # Check that all key fields in extraction are present in the original text
    data = extraction.data
    text_lower = full_text.lower()
    if data.reference_id:
        assert data.reference_id.lower() in text_lower
    if data.carrier and data.carrier.carrier_name:
        assert data.carrier.carrier_name.lower() in text_lower
    if data.driver and data.driver.driver_name:
        assert data.driver.driver_name.lower() in text_lower
    if data.pickup and data.pickup.name:
        assert data.pickup.name.lower() in text_lower
    if data.drop and data.drop.name:
        assert data.drop.name.lower() in text_lower
    if data.special_instructions:
        assert data.special_instructions.lower() in text_lower

def test_aggregate_consistency_across_pipeline(sample_txt_file, sample_txt_file_missing):
    # Ingest, extract, and store both files
    file_content1, filename1 = sample_txt_file
    file_content2, filename2 = sample_txt_file_missing
    chunks1 = ingestion_service.process_file(file_content1, filename1)
    chunks2 = ingestion_service.process_file(file_content2, filename2)
    extraction1 = extraction_service.extract_data("\n".join([c.text for c in chunks1]), filename1)
    extraction2 = extraction_service.extract_data("\n".join([c.text for c in chunks2]), filename2)
    # Add all to vector store
    vector_store_service.vector_store = None
    vector_store_service.add_documents(chunks1)
    vector_store_service.add_documents(chunks2)
    structured_chunk1 = extraction_service.create_structured_chunk(extraction1, filename1)
    structured_chunk2 = extraction_service.create_structured_chunk(extraction2, filename2)
    vector_store_service.add_documents([structured_chunk1, structured_chunk2])
    vs = vector_store_service.vector_store
    # Aggregate: total number of records = all chunks + 2 structured
    expected_total = len(chunks1) + len(chunks2) + 2
    assert len(vs.docstore._dict) == expected_total, "Aggregate record count mismatch"
    # Aggregate: sum of all chunk_ids (excluding structured) matches expected
    chunk_ids = [doc.metadata["chunk_id"] for doc in vs.docstore._dict.values() if doc.metadata["chunk_type"] != "structured_data"]
    assert set(chunk_ids) == set(range(len(chunk_ids))), "Chunk IDs are not sequential or unique"
    # Aggregate: all filenames present
    filenames = set(doc.metadata["filename"] for doc in vs.docstore._dict.values())
    assert {filename1, filename2} <= filenames
