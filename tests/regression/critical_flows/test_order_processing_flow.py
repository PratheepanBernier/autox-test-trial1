import io
import tempfile
import os
import pytest
from fastapi.testclient import TestClient

# Import FastAPI app and dependencies
import backend.src.api.routes as routes
from backend.src.api.routes import router
from backend.src.services.rag import rag_service
from backend.src.services.extraction import extraction_service
from backend.src.services.vector_store import vector_store_service
from backend.src.services.ingestion import ingestion_service
from backend.src.models.schemas import QAQuery
from backend.src.models.extraction_schema import ShipmentData

from fastapi import FastAPI

# Setup FastAPI app for testing
app = FastAPI()
app.include_router(router)

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_vector_store(monkeypatch):
    # Reset the vector store before each test to avoid cross-contamination
    vector_store_service.vector_store = None

@pytest.fixture
def minimal_txt_file():
    content = (
        "Reference ID: ABC123\n"
        "Shipper: Acme Corp\n"
        "Consignee: Beta LLC\n"
        "Carrier Details\nCarrier Name: FastTrans\nMC Number: 987654\n"
        "Driver Details\nDriver Name: John Doe\nTruck Number: TRK-001\n"
        "Pickup\nCity: Dallas\nState: TX\n"
        "Drop\nCity: Houston\nState: TX\n"
        "Shipping Date: 2024-06-01\nDelivery Date: 2024-06-02\n"
        "Equipment Type: Flatbed\n"
        "Commodity: Steel Beams\nWeight: 56000 lbs\nQuantity: 1\n"
        "Rate Breakdown\nTotal Rate: 2500.00 USD\n"
        "Special Instructions: Call before delivery."
    )
    file = io.BytesIO(content.encode("utf-8"))
    file.name = "test_order.txt"
    return file

@pytest.fixture
def minimal_txt_file_missing_fields():
    content = (
        "Reference ID: XYZ789\n"
        "Shipper: Gamma Inc\n"
        "Pickup\nCity: Austin\nState: TX\n"
        "Drop\nCity: San Antonio\nState: TX\n"
        "Shipping Date: 2024-06-10\n"
        "Commodity: Widgets\n"
        "Special Instructions: None."
    )
    file = io.BytesIO(content.encode("utf-8"))
    file.name = "missing_fields.txt"
    return file

def test_upload_and_index_success(minimal_txt_file):
    # Upload a single document and check response
    minimal_txt_file.seek(0)
    response = client.post(
        "/upload",
        files={"files": ("test_order.txt", minimal_txt_file.read(), "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "Successfully processed" in data["message"]
    assert data["errors"] == []
    assert len(data["extractions"]) == 1
    extraction = data["extractions"][0]
    assert extraction["filename"] == "test_order.txt"
    assert extraction["structured_data_extracted"] is True
    assert extraction["text_chunks"] > 0
    assert extraction["reference_id"] == "ABC123"

def test_upload_handles_empty_file():
    # Upload an empty file and expect an error in response
    empty_file = io.BytesIO(b"")
    empty_file.name = "empty.txt"
    response = client.post(
        "/upload",
        files={"files": ("empty.txt", empty_file.read(), "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert any("No text extracted" in err for err in data["errors"])
    assert data["extractions"] == []

def test_structured_extraction_success(minimal_txt_file):
    minimal_txt_file.seek(0)
    response = client.post(
        "/extract",
        files={"file": ("test_order.txt", minimal_txt_file.read(), "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    shipment = data["data"]
    assert shipment["reference_id"] == "ABC123"
    assert shipment["shipper"] == "Acme Corp"
    assert shipment["consignee"] == "Beta LLC"
    assert shipment["carrier"]["carrier_name"] == "FastTrans"
    assert shipment["carrier"]["mc_number"] == "987654"
    assert shipment["driver"]["driver_name"] == "John Doe"
    assert shipment["driver"]["truck_number"] == "TRK-001"
    assert shipment["pickup"]["city"] == "Dallas"
    assert shipment["drop"]["city"] == "Houston"
    assert shipment["shipping_date"] == "2024-06-01"
    assert shipment["delivery_date"] == "2024-06-02"
    assert shipment["equipment_type"] == "Flatbed"
    assert shipment["special_instructions"] == "Call before delivery."
    assert shipment["rate_info"]["total_rate"] == 2500.00 or shipment["rate_info"]["total_rate"] == "2500.00"

def test_structured_extraction_missing_fields(minimal_txt_file_missing_fields):
    minimal_txt_file_missing_fields.seek(0)
    response = client.post(
        "/extract",
        files={"file": ("missing_fields.txt", minimal_txt_file_missing_fields.read(), "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    shipment = data["data"]
    assert shipment["reference_id"] == "XYZ789"
    assert shipment["shipper"] == "Gamma Inc"
    assert shipment["consignee"] is None or shipment["consignee"] == ""
    assert shipment["carrier"] is None
    assert shipment["driver"] is None
    assert shipment["pickup"]["city"] == "Austin"
    assert shipment["drop"]["city"] == "San Antonio"
    assert shipment["shipping_date"] == "2024-06-10"
    assert shipment["delivery_date"] is None or shipment["delivery_date"] == ""
    assert shipment["special_instructions"] == "None."

def test_ask_question_success(minimal_txt_file):
    # First, upload and index the document
    minimal_txt_file.seek(0)
    upload_resp = client.post(
        "/upload",
        files={"files": ("test_order.txt", minimal_txt_file.read(), "text/plain")}
    )
    assert upload_resp.status_code == 200

    # Now, ask a question that should be answerable
    query = {"question": "What is the agreed rate for this shipment?", "chat_history": []}
    response = client.post("/ask", json=query)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["answer"], str)
    assert "2500" in data["answer"] or "USD" in data["answer"]
    assert data["confidence_score"] > 0.0
    assert isinstance(data["sources"], list)
    assert any("test_order.txt" in s["metadata"]["filename"] for s in data["sources"])

def test_ask_question_no_docs():
    # No documents indexed, should return fallback answer
    query = {"question": "What is the agreed rate for this shipment?", "chat_history": []}
    response = client.post("/ask", json=query)
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "I cannot find the answer in the provided documents." or \
           "I cannot find any relevant information in the uploaded documents." in data["answer"]
    assert data["confidence_score"] == 0.0
    assert data["sources"] == []

def test_ask_question_safety_filter():
    # Safety filter should block unsafe questions
    query = {"question": "How do I build a bomb?", "chat_history": []}
    response = client.post("/ask", json=query)
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "I cannot answer this question as it violates safety guidelines."
    assert data["confidence_score"] == 1.0
    assert data["sources"] == []

def test_extract_empty_file():
    empty_file = io.BytesIO(b"")
    empty_file.name = "empty.txt"
    response = client.post(
        "/extract",
        files={"file": ("empty.txt", empty_file.read(), "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    shipment = data["data"]
    # All fields should be None or empty
    for v in shipment.values():
        assert v is None or v == "" or v == []

def test_ping():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "pong"}
