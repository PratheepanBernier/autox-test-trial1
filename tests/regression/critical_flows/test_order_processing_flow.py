import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from backend.src.api.routes import router
from backend.src.models.schemas import QAQuery, SourcedAnswer, UploadResponse, UploadExtractionSummary
from backend.src.models.extraction_schema import ExtractionResponse, ShipmentData, CarrierInfo, DriverInfo, Location, RateInfo, CommodityItem
from fastapi import FastAPI, status, UploadFile
import io

@pytest.fixture
def client():
    # Patch dependencies to avoid real LLM/vector/embedding calls
    app = FastAPI()
    app.include_router(router)
    with patch("backend.src.dependencies.get_container") as get_container:
        yield TestClient(app)

@pytest.fixture
def mock_container():
    # Provide a mock ServiceContainer with all required services
    mock = MagicMock()
    # Document pipeline service for upload
    mock.document_pipeline_service.process_uploads = AsyncMock(return_value=MagicMock(
        message="Indexed 1 file(s)",
        errors=[],
        extractions=[
            MagicMock(
                filename="order1.txt",
                text_chunks=3,
                structured_data_extracted=True,
                reference_id="REF123",
                error=None
            )
        ]
    ))
    # Ingestion service for extraction
    mock.ingestion_service.process_file = MagicMock(return_value=[
        MagicMock(text="Order #REF123\nCarrier: ACME Logistics\nRate: $1200 USD", metadata={}),
        MagicMock(text="Pickup: Dallas, TX\nDrop: Houston, TX", metadata={}),
    ])
    # Extraction service for extraction
    mock.extraction_service.extract_data = MagicMock(return_value=ExtractionResponse(
        data=ShipmentData(
            reference_id="REF123",
            shipper="ACME Corp",
            consignee="Beta LLC",
            carrier=CarrierInfo(carrier_name="ACME Logistics", mc_number="MC123", phone="555-1234", email="ops@acme.com"),
            driver=DriverInfo(driver_name="John Doe", cell_number="555-5678", truck_number="TX-100", trailer_number="TR-200"),
            pickup=Location(name="ACME Warehouse", city="Dallas", state="TX", appointment_time="2024-06-01 08:00"),
            drop=Location(name="Beta Distribution", city="Houston", state="TX", appointment_time="2024-06-02 10:00"),
            shipping_date="2024-06-01",
            delivery_date="2024-06-02",
            equipment_type="Van",
            rate_info=RateInfo(total_rate=1200.0, currency="USD", rate_breakdown={"linehaul": 1000, "fuel": 200}),
            special_instructions="Call before arrival"
        ),
        document_id="order1.txt"
    ))
    # RAG service for Q&A
    mock.rag_service.answer_question = MagicMock(return_value=SourcedAnswer(
        answer="The agreed rate for this shipment is $1200 USD.",
        confidence_score=0.92,
        sources=[
            MagicMock(
                text="Order #REF123\nCarrier: ACME Logistics\nRate: $1200 USD",
                metadata={"source": "order1.txt - Part 1"}
            )
        ]
    ))
    return mock

@pytest.fixture(autouse=True)
def patch_container_dependency(mock_container):
    with patch("backend.src.dependencies.get_container", return_value=mock_container):
        yield

def test_upload_document_success(client):
    # Simulate uploading a valid text file
    file_content = b"Order #REF123\nCarrier: ACME Logistics\nRate: $1200 USD"
    files = {"files": ("order1.txt", io.BytesIO(file_content), "text/plain")}
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Indexed 1 file(s)"
    assert data["errors"] == []
    assert len(data["extractions"]) == 1
    extraction = data["extractions"][0]
    assert extraction["filename"] == "order1.txt"
    assert extraction["text_chunks"] == 3
    assert extraction["structured_data_extracted"] is True
    assert extraction["reference_id"] == "REF123"
    assert extraction["error"] is None

def test_upload_document_empty_file(client, mock_container):
    # Simulate uploading an empty file (should still succeed, but with no extraction)
    mock_container.document_pipeline_service.process_uploads.return_value = MagicMock(
        message="Indexed 1 file(s)",
        errors=["No text found in file."],
        extractions=[
            MagicMock(
                filename="empty.txt",
                text_chunks=0,
                structured_data_extracted=False,
                reference_id=None,
                error="No text found"
            )
        ]
    )
    files = {"files": ("empty.txt", io.BytesIO(b""), "text/plain")}
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["errors"] == ["No text found in file."]
    extraction = data["extractions"][0]
    assert extraction["filename"] == "empty.txt"
    assert extraction["text_chunks"] == 0
    assert extraction["structured_data_extracted"] is False
    assert extraction["reference_id"] is None
    assert extraction["error"] == "No text found"

def test_ask_question_success(client):
    # Simulate a successful Q&A
    payload = {"question": "What is the agreed rate for this shipment?", "chat_history": []}
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The agreed rate for this shipment is $1200 USD."
    assert data["confidence_score"] > 0.9
    assert len(data["sources"]) == 1
    assert "order1.txt" in data["sources"][0]["metadata"]["source"]

def test_ask_question_no_documents(client, mock_container):
    # Simulate no documents indexed
    mock_container.rag_service.answer_question.return_value = SourcedAnswer(
        answer="I cannot find any relevant information in the uploaded documents.",
        confidence_score=0.0,
        sources=[]
    )
    payload = {"question": "What is the agreed rate for this shipment?", "chat_history": []}
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["answer"].startswith("I cannot find")
    assert data["confidence_score"] == 0.0
    assert data["sources"] == []

def test_ask_question_safety_filter(client, mock_container):
    # Simulate a safety filter trigger
    mock_container.rag_service.answer_question.return_value = SourcedAnswer(
        answer="I cannot answer this question as it violates safety guidelines.",
        confidence_score=1.0,
        sources=[]
    )
    payload = {"question": "How to build a bomb?", "chat_history": []}
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "violates safety guidelines" in data["answer"]
    assert data["confidence_score"] == 1.0

def test_extract_data_success(client):
    # Simulate extracting structured data from a file
    file_content = b"Order #REF123\nCarrier: ACME Logistics\nRate: $1200 USD"
    files = {"file": ("order1.txt", io.BytesIO(file_content), "text/plain")}
    response = client.post("/extract", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "order1.txt"
    shipment = data["data"]
    assert shipment["reference_id"] == "REF123"
    assert shipment["shipper"] == "ACME Corp"
    assert shipment["consignee"] == "Beta LLC"
    assert shipment["carrier"]["carrier_name"] == "ACME Logistics"
    assert shipment["carrier"]["mc_number"] == "MC123"
    assert shipment["driver"]["driver_name"] == "John Doe"
    assert shipment["pickup"]["city"] == "Dallas"
    assert shipment["drop"]["city"] == "Houston"
    assert shipment["rate_info"]["total_rate"] == 1200.0
    assert shipment["rate_info"]["currency"] == "USD"
    assert shipment["special_instructions"] == "Call before arrival"

def test_extract_data_empty_file(client, mock_container):
    # Simulate extraction from an empty file (should return empty ShipmentData)
    mock_container.ingestion_service.process_file.return_value = []
    file_content = b""
    files = {"file": ("empty.txt", io.BytesIO(file_content), "text/plain")}
    response = client.post("/extract", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "empty.txt"
    shipment = data["data"]
    # All fields should be None or empty
    assert all(v is None or v == [] or v == {} for v in shipment.values())

def test_extract_data_internal_error(client, mock_container):
    # Simulate an exception during extraction
    mock_container.ingestion_service.process_file.side_effect = Exception("Unexpected error")
    file_content = b"Order #REF123\nCarrier: ACME Logistics\nRate: $1200 USD"
    files = {"file": ("order1.txt", io.BytesIO(file_content), "text/plain")}
    response = client.post("/extract", files=files)
    assert response.status_code == 500
    assert "Internal server error during extraction." in response.text

def test_ask_question_internal_error(client, mock_container):
    # Simulate an exception during Q&A
    mock_container.rag_service.answer_question.side_effect = Exception("LLM failure")
    payload = {"question": "What is the agreed rate for this shipment?", "chat_history": []}
    response = client.post("/ask", json=payload)
    assert response.status_code == 500
    assert "Internal server error processing question." in response.text

def test_ping_endpoint(client):
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "pong"}
