import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException, UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile
from backend.src.api import routes
from backend.src.models.extraction_schema import (
    ExtractionResponse,
    ShipmentData,
    CarrierInfo,
    DriverInfo,
    Location,
    CommodityItem,
    RateInfo,
)
from backend.src.models.schemas import (
    QAQuery,
    SourcedAnswer,
    UploadResponse,
    UploadExtractionSummary,
)
import io

@pytest.fixture
def mock_container():
    container = MagicMock()
    container.document_pipeline_service.process_uploads = AsyncMock()
    container.rag_service.answer_question = MagicMock()
    container.ingestion_service.process_file = MagicMock()
    container.extraction_service.extract_data = MagicMock()
    return container

@pytest.fixture
def sample_upload_file():
    file_content = b"Sample file content"
    file = MagicMock(spec=UploadFile)
    file.filename = "test.pdf"
    file.read = AsyncMock(return_value=file_content)
    return file

@pytest.fixture
def sample_starlette_upload_file():
    # Used for FastAPI UploadFile (starlette.datastructures.UploadFile)
    file_content = b"Sample file content"
    file = StarletteUploadFile(filename="test.pdf", file=io.BytesIO(file_content))
    return file

@pytest.mark.asyncio
async def test_upload_document_happy_path(mock_container):
    # Arrange
    files = [MagicMock(spec=UploadFile)]
    files[0].filename = "doc1.pdf"
    summary = MagicMock()
    summary.filename = "doc1.pdf"
    summary.text_chunks = 3
    summary.structured_data_extracted = True
    summary.reference_id = "REF123"
    summary.error = None
    result = MagicMock()
    result.message = "Upload successful"
    result.errors = []
    result.extractions = [summary]
    mock_container.document_pipeline_service.process_uploads.return_value = result

    # Act
    response = await routes.upload_document(files=files, container=mock_container)

    # Assert
    assert isinstance(response, UploadResponse)
    assert response.message == "Upload successful"
    assert response.errors == []
    assert len(response.extractions) == 1
    assert response.extractions[0].filename == "doc1.pdf"
    assert response.extractions[0].structured_data_extracted is True
    assert response.extractions[0].reference_id == "REF123"
    assert response.extractions[0].error is None

@pytest.mark.asyncio
async def test_upload_document_with_errors(mock_container):
    # Arrange
    files = [MagicMock(spec=UploadFile)]
    files[0].filename = "doc2.pdf"
    summary = MagicMock()
    summary.filename = "doc2.pdf"
    summary.text_chunks = 0
    summary.structured_data_extracted = False
    summary.reference_id = None
    summary.error = "Extraction failed"
    result = MagicMock()
    result.message = "Partial failure"
    result.errors = ["Extraction failed"]
    result.extractions = [summary]
    mock_container.document_pipeline_service.process_uploads.return_value = result

    # Act
    response = await routes.upload_document(files=files, container=mock_container)

    # Assert
    assert response.message == "Partial failure"
    assert response.errors == ["Extraction failed"]
    assert len(response.extractions) == 1
    assert response.extractions[0].error == "Extraction failed"
    assert response.extractions[0].structured_data_extracted is False

@pytest.mark.asyncio
async def test_upload_document_empty_file_list(mock_container):
    # Arrange
    files = []
    result = MagicMock()
    result.message = "No files uploaded"
    result.errors = ["No files"]
    result.extractions = []
    mock_container.document_pipeline_service.process_uploads.return_value = result

    # Act
    response = await routes.upload_document(files=files, container=mock_container)

    # Assert
    assert response.message == "No files uploaded"
    assert response.errors == ["No files"]
    assert response.extractions == []

@pytest.mark.asyncio
async def test_ask_question_happy_path(mock_container):
    # Arrange
    query = QAQuery(question="What is the delivery date?", filters=None)
    answer = SourcedAnswer(
        answer="The delivery date is 2024-06-01.",
        confidence_score=0.95,
        sources=[],
    )
    mock_container.rag_service.answer_question.return_value = answer

    # Act
    response = await routes.ask_question(query=query, container=mock_container)

    # Assert
    assert isinstance(response, SourcedAnswer)
    assert response.answer == "The delivery date is 2024-06-01."
    assert response.confidence_score == 0.95
    assert response.sources == []

@pytest.mark.asyncio
async def test_ask_question_safety_violation(mock_container):
    # Arrange
    query = QAQuery(question="How to hack the system?", filters=None)
    answer = SourcedAnswer(
        answer="I cannot answer this question as it violates safety guidelines.",
        confidence_score=1.0,
        sources=[],
    )
    mock_container.rag_service.answer_question.return_value = answer

    # Act
    response = await routes.ask_question(query=query, container=mock_container)

    # Assert
    assert response.answer == "I cannot answer this question as it violates safety guidelines."
    assert response.confidence_score == 1.0

@pytest.mark.asyncio
async def test_ask_question_rag_service_exception(mock_container):
    # Arrange
    query = QAQuery(question="What is the shipper name?", filters=None)
    mock_container.rag_service.answer_question.side_effect = Exception("RAG error")

    # Act & Assert
    with pytest.raises(HTTPException) as excinfo:
        await routes.ask_question(query=query, container=mock_container)
    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Internal server error processing question."

@pytest.mark.asyncio
async def test_extract_data_happy_path(sample_upload_file, mock_container):
    # Arrange
    content = b"Bill of Lading: REF123\nShipper: ACME Corp\nDelivery Date: 2024-06-01"
    sample_upload_file.read.return_value = content
    chunks = [MagicMock()]
    chunks[0].text = "Bill of Lading: REF123\nShipper: ACME Corp\nDelivery Date: 2024-06-01"
    mock_container.ingestion_service.process_file.return_value = chunks

    shipment_data = ShipmentData(
        reference_id="REF123",
        shipper="ACME Corp",
        delivery_date="2024-06-01"
    )
    extraction_response = ExtractionResponse(data=shipment_data, document_id="test.pdf")
    mock_container.extraction_service.extract_data.return_value = extraction_response

    # Act
    response = await routes.extract_data(file=sample_upload_file, container=mock_container)

    # Assert
    assert isinstance(response, ExtractionResponse)
    assert response.document_id == "test.pdf"
    assert response.data.reference_id == "REF123"
    assert response.data.shipper == "ACME Corp"
    assert response.data.delivery_date == "2024-06-01"

@pytest.mark.asyncio
async def test_extract_data_empty_text(sample_upload_file, mock_container):
    # Arrange
    sample_upload_file.read.return_value = b""
    mock_container.ingestion_service.process_file.return_value = []
    # Should not call extraction_service.extract_data

    # Act
    response = await routes.extract_data(file=sample_upload_file, container=mock_container)

    # Assert
    assert isinstance(response, ExtractionResponse)
    assert response.document_id == "test.pdf"
    assert response.data == ShipmentData()

@pytest.mark.asyncio
async def test_extract_data_extraction_service_exception(sample_upload_file, mock_container):
    # Arrange
    content = b"Some content"
    sample_upload_file.read.return_value = content
    chunks = [MagicMock()]
    chunks[0].text = "Some content"
    mock_container.ingestion_service.process_file.return_value = chunks
    mock_container.extraction_service.extract_data.side_effect = Exception("Extraction failed")

    # Act & Assert
    with pytest.raises(HTTPException) as excinfo:
        await routes.extract_data(file=sample_upload_file, container=mock_container)
    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Internal server error during extraction."

@pytest.mark.asyncio
async def test_extract_data_boundary_large_text(sample_upload_file, mock_container):
    # Arrange
    large_text = "A" * 10000
    sample_upload_file.read.return_value = large_text.encode()
    chunk = MagicMock()
    chunk.text = large_text
    mock_container.ingestion_service.process_file.return_value = [chunk]
    shipment_data = ShipmentData(reference_id="LARGE", shipper="Big Shipper")
    extraction_response = ExtractionResponse(data=shipment_data, document_id="test.pdf")
    mock_container.extraction_service.extract_data.return_value = extraction_response

    # Act
    response = await routes.extract_data(file=sample_upload_file, container=mock_container)

    # Assert
    assert response.data.reference_id == "LARGE"
    assert response.data.shipper == "Big Shipper"

@pytest.mark.asyncio
async def test_extract_data_invalid_file_type(mock_container):
    # Arrange
    file = MagicMock(spec=UploadFile)
    file.filename = "malicious.exe"
    file.read = AsyncMock(return_value=b"not a pdf")
    mock_container.ingestion_service.process_file.side_effect = Exception("Invalid file type")

    # Act & Assert
    with pytest.raises(HTTPException) as excinfo:
        await routes.extract_data(file=file, container=mock_container)
    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Internal server error during extraction."

@pytest.mark.asyncio
async def test_extract_data_partial_fields(sample_upload_file, mock_container):
    # Arrange
    content = b"Carrier: XYZ Logistics\nDriver: John Doe"
    sample_upload_file.read.return_value = content
    chunk = MagicMock()
    chunk.text = "Carrier: XYZ Logistics\nDriver: John Doe"
    mock_container.ingestion_service.process_file.return_value = [chunk]
    shipment_data = ShipmentData(
        carrier=CarrierInfo(carrier_name="XYZ Logistics"),
        driver=DriverInfo(driver_name="John Doe")
    )
    extraction_response = ExtractionResponse(data=shipment_data, document_id="test.pdf")
    mock_container.extraction_service.extract_data.return_value = extraction_response

    # Act
    response = await routes.extract_data(file=sample_upload_file, container=mock_container)

    # Assert
    assert response.data.carrier.carrier_name == "XYZ Logistics"
    assert response.data.driver.driver_name == "John Doe"

@pytest.mark.asyncio
async def test_extract_data_all_fields(sample_upload_file, mock_container):
    # Arrange
    content = b"""
    Reference ID: REF999
    Load ID: LID888
    PO Number: PO777
    Shipper: MegaCorp
    Consignee: RetailerX
    Carrier: FastTrans
    MC Number: MC12345
    Carrier Phone: 555-1234
    Driver: Alice Smith
    Driver Phone: 555-5678
    Truck Number: T123
    Trailer Number: TR456
    Pickup: Warehouse A, 123 Main St, CityA, ST, 12345, USA, 2024-06-01 08:00
    Drop: Store B, 456 Elm St, CityB, ST, 67890, USA, 2024-06-02 10:00
    Shipping Date: 2024-06-01
    Delivery Date: 2024-06-02
    Equipment Type: Flatbed
    Equipment Size: 53
    Load Type: FTL
    Commodity: Steel Beams, 56000.00 lbs, 10000 units, Heavy steel beams
    Rate: 2500.00 USD
    Special Instructions: Call before delivery
    Shipper Instructions: Load from dock 3
    Carrier Instructions: Secure with straps
    Dispatcher: Bob Manager, 555-9999, bob@megacorp.com
    """
    sample_upload_file.read.return_value = content
    chunk = MagicMock()
    chunk.text = content.decode()
    mock_container.ingestion_service.process_file.return_value = [chunk]
    shipment_data = ShipmentData(
        reference_id="REF999",
        load_id="LID888",
        po_number="PO777",
        shipper="MegaCorp",
        consignee="RetailerX",
        carrier=CarrierInfo(carrier_name="FastTrans", mc_number="MC12345", phone="555-1234"),
        driver=DriverInfo(driver_name="Alice Smith", cell_number="555-5678", truck_number="T123", trailer_number="TR456"),
        pickup=Location(
            name="Warehouse A",
            address="123 Main St",
            city="CityA",
            state="ST",
            zip_code="12345",
            country="USA",
            appointment_time="2024-06-01 08:00"
        ),
        drop=Location(
            name="Store B",
            address="456 Elm St",
            city="CityB",
            state="ST",
            zip_code="67890",
            country="USA",
            appointment_time="2024-06-02 10:00"
        ),
        shipping_date="2024-06-01",
        delivery_date="2024-06-02",
        equipment_type="Flatbed",
        equipment_size="53",
        load_type="FTL",
        commodities=[
            CommodityItem(
                commodity_name="Steel Beams",
                weight="56000.00 lbs",
                quantity="10000 units",
                description="Heavy steel beams"
            )
        ],
        rate_info=RateInfo(total_rate=2500.00, currency="USD"),
        special_instructions="Call before delivery",
        shipper_instructions="Load from dock 3",
        carrier_instructions="Secure with straps",
        dispatcher_name="Bob Manager",
        dispatcher_phone="555-9999",
        dispatcher_email="bob@megacorp.com"
    )
    extraction_response = ExtractionResponse(data=shipment_data, document_id="test.pdf")
    mock_container.extraction_service.extract_data.return_value = extraction_response

    # Act
    response = await routes.extract_data(file=sample_upload_file, container=mock_container)

    # Assert
    assert response.data.reference_id == "REF999"
    assert response.data.load_id == "LID888"
    assert response.data.po_number == "PO777"
    assert response.data.shipper == "MegaCorp"
    assert response.data.consignee == "RetailerX"
    assert response.data.carrier.carrier_name == "FastTrans"
    assert response.data.carrier.mc_number == "MC12345"
    assert response.data.carrier.phone == "555-1234"
    assert response.data.driver.driver_name == "Alice Smith"
    assert response.data.driver.cell_number == "555-5678"
    assert response.data.driver.truck_number == "T123"
    assert response.data.driver.trailer_number == "TR456"
    assert response.data.pickup.name == "Warehouse A"
    assert response.data.pickup.address == "123 Main St"
    assert response.data.pickup.city == "CityA"
    assert response.data.pickup.state == "ST"
    assert response.data.pickup.zip_code == "12345"
    assert response.data.pickup.country == "USA"
    assert response.data.pickup.appointment_time == "2024-06-01 08:00"
    assert response.data.drop.name == "Store B"
    assert response.data.drop.address == "456 Elm St"
    assert response.data.drop.city == "CityB"
    assert response.data.drop.state == "ST"
    assert response.data.drop.zip_code == "67890"
    assert response.data.drop.country == "USA"
    assert response.data.drop.appointment_time == "2024-06-02 10:00"
    assert response.data.shipping_date == "2024-06-01"
    assert response.data.delivery_date == "2024-06-02"
    assert response.data.equipment_type == "Flatbed"
    assert response.data.equipment_size == "53"
    assert response.data.load_type == "FTL"
    assert len(response.data.commodities) == 1
    assert response.data.commodities[0].commodity_name == "Steel Beams"
    assert response.data.commodities[0].weight == "56000.00 lbs"
    assert response.data.commodities[0].quantity == "10000 units"
    assert response.data.commodities[0].description == "Heavy steel beams"
    assert response.data.rate_info.total_rate == 2500.00
    assert response.data.rate_info.currency == "USD"
    assert response.data.special_instructions == "Call before delivery"
    assert response.data.shipper_instructions == "Load from dock 3"
    assert response.data.carrier_instructions == "Secure with straps"
    assert response.data.dispatcher_name == "Bob Manager"
    assert response.data.dispatcher_phone == "555-9999"
    assert response.data.dispatcher_email == "bob@megacorp.com"

@pytest.mark.asyncio
async def test_ping_endpoint():
    # Act
    response = await routes.ping()
    # Assert
    assert response == {"status": "pong"}
