import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException, UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile

from backend.src.api import routes
from backend.src.models.extraction_schema import (
    ExtractionResponse,
    ShipmentData,
    Location,
    CommodityItem,
    CarrierInfo,
    DriverInfo,
    RateInfo,
)
from backend.src.models.schemas import UploadResponse, UploadExtractionSummary, QAQuery, SourcedAnswer

@pytest.mark.asyncio
class TestOrderProcessingWorkflow:
    @pytest.fixture
    def mock_container(self):
        # Mock all services in the ServiceContainer
        container = MagicMock()
        container.document_pipeline_service.process_uploads = AsyncMock()
        container.ingestion_service.process_file = MagicMock()
        container.extraction_service.extract_data = MagicMock()
        container.rag_service.answer_question = MagicMock()
        return container

    @pytest.fixture
    def sample_upload_file(self):
        # Simulate a FastAPI UploadFile
        file = MagicMock(spec=UploadFile)
        file.filename = "test_document.pdf"
        file.read = AsyncMock(return_value=b"fake file content")
        return file

    @pytest.fixture
    def sample_upload_files(self, sample_upload_file):
        # List of UploadFile mocks
        file2 = MagicMock(spec=UploadFile)
        file2.filename = "test_document2.pdf"
        file2.read = AsyncMock(return_value=b"another fake content")
        return [sample_upload_file, file2]

    @pytest.fixture
    def sample_extraction_summary(self):
        return [
            MagicMock(
                filename="test_document.pdf",
                text_chunks=5,
                structured_data_extracted=True,
                reference_id="REF123",
                error=None,
            ),
            MagicMock(
                filename="test_document2.pdf",
                text_chunks=0,
                structured_data_extracted=False,
                reference_id=None,
                error="Extraction failed",
            ),
        ]

    @pytest.fixture
    def sample_upload_result(self, sample_extraction_summary):
        result = MagicMock()
        result.message = "2 files processed"
        result.errors = []
        result.extractions = sample_extraction_summary
        return result

    @pytest.fixture
    def sample_extraction_response(self):
        return ExtractionResponse(
            data=ShipmentData(
                reference_id="REF123",
                load_id="LOAD456",
                po_number="PO789",
                shipper="Acme Corp",
                consignee="Beta LLC",
                carrier=CarrierInfo(
                    carrier_name="CarrierX",
                    mc_number="MC123",
                    phone="555-1234",
                    email="carrierx@example.com",
                ),
                driver=DriverInfo(
                    driver_name="John Doe",
                    cell_number="555-5678",
                    truck_number="TRK001",
                    trailer_number="TRL002",
                ),
                pickup=Location(
                    name="Warehouse A",
                    address="123 Main St",
                    city="Metropolis",
                    state="NY",
                    zip_code="10001",
                    country="USA",
                    appointment_time="2024-06-01T09:00:00",
                    po_number="PO789",
                ),
                drop=Location(
                    name="Distribution Center",
                    address="456 Elm St",
                    city="Gotham",
                    state="NJ",
                    zip_code="07001",
                    country="USA",
                    appointment_time="2024-06-02T14:00:00",
                    po_number="PO789",
                ),
                shipping_date="2024-06-01",
                delivery_date="2024-06-02",
                created_on="2024-05-30",
                booking_date="2024-05-29",
                equipment_type="Flatbed",
                equipment_size="53",
                load_type="FTL",
                commodities=[
                    CommodityItem(
                        commodity_name="Steel Beams",
                        weight="56000.00 lbs",
                        quantity="10000 units",
                        description="High-grade steel beams",
                    )
                ],
                rate_info=RateInfo(
                    total_rate=2500.0,
                    currency="USD",
                    rate_breakdown={"base": 2000, "fuel": 500},
                ),
                special_instructions="Handle with care",
                shipper_instructions="Call before arrival",
                carrier_instructions="No overnight parking",
                dispatcher_name="Alice Smith",
                dispatcher_phone="555-9999",
                dispatcher_email="alice@acme.com",
                additional_data={"custom_field": "custom_value"},
            ),
            document_id="test_document.pdf",
        )

    @pytest.fixture
    def sample_empty_extraction_response(self):
        return ExtractionResponse(
            data=ShipmentData(),
            document_id="test_document.pdf"
        )

    @pytest.fixture
    def sample_qa_query(self):
        return QAQuery(question="What is the delivery date for REF123?")

    @pytest.fixture
    def sample_sourced_answer(self):
        return SourcedAnswer(
            answer="The delivery date for REF123 is 2024-06-02.",
            confidence_score=0.95,
            sources=[],
        )

    # --- /upload endpoint ---

    async def test_upload_document_happy_path(self, mock_container, sample_upload_files, sample_upload_result, sample_extraction_summary):
        # Arrange
        mock_container.document_pipeline_service.process_uploads.return_value = sample_upload_result

        # Act
        response = await routes.upload_document(files=sample_upload_files, container=mock_container)

        # Assert
        assert isinstance(response, UploadResponse)
        assert response.message == "2 files processed"
        assert response.errors == []
        assert len(response.extractions) == 2
        assert response.extractions[0].filename == "test_document.pdf"
        assert response.extractions[0].structured_data_extracted is True
        assert response.extractions[1].filename == "test_document2.pdf"
        assert response.extractions[1].error == "Extraction failed"

    async def test_upload_document_with_errors(self, mock_container, sample_upload_files, sample_upload_result):
        # Arrange
        sample_upload_result.errors = ["File corrupted"]
        mock_container.document_pipeline_service.process_uploads.return_value = sample_upload_result

        # Act
        response = await routes.upload_document(files=sample_upload_files, container=mock_container)

        # Assert
        assert response.errors == ["File corrupted"]

    async def test_upload_document_empty_file_list(self, mock_container):
        # Arrange
        files = []
        mock_container.document_pipeline_service.process_uploads.return_value = MagicMock(
            message="No files uploaded", errors=["No files"], extractions=[]
        )

        # Act
        response = await routes.upload_document(files=files, container=mock_container)

        # Assert
        assert response.message == "No files uploaded"
        assert response.errors == ["No files"]
        assert response.extractions == []

    # --- /extract endpoint ---

    async def test_extract_data_happy_path(self, mock_container, sample_upload_file, sample_extraction_response):
        # Arrange
        mock_container.ingestion_service.process_file.return_value = [
            MagicMock(text="Shipment for REF123. Delivery date: 2024-06-02.")
        ]
        mock_container.extraction_service.extract_data.return_value = sample_extraction_response

        # Act
        response = await routes.extract_data(file=sample_upload_file, container=mock_container)

        # Assert
        assert isinstance(response, ExtractionResponse)
        assert response.document_id == "test_document.pdf"
        assert response.data.reference_id == "REF123"
        assert response.data.delivery_date == "2024-06-02"
        assert response.data.carrier.carrier_name == "CarrierX"
        assert response.data.commodities[0].commodity_name == "Steel Beams"

    async def test_extract_data_empty_text(self, mock_container, sample_upload_file, sample_empty_extraction_response):
        # Arrange
        mock_container.ingestion_service.process_file.return_value = []
        # extraction_service.extract_data should not be called in this case

        # Act
        response = await routes.extract_data(file=sample_upload_file, container=mock_container)

        # Assert
        assert isinstance(response, ExtractionResponse)
        assert response.document_id == "test_document.pdf"
        assert response.data.reference_id is None
        assert response.data.commodities is None

    async def test_extract_data_extraction_service_error(self, mock_container, sample_upload_file):
        # Arrange
        mock_container.ingestion_service.process_file.return_value = [
            MagicMock(text="Some text")
        ]
        mock_container.extraction_service.extract_data.side_effect = Exception("Extraction failed")

        # Act & Assert
        with pytest.raises(HTTPException) as excinfo:
            await routes.extract_data(file=sample_upload_file, container=mock_container)
        assert excinfo.value.status_code == 500
        assert "Internal server error during extraction." in str(excinfo.value.detail)

    async def test_extract_data_file_read_error(self, mock_container, sample_upload_file):
        # Arrange
        sample_upload_file.read.side_effect = Exception("Read failed")

        # Act & Assert
        with pytest.raises(HTTPException) as excinfo:
            await routes.extract_data(file=sample_upload_file, container=mock_container)
        assert excinfo.value.status_code == 500
        assert "Internal server error during extraction." in str(excinfo.value.detail)

    # --- /ask endpoint ---

    async def test_ask_question_happy_path(self, mock_container, sample_qa_query, sample_sourced_answer):
        # Arrange
        mock_container.rag_service.answer_question.return_value = sample_sourced_answer

        # Act
        response = await routes.ask_question(query=sample_qa_query, container=mock_container)

        # Assert
        assert isinstance(response, SourcedAnswer)
        assert response.answer == "The delivery date for REF123 is 2024-06-02."
        assert response.confidence_score == 0.95

    async def test_ask_question_rag_service_error(self, mock_container, sample_qa_query):
        # Arrange
        mock_container.rag_service.answer_question.side_effect = Exception("RAG error")

        # Act & Assert
        with pytest.raises(HTTPException) as excinfo:
            await routes.ask_question(query=sample_qa_query, container=mock_container)
        assert excinfo.value.status_code == 500
        assert "Internal server error processing question." in str(excinfo.value.detail)

    async def test_ask_question_edge_case_empty_answer(self, mock_container, sample_qa_query):
        # Arrange
        mock_container.rag_service.answer_question.return_value = SourcedAnswer(
            answer="I cannot find the answer in the provided documents.",
            confidence_score=0.0,
            sources=[],
        )

        # Act
        response = await routes.ask_question(query=sample_qa_query, container=mock_container)

        # Assert
        assert response.answer == "I cannot find the answer in the provided documents."
        assert response.confidence_score == 0.0

    # --- /ping endpoint ---

    async def test_ping(self):
        # Act
        response = await routes.ping()

        # Assert
        assert response == {"status": "pong"}
