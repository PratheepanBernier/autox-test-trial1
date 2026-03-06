import logging
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.src.dependencies import ServiceContainer, get_container
from backend.src.models.extraction_schema import ExtractionResponse, ShipmentData
from backend.src.models.schemas import QAQuery, SourcedAnswer, UploadResponse, UploadExtractionSummary

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    files: List[UploadFile] = File(...),
    container: ServiceContainer = Depends(get_container),
):
    """
    Upload one or more documents.
    Automatically extracts structured data and stores both text chunks and structured data in vector DB.
    """
    logger.info("Received upload request for %d files", len(files))
    result = await container.document_pipeline_service.process_uploads(files)
    return UploadResponse(
        message=result.message,
        errors=result.errors,
        extractions=[
            UploadExtractionSummary(
                filename=summary.filename,
                text_chunks=summary.text_chunks,
                structured_data_extracted=summary.structured_data_extracted,
                reference_id=summary.reference_id,
                error=summary.error,
            )
            for summary in result.extractions
        ],
    )

@router.post("/ask", response_model=SourcedAnswer)
async def ask_question(
    query: QAQuery,
    container: ServiceContainer = Depends(get_container),
):
    """
    Ask a question about the uploaded documents.
    """
    logger.info(f"Received question: {query.question}")
    try:
        response = container.rag_service.answer_question(query)
        logger.info(f"Answer generated with confidence {response.confidence_score}")
        return response
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error processing question.")

@router.post("/extract", response_model=ExtractionResponse)
async def extract_data(
    file: UploadFile = File(...),
    container: ServiceContainer = Depends(get_container),
):
    """
    Extract structured shipment data from a document.
    """
    logger.info(f"Received extraction request for file: {file.filename}")
    try:
        content = await file.read()
        logger.info(f"Extracting text from {file.filename}...")
        chunks = container.ingestion_service.process_file(content, file.filename)
        full_text = "\n".join([c.text for c in chunks])
        
        if not full_text:
            logger.warning(f"No text extracted from {file.filename}, returning empty data.")
            return ExtractionResponse(data=ShipmentData(), document_id=file.filename)

        logger.info(f"Text extracted ({len(full_text)} chars). Proceeding to structured extraction.")
        result = container.extraction_service.extract_data(full_text, file.filename)
        logger.info("Extraction completed successfully.")
        return result
        
    except Exception as e:
        logger.error(f"Error during extraction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during extraction.")

@router.get("/ping")
async def ping():
    logger.debug("Ping endpoint called.")
    return {"status": "pong"}
