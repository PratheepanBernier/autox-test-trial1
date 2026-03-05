from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from services.ingestion import ingestion_service
from services.vector_store import vector_store_service
from services.rag import rag_service
from services.extraction import extraction_service
from models.schemas import QAQuery, SourcedAnswer
from models.extraction_schema import ExtractionResponse, ShipmentData
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/upload", response_model=dict)
async def upload_document(files: List[UploadFile] = File(...)):
    """
    Upload one or more documents.
    Automatically extracts structured data and stores both text chunks and structured data in vector DB.
    """
    logger.info(f"Received upload request for {len(files)} files.")
    processed_count = 0
    errors = []
    extractions = []
    
    for file in files:
        try:
            logger.info(f"Processing file: {file.filename}")
            content = await file.read()
            
            # Step 1: Process file into semantic text chunks
            chunks = ingestion_service.process_file(content, file.filename)
            if not chunks:
                msg = f"No text extracted from {file.filename}"
                logger.warning(msg)
                errors.append(msg)
                continue
            
            # Step 2: Store text chunks in vector DB
            vector_store_service.add_documents(chunks)
            logger.info(f"Stored {len(chunks)} text chunks from {file.filename}")
            
            # Step 3: Auto-extract structured data
            try:
                full_text = "\n".join([c.text for c in chunks])
                logger.info(f"Auto-extracting structured data from {file.filename}...")
                
                extraction_result = extraction_service.extract_data(full_text, file.filename)
                
                # Step 4: Create structured data chunk and store in vector DB
                structured_chunk = extraction_service.create_structured_chunk(extraction_result, file.filename)
                vector_store_service.add_documents([structured_chunk])
                logger.info(f"Stored structured data chunk for {file.filename}")
                
                # Track extraction for response
                extractions.append({
                    "filename": file.filename,
                    "text_chunks": len(chunks),
                    "structured_data_extracted": True,
                    "reference_id": extraction_result.data.reference_id
                })
            except Exception as extraction_error:
                logger.error(f"Extraction failed for {file.filename}: {str(extraction_error)}", exc_info=True)
                extractions.append({
                    "filename": file.filename,
                    "text_chunks": len(chunks),
                    "structured_data_extracted": False,
                    "error": str(extraction_error)
                })
            
            processed_count += 1
            logger.info(f"Successfully processed {file.filename}: {len(chunks)} text chunks")
            
        except Exception as e:
            msg = f"Error processing {file.filename}: {str(e)}"
            logger.error(msg, exc_info=True)
            errors.append(msg)
            
    return {
        "message": f"Successfully processed {processed_count} documents.",
        "errors": errors,
        "extractions": extractions
    }

@router.post("/ask", response_model=SourcedAnswer)
async def ask_question(query: QAQuery):
    """
    Ask a question about the uploaded documents.
    """
    logger.info(f"Received question: {query.question}")
    try:
        response = rag_service.answer_question(query)
        logger.info(f"Answer generated with confidence {response.confidence_score}")
        return response
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error processing question.")

@router.post("/extract", response_model=ExtractionResponse)
async def extract_data(file: UploadFile = File(...)):
    """
    Extract structured shipment data from a document.
    """
    logger.info(f"Received extraction request for file: {file.filename}")
    try:
        content = await file.read()
        logger.info(f"Extracting text from {file.filename}...")
        chunks = ingestion_service.process_file(content, file.filename)
        full_text = "\n".join([c.text for c in chunks])
        
        if not full_text:
            logger.warning(f"No text extracted from {file.filename}, returning empty data.")
            return ExtractionResponse(data=ShipmentData(), document_id=file.filename)

        logger.info(f"Text extracted ({len(full_text)} chars). Proceeding to structured extraction.")
        result = extraction_service.extract_data(full_text, file.filename)
        logger.info("Extraction completed successfully.")
        return result
        
    except Exception as e:
        logger.error(f"Error during extraction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during extraction.")

@router.get("/ping")
async def ping():
    logger.debug("Ping endpoint called.")
    return {"status": "pong"}
