from typing import Dict, List, Optional

from pydantic import BaseModel, Field

class DocumentMetadata(BaseModel):
    filename: str
    page_number: Optional[int] = None
    chunk_id: int
    source: str
    chunk_type: str = "text"  # "text" or "structured_data"
    
class Chunk(BaseModel):
    text: str
    metadata: DocumentMetadata

class QAQuery(BaseModel):
    question: str
    chat_history: List[Dict[str, str]] = Field(default_factory=list)

class SourcedAnswer(BaseModel):
    answer: str
    confidence_score: float
    sources: List[Chunk]

class ExtractionRequest(BaseModel):
    document_text: str


class UploadExtractionSummary(BaseModel):
    filename: str
    text_chunks: int
    structured_data_extracted: bool
    reference_id: Optional[str] = None
    error: Optional[str] = None


class UploadResponse(BaseModel):
    message: str
    errors: List[str]
    extractions: List[UploadExtractionSummary]
