from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

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
    chat_history: List[Dict[str, str]] = []

class SourcedAnswer(BaseModel):
    answer: str
    confidence_score: float
    sources: List[Chunk]

class ExtractionRequest(BaseModel):
    document_text: str
