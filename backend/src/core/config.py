from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "Logistics Document Intelligence Assistant"
    API_V1_STR: str = "/api/v1"
    
    # LLM Settings
    GROQ_API_KEY: str = "gsk_your_api_key"
    QA_MODEL: str = "llama-3.3-70b-versatile"  # Updated to current supported model
    VISION_MODEL: str = "llama-3.2-11b-vision-preview" # For OCR if needed
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunking Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Retrieval Settings
    TOP_K: int = 4
    SIMILARITY_THRESHOLD: float = 0.5
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "ignore"

settings = Settings()
