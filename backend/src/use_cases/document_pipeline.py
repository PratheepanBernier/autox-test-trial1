from dataclasses import dataclass, field
from typing import List, Protocol

from fastapi import UploadFile

from backend.src.models.extraction_schema import ExtractionResponse
from backend.src.models.schemas import Chunk


class IngestionServicePort(Protocol):
    def process_file(self, file_content: bytes, filename: str) -> List[Chunk]:
        ...


class VectorStoreServicePort(Protocol):
    def add_documents(self, chunks: List[Chunk]) -> None:
        ...


class ExtractionServicePort(Protocol):
    def extract_data(self, text: str, filename: str = "unknown") -> ExtractionResponse:
        ...

    def create_structured_chunk(self, extraction: ExtractionResponse, filename: str) -> Chunk:
        ...


@dataclass
class UploadExtractionSummary:
    filename: str
    text_chunks: int
    structured_data_extracted: bool
    reference_id: str | None = None
    error: str | None = None


@dataclass
class UploadResult:
    processed_count: int = 0
    errors: List[str] = field(default_factory=list)
    extractions: List[UploadExtractionSummary] = field(default_factory=list)

    @property
    def message(self) -> str:
        return f"Successfully processed {self.processed_count} documents."


class DocumentPipelineService:
    """Coordinates ingestion, indexing, and structured extraction."""

    def __init__(
        self,
        ingestion_service: IngestionServicePort,
        vector_store_service: VectorStoreServicePort,
        extraction_service: ExtractionServicePort,
    ) -> None:
        self._ingestion_service = ingestion_service
        self._vector_store_service = vector_store_service
        self._extraction_service = extraction_service

    async def process_uploads(self, files: List[UploadFile]) -> UploadResult:
        result = UploadResult()

        for file in files:
            try:
                content = await file.read()
                chunks = self._ingestion_service.process_file(content, file.filename)
                if not chunks:
                    result.errors.append(f"No text extracted from {file.filename}")
                    continue

                self._vector_store_service.add_documents(chunks)
                full_text = "\n".join(chunk.text for chunk in chunks)

                try:
                    extraction = self._extraction_service.extract_data(full_text, file.filename)
                    structured_chunk = self._extraction_service.create_structured_chunk(
                        extraction, file.filename
                    )
                    self._vector_store_service.add_documents([structured_chunk])

                    result.extractions.append(
                        UploadExtractionSummary(
                            filename=file.filename,
                            text_chunks=len(chunks),
                            structured_data_extracted=True,
                            reference_id=extraction.data.reference_id,
                        )
                    )
                except Exception as extraction_error:
                    result.extractions.append(
                        UploadExtractionSummary(
                            filename=file.filename,
                            text_chunks=len(chunks),
                            structured_data_extracted=False,
                            error=str(extraction_error),
                        )
                    )

                result.processed_count += 1
            except Exception as process_error:
                result.errors.append(f"Error processing {file.filename}: {process_error}")

        return result
