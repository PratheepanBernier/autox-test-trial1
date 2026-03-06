from functools import cached_property

from backend.src.services.extraction import ExtractionService
from backend.src.services.ingestion import DocumentIngestionService
from backend.src.services.rag import RAGService
from backend.src.services.vector_store import VectorStoreService
from backend.src.use_cases.document_pipeline import DocumentPipelineService


class ServiceContainer:
    @cached_property
    def ingestion_service(self) -> DocumentIngestionService:
        return DocumentIngestionService()

    @cached_property
    def vector_store_service(self) -> VectorStoreService:
        return VectorStoreService()

    @cached_property
    def extraction_service(self) -> ExtractionService:
        return ExtractionService()

    @cached_property
    def rag_service(self) -> RAGService:
        return RAGService(vector_store_service=self.vector_store_service)

    @cached_property
    def document_pipeline_service(self) -> DocumentPipelineService:
        return DocumentPipelineService(
            ingestion_service=self.ingestion_service,
            vector_store_service=self.vector_store_service,
            extraction_service=self.extraction_service,
        )


container = ServiceContainer()


def get_container() -> ServiceContainer:
    return container
