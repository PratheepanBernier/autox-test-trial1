import pytest
from unittest.mock import patch, MagicMock

import backend.src.dependencies as dependencies


@pytest.fixture(autouse=True)
def reset_cached_properties():
    # Reset cached_property cache for ServiceContainer between tests
    yield
    for attr in [
        "_cached_ingestion_service",
        "_cached_vector_store_service",
        "_cached_extraction_service",
        "_cached_rag_service",
        "_cached_document_pipeline_service",
    ]:
        dependencies.ServiceContainer.__dict__.get(attr, None)
        dependencies.ServiceContainer.__dict__.pop(attr, None)
    for attr in [
        "ingestion_service",
        "vector_store_service",
        "extraction_service",
        "rag_service",
        "document_pipeline_service",
    ]:
        dependencies.ServiceContainer.__dict__.get(attr, None)
    # Remove cached_property values from the instance
    for attr in [
        "ingestion_service",
        "vector_store_service",
        "extraction_service",
        "rag_service",
        "document_pipeline_service",
    ]:
        dependencies.container.__dict__.pop(attr, None)


def test_get_container_returns_singleton():
    # Arrange & Act
    c1 = dependencies.get_container()
    c2 = dependencies.get_container()
    # Assert
    assert isinstance(c1, dependencies.ServiceContainer)
    assert c1 is c2  # Singleton instance


def test_ingestion_service_is_singleton_and_type():
    with patch("backend.src.services.ingestion.DocumentIngestionService") as MockIngestion:
        MockIngestion.return_value = MagicMock(name="MockIngestion")
        container = dependencies.ServiceContainer()
        # Act
        service1 = container.ingestion_service
        service2 = container.ingestion_service
        # Assert
        assert service1 is service2
        assert service1 is MockIngestion.return_value


def test_vector_store_service_is_singleton_and_type():
    with patch("backend.src.services.vector_store.VectorStoreService") as MockVector:
        MockVector.return_value = MagicMock(name="MockVector")
        container = dependencies.ServiceContainer()
        service1 = container.vector_store_service
        service2 = container.vector_store_service
        assert service1 is service2
        assert service1 is MockVector.return_value


def test_extraction_service_is_singleton_and_type():
    with patch("backend.src.services.extraction.ExtractionService") as MockExtraction:
        MockExtraction.return_value = MagicMock(name="MockExtraction")
        container = dependencies.ServiceContainer()
        service1 = container.extraction_service
        service2 = container.extraction_service
        assert service1 is service2
        assert service1 is MockExtraction.return_value


def test_rag_service_is_singleton_and_type_and_depends_on_vector_store():
    with patch("backend.src.services.rag.RAGService") as MockRAG, \
         patch("backend.src.services.vector_store.VectorStoreService") as MockVector:
        mock_vector_instance = MagicMock(name="MockVector")
        MockVector.return_value = mock_vector_instance
        mock_rag_instance = MagicMock(name="MockRAG")
        MockRAG.return_value = mock_rag_instance

        container = dependencies.ServiceContainer()
        # Act
        rag1 = container.rag_service
        rag2 = container.rag_service
        # Assert
        assert rag1 is rag2
        assert rag1 is mock_rag_instance
        MockRAG.assert_called_once_with(vector_store_service=mock_vector_instance)


def test_document_pipeline_service_is_singleton_and_type_and_depends_on_others():
    with patch("backend.src.services.ingestion.DocumentIngestionService") as MockIngestion, \
         patch("backend.src.services.vector_store.VectorStoreService") as MockVector, \
         patch("backend.src.services.extraction.ExtractionService") as MockExtraction, \
         patch("backend.src.use_cases.document_pipeline.DocumentPipelineService") as MockPipeline:

        mock_ingestion = MagicMock(name="MockIngestion")
        mock_vector = MagicMock(name="MockVector")
        mock_extraction = MagicMock(name="MockExtraction")
        mock_pipeline = MagicMock(name="MockPipeline")

        MockIngestion.return_value = mock_ingestion
        MockVector.return_value = mock_vector
        MockExtraction.return_value = mock_extraction
        MockPipeline.return_value = mock_pipeline

        container = dependencies.ServiceContainer()
        # Act
        pipeline1 = container.document_pipeline_service
        pipeline2 = container.document_pipeline_service
        # Assert
        assert pipeline1 is pipeline2
        assert pipeline1 is mock_pipeline
        MockPipeline.assert_called_once_with(
            ingestion_service=mock_ingestion,
            vector_store_service=mock_vector,
            extraction_service=mock_extraction,
        )


def test_services_are_independent_instances():
    # Each ServiceContainer instance should have its own cached services
    with patch("backend.src.services.ingestion.DocumentIngestionService") as MockIngestion:
        MockIngestion.side_effect = [MagicMock(name="MockIngestion1"), MagicMock(name="MockIngestion2")]
        c1 = dependencies.ServiceContainer()
        c2 = dependencies.ServiceContainer()
        s1 = c1.ingestion_service
        s2 = c2.ingestion_service
        assert s1 is not s2
        assert MockIngestion.call_count == 2


def test_cached_property_isolation_between_services():
    # Changing one service's cached property should not affect another's
    with patch("backend.src.services.ingestion.DocumentIngestionService") as MockIngestion:
        mock1 = MagicMock(name="MockIngestion1")
        mock2 = MagicMock(name="MockIngestion2")
        MockIngestion.side_effect = [mock1, mock2]
        c1 = dependencies.ServiceContainer()
        c2 = dependencies.ServiceContainer()
        assert c1.ingestion_service is mock1
        assert c2.ingestion_service is mock2


def test_error_on_service_init_propagates():
    # If a service constructor raises, the error should propagate
    with patch("backend.src.services.extraction.ExtractionService", side_effect=RuntimeError("fail")):
        container = dependencies.ServiceContainer()
        with pytest.raises(RuntimeError, match="fail"):
            _ = container.extraction_service


def test_document_pipeline_service_dependency_error_propagates():
    # If a dependency fails, the pipeline service should not be constructed
    with patch("backend.src.services.ingestion.DocumentIngestionService", side_effect=ValueError("ingest fail")), \
         patch("backend.src.services.vector_store.VectorStoreService") as MockVector, \
         patch("backend.src.services.extraction.ExtractionService") as MockExtraction, \
         patch("backend.src.use_cases.document_pipeline.DocumentPipelineService") as MockPipeline:
        MockVector.return_value = MagicMock()
        MockExtraction.return_value = MagicMock()
        MockPipeline.return_value = MagicMock()
        container = dependencies.ServiceContainer()
        with pytest.raises(ValueError, match="ingest fail"):
            _ = container.document_pipeline_service


def test_rag_service_dependency_error_propagates():
    # If vector_store_service fails, rag_service should not be constructed
    with patch("backend.src.services.vector_store.VectorStoreService", side_effect=KeyError("vector fail")), \
         patch("backend.src.services.rag.RAGService") as MockRAG:
        MockRAG.return_value = MagicMock()
        container = dependencies.ServiceContainer()
        with pytest.raises(KeyError, match="vector fail"):
            _ = container.rag_service
