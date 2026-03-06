import pytest
from unittest.mock import patch, MagicMock

import backend.src.dependencies as dependencies


@pytest.fixture(autouse=True)
def reset_cached_properties():
    # Reset cached properties between tests to avoid cross-test pollution
    yield
    for attr in [
        "_cached_ingestion_service",
        "_cached_vector_store_service",
        "_cached_extraction_service",
        "_cached_rag_service",
        "_cached_document_pipeline_service",
    ]:
        dependencies.ServiceContainer.__dict__.pop(attr, None)
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


def test_ingestion_service_returns_instance_and_is_cached():
    with patch("backend.src.services.ingestion.DocumentIngestionService") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        container = dependencies.ServiceContainer()
        # Act
        service1 = container.ingestion_service
        service2 = container.ingestion_service
        # Assert
        assert service1 is mock_instance
        assert service2 is service1
        mock_cls.assert_called_once()


def test_vector_store_service_returns_instance_and_is_cached():
    with patch("backend.src.services.vector_store.VectorStoreService") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        container = dependencies.ServiceContainer()
        service1 = container.vector_store_service
        service2 = container.vector_store_service
        assert service1 is mock_instance
        assert service2 is service1
        mock_cls.assert_called_once()


def test_extraction_service_returns_instance_and_is_cached():
    with patch("backend.src.services.extraction.ExtractionService") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        container = dependencies.ServiceContainer()
        service1 = container.extraction_service
        service2 = container.extraction_service
        assert service1 is mock_instance
        assert service2 is service1
        mock_cls.assert_called_once()


def test_rag_service_returns_instance_and_is_cached_and_uses_vector_store():
    with patch("backend.src.services.rag.RAGService") as mock_rag_cls, \
         patch("backend.src.services.vector_store.VectorStoreService") as mock_vector_cls:
        mock_vector_instance = MagicMock()
        mock_vector_cls.return_value = mock_vector_instance
        mock_rag_instance = MagicMock()
        mock_rag_cls.return_value = mock_rag_instance
        container = dependencies.ServiceContainer()
        # Act
        rag1 = container.rag_service
        rag2 = container.rag_service
        # Assert
        assert rag1 is mock_rag_instance
        assert rag2 is rag1
        mock_rag_cls.assert_called_once_with(vector_store_service=mock_vector_instance)
        mock_vector_cls.assert_called_once()


def test_document_pipeline_service_returns_instance_and_is_cached_and_uses_dependencies():
    with patch("backend.src.services.ingestion.DocumentIngestionService") as mock_ingest_cls, \
         patch("backend.src.services.vector_store.VectorStoreService") as mock_vector_cls, \
         patch("backend.src.services.extraction.ExtractionService") as mock_extract_cls, \
         patch("backend.src.use_cases.document_pipeline.DocumentPipelineService") as mock_pipeline_cls:
        mock_ingest_instance = MagicMock()
        mock_vector_instance = MagicMock()
        mock_extract_instance = MagicMock()
        mock_pipeline_instance = MagicMock()
        mock_ingest_cls.return_value = mock_ingest_instance
        mock_vector_cls.return_value = mock_vector_instance
        mock_extract_cls.return_value = mock_extract_instance
        mock_pipeline_cls.return_value = mock_pipeline_instance

        container = dependencies.ServiceContainer()
        pipeline1 = container.document_pipeline_service
        pipeline2 = container.document_pipeline_service

        assert pipeline1 is mock_pipeline_instance
        assert pipeline2 is pipeline1
        mock_pipeline_cls.assert_called_once_with(
            ingestion_service=mock_ingest_instance,
            vector_store_service=mock_vector_instance,
            extraction_service=mock_extract_instance,
        )
        mock_ingest_cls.assert_called_once()
        mock_vector_cls.assert_called_once()
        mock_extract_cls.assert_called_once()


def test_services_are_independent_instances():
    # Each service property should be a different object (except for caching)
    with patch("backend.src.services.ingestion.DocumentIngestionService") as mock_ingest_cls, \
         patch("backend.src.services.vector_store.VectorStoreService") as mock_vector_cls, \
         patch("backend.src.services.extraction.ExtractionService") as mock_extract_cls, \
         patch("backend.src.services.rag.RAGService") as mock_rag_cls, \
         patch("backend.src.use_cases.document_pipeline.DocumentPipelineService") as mock_pipeline_cls:
        mock_ingest_cls.return_value = MagicMock(name="ingest")
        mock_vector_cls.return_value = MagicMock(name="vector")
        mock_extract_cls.return_value = MagicMock(name="extract")
        mock_rag_cls.return_value = MagicMock(name="rag")
        mock_pipeline_cls.return_value = MagicMock(name="pipeline")

        container = dependencies.ServiceContainer()
        ingestion = container.ingestion_service
        vector = container.vector_store_service
        extraction = container.extraction_service
        rag = container.rag_service
        pipeline = container.document_pipeline_service

        assert ingestion is not vector
        assert ingestion is not extraction
        assert rag is not pipeline
        assert pipeline is not extraction


def test_cached_property_isolation_between_instances():
    # Arrange
    with patch("backend.src.services.ingestion.DocumentIngestionService") as mock_ingest_cls:
        mock_ingest_cls.return_value = MagicMock()
        c1 = dependencies.ServiceContainer()
        c2 = dependencies.ServiceContainer()
        # Act
        s1 = c1.ingestion_service
        s2 = c2.ingestion_service
        # Assert
        assert s1 is not s2  # Different containers, different cached properties


def test_error_on_service_instantiation_propagates():
    # Simulate error in service constructor
    with patch("backend.src.services.extraction.ExtractionService", side_effect=RuntimeError("fail")):
        container = dependencies.ServiceContainer()
        with pytest.raises(RuntimeError, match="fail"):
            _ = container.extraction_service


def test_document_pipeline_service_dependency_order():
    # Ensure dependencies are constructed in the correct order
    call_order = []
    def make_mock(name):
        def ctor(*a, **k):
            call_order.append(name)
            return MagicMock(name=name)
        return ctor

    with patch("backend.src.services.ingestion.DocumentIngestionService", side_effect=make_mock("ingest")), \
         patch("backend.src.services.vector_store.VectorStoreService", side_effect=make_mock("vector")), \
         patch("backend.src.services.extraction.ExtractionService", side_effect=make_mock("extract")), \
         patch("backend.src.use_cases.document_pipeline.DocumentPipelineService", side_effect=make_mock("pipeline")):
        container = dependencies.ServiceContainer()
        _ = container.document_pipeline_service
        # Assert that dependencies are constructed before pipeline
        assert call_order[:3] == ["ingest", "vector", "extract"]
        assert call_order[-1] == "pipeline"
        assert len(call_order) == 4
