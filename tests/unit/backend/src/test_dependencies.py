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
        if hasattr(dependencies.container, attr):
            dependencies.container.__dict__.pop(attr, None)


@pytest.fixture
def mock_services():
    with patch("backend.src.services.ingestion.DocumentIngestionService") as mock_ingest, \
         patch("backend.src.services.vector_store.VectorStoreService") as mock_vector, \
         patch("backend.src.services.extraction.ExtractionService") as mock_extract, \
         patch("backend.src.services.rag.RAGService") as mock_rag, \
         patch("backend.src.use_cases.document_pipeline.DocumentPipelineService") as mock_pipeline:
        yield {
            "ingestion": mock_ingest,
            "vector": mock_vector,
            "extraction": mock_extract,
            "rag": mock_rag,
            "pipeline": mock_pipeline,
        }


def test_ingestion_service_happy_path(mock_services):
    mock_instance = MagicMock(name="IngestionServiceInstance")
    mock_services["ingestion"].return_value = mock_instance
    container = dependencies.ServiceContainer()
    result = container.ingestion_service
    assert result is mock_instance
    # Cached property: should return same instance
    assert container.ingestion_service is result
    mock_services["ingestion"].assert_called_once()


def test_vector_store_service_happy_path(mock_services):
    mock_instance = MagicMock(name="VectorStoreServiceInstance")
    mock_services["vector"].return_value = mock_instance
    container = dependencies.ServiceContainer()
    result = container.vector_store_service
    assert result is mock_instance
    assert container.vector_store_service is result
    mock_services["vector"].assert_called_once()


def test_extraction_service_happy_path(mock_services):
    mock_instance = MagicMock(name="ExtractionServiceInstance")
    mock_services["extraction"].return_value = mock_instance
    container = dependencies.ServiceContainer()
    result = container.extraction_service
    assert result is mock_instance
    assert container.extraction_service is result
    mock_services["extraction"].assert_called_once()


def test_rag_service_happy_path_and_dependency(mock_services):
    mock_vector_instance = MagicMock(name="VectorStoreServiceInstance")
    mock_services["vector"].return_value = mock_vector_instance
    mock_rag_instance = MagicMock(name="RAGServiceInstance")
    mock_services["rag"].return_value = mock_rag_instance
    container = dependencies.ServiceContainer()
    # Access vector_store_service first to ensure it's cached
    vector_service = container.vector_store_service
    assert vector_service is mock_vector_instance
    rag_service = container.rag_service
    assert rag_service is mock_rag_instance
    # RAGService should be constructed with the vector_store_service instance
    mock_services["rag"].assert_called_once_with(vector_store_service=mock_vector_instance)
    # Cached property: should return same instance
    assert container.rag_service is rag_service


def test_document_pipeline_service_happy_path_and_dependencies(mock_services):
    mock_ingest_instance = MagicMock(name="IngestionServiceInstance")
    mock_vector_instance = MagicMock(name="VectorStoreServiceInstance")
    mock_extract_instance = MagicMock(name="ExtractionServiceInstance")
    mock_pipeline_instance = MagicMock(name="DocumentPipelineServiceInstance")
    mock_services["ingestion"].return_value = mock_ingest_instance
    mock_services["vector"].return_value = mock_vector_instance
    mock_services["extraction"].return_value = mock_extract_instance
    mock_services["pipeline"].return_value = mock_pipeline_instance

    container = dependencies.ServiceContainer()
    # Access dependencies first to ensure they are cached
    ingestion_service = container.ingestion_service
    vector_service = container.vector_store_service
    extraction_service = container.extraction_service

    pipeline_service = container.document_pipeline_service
    assert pipeline_service is mock_pipeline_instance
    mock_services["pipeline"].assert_called_once_with(
        ingestion_service=ingestion_service,
        vector_store_service=vector_service,
        extraction_service=extraction_service,
    )
    # Cached property: should return same instance
    assert container.document_pipeline_service is pipeline_service


def test_services_are_singletons_per_container_instance(mock_services):
    # Each property should be cached and return the same instance per container
    mock_services["ingestion"].return_value = MagicMock()
    mock_services["vector"].return_value = MagicMock()
    mock_services["extraction"].return_value = MagicMock()
    mock_services["rag"].return_value = MagicMock()
    mock_services["pipeline"].return_value = MagicMock()
    container = dependencies.ServiceContainer()
    assert container.ingestion_service is container.ingestion_service
    assert container.vector_store_service is container.vector_store_service
    assert container.extraction_service is container.extraction_service
    assert container.rag_service is container.rag_service
    assert container.document_pipeline_service is container.document_pipeline_service


def test_services_are_not_shared_across_containers(mock_services):
    # Each ServiceContainer instance should have its own cached properties
    mock_services["ingestion"].side_effect = [MagicMock(name="A"), MagicMock(name="B")]
    container1 = dependencies.ServiceContainer()
    container2 = dependencies.ServiceContainer()
    assert container1.ingestion_service is not container2.ingestion_service
    assert mock_services["ingestion"].call_count == 2


def test_get_container_returns_singleton():
    # Should always return the same container instance
    c1 = dependencies.get_container()
    c2 = dependencies.get_container()
    assert c1 is c2
    assert isinstance(c1, dependencies.ServiceContainer)


def test_cached_property_exception_does_not_cache(mock_services):
    # If service constructor raises, property should not be cached
    mock_services["ingestion"].side_effect = Exception("fail")
    container = dependencies.ServiceContainer()
    with pytest.raises(Exception, match="fail"):
        _ = container.ingestion_service
    # Next access should try again (not cached)
    mock_services["ingestion"].side_effect = MagicMock()
    # Remove the failed cached_property if present
    if "ingestion_service" in container.__dict__:
        del container.__dict__["ingestion_service"]
    # Now should succeed
    mock_services["ingestion"].side_effect = None
    mock_instance = MagicMock()
    mock_services["ingestion"].return_value = mock_instance
    result = container.ingestion_service
    assert result is mock_instance


def test_document_pipeline_service_dependency_order(mock_services):
    # Ensure dependencies are constructed in the correct order
    order = []
    def ingest_ctor(*a, **k):
        order.append("ingest")
        return MagicMock()
    def vector_ctor(*a, **k):
        order.append("vector")
        return MagicMock()
    def extract_ctor(*a, **k):
        order.append("extract")
        return MagicMock()
    def pipeline_ctor(*a, **k):
        order.append("pipeline")
        return MagicMock()
    mock_services["ingestion"].side_effect = ingest_ctor
    mock_services["vector"].side_effect = vector_ctor
    mock_services["extraction"].side_effect = extract_ctor
    mock_services["pipeline"].side_effect = pipeline_ctor

    container = dependencies.ServiceContainer()
    _ = container.document_pipeline_service
    # The three dependencies must be constructed before pipeline
    assert order[:3] == sorted(order[:3])
    assert order[-1] == "pipeline"
    assert len(order) == 4


def test_rag_service_dependency_is_vector_store_service_instance(mock_services):
    # RAGService must receive the same vector_store_service instance as constructed by container
    mock_vector_instance = MagicMock(name="VectorStoreServiceInstance")
    mock_services["vector"].return_value = mock_vector_instance
    mock_rag_instance = MagicMock(name="RAGServiceInstance")
    def rag_ctor(vector_store_service):
        assert vector_store_service is mock_vector_instance
        return mock_rag_instance
    mock_services["rag"].side_effect = rag_ctor
    container = dependencies.ServiceContainer()
    result = container.rag_service
    assert result is mock_rag_instance
    # Should not call rag_ctor again
    assert container.rag_service is result
    assert mock_services["rag"].call_count == 1


def test_document_pipeline_service_dependency_instances_are_consistent(mock_services):
    # The dependencies passed to DocumentPipelineService must be the same as container's properties
    mock_ingest_instance = MagicMock()
    mock_vector_instance = MagicMock()
    mock_extract_instance = MagicMock()
    mock_services["ingestion"].return_value = mock_ingest_instance
    mock_services["vector"].return_value = mock_vector_instance
    mock_services["extraction"].return_value = mock_extract_instance
    def pipeline_ctor(ingestion_service, vector_store_service, extraction_service):
        assert ingestion_service is mock_ingest_instance
        assert vector_store_service is mock_vector_instance
        assert extraction_service is mock_extract_instance
        return MagicMock()
    mock_services["pipeline"].side_effect = pipeline_ctor
    container = dependencies.ServiceContainer()
    result = container.document_pipeline_service
    assert result is container.document_pipeline_service
    assert mock_services["pipeline"].call_count == 1
