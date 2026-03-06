import pytest
from unittest.mock import patch, MagicMock

import backend.src.dependencies as dependencies


@pytest.fixture(autouse=True)
def reset_cached_properties():
    # Clear cached_property cache between tests for isolation
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
        if hasattr(dependencies.ServiceContainer, attr):
            try:
                delattr(dependencies.ServiceContainer, attr)
            except Exception:
                pass
    # Also clear instance-level cache
    for attr in [
        "ingestion_service",
        "vector_store_service",
        "extraction_service",
        "rag_service",
        "document_pipeline_service",
    ]:
        if hasattr(dependencies.container, attr):
            try:
                delattr(dependencies.container, attr)
            except Exception:
                pass


def test_get_container_returns_singleton():
    c1 = dependencies.get_container()
    c2 = dependencies.get_container()
    assert c1 is c2
    assert isinstance(c1, dependencies.ServiceContainer)


def test_ingestion_service_is_singleton_and_type():
    with patch("backend.src.services.ingestion.DocumentIngestionService") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        container = dependencies.ServiceContainer()
        s1 = container.ingestion_service
        s2 = container.ingestion_service
        assert s1 is s2
        assert s1 is mock_instance
        mock_cls.assert_called_once()


def test_vector_store_service_is_singleton_and_type():
    with patch("backend.src.services.vector_store.VectorStoreService") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        container = dependencies.ServiceContainer()
        s1 = container.vector_store_service
        s2 = container.vector_store_service
        assert s1 is s2
        assert s1 is mock_instance
        mock_cls.assert_called_once()


def test_extraction_service_is_singleton_and_type():
    with patch("backend.src.services.extraction.ExtractionService") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        container = dependencies.ServiceContainer()
        s1 = container.extraction_service
        s2 = container.extraction_service
        assert s1 is s2
        assert s1 is mock_instance
        mock_cls.assert_called_once()


def test_rag_service_is_singleton_and_type_and_depends_on_vector_store():
    with patch("backend.src.services.vector_store.VectorStoreService") as mock_vector_cls, \
         patch("backend.src.services.rag.RAGService") as mock_rag_cls:
        mock_vector_instance = MagicMock()
        mock_vector_cls.return_value = mock_vector_instance
        mock_rag_instance = MagicMock()
        mock_rag_cls.return_value = mock_rag_instance
        container = dependencies.ServiceContainer()
        s1 = container.rag_service
        s2 = container.rag_service
        assert s1 is s2
        assert s1 is mock_rag_instance
        mock_rag_cls.assert_called_once_with(vector_store_service=mock_vector_instance)
        mock_vector_cls.assert_called_once()


def test_document_pipeline_service_is_singleton_and_type_and_depends_on_services():
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
        s1 = container.document_pipeline_service
        s2 = container.document_pipeline_service
        assert s1 is s2
        assert s1 is mock_pipeline_instance
        mock_pipeline_cls.assert_called_once_with(
            ingestion_service=mock_ingest_instance,
            vector_store_service=mock_vector_instance,
            extraction_service=mock_extract_instance,
        )
        mock_ingest_cls.assert_called_once()
        mock_vector_cls.assert_called_once()
        mock_extract_cls.assert_called_once()


def test_services_are_independent_instances():
    # Each ServiceContainer instance should have its own cached services
    with patch("backend.src.services.ingestion.DocumentIngestionService") as mock_ingest_cls, \
         patch("backend.src.services.vector_store.VectorStoreService") as mock_vector_cls, \
         patch("backend.src.services.extraction.ExtractionService") as mock_extract_cls, \
         patch("backend.src.services.rag.RAGService") as mock_rag_cls, \
         patch("backend.src.use_cases.document_pipeline.DocumentPipelineService") as mock_pipeline_cls:
        mock_ingest_cls.side_effect = [MagicMock(), MagicMock()]
        mock_vector_cls.side_effect = [MagicMock(), MagicMock()]
        mock_extract_cls.side_effect = [MagicMock(), MagicMock()]
        mock_rag_cls.side_effect = [MagicMock(), MagicMock()]
        mock_pipeline_cls.side_effect = [MagicMock(), MagicMock()]

        c1 = dependencies.ServiceContainer()
        c2 = dependencies.ServiceContainer()
        assert c1 is not c2

        s1 = c1.ingestion_service
        s2 = c2.ingestion_service
        assert s1 is not s2

        v1 = c1.vector_store_service
        v2 = c2.vector_store_service
        assert v1 is not v2

        e1 = c1.extraction_service
        e2 = c2.extraction_service
        assert e1 is not e2

        r1 = c1.rag_service
        r2 = c2.rag_service
        assert r1 is not r2

        p1 = c1.document_pipeline_service
        p2 = c2.document_pipeline_service
        assert p1 is not p2


def test_cached_property_boundary_conditions():
    # Accessing properties in different orders should not affect correctness
    with patch("backend.src.services.ingestion.DocumentIngestionService") as mock_ingest_cls, \
         patch("backend.src.services.vector_store.VectorStoreService") as mock_vector_cls, \
         patch("backend.src.services.extraction.ExtractionService") as mock_extract_cls, \
         patch("backend.src.services.rag.RAGService") as mock_rag_cls, \
         patch("backend.src.use_cases.document_pipeline.DocumentPipelineService") as mock_pipeline_cls:
        mock_ingest_instance = MagicMock()
        mock_vector_instance = MagicMock()
        mock_extract_instance = MagicMock()
        mock_rag_instance = MagicMock()
        mock_pipeline_instance = MagicMock()
        mock_ingest_cls.return_value = mock_ingest_instance
        mock_vector_cls.return_value = mock_vector_instance
        mock_extract_cls.return_value = mock_extract_instance
        mock_rag_cls.return_value = mock_rag_instance
        mock_pipeline_cls.return_value = mock_pipeline_instance

        container = dependencies.ServiceContainer()
        # Access rag_service before vector_store_service
        rag = container.rag_service
        vector = container.vector_store_service
        assert rag is mock_rag_instance
        assert vector is mock_vector_instance

        # Access document_pipeline_service before ingestion_service
        pipeline = container.document_pipeline_service
        ingestion = container.ingestion_service
        assert pipeline is mock_pipeline_instance
        assert ingestion is mock_ingest_instance


def test_error_handling_on_service_init_failure():
    # If a service constructor raises, the error should propagate and not cache a broken value
    with patch("backend.src.services.ingestion.DocumentIngestionService", side_effect=RuntimeError("fail")) as mock_ingest_cls:
        container = dependencies.ServiceContainer()
        with pytest.raises(RuntimeError, match="fail"):
            _ = container.ingestion_service
        # On next access, should try again (not cached)
        with pytest.raises(RuntimeError, match="fail"):
            _ = container.ingestion_service
        assert mock_ingest_cls.call_count == 2
