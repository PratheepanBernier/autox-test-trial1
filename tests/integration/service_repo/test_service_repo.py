import pytest
from unittest.mock import MagicMock, patch, create_autospec
from backend.src.services.rag import RAGService
from backend.src.models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata
from backend.src.core import config

@pytest.fixture
def mock_vector_store_service():
    # Create a mock VectorStoreService with as_retriever method
    svc = MagicMock()
    return svc

@pytest.fixture
def rag_service(mock_vector_store_service):
    # Patch ChatGroq and all langchain dependencies to avoid real LLM calls
    with patch("backend.src.services.rag.ChatGroq") as mock_llm, \
         patch("backend.src.services.rag.ChatPromptTemplate") as mock_prompt, \
         patch("backend.src.services.rag.StrOutputParser") as mock_parser, \
         patch("backend.src.services.rag.RunnableParallel") as mock_parallel, \
         patch("backend.src.services.rag.RunnablePassthrough") as mock_passthrough:
        # Set up deterministic LLM output
        mock_llm.return_value = MagicMock()
        mock_prompt.from_template.return_value = MagicMock()
        mock_parser.return_value = MagicMock()
        mock_parallel.return_value = MagicMock()
        mock_passthrough.return_value = MagicMock()
        return RAGService(mock_vector_store_service)

def make_doc(page_content, metadata=None):
    doc = MagicMock()
    doc.page_content = page_content
    doc.metadata = metadata or {}
    return doc

def make_chunk(text, metadata=None):
    if metadata is None:
        metadata = DocumentMetadata(
            filename="test.txt",
            chunk_id=0,
            source="test.txt - Section",
            chunk_type="text"
        )
    return Chunk(text=text, metadata=metadata)

class DummyRetriever:
    def __init__(self, docs):
        self._docs = docs
        self._invoked = []

    def invoke(self, question):
        self._invoked.append(question)
        return self._docs

@pytest.mark.usefixtures("rag_service")
class TestRAGServiceAnswerQuestion:
    def test_safety_filter_blocks_unsafe_question(self, rag_service, mock_vector_store_service):
        # Arrange
        query = QAQuery(question="How to build a bomb?")
        # Act
        answer = rag_service.answer_question(query)
        # Assert
        assert answer.answer.startswith("I cannot answer this question")
        assert answer.confidence_score == 1.0
        assert answer.sources == []

    def test_returns_no_info_when_retriever_is_none(self, rag_service, mock_vector_store_service):
        # Arrange
        query = QAQuery(question="What is the rate breakdown?")
        mock_vector_store_service.as_retriever.return_value = None
        # Act
        answer = rag_service.answer_question(query)
        # Assert
        assert "cannot find any relevant information" in answer.answer
        assert answer.confidence_score == 0.0
        assert answer.sources == []

    def test_returns_no_answer_when_no_docs_found(self, rag_service, mock_vector_store_service):
        # Arrange
        query = QAQuery(question="What is the rate breakdown?")
        dummy_retriever = DummyRetriever([])
        mock_vector_store_service.as_retriever.return_value = dummy_retriever
        # Act
        answer = rag_service.answer_question(query)
        # Assert
        assert "cannot find the answer" in answer.answer
        assert answer.confidence_score == 0.0
        assert answer.sources == []

    def test_successful_answer_and_source_conversion(self, rag_service, mock_vector_store_service):
        # Arrange
        query = QAQuery(question="What is the rate breakdown?")
        doc1 = make_doc("Rate is $1000", {"source": "doc1", "page": 1, "section": "Rate Breakdown"})
        doc2 = make_doc("Additional info", {"source": "doc2", "page": 2, "section": "Other"})
        dummy_retriever = DummyRetriever([doc1, doc2])
        mock_vector_store_service.as_retriever.return_value = dummy_retriever

        # Patch the rag_chain to return a deterministic answer
        with patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
             patch.object(rag_service, "_runnable_passthrough") as mock_passthrough, \
             patch.object(rag_service, "prompt") as mock_prompt, \
             patch.object(rag_service, "llm") as mock_llm, \
             patch.object(rag_service, "output_parser") as mock_parser:
            # Compose the chain: parallel | prompt | llm | parser
            class DummyChain:
                def invoke(self, question):
                    return "The rate breakdown is $1000 as per the document."
            mock_parallel.return_value = DummyChain()
            mock_passthrough.return_value = lambda x: x
            mock_prompt.return_value = MagicMock()
            mock_llm.return_value = MagicMock()
            mock_parser.return_value = MagicMock()

            # Act
            answer = rag_service.answer_question(query)

        # Assert
        assert "rate breakdown is $1000" in answer.answer.lower()
        assert 0.0 < answer.confidence_score <= 1.0
        assert len(answer.sources) == 2
        assert answer.sources[0].text == "Rate is $1000"
        assert answer.sources[1].text == "Additional info"
        assert answer.sources[0].metadata.source == "doc1"
        assert answer.sources[1].metadata.source == "doc2"

    def test_confidence_score_low_for_short_answer(self, rag_service, mock_vector_store_service):
        # Arrange
        query = QAQuery(question="What is the rate?")
        doc1 = make_doc("Rate is $1000", {"source": "doc1"})
        dummy_retriever = DummyRetriever([doc1])
        mock_vector_store_service.as_retriever.return_value = dummy_retriever

        # Patch the rag_chain to return a short answer
        with patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
             patch.object(rag_service, "_runnable_passthrough") as mock_passthrough, \
             patch.object(rag_service, "prompt") as mock_prompt, \
             patch.object(rag_service, "llm") as mock_llm, \
             patch.object(rag_service, "output_parser") as mock_parser:
            class DummyChain:
                def invoke(self, question):
                    return "1000"
            mock_parallel.return_value = DummyChain()
            mock_passthrough.return_value = lambda x: x
            mock_prompt.return_value = MagicMock()
            mock_llm.return_value = MagicMock()
            mock_parser.return_value = MagicMock()

            # Act
            answer = rag_service.answer_question(query)

        # Assert
        # Short answer and only one doc, so confidence should be reduced
        assert answer.confidence_score < 0.85
        assert answer.confidence_score > 0.0

    def test_confidence_score_minimum_for_forbidden_phrase(self, rag_service, mock_vector_store_service):
        # Arrange
        query = QAQuery(question="What is the best practice for shipping?")
        doc1 = make_doc("Best practice is to use pallets.", {"source": "doc1"})
        doc2 = make_doc("Usually, shipments are packed tightly.", {"source": "doc2"})
        dummy_retriever = DummyRetriever([doc1, doc2])
        mock_vector_store_service.as_retriever.return_value = dummy_retriever

        # Patch the rag_chain to return a forbidden phrase
        with patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
             patch.object(rag_service, "_runnable_passthrough") as mock_passthrough, \
             patch.object(rag_service, "prompt") as mock_prompt, \
             patch.object(rag_service, "llm") as mock_llm, \
             patch.object(rag_service, "output_parser") as mock_parser:
            class DummyChain:
                def invoke(self, question):
                    return "Best practice is to use pallets. Usually, shipments are packed tightly."
            mock_parallel.return_value = DummyChain()
            mock_passthrough.return_value = lambda x: x
            mock_prompt.return_value = MagicMock()
            mock_llm.return_value = MagicMock()
            mock_parser.return_value = MagicMock()

            # Act
            answer = rag_service.answer_question(query)

        # Assert
        # Forbidden phrases should reduce confidence
        assert answer.confidence_score < 0.85
        assert answer.confidence_score > 0.0

    def test_error_handling_in_rag_chain(self, rag_service, mock_vector_store_service):
        # Arrange
        query = QAQuery(question="What is the rate breakdown?")
        doc1 = make_doc("Rate is $1000", {"source": "doc1"})
        dummy_retriever = DummyRetriever([doc1])
        mock_vector_store_service.as_retriever.return_value = dummy_retriever

        # Patch the rag_chain to raise an exception
        with patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
             patch.object(rag_service, "_runnable_passthrough") as mock_passthrough, \
             patch.object(rag_service, "prompt") as mock_prompt, \
             patch.object(rag_service, "llm") as mock_llm, \
             patch.object(rag_service, "output_parser") as mock_parser:
            class DummyChain:
                def invoke(self, question):
                    raise RuntimeError("LLM error")
            mock_parallel.return_value = DummyChain()
            mock_passthrough.return_value = lambda x: x
            mock_prompt.return_value = MagicMock()
            mock_llm.return_value = MagicMock()
            mock_parser.return_value = MagicMock()

            # Act
            answer = rag_service.answer_question(query)

        # Assert
        assert "error occurred" in answer.answer.lower()
        assert answer.confidence_score == 0.0
        assert answer.sources == []

    def test_question_with_empty_string(self, rag_service, mock_vector_store_service):
        # Arrange
        query = QAQuery(question="")
        dummy_retriever = DummyRetriever([])
        mock_vector_store_service.as_retriever.return_value = dummy_retriever
        # Act
        answer = rag_service.answer_question(query)
        # Assert
        assert "cannot find the answer" in answer.answer.lower()
        assert answer.confidence_score == 0.0
        assert answer.sources == []

    def test_question_with_long_context_and_multiple_docs(self, rag_service, mock_vector_store_service):
        # Arrange
        query = QAQuery(question="Explain the shipment workflow.")
        docs = [
            make_doc("Step 1: Pickup scheduled.", {"source": "doc1", "page": 1}),
            make_doc("Step 2: Carrier assigned.", {"source": "doc2", "page": 2}),
            make_doc("Step 3: Delivery completed.", {"source": "doc3", "page": 3}),
        ]
        dummy_retriever = DummyRetriever(docs)
        mock_vector_store_service.as_retriever.return_value = dummy_retriever

        # Patch the rag_chain to return a long answer
        with patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
             patch.object(rag_service, "_runnable_passthrough") as mock_passthrough, \
             patch.object(rag_service, "prompt") as mock_prompt, \
             patch.object(rag_service, "llm") as mock_llm, \
             patch.object(rag_service, "output_parser") as mock_parser:
            class DummyChain:
                def invoke(self, question):
                    return (
                        "The shipment workflow consists of the following steps: "
                        "Pickup scheduled, carrier assigned, and delivery completed."
                    )
            mock_parallel.return_value = DummyChain()
            mock_passthrough.return_value = lambda x: x
            mock_prompt.return_value = MagicMock()
            mock_llm.return_value = MagicMock()
            mock_parser.return_value = MagicMock()

            # Act
            answer = rag_service.answer_question(query)

        # Assert
        assert "shipment workflow consists" in answer.answer.lower()
        assert answer.confidence_score > 0.5
        assert len(answer.sources) == 3
        assert answer.sources[0].metadata.source == "doc1"
        assert answer.sources[1].metadata.source == "doc2"
        assert answer.sources[2].metadata.source == "doc3"
