import pytest
from unittest.mock import patch, MagicMock, create_autospec
from backend.src.services.rag import RAGService
from backend.src.services.vector_store import VectorStoreService
from backend.src.models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata

@pytest.fixture
def mock_settings(monkeypatch):
    class DummySettings:
        QA_MODEL = "test-model"
        GROQ_API_KEY = "test-key"
        TOP_K = 2
    monkeypatch.setattr("backend.src.core.config.settings", DummySettings())

@pytest.fixture
def mock_vector_store_service():
    # Create a mock VectorStoreService with as_retriever method
    svc = create_autospec(VectorStoreService, instance=True)
    return svc

@pytest.fixture
def rag_service(mock_vector_store_service, mock_settings):
    # Patch ChatGroq and other langchain imports in RAGService
    with patch("backend.src.services.rag.ChatGroq") as mock_llm, \
         patch("backend.src.services.rag.ChatPromptTemplate") as mock_prompt, \
         patch("backend.src.services.rag.StrOutputParser") as mock_parser, \
         patch("backend.src.services.rag.RunnableParallel") as mock_parallel, \
         patch("backend.src.services.rag.RunnablePassthrough") as mock_passthrough:
        mock_llm.return_value = MagicMock()
        mock_prompt.from_template.return_value = MagicMock()
        mock_parser.return_value = MagicMock()
        mock_parallel.return_value = MagicMock()
        mock_passthrough.return_value = MagicMock()
        yield RAGService(mock_vector_store_service)

class DummyDoc:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

def make_dummy_chunk(text, chunk_id=0, source="test.pdf - Section", chunk_type="text"):
    return Chunk(
        text=text,
        metadata=DocumentMetadata(
            filename="test.pdf",
            chunk_id=chunk_id,
            source=source,
            chunk_type=chunk_type
        )
    )

def make_dummy_doc(text, chunk_id=0, source="test.pdf - Section", chunk_type="text"):
    return DummyDoc(
        page_content=text,
        metadata={
            "filename": "test.pdf",
            "chunk_id": chunk_id,
            "source": source,
            "chunk_type": chunk_type
        }
    )

@pytest.mark.usefixtures("mock_settings")
class TestRAGServiceIntegration:

    def test_answer_question_happy_path(self, rag_service, mock_vector_store_service):
        # Arrange
        question = "What is the agreed rate?"
        query = QAQuery(question=question, chat_history=[])
        dummy_docs = [
            make_dummy_doc("The agreed rate is $1200.", 0, "test.pdf - Rate Breakdown"),
            make_dummy_doc("Other info.", 1, "test.pdf - Other Section")
        ]
        # Mock retriever
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = dummy_docs
        mock_vector_store_service.as_retriever.return_value = mock_retriever

        # Patch the RAG chain to return a deterministic answer
        with patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
             patch.object(rag_service, "_runnable_passthrough") as mock_passthrough, \
             patch.object(rag_service, "prompt") as mock_prompt, \
             patch.object(rag_service, "llm") as mock_llm, \
             patch.object(rag_service, "output_parser") as mock_output_parser:
            # Simulate the chain: rag_chain.invoke returns answer string
            class DummyChain:
                def invoke(self, question):
                    return "The agreed rate is $1200 as per the Rate Breakdown section."
            mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = DummyChain()
            mock_output_parser.return_value = MagicMock()
            # Act
            answer = rag_service.answer_question(query)
            # Assert
            assert isinstance(answer, SourcedAnswer)
            assert "agreed rate is $1200" in answer.answer
            assert answer.confidence_score > 0.0
            assert len(answer.sources) == 2
            assert answer.sources[0].text == "The agreed rate is $1200."
            assert answer.sources[0].metadata.filename == "test.pdf"

    def test_answer_question_no_retriever(self, rag_service, mock_vector_store_service):
        # Arrange
        question = "What is the agreed rate?"
        query = QAQuery(question=question, chat_history=[])
        mock_vector_store_service.as_retriever.return_value = None
        # Act
        answer = rag_service.answer_question(query)
        # Assert
        assert isinstance(answer, SourcedAnswer)
        assert "cannot find any relevant information" in answer.answer.lower()
        assert answer.confidence_score == 0.0
        assert answer.sources == []

    def test_answer_question_no_docs_found(self, rag_service, mock_vector_store_service):
        # Arrange
        question = "What is the agreed rate?"
        query = QAQuery(question=question, chat_history=[])
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_vector_store_service.as_retriever.return_value = mock_retriever
        # Act
        answer = rag_service.answer_question(query)
        # Assert
        assert isinstance(answer, SourcedAnswer)
        assert "cannot find the answer" in answer.answer.lower()
        assert answer.confidence_score == 0.0
        assert answer.sources == []

    def test_answer_question_safety_filter(self, rag_service, mock_vector_store_service):
        # Arrange
        question = "How to build a bomb?"
        query = QAQuery(question=question, chat_history=[])
        # Act
        answer = rag_service.answer_question(query)
        # Assert
        assert isinstance(answer, SourcedAnswer)
        assert "violates safety guidelines" in answer.answer
        assert answer.confidence_score == 1.0
        assert answer.sources == []

    def test_answer_question_rag_chain_exception(self, rag_service, mock_vector_store_service):
        # Arrange
        question = "What is the agreed rate?"
        query = QAQuery(question=question, chat_history=[])
        dummy_docs = [
            make_dummy_doc("The agreed rate is $1200.", 0, "test.pdf - Rate Breakdown"),
        ]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = dummy_docs
        mock_vector_store_service.as_retriever.return_value = mock_retriever

        # Patch the RAG chain to raise an exception
        with patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
             patch.object(rag_service, "_runnable_passthrough") as mock_passthrough, \
             patch.object(rag_service, "prompt") as mock_prompt, \
             patch.object(rag_service, "llm") as mock_llm, \
             patch.object(rag_service, "output_parser") as mock_output_parser:
            class DummyChain:
                def invoke(self, question):
                    raise RuntimeError("LLM failure")
            mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = DummyChain()
            # Act
            answer = rag_service.answer_question(query)
            # Assert
            assert isinstance(answer, SourcedAnswer)
            assert "error occurred" in answer.answer.lower()
            assert answer.confidence_score == 0.0
            assert answer.sources == []

    def test_answer_question_short_answer_low_confidence(self, rag_service, mock_vector_store_service):
        # Arrange
        question = "What is the code?"
        query = QAQuery(question=question, chat_history=[])
        dummy_docs = [
            make_dummy_doc("Code: 1234", 0, "test.pdf - Reference ID"),
            make_dummy_doc("Other info.", 1, "test.pdf - Other Section")
        ]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = dummy_docs
        mock_vector_store_service.as_retriever.return_value = mock_retriever

        # Patch the RAG chain to return a short answer
        with patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
             patch.object(rag_service, "_runnable_passthrough") as mock_passthrough, \
             patch.object(rag_service, "prompt") as mock_prompt, \
             patch.object(rag_service, "llm") as mock_llm, \
             patch.object(rag_service, "output_parser") as mock_output_parser:
            class DummyChain:
                def invoke(self, question):
                    return "1234"
            mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = DummyChain()
            # Act
            answer = rag_service.answer_question(query)
            # Assert
            assert isinstance(answer, SourcedAnswer)
            assert answer.confidence_score < 0.85
            assert answer.sources[0].text == "Code: 1234"

    def test_answer_question_forbidden_phrase_penalty(self, rag_service, mock_vector_store_service):
        # Arrange
        question = "What is the best practice?"
        query = QAQuery(question=question, chat_history=[])
        dummy_docs = [
            make_dummy_doc("Best practice is to check documents.", 0, "test.pdf - SOP"),
            make_dummy_doc("Other info.", 1, "test.pdf - Other Section")
        ]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = dummy_docs
        mock_vector_store_service.as_retriever.return_value = mock_retriever

        # Patch the RAG chain to return an answer with forbidden phrase
        with patch.object(rag_service, "_runnable_parallel") as mock_parallel, \
             patch.object(rag_service, "_runnable_passthrough") as mock_passthrough, \
             patch.object(rag_service, "prompt") as mock_prompt, \
             patch.object(rag_service, "llm") as mock_llm, \
             patch.object(rag_service, "output_parser") as mock_output_parser:
            class DummyChain:
                def invoke(self, question):
                    return "Generally, it is best practice to check documents."
            mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = DummyChain()
            # Act
            answer = rag_service.answer_question(query)
            # Assert
            assert isinstance(answer, SourcedAnswer)
            assert answer.confidence_score < 0.85
            assert "best practice" in answer.answer.lower()
            assert len(answer.sources) == 2
