# source_hash: e871aa5cb722f502
# import_target: backend.src.services.rag
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import pytest
from unittest.mock import patch, MagicMock, create_autospec

from backend.src.services import rag as rag_module
from backend.src.services.rag import RAGService, format_docs
from langchain_core.documents import Document
from models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata

@pytest.fixture
def sample_docs():
    return [
        Document(
            page_content="Content 1",
            metadata={"source": "Doc1.pdf", "page": 1, "section": "Intro"}
        ),
        Document(
            page_content="Content 2",
            metadata={"source": "Doc2.pdf", "page": 2, "section": "Details"}
        ),
    ]

def test_format_docs_returns_empty_string_on_empty_list():
    assert format_docs([]) == ""

def test_format_docs_formats_documents_correctly(sample_docs):
    result = format_docs(sample_docs)
    assert "[Document 1]" in result
    assert "Source: Doc1.pdf" in result
    assert "Page: 1" in result
    assert "Section: Intro" in result
    assert "Content 1" in result
    assert "[Document 2]" in result
    assert "Source: Doc2.pdf" in result
    assert "Page: 2" in result
    assert "Section: Details" in result
    assert "Content 2" in result

def test_format_docs_handles_missing_metadata_fields():
    doc = Document(page_content="Test", metadata={})
    result = format_docs([doc])
    assert "Source: Unknown" in result
    assert "Page: N/A" in result
    assert "Section: N/A" in result

def test_check_safety_returns_none_for_safe_question():
    service = RAGService()
    assert service._check_safety("What is the shipment process?") is None

@pytest.mark.parametrize("unsafe_word", ["bomb", "kill", "suicide", "hack", "exploit", "weapon"])
def test_check_safety_detects_unsafe_keywords(unsafe_word):
    service = RAGService()
    question = f"How to {unsafe_word} a system?"
    assert "violates safety guidelines" in service._check_safety(question)

def test_calculate_confidence_returns_0_for_empty_answer():
    service = RAGService()
    assert service._calculate_confidence("", [Document(page_content="abc", metadata={})]) == 0.0

def test_calculate_confidence_low_for_cannot_find_phrase(sample_docs):
    service = RAGService()
    answer = "I cannot find the answer in the provided documents."
    assert service._calculate_confidence(answer, sample_docs) == 0.05

def test_calculate_confidence_penalizes_short_answer(sample_docs):
    service = RAGService()
    answer = "Short answer."
    conf = service._calculate_confidence(answer, sample_docs)
    assert conf < 0.85

def test_calculate_confidence_penalizes_few_docs():
    service = RAGService()
    answer = "This is a sufficiently long answer that should not be penalized for length."
    docs = [Document(page_content="abc", metadata={})]
    conf = service._calculate_confidence(answer, docs)
    assert conf < 0.85

def test_calculate_confidence_penalizes_forbidden_phrases(sample_docs):
    service = RAGService()
    for phrase in service.forbidden_phrases:
        answer = f"This is {phrase} the case."
        conf = service._calculate_confidence(answer, sample_docs)
        assert conf < 0.85

def test_calculate_confidence_clamped_between_0_and_1(sample_docs):
    service = RAGService()
    answer = "generally usually in most cases best practice typically commonly"
    conf = service._calculate_confidence(answer, [])
    assert 0.0 <= conf <= 1.0

@patch("backend.src.services.rag.vector_store_service")
def test_answer_question_returns_safety_error(mock_vector_store_service):
    service = RAGService()
    query = QAQuery(question="How to make a bomb?")
    result = service.answer_question(query)
    assert "violates safety guidelines" in result.answer
    assert result.confidence_score == 1.0
    assert result.sources == []

@patch("backend.src.services.rag.vector_store_service")
def test_answer_question_returns_no_retriever(mock_vector_store_service):
    service = RAGService()
    query = QAQuery(question="What is logistics?")
    mock_vector_store_service.as_retriever.return_value = None
    result = service.answer_question(query)
    assert "I cannot find any relevant information" in result.answer
    assert result.confidence_score == 0.0
    assert result.sources == []

@patch("backend.src.services.rag.vector_store_service")
def test_answer_question_returns_no_docs(mock_vector_store_service):
    service = RAGService()
    query = QAQuery(question="What is logistics?")
    retriever = MagicMock()
    retriever.invoke.return_value = []
    mock_vector_store_service.as_retriever.return_value = retriever
    result = service.answer_question(query)
    assert "I cannot find the answer in the provided documents." in result.answer
    assert result.confidence_score == 0.0
    assert result.sources == []

@patch("backend.src.services.rag.vector_store_service")
@patch.object(rag_module, "format_docs")
@patch.object(rag_module, "ChatPromptTemplate")
@patch.object(rag_module, "StrOutputParser")
@patch.object(rag_module, "ChatGroq")
def test_answer_question_happy_path(
    mock_chatgroq, mock_output_parser, mock_prompt_template, mock_format_docs, mock_vector_store_service
):
    # Setup
    service = RAGService()
    query = QAQuery(question="What is the process?")
    doc1 = Document(page_content="Doc1 content", metadata={"source": "A", "page": 1, "section": "X"})
    doc2 = Document(page_content="Doc2 content", metadata={"source": "B", "page": 2, "section": "Y"})
    retrieved_docs = [doc1, doc2]

    retriever = MagicMock()
    retriever.invoke.return_value = retrieved_docs
    mock_vector_store_service.as_retriever.return_value = retriever

    # Mock LCEL chain
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "This is the answer from context."
    # Patch the chain construction
    class DummyPrompt:
        def __ror__(self, other): return mock_chain
        def __or__(self, other): return mock_chain
    mock_prompt_template.from_template.return_value = DummyPrompt()
    service.prompt = DummyPrompt()
    service.llm = MagicMock()
    service.output_parser = MagicMock()
    service.output_parser.__ror__ = lambda self, other: mock_chain
    service.output_parser.__or__ = lambda self, other: mock_chain

    # Patch format_docs
    mock_format_docs.return_value = "Formatted context"

    # Patch RunnableParallel and RunnablePassthrough
    with patch("backend.src.services.rag.RunnableParallel", return_value=MagicMock()) as mock_runnable_parallel, \
         patch("backend.src.services.rag.RunnablePassthrough", return_value=MagicMock()):
        result = service.answer_question(query)

    assert isinstance(result, SourcedAnswer)
    assert result.answer == "This is the answer from context."
    assert 0.0 < result.confidence_score <= 1.0
    assert len(result.sources) == 2
    assert all(isinstance(chunk, Chunk) for chunk in result.sources)
    assert result.sources[0].text == "Doc1 content"
    assert result.sources[1].text == "Doc2 content"

@patch("backend.src.services.rag.vector_store_service")
def test_answer_question_handles_exception(mock_vector_store_service):
    service = RAGService()
    query = QAQuery(question="What is logistics?")
    retriever = MagicMock()
    retriever.invoke.side_effect = Exception("DB error")
    mock_vector_store_service.as_retriever.return_value = retriever
    result = service.answer_question(query)
    assert "An error occurred while generating the answer." in result.answer
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_rag_service_singleton_instance_exists():
    assert isinstance(rag_module.rag_service, RAGService)
