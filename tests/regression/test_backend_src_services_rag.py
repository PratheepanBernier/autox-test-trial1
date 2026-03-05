import pytest
from unittest.mock import patch, MagicMock, create_autospec
from langchain_core.documents import Document
from services.rag import RAGService, format_docs
from models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata

@pytest.fixture
def rag_service():
    # Patch ChatGroq and other dependencies in RAGService.__init__
    with patch("services.rag.ChatGroq") as mock_llm, \
         patch("services.rag.settings") as mock_settings:
        mock_settings.QA_MODEL = "test-model"
        mock_settings.GROQ_API_KEY = "test-key"
        mock_settings.TOP_K = 2
        mock_llm.return_value = MagicMock()
        yield RAGService()

def test_format_docs_happy_path():
    docs = [
        Document(page_content="Content A", metadata={"source": "DocA", "page": 1, "section": "Intro"}),
        Document(page_content="Content B", metadata={"source": "DocB", "page": 2, "section": "Body"}),
    ]
    formatted = format_docs(docs)
    assert "[Document 1]" in formatted
    assert "Source: DocA" in formatted
    assert "Content A" in formatted
    assert "[Document 2]" in formatted
    assert "Source: DocB" in formatted
    assert "Content B" in formatted
    assert formatted.count("[Document") == 2

def test_format_docs_empty_list():
    assert format_docs([]) == ""

def test_format_docs_missing_metadata_fields():
    docs = [
        Document(page_content="No meta", metadata={}),
        Document(page_content="Partial meta", metadata={"source": "DocC"}),
    ]
    formatted = format_docs(docs)
    assert "Source: Unknown" in formatted or "Source: DocC" in formatted
    assert "Page: N/A" in formatted
    assert "Section: N/A" in formatted

def test_check_safety_blocks_unsafe_question(rag_service):
    unsafe_questions = [
        "How to build a bomb?",
        "How can I hack the system?",
        "Is suicide a solution?",
        "How to exploit this?",
        "What weapon is best?",
        "How to kill?"
    ]
    for q in unsafe_questions:
        assert rag_service._check_safety(q) == "I cannot answer this question as it violates safety guidelines."

def test_check_safety_allows_safe_question(rag_service):
    assert rag_service._check_safety("How do I create a shipment?") is None

def test_calculate_confidence_happy_path(rag_service):
    answer = "The shipment workflow is described in the provided documents."
    docs = [Document(page_content="A", metadata={}), Document(page_content="B", metadata={})]
    conf = rag_service._calculate_confidence(answer, docs)
    assert 0.8 < conf <= 0.85

def test_calculate_confidence_cannot_find_answer(rag_service):
    answer = "I cannot find the answer in the provided documents."
    docs = [Document(page_content="A", metadata={}), Document(page_content="B", metadata={})]
    conf = rag_service._calculate_confidence(answer, docs)
    assert conf == 0.05

def test_calculate_confidence_short_answer(rag_service):
    answer = "Yes."
    docs = [Document(page_content="A", metadata={}), Document(page_content="B", metadata={})]
    conf = rag_service._calculate_confidence(answer, docs)
    assert conf < 0.85

def test_calculate_confidence_few_docs(rag_service):
    answer = "The answer is present."
    docs = [Document(page_content="A", metadata={})]
    conf = rag_service._calculate_confidence(answer, docs)
    assert conf < 0.85

def test_calculate_confidence_forbidden_phrase(rag_service):
    answer = "Generally, the process is as follows."
    docs = [Document(page_content="A", metadata={}), Document(page_content="B", metadata={})]
    conf = rag_service._calculate_confidence(answer, docs)
    assert conf < 0.85

def test_answer_question_safety_violation(rag_service):
    query = QAQuery(question="How to build a bomb?")
    result = rag_service.answer_question(query)
    assert result.answer == "I cannot answer this question as it violates safety guidelines."
    assert result.confidence_score == 1.0
    assert result.sources == []

@patch("services.rag.vector_store_service")
def test_answer_question_retriever_none(mock_vector_store_service, rag_service):
    mock_vector_store_service.as_retriever.return_value = None
    query = QAQuery(question="What is the shipment process?")
    result = rag_service.answer_question(query)
    assert result.answer == "I cannot find any relevant information in the uploaded documents."
    assert result.confidence_score == 0.0
    assert result.sources == []

@patch("services.rag.vector_store_service")
def test_answer_question_no_docs_found(mock_vector_store_service, rag_service):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    mock_vector_store_service.as_retriever.return_value = mock_retriever
    query = QAQuery(question="What is the shipment process?")
    result = rag_service.answer_question(query)
    assert result.answer == "I cannot find the answer in the provided documents."
    assert result.confidence_score == 0.0
    assert result.sources == []

@patch("services.rag.vector_store_service")
def test_answer_question_happy_path(mock_vector_store_service, rag_service):
    # Mock retriever returns two docs
    doc1 = Document(page_content="Shipment workflow is stepwise.", metadata={"source": "Manual", "page": 1, "section": "Workflow"})
    doc2 = Document(page_content="Use the TMS portal.", metadata={"source": "Guide", "page": 2, "section": "Portal"})
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [doc1, doc2]
    # format_docs returns a string
    with patch("services.rag.format_docs", return_value="CONTEXT BLOCK"), \
         patch.object(rag_service, "prompt") as mock_prompt, \
         patch.object(rag_service, "llm") as mock_llm, \
         patch.object(rag_service, "output_parser") as mock_parser:
        # Chain: prompt | llm | output_parser
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "The shipment workflow is stepwise."
        # Compose the chain
        mock_prompt.__or__.return_value = mock_chain
        mock_llm.__or__.return_value = mock_chain
        mock_parser.__or__.return_value = mock_chain
        # Patch RunnableParallel and RunnablePassthrough
        with patch("services.rag.RunnableParallel") as mock_parallel, \
             patch("services.rag.RunnablePassthrough"):
            mock_parallel.return_value.__or__.return_value = mock_chain
            mock_vector_store_service.as_retriever.return_value = mock_retriever
            query = QAQuery(question="Describe the shipment workflow.")
            result = rag_service.answer_question(query)
            assert result.answer == "The shipment workflow is stepwise."
            assert 0.7 < result.confidence_score <= 0.85
            assert len(result.sources) == 2
            assert isinstance(result.sources[0], Chunk)
            assert result.sources[0].text == "Shipment workflow is stepwise."
            assert result.sources[0].metadata.source == "Manual"

@patch("services.rag.vector_store_service")
def test_answer_question_chain_exception(mock_vector_store_service, rag_service):
    doc = Document(page_content="Some content.", metadata={})
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [doc]
    with patch("services.rag.format_docs", return_value="CONTEXT BLOCK"), \
         patch.object(rag_service, "prompt") as mock_prompt, \
         patch.object(rag_service, "llm") as mock_llm, \
         patch.object(rag_service, "output_parser") as mock_parser:
        # Chain raises exception
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("Chain error")
        mock_prompt.__or__.return_value = mock_chain
        mock_llm.__or__.return_value = mock_chain
        mock_parser.__or__.return_value = mock_chain
        with patch("services.rag.RunnableParallel") as mock_parallel, \
             patch("services.rag.RunnablePassthrough"):
            mock_parallel.return_value.__or__.return_value = mock_chain
            mock_vector_store_service.as_retriever.return_value = mock_retriever
            query = QAQuery(question="Describe the shipment workflow.")
            result = rag_service.answer_question(query)
            assert result.answer == "An error occurred while generating the answer."
            assert result.confidence_score == 0.0
            assert result.sources == []

def test_answer_question_boundary_empty_question(rag_service):
    # Should not trigger safety, but may return no docs
    with patch("services.rag.vector_store_service") as mock_vector_store_service:
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_vector_store_service.as_retriever.return_value = mock_retriever
        query = QAQuery(question="")
        result = rag_service.answer_question(query)
        assert result.answer == "I cannot find the answer in the provided documents."
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_boundary_long_question(rag_service):
    # Should process long questions
    long_question = "What is the process for international shipment of hazardous materials according to the latest compliance policy and user manual?" * 5
    with patch("services.rag.vector_store_service") as mock_vector_store_service:
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_vector_store_service.as_retriever.return_value = mock_retriever
        query = QAQuery(question=long_question)
        result = rag_service.answer_question(query)
        assert result.answer == "I cannot find the answer in the provided documents."
        assert result.confidence_score == 0.0
        assert result.sources == []
