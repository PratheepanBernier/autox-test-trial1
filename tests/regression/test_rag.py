# source_hash: e871aa5cb722f502
import pytest
from unittest.mock import patch, MagicMock, create_autospec
from langchain_core.documents import Document
from backend.src.services.rag import RAGService, format_docs
from models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata

@pytest.fixture
def rag_service():
    # Patch ChatGroq and settings for deterministic behavior
    with patch("backend.src.services.rag.ChatGroq") as mock_llm, \
         patch("backend.src.services.rag.settings") as mock_settings:
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
    result = format_docs(docs)
    assert "[Document 1]" in result
    assert "Source: DocA" in result
    assert "Content A" in result
    assert "[Document 2]" in result
    assert "Source: DocB" in result
    assert "Content B" in result

def test_format_docs_empty_list():
    assert format_docs([]) == ""

def test_format_docs_missing_metadata_fields():
    docs = [
        Document(page_content="Test", metadata={}),
        Document(page_content="Test2", metadata=None),
    ]
    result = format_docs(docs)
    assert "Source: Unknown" in result
    assert "Page: N/A" in result
    assert "Section: N/A" in result

def test_check_safety_blocks_unsafe_question(rag_service):
    unsafe_questions = [
        "How to make a bomb?",
        "How can I hack the system?",
        "Is suicide an option?",
        "How to exploit this?",
        "What weapon is best?",
        "How to kill?"
    ]
    for q in unsafe_questions:
        assert rag_service._check_safety(q) == "I cannot answer this question as it violates safety guidelines."

def test_check_safety_allows_safe_question(rag_service):
    assert rag_service._check_safety("How do I create a shipment?") is None

@pytest.mark.parametrize(
    "answer,retrieved_docs,expected",
    [
        ("I cannot find the answer in the provided documents.", [Document(page_content="A")], 0.05),
        ("Short answer.", [Document(page_content="A"), Document(page_content="B")], 0.55),
        ("This is a sufficiently long answer that should not be penalized.", [Document(page_content="A"), Document(page_content="B")], 0.85),
        ("This is a generally accepted answer.", [Document(page_content="A"), Document(page_content="B")], 0.55),
        ("", [Document(page_content="A")], 0.0),
        ("Short answer.", [Document(page_content="A")], 0.4),
        ("This is a best practice.", [Document(page_content="A")], 0.25),
    ]
)
def test_calculate_confidence_various_cases(rag_service, answer, retrieved_docs, expected):
    result = rag_service._calculate_confidence(answer, retrieved_docs)
    assert abs(result - expected) < 1e-6

def test_answer_question_happy_path(monkeypatch, rag_service):
    # Mock retriever and chain
    mock_retriever = MagicMock()
    doc1 = Document(page_content="Doc1 content", metadata={"source": "S1", "page": 1, "section": "A"})
    doc2 = Document(page_content="Doc2 content", metadata={"source": "S2", "page": 2, "section": "B"})
    mock_retriever.invoke.return_value = [doc1, doc2]

    # Patch vector_store_service.as_retriever
    with patch("backend.src.services.rag.vector_store_service") as mock_vs:
        mock_vs.as_retriever.return_value = mock_retriever

        # Patch prompt, llm, output_parser, and rag_chain
        mock_prompt = MagicMock()
        mock_llm = MagicMock()
        mock_output_parser = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "This is the answer from the chain."

        with patch.object(rag_service, "prompt", mock_prompt), \
             patch.object(rag_service, "llm", mock_llm), \
             patch.object(rag_service, "output_parser", mock_output_parser), \
             patch("backend.src.services.rag.RunnableParallel", return_value=mock_chain), \
             patch("backend.src.services.rag.RunnablePassthrough"):
            mock_output_parser.__or__.return_value = mock_chain
            query = QAQuery(question="What is the shipment process?")
            result = rag_service.answer_question(query)
            assert isinstance(result, SourcedAnswer)
            assert result.answer == "This is the answer from the chain."
            assert result.confidence_score > 0
            assert len(result.sources) == 2
            assert result.sources[0].text == "Doc1 content"
            assert result.sources[1].text == "Doc2 content"

def test_answer_question_safety_violation(monkeypatch, rag_service):
    query = QAQuery(question="How to make a bomb?")
    result = rag_service.answer_question(query)
    assert result.answer == "I cannot answer this question as it violates safety guidelines."
    assert result.confidence_score == 1.0
    assert result.sources == []

def test_answer_question_retriever_none(monkeypatch, rag_service):
    with patch("backend.src.services.rag.vector_store_service") as mock_vs:
        mock_vs.as_retriever.return_value = None
        query = QAQuery(question="What is the shipment process?")
        result = rag_service.answer_question(query)
        assert result.answer == "I cannot find any relevant information in the uploaded documents."
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_no_retrieved_docs(monkeypatch, rag_service):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    with patch("backend.src.services.rag.vector_store_service") as mock_vs:
        mock_vs.as_retriever.return_value = mock_retriever
        query = QAQuery(question="What is the shipment process?")
        result = rag_service.answer_question(query)
        assert result.answer == "I cannot find the answer in the provided documents."
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_chain_exception(monkeypatch, rag_service):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [Document(page_content="Doc1", metadata={})]
    with patch("backend.src.services.rag.vector_store_service") as mock_vs:
        mock_vs.as_retriever.return_value = mock_retriever
        # Patch chain to raise exception
        with patch("backend.src.services.rag.RunnableParallel") as mock_runnable_parallel, \
             patch("backend.src.services.rag.RunnablePassthrough"):
            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = Exception("Chain error")
            mock_runnable_parallel.return_value = mock_chain
            query = QAQuery(question="What is the shipment process?")
            result = rag_service.answer_question(query)
            assert result.answer == "An error occurred while generating the answer."
            assert result.confidence_score == 0.0
            assert result.sources == []

def test_answer_question_reconciliation_equivalent_paths(monkeypatch, rag_service):
    # This test checks that two equivalent queries with same docs and answer yield same output
    mock_retriever = MagicMock()
    doc = Document(page_content="Doc content", metadata={"source": "S", "page": 1, "section": "A"})
    mock_retriever.invoke.return_value = [doc, doc]
    with patch("backend.src.services.rag.vector_store_service") as mock_vs:
        mock_vs.as_retriever.return_value = mock_retriever
        mock_prompt = MagicMock()
        mock_llm = MagicMock()
        mock_output_parser = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Identical answer."
        with patch.object(rag_service, "prompt", mock_prompt), \
             patch.object(rag_service, "llm", mock_llm), \
             patch.object(rag_service, "output_parser", mock_output_parser), \
             patch("backend.src.services.rag.RunnableParallel", return_value=mock_chain), \
             patch("backend.src.services.rag.RunnablePassthrough"):
            mock_output_parser.__or__.return_value = mock_chain
            query1 = QAQuery(question="Q1?")
            query2 = QAQuery(question="Q1?")
            result1 = rag_service.answer_question(query1)
            result2 = rag_service.answer_question(query2)
            assert result1.answer == result2.answer
            assert result1.confidence_score == result2.confidence_score
            assert [c.text for c in result1.sources] == [c.text for c in result2.sources]
