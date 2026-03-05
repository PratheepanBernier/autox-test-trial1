# source_hash: e871aa5cb722f502
import pytest
from unittest.mock import patch, MagicMock, create_autospec
from langchain_core.documents import Document
from services.rag import RAGService, format_docs
from models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata

@pytest.fixture
def rag_service():
    # Patch ChatGroq and settings for deterministic behavior
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
        Document(page_content="Content", metadata={}),
        Document(page_content="Content2", metadata=None),
    ]
    result = format_docs(docs)
    assert "Source: Unknown" in result
    assert "Page: N/A" in result
    assert "Section: N/A" in result

def test_check_safety_blocks_unsafe_question(rag_service):
    unsafe_questions = [
        "How to build a bomb?",
        "How can I kill someone?",
        "Tell me about suicide.",
        "How to hack a system?",
        "How to exploit a vulnerability?",
        "What is a weapon of mass destruction?"
    ]
    for q in unsafe_questions:
        assert rag_service._check_safety(q) == "I cannot answer this question as it violates safety guidelines."

def test_check_safety_allows_safe_question(rag_service):
    assert rag_service._check_safety("How do I create a shipment?") is None

@pytest.mark.parametrize(
    "answer,retrieved_docs,expected",
    [
        ("", [Document(page_content="A", metadata={})], 0.0),
        ("I cannot find the answer in the provided documents.", [Document(page_content="A", metadata={})], 0.05),
        ("Short answer.", [Document(page_content="A", metadata={}), Document(page_content="B", metadata={})], 0.55),
        ("This is a sufficiently long answer that should not be penalized.", [Document(page_content="A", metadata={}), Document(page_content="B", metadata={})], 0.85),
        ("This is a generally accepted answer.", [Document(page_content="A", metadata={}), Document(page_content="B", metadata={})], 0.55),
        ("Long answer.", [Document(page_content="A", metadata={})], 0.4),
        ("", [], 0.0),
    ]
)
def test_calculate_confidence_various_cases(rag_service, answer, retrieved_docs, expected):
    result = rag_service._calculate_confidence(answer, retrieved_docs)
    assert abs(result - expected) < 0.01

def test_answer_question_happy_path(monkeypatch, rag_service):
    query = QAQuery(question="What is the shipment process?")
    fake_docs = [
        Document(page_content="Step 1: Do X.", metadata={"source": "Manual", "page": 1, "section": "Process"}),
        Document(page_content="Step 2: Do Y.", metadata={"source": "Manual", "page": 2, "section": "Process"}),
    ]
    # Mock retriever
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = fake_docs
    # Patch vector_store_service.as_retriever
    with patch("services.rag.vector_store_service") as mock_vs, \
         patch.object(rag_service, "llm") as mock_llm, \
         patch.object(rag_service, "prompt") as mock_prompt, \
         patch.object(rag_service, "output_parser") as mock_parser:
        mock_vs.as_retriever.return_value = fake_retriever
        # Simulate LCEL chain
        class DummyChain:
            def invoke(self, question):
                return "The shipment process is described in the provided documents."
        dummy_chain = DummyChain()
        # Patch the chain construction to return dummy_chain
        with patch("services.rag.RunnableParallel", return_value=lambda x: dummy_chain):
            # Patch the chain invocation
            with patch("services.rag.RunnablePassthrough"):
                with patch("services.rag.format_docs", side_effect=lambda docs: format_docs(docs)):
                    # Patch prompt, llm, output_parser to be identity
                    mock_prompt.__or__.return_value = mock_prompt
                    mock_llm.__or__.return_value = mock_llm
                    mock_parser.__or__.return_value = mock_parser
                    mock_parser.invoke = dummy_chain.invoke
                    result = rag_service.answer_question(query)
    assert isinstance(result, SourcedAnswer)
    assert "shipment process" in result.answer
    assert result.confidence_score > 0.0
    assert len(result.sources) == 2
    assert isinstance(result.sources[0], Chunk)
    assert result.sources[0].metadata.source == "Manual"

def test_answer_question_safety_block(monkeypatch, rag_service):
    query = QAQuery(question="How to build a bomb?")
    result = rag_service.answer_question(query)
    assert result.answer == "I cannot answer this question as it violates safety guidelines."
    assert result.confidence_score == 1.0
    assert result.sources == []

def test_answer_question_retriever_none(monkeypatch, rag_service):
    query = QAQuery(question="What is the shipment process?")
    with patch("services.rag.vector_store_service") as mock_vs:
        mock_vs.as_retriever.return_value = None
        result = rag_service.answer_question(query)
    assert result.answer == "I cannot find any relevant information in the uploaded documents."
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_answer_question_no_documents(monkeypatch, rag_service):
    query = QAQuery(question="What is the shipment process?")
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = []
    with patch("services.rag.vector_store_service") as mock_vs:
        mock_vs.as_retriever.return_value = fake_retriever
        result = rag_service.answer_question(query)
    assert result.answer == "I cannot find the answer in the provided documents."
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_answer_question_chain_exception(monkeypatch, rag_service):
    query = QAQuery(question="What is the shipment process?")
    fake_docs = [
        Document(page_content="Step 1: Do X.", metadata={"source": "Manual", "page": 1, "section": "Process"}),
    ]
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = fake_docs
    with patch("services.rag.vector_store_service") as mock_vs, \
         patch.object(rag_service, "llm") as mock_llm, \
         patch.object(rag_service, "prompt") as mock_prompt, \
         patch.object(rag_service, "output_parser") as mock_parser:
        mock_vs.as_retriever.return_value = fake_retriever
        # Patch the chain to raise exception
        class DummyChain:
            def invoke(self, question):
                raise RuntimeError("Chain error")
        with patch("services.rag.RunnableParallel", return_value=lambda x: DummyChain()):
            with patch("services.rag.RunnablePassthrough"):
                with patch("services.rag.format_docs", side_effect=lambda docs: format_docs(docs)):
                    mock_prompt.__or__.return_value = mock_prompt
                    mock_llm.__or__.return_value = mock_llm
                    mock_parser.__or__.return_value = mock_parser
                    mock_parser.invoke = DummyChain().invoke
                    result = rag_service.answer_question(query)
    assert result.answer == "An error occurred while generating the answer."
    assert result.confidence_score == 0.0
    assert result.sources == []

def test_answer_question_forbidden_phrase_penalty(monkeypatch, rag_service):
    query = QAQuery(question="What is the best practice for shipment?")
    fake_docs = [
        Document(page_content="Best practice is to...", metadata={"source": "Manual", "page": 1, "section": "Best Practices"}),
        Document(page_content="Another tip...", metadata={"source": "Manual", "page": 2, "section": "Tips"}),
    ]
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = fake_docs
    with patch("services.rag.vector_store_service") as mock_vs, \
         patch.object(rag_service, "llm") as mock_llm, \
         patch.object(rag_service, "prompt") as mock_prompt, \
         patch.object(rag_service, "output_parser") as mock_parser:
        mock_vs.as_retriever.return_value = fake_retriever
        class DummyChain:
            def invoke(self, question):
                return "The best practice is to always check the documents."
        dummy_chain = DummyChain()
        with patch("services.rag.RunnableParallel", return_value=lambda x: dummy_chain):
            with patch("services.rag.RunnablePassthrough"):
                with patch("services.rag.format_docs", side_effect=lambda docs: format_docs(docs)):
                    mock_prompt.__or__.return_value = mock_prompt
                    mock_llm.__or__.return_value = mock_llm
                    mock_parser.__or__.return_value = mock_parser
                    mock_parser.invoke = dummy_chain.invoke
                    result = rag_service.answer_question(query)
    # Confidence should be penalized due to forbidden phrase
    assert result.confidence_score < 0.85
    assert "best practice" in result.answer.lower()
    assert len(result.sources) == 2

def test_answer_question_short_answer_penalty(monkeypatch, rag_service):
    query = QAQuery(question="What is X?")
    fake_docs = [
        Document(page_content="X is Y.", metadata={"source": "Manual", "page": 1, "section": "Defs"}),
        Document(page_content="More info...", metadata={"source": "Manual", "page": 2, "section": "Defs"}),
    ]
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = fake_docs
    with patch("services.rag.vector_store_service") as mock_vs, \
         patch.object(rag_service, "llm") as mock_llm, \
         patch.object(rag_service, "prompt") as mock_prompt, \
         patch.object(rag_service, "output_parser") as mock_parser:
        mock_vs.as_retriever.return_value = fake_retriever
        class DummyChain:
            def invoke(self, question):
                return "X is Y."
        dummy_chain = DummyChain()
        with patch("services.rag.RunnableParallel", return_value=lambda x: dummy_chain):
            with patch("services.rag.RunnablePassthrough"):
                with patch("services.rag.format_docs", side_effect=lambda docs: format_docs(docs)):
                    mock_prompt.__or__.return_value = mock_prompt
                    mock_llm.__or__.return_value = mock_llm
                    mock_parser.__or__.return_value = mock_parser
                    mock_parser.invoke = dummy_chain.invoke
                    result = rag_service.answer_question(query)
    # Confidence should be penalized due to short answer
    assert result.confidence_score < 0.85
    assert result.answer == "X is Y."
    assert len(result.sources) == 2
