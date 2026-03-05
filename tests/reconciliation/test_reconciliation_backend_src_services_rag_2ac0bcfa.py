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
from backend.src.services.rag import Document
from models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata

class DummyRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.invoked = False
    def invoke(self, question):
        self.invoked = True
        return self.docs

@pytest.fixture
def dummy_docs():
    return [
        Document(
            page_content="Content A",
            metadata={"source": "DocA", "page": 1, "section": "Intro"}
        ),
        Document(
            page_content="Content B",
            metadata={"source": "DocB", "page": 2, "section": "Details"}
        ),
    ]

@pytest.fixture
def dummy_query():
    return QAQuery(question="What is the shipment workflow?")

@pytest.fixture
def rag_service():
    # Patch ChatGroq, ChatPromptTemplate, StrOutputParser to avoid real LLM calls
    with patch.object(rag_module, "ChatGroq") as mock_llm, \
         patch.object(rag_module, "ChatPromptTemplate") as mock_prompt, \
         patch.object(rag_module, "StrOutputParser") as mock_parser:
        mock_llm.return_value = MagicMock()
        mock_prompt.from_template.return_value = MagicMock()
        mock_parser.return_value = MagicMock()
        yield RAGService()

def test_format_docs_happy_path(dummy_docs):
    formatted = format_docs(dummy_docs)
    assert "[Document 1]" in formatted
    assert "Source: DocA" in formatted
    assert "Content A" in formatted
    assert "[Document 2]" in formatted
    assert "Source: DocB" in formatted
    assert "Content B" in formatted
    assert formatted.count("[Document") == 2

def test_format_docs_empty_list():
    assert format_docs([]) == ""

def test_format_docs_missing_metadata():
    docs = [Document(page_content="No meta", metadata=None)]
    formatted = format_docs(docs)
    assert "Unknown" in formatted
    assert "N/A" in formatted
    assert "No meta" in formatted

def test_check_safety_blocks_unsafe_question(rag_service):
    unsafe = rag_service._check_safety("How to build a bomb?")
    assert unsafe is not None
    assert "violates safety guidelines" in unsafe

def test_check_safety_allows_safe_question(rag_service):
    safe = rag_service._check_safety("How to create a shipment?")
    assert safe is None

@pytest.mark.parametrize("answer,docs,expected", [
    ("I cannot find the answer in the provided documents.", [Document(page_content="x", metadata={})], 0.05),
    ("Short answer.", [Document(page_content="x", metadata={}), Document(page_content="y", metadata={})], 0.55),
    ("A long enough answer that exceeds thirty characters.", [Document(page_content="x", metadata={}), Document(page_content="y", metadata={})], 0.85),
    ("A long answer with generally used.", [Document(page_content="x", metadata={}), Document(page_content="y", metadata={})], 0.55),
    ("", [Document(page_content="x", metadata={})], 0.0),
    ("Short.", [], 0.25),
])
def test_calculate_confidence_various_cases(rag_service, answer, docs, expected):
    result = rag_service._calculate_confidence(answer, docs)
    assert abs(result - expected) < 1e-6

def test_answer_question_safety_violation(rag_service, dummy_query):
    with patch.object(rag_service, "_check_safety", return_value="Blocked for safety"):
        result = rag_service.answer_question(dummy_query)
        assert result.answer == "Blocked for safety"
        assert result.confidence_score == 1.0
        assert result.sources == []

def test_answer_question_retriever_none(rag_service, dummy_query):
    with patch("backend.src.services.rag.vector_store_service") as mock_vs:
        mock_vs.as_retriever.return_value = None
        result = rag_service.answer_question(dummy_query)
        assert "cannot find any relevant information" in result.answer
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_no_docs(rag_service, dummy_query):
    dummy_retriever = DummyRetriever([])
    with patch("backend.src.services.rag.vector_store_service") as mock_vs:
        mock_vs.as_retriever.return_value = dummy_retriever
        result = rag_service.answer_question(dummy_query)
        assert "cannot find the answer in the provided documents" in result.answer
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_happy_path(rag_service, dummy_query, dummy_docs):
    dummy_retriever = DummyRetriever(dummy_docs)
    fake_answer = "The shipment workflow is described in section 2."
    with patch("backend.src.services.rag.vector_store_service") as mock_vs, \
         patch.object(rag_service, "prompt") as mock_prompt, \
         patch.object(rag_service, "llm") as mock_llm, \
         patch.object(rag_service, "output_parser") as mock_parser, \
         patch.object(rag_service, "_calculate_confidence", return_value=0.77) as mock_conf:
        mock_vs.as_retriever.return_value = dummy_retriever

        # Simulate LCEL chain: rag_chain.invoke returns fake_answer
        class DummyChain:
            def invoke(self, question):
                return fake_answer
        mock_prompt.__or__.return_value = DummyChain()
        mock_llm.__or__.return_value = DummyChain()
        mock_parser.__or__.return_value = DummyChain()

        # Patch RunnableParallel and RunnablePassthrough to just pass through
        with patch("backend.src.services.rag.RunnableParallel", side_effect=lambda x: DummyChain()), \
             patch("backend.src.services.rag.RunnablePassthrough", side_effect=lambda: lambda x: x):
            result = rag_service.answer_question(dummy_query)
            assert result.answer == fake_answer
            assert result.confidence_score == 0.77
            assert len(result.sources) == len(dummy_docs)
            for chunk, doc in zip(result.sources, dummy_docs):
                assert chunk.text == doc.page_content
                assert isinstance(chunk.metadata, DocumentMetadata)

def test_answer_question_chain_exception(rag_service, dummy_query, dummy_docs):
    dummy_retriever = DummyRetriever(dummy_docs)
    with patch("backend.src.services.rag.vector_store_service") as mock_vs, \
         patch.object(rag_service, "prompt") as mock_prompt, \
         patch.object(rag_service, "llm") as mock_llm, \
         patch.object(rag_service, "output_parser") as mock_parser:
        mock_vs.as_retriever.return_value = dummy_retriever

        class DummyChain:
            def invoke(self, question):
                raise RuntimeError("Chain failed")
        mock_prompt.__or__.return_value = DummyChain()
        mock_llm.__or__.return_value = DummyChain()
        mock_parser.__or__.return_value = DummyChain()

        with patch("backend.src.services.rag.RunnableParallel", side_effect=lambda x: DummyChain()), \
             patch("backend.src.services.rag.RunnablePassthrough", side_effect=lambda: lambda x: x):
            result = rag_service.answer_question(dummy_query)
            assert "error occurred" in result.answer.lower()
            assert result.confidence_score == 0.0
            assert result.sources == []

def test_answer_question_equivalent_paths_retriever_none_vs_no_docs(rag_service, dummy_query):
    # Both should return similar "cannot find" answers but with different confidence
    with patch("backend.src.services.rag.vector_store_service") as mock_vs:
        # Path 1: retriever is None
        mock_vs.as_retriever.return_value = None
        result_none = rag_service.answer_question(dummy_query)

        # Path 2: retriever returns empty docs
        dummy_retriever = DummyRetriever([])
        mock_vs.as_retriever.return_value = dummy_retriever
        result_empty = rag_service.answer_question(dummy_query)

        assert "cannot find" in result_none.answer.lower()
        assert "cannot find" in result_empty.answer.lower()
        assert result_none.confidence_score <= result_empty.confidence_score
        assert result_none.sources == []
        assert result_empty.sources == []

def test_answer_question_equivalent_paths_safety_vs_chain_exception(rag_service, dummy_query, dummy_docs):
    # Path 1: Safety violation
    with patch.object(rag_service, "_check_safety", return_value="Blocked for safety"):
        result_safety = rag_service.answer_question(dummy_query)
    # Path 2: Chain exception
    dummy_retriever = DummyRetriever(dummy_docs)
    with patch("backend.src.services.rag.vector_store_service") as mock_vs, \
         patch.object(rag_service, "prompt") as mock_prompt, \
         patch.object(rag_service, "llm") as mock_llm, \
         patch.object(rag_service, "output_parser") as mock_parser:
        mock_vs.as_retriever.return_value = dummy_retriever
        class DummyChain:
            def invoke(self, question):
                raise RuntimeError("fail")
        mock_prompt.__or__.return_value = DummyChain()
        mock_llm.__or__.return_value = DummyChain()
        mock_parser.__or__.return_value = DummyChain()
        with patch("backend.src.services.rag.RunnableParallel", side_effect=lambda x: DummyChain()), \
             patch("backend.src.services.rag.RunnablePassthrough", side_effect=lambda: lambda x: x):
            result_chain = rag_service.answer_question(dummy_query)
    # Both should not return sources, but answers and confidence differ
    assert result_safety.sources == []
    assert result_chain.sources == []
    assert result_safety.confidence_score != result_chain.confidence_score
    assert result_safety.answer != result_chain.answer
