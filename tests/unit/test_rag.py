import pytest
from unittest.mock import patch, MagicMock, create_autospec
from langchain_core.documents import Document

from backend.src.services.rag import RAGService, format_docs
from backend.src.models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata

# ---------------------------
# format_docs tests
# ---------------------------

def test_format_docs_empty_list_returns_empty_string():
    assert format_docs([]) == ""

def test_format_docs_single_document_with_full_metadata():
    doc = Document(
        page_content="This is the content.",
        metadata={"source": "Manual.pdf", "page": 5, "section": "2.1"}
    )
    result = format_docs([doc])
    assert "[Document 1]" in result
    assert "Source: Manual.pdf" in result
    assert "Page: 5" in result
    assert "Section: 2.1" in result
    assert "This is the content." in result

def test_format_docs_multiple_documents_with_partial_metadata():
    doc1 = Document(page_content="Doc1", metadata={"source": "A", "page": 1})
    doc2 = Document(page_content="Doc2", metadata={})
    doc3 = Document(page_content="Doc3", metadata=None)
    result = format_docs([doc1, doc2, doc3])
    assert result.count("[Document") == 3
    assert "Source: A" in result
    assert "Page: 1" in result
    assert "Section: N/A" in result
    assert "Source: Unknown" in result
    assert "Doc3" in result

# ---------------------------
# RAGService._check_safety tests
# ---------------------------

@pytest.mark.parametrize("question,expected", [
    ("How to hack the system?", "I cannot answer this question as it violates safety guidelines."),
    ("What is the bomb code?", "I cannot answer this question as it violates safety guidelines."),
    ("Tell me about logistics.", None),
    ("Is suicide a topic?", "I cannot answer this question as it violates safety guidelines."),
    ("How to optimize shipments?", None),
])
def test_check_safety_various_inputs(question, expected):
    service = RAGService()
    assert service._check_safety(question) == expected

# ---------------------------
# RAGService._calculate_confidence tests
# ---------------------------

def test_calculate_confidence_empty_answer_returns_zero():
    service = RAGService()
    docs = [Document(page_content="abc", metadata={})]
    assert service._calculate_confidence("", docs) == 0.0

def test_calculate_confidence_cannot_find_phrase_low_confidence():
    service = RAGService()
    docs = [Document(page_content="abc", metadata={})]
    answer = "I cannot find the answer in the provided documents."
    assert service._calculate_confidence(answer, docs) == 0.05

def test_calculate_confidence_short_answer_and_few_docs_and_forbidden_phrase():
    service = RAGService()
    docs = [Document(page_content="abc", metadata={})]
    answer = "Generally, yes."
    # -0.3 for short, -0.15 for <2 docs, -0.3 for forbidden phrase, base 0.85
    expected = max(0.0, min(0.85 - 0.3 - 0.15 - 0.3, 1.0))
    assert service._calculate_confidence(answer, docs) == expected

def test_calculate_confidence_long_answer_many_docs_no_forbidden():
    service = RAGService()
    docs = [Document(page_content="abc", metadata={}) for _ in range(3)]
    answer = "This is a sufficiently long answer that does not contain forbidden phrases and is based on the documents."
    assert service._calculate_confidence(answer, docs) == 0.85

def test_calculate_confidence_confidence_cannot_exceed_one_or_below_zero():
    service = RAGService()
    docs = [Document(page_content="abc", metadata={}) for _ in range(10)]
    answer = "This is a normal answer."
    conf = service._calculate_confidence(answer, docs)
    assert 0.0 <= conf <= 1.0

# ---------------------------
# RAGService.answer_question tests
# ---------------------------

@patch("backend.src.services.rag.vector_store_service")
@patch("backend.src.services.rag.settings")
def test_answer_question_safety_violation_returns_safety_message(mock_settings, mock_vector_store_service):
    service = RAGService()
    query = QAQuery(question="How to build a bomb?")
    result = service.answer_question(query)
    assert result.answer == "I cannot answer this question as it violates safety guidelines."
    assert result.confidence_score == 1.0
    assert result.sources == []

@patch("backend.src.services.rag.vector_store_service")
@patch("backend.src.services.rag.settings")
def test_answer_question_retriever_none_returns_no_info_message(mock_settings, mock_vector_store_service):
    service = RAGService()
    query = QAQuery(question="What is the shipment process?")
    mock_vector_store_service.as_retriever.return_value = None
    result = service.answer_question(query)
    assert result.answer == "I cannot find any relevant information in the uploaded documents."
    assert result.confidence_score == 0.0
    assert result.sources == []

@patch("backend.src.services.rag.vector_store_service")
@patch("backend.src.services.rag.settings")
def test_answer_question_no_retrieved_docs_returns_no_answer_message(mock_settings, mock_vector_store_service):
    service = RAGService()
    query = QAQuery(question="What is the shipment process?")
    retriever = MagicMock()
    retriever.invoke.return_value = []
    mock_vector_store_service.as_retriever.return_value = retriever
    result = service.answer_question(query)
    assert result.answer == "I cannot find the answer in the provided documents."
    assert result.confidence_score == 0.0
    assert result.sources == []

@patch("backend.src.services.rag.vector_store_service")
@patch("backend.src.services.rag.settings")
def test_answer_question_happy_path_returns_sourced_answer(mock_settings, mock_vector_store_service):
    # Setup
    service = RAGService()
    query = QAQuery(question="What is the shipment process?")
    doc1 = Document(page_content="Step 1: Do X.", metadata={"source": "Manual.pdf", "page": 1})
    doc2 = Document(page_content="Step 2: Do Y.", metadata={"source": "Manual.pdf", "page": 2})
    retrieved_docs = [doc1, doc2]

    # Mock retriever
    retriever = MagicMock()
    retriever.invoke.return_value = retrieved_docs
    # The retriever is also used as a pipe in RunnableParallel, so we need to support __or__ and __call__
    retriever.__or__ = lambda self, other: lambda q: format_docs(retrieved_docs)
    retriever.__call__ = lambda self, q: retrieved_docs
    mock_vector_store_service.as_retriever.return_value = retriever

    # Patch settings
    mock_settings.QA_MODEL = "mock-model"
    mock_settings.GROQ_API_KEY = "mock-key"
    mock_settings.TOP_K = 2

    # Patch LLM and chain
    with patch.object(service, "llm", autospec=True) as mock_llm, \
         patch.object(service, "prompt", autospec=True) as mock_prompt, \
         patch.object(service, "output_parser", autospec=True) as mock_parser, \
         patch("backend.src.services.rag.RunnableParallel") as mock_runnable_parallel, \
         patch("backend.src.services.rag.RunnablePassthrough"):

        # Simulate chain: rag_chain.invoke returns answer
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "The shipment process is as follows: Step 1: Do X. Step 2: Do Y."
        mock_runnable_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = mock_chain

        result = service.answer_question(query)

        assert isinstance(result, SourcedAnswer)
        assert "shipment process" in result.answer
        assert result.confidence_score > 0.0
        assert len(result.sources) == 2
        assert all(isinstance(chunk, Chunk) for chunk in result.sources)
        assert result.sources[0].text == "Step 1: Do X."
        assert result.sources[1].text == "Step 2: Do Y."
        assert result.sources[0].metadata.source == "Manual.pdf"

@patch("backend.src.services.rag.vector_store_service")
@patch("backend.src.services.rag.settings")
def test_answer_question_chain_raises_exception_returns_error_message(mock_settings, mock_vector_store_service):
    service = RAGService()
    query = QAQuery(question="What is the shipment process?")
    doc1 = Document(page_content="Step 1: Do X.", metadata={"source": "Manual.pdf", "page": 1})
    retrieved_docs = [doc1]
    retriever = MagicMock()
    retriever.invoke.return_value = retrieved_docs
    retriever.__or__ = lambda self, other: lambda q: format_docs(retrieved_docs)
    retriever.__call__ = lambda self, q: retrieved_docs
    mock_vector_store_service.as_retriever.return_value = retriever
    mock_settings.QA_MODEL = "mock-model"
    mock_settings.GROQ_API_KEY = "mock-key"
    mock_settings.TOP_K = 1

    with patch.object(service, "llm", autospec=True) as mock_llm, \
         patch.object(service, "prompt", autospec=True) as mock_prompt, \
         patch.object(service, "output_parser", autospec=True) as mock_parser, \
         patch("backend.src.services.rag.RunnableParallel") as mock_runnable_parallel, \
         patch("backend.src.services.rag.RunnablePassthrough"):

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("Chain error")
        mock_runnable_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = mock_chain

        result = service.answer_question(query)
        assert result.answer == "An error occurred while generating the answer."
        assert result.confidence_score == 0.0
        assert result.sources == []

# ---------------------------
# RAGService singleton instance test
# ---------------------------

def test_rag_service_singleton_instance_exists():
    from backend.src.services.rag import rag_service
    assert isinstance(rag_service, RAGService)
