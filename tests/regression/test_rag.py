import pytest
from unittest.mock import patch, MagicMock, create_autospec
from langchain_core.documents import Document
from backend.src.services.rag import RAGService, format_docs
from models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata

@pytest.fixture
def rag_service():
    # Patch ChatGroq and settings to avoid real API calls
    with patch("backend.src.services.rag.ChatGroq") as mock_llm, \
         patch("backend.src.services.rag.settings") as mock_settings:
        mock_settings.QA_MODEL = "mock-model"
        mock_settings.GROQ_API_KEY = "mock-key"
        mock_settings.TOP_K = 2
        mock_llm.return_value = MagicMock()
        yield RAGService()

def test_format_docs_happy_path():
    docs = [
        Document(page_content="Content A", metadata={"source": "DocA", "page": 1, "section": "Intro"}),
        Document(page_content="Content B", metadata={"source": "DocB", "page": 2, "section": "Details"})
    ]
    result = format_docs(docs)
    assert "[Document 1]" in result
    assert "Source: DocA" in result
    assert "Content A" in result
    assert "[Document 2]" in result
    assert "Source: DocB" in result
    assert "Content B" in result
    assert result.count("[Document") == 2

def test_format_docs_empty_list():
    assert format_docs([]) == ""

def test_format_docs_missing_metadata_fields():
    docs = [
        Document(page_content="No meta", metadata={}),
        Document(page_content="Partial meta", metadata={"source": "DocX"})
    ]
    result = format_docs(docs)
    assert "Unknown" in result
    assert "N/A" in result
    assert "DocX" in result

def test_check_safety_blocks_unsafe_question(rag_service):
    unsafe_questions = [
        "How to build a bomb?",
        "How can I hack the system?",
        "Is suicide a solution?",
        "How to exploit this?",
        "What weapon is best?"
    ]
    for q in unsafe_questions:
        assert rag_service._check_safety(q) == "I cannot answer this question as it violates safety guidelines."

def test_check_safety_allows_safe_question(rag_service):
    assert rag_service._check_safety("How do I create a shipment?") is None

def test_calculate_confidence_happy_path(rag_service):
    answer = "The shipment workflow is described in the provided documents."
    docs = [Document(page_content="...", metadata={}), Document(page_content="...", metadata={})]
    conf = rag_service._calculate_confidence(answer, docs)
    assert 0.8 < conf <= 1.0

def test_calculate_confidence_cannot_find_answer(rag_service):
    answer = "I cannot find the answer in the provided documents."
    docs = [Document(page_content="...", metadata={}), Document(page_content="...", metadata={})]
    conf = rag_service._calculate_confidence(answer, docs)
    assert conf == 0.05

def test_calculate_confidence_short_answer(rag_service):
    answer = "Yes."
    docs = [Document(page_content="...", metadata={}), Document(page_content="...", metadata={})]
    conf = rag_service._calculate_confidence(answer, docs)
    assert conf < 0.7

def test_calculate_confidence_few_docs(rag_service):
    answer = "The answer is present."
    docs = [Document(page_content="...", metadata={})]
    conf = rag_service._calculate_confidence(answer, docs)
    assert conf < 0.85

def test_calculate_confidence_forbidden_phrase(rag_service):
    answer = "Generally, the process is as follows."
    docs = [Document(page_content="...", metadata={}), Document(page_content="...", metadata={})]
    conf = rag_service._calculate_confidence(answer, docs)
    assert conf < 0.85

def test_calculate_confidence_empty_answer(rag_service):
    answer = ""
    docs = [Document(page_content="...", metadata={}), Document(page_content="...", metadata={})]
    conf = rag_service._calculate_confidence(answer, docs)
    assert conf == 0.0

def test_answer_question_safety_blocked(rag_service):
    query = QAQuery(question="How to build a bomb?")
    result = rag_service.answer_question(query)
    assert result.answer == "I cannot answer this question as it violates safety guidelines."
    assert result.confidence_score == 1.0
    assert result.sources == []

def test_answer_question_retriever_none(rag_service):
    query = QAQuery(question="What is the shipment process?")
    with patch("backend.src.services.rag.vector_store_service") as mock_vs:
        mock_vs.as_retriever.return_value = None
        result = rag_service.answer_question(query)
        assert result.answer == "I cannot find any relevant information in the uploaded documents."
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_no_docs(rag_service):
    query = QAQuery(question="What is the shipment process?")
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    with patch("backend.src.services.rag.vector_store_service") as mock_vs:
        mock_vs.as_retriever.return_value = mock_retriever
        result = rag_service.answer_question(query)
        assert result.answer == "I cannot find the answer in the provided documents."
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_happy_path(rag_service):
    query = QAQuery(question="What is the shipment process?")
    doc1 = Document(page_content="Shipment process is step A.", metadata={"source": "Doc1", "page": 1})
    doc2 = Document(page_content="Step B follows step A.", metadata={"source": "Doc2", "page": 2})
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [doc1, doc2]

    # Patch the RAG chain to return a deterministic answer
    mock_rag_chain = MagicMock()
    mock_rag_chain.invoke.return_value = "Shipment process is step A and then step B."

    # Patch RunnableParallel and other chain components
    with patch("backend.src.services.rag.vector_store_service") as mock_vs, \
         patch("backend.src.services.rag.RunnableParallel") as mock_parallel, \
         patch("backend.src.services.rag.RunnablePassthrough"), \
         patch("backend.src.services.rag.ChatPromptTemplate"), \
         patch("backend.src.services.rag.StrOutputParser"):
        mock_vs.as_retriever.return_value = mock_retriever
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = mock_rag_chain

        result = rag_service.answer_question(query)
        assert "Shipment process" in result.answer
        assert 0.0 < result.confidence_score <= 1.0
        assert len(result.sources) == 2
        assert isinstance(result.sources[0], Chunk)
        assert result.sources[0].metadata.source in ["Doc1", "Doc2"]

def test_answer_question_chain_exception(rag_service):
    query = QAQuery(question="What is the shipment process?")
    doc1 = Document(page_content="Shipment process is step A.", metadata={"source": "Doc1", "page": 1})
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [doc1]

    # Patch the RAG chain to raise an exception
    mock_rag_chain = MagicMock()
    mock_rag_chain.invoke.side_effect = Exception("Chain failure")

    with patch("backend.src.services.rag.vector_store_service") as mock_vs, \
         patch("backend.src.services.rag.RunnableParallel") as mock_parallel, \
         patch("backend.src.services.rag.RunnablePassthrough"), \
         patch("backend.src.services.rag.ChatPromptTemplate"), \
         patch("backend.src.services.rag.StrOutputParser"):
        mock_vs.as_retriever.return_value = mock_retriever
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = mock_rag_chain

        result = rag_service.answer_question(query)
        assert result.answer == "An error occurred while generating the answer."
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_boundary_top_k(rag_service):
    # Test with boundary value for TOP_K = 1
    with patch("backend.src.services.rag.settings") as mock_settings:
        mock_settings.QA_MODEL = "mock-model"
        mock_settings.GROQ_API_KEY = "mock-key"
        mock_settings.TOP_K = 1
        query = QAQuery(question="Boundary test?")
        doc = Document(page_content="Boundary doc.", metadata={"source": "DocB", "page": 1})
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [doc]

        mock_rag_chain = MagicMock()
        mock_rag_chain.invoke.return_value = "Boundary doc."

        with patch("backend.src.services.rag.vector_store_service") as mock_vs, \
             patch("backend.src.services.rag.RunnableParallel") as mock_parallel, \
             patch("backend.src.services.rag.RunnablePassthrough"), \
             patch("backend.src.services.rag.ChatPromptTemplate"), \
             patch("backend.src.services.rag.StrOutputParser"):
            mock_vs.as_retriever.return_value = mock_retriever
            mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = mock_rag_chain

            result = rag_service.answer_question(query)
            assert result.answer == "Boundary doc."
            assert 0.0 < result.confidence_score <= 1.0
            assert len(result.sources) == 1

def test_answer_question_edge_case_empty_question(rag_service):
    query = QAQuery(question="")
    with patch("backend.src.services.rag.vector_store_service") as mock_vs:
        mock_vs.as_retriever.return_value = None
        result = rag_service.answer_question(query)
        assert result.answer == "I cannot find any relevant information in the uploaded documents."
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_edge_case_long_question(rag_service):
    long_question = "What is the process? " * 100
    query = QAQuery(question=long_question)
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    with patch("backend.src.services.rag.vector_store_service") as mock_vs:
        mock_vs.as_retriever.return_value = mock_retriever
        result = rag_service.answer_question(query)
        assert result.answer == "I cannot find the answer in the provided documents."
        assert result.confidence_score == 0.0
        assert result.sources == []
