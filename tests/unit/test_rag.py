# source_hash: e871aa5cb722f502
import pytest
from unittest.mock import patch, MagicMock, create_autospec
from langchain_core.documents import Document
from services.rag import RAGService, format_docs
from models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata

# ---------- format_docs tests ----------

def test_format_docs_empty_list_returns_empty_string():
    assert format_docs([]) == ""

def test_format_docs_single_document_with_metadata():
    doc = Document(
        page_content="This is the content.",
        metadata={"source": "Manual.pdf", "page": 5, "section": "Intro"}
    )
    result = format_docs([doc])
    assert "[Document 1]" in result
    assert "Manual.pdf" in result
    assert "Page: 5" in result
    assert "Section: Intro" in result
    assert "This is the content." in result

def test_format_docs_multiple_documents_with_partial_metadata():
    doc1 = Document(page_content="Doc1", metadata={"source": "A", "page": 1})
    doc2 = Document(page_content="Doc2", metadata={})
    doc3 = Document(page_content="Doc3", metadata=None)
    result = format_docs([doc1, doc2, doc3])
    assert result.count("[Document") == 3
    assert "Source: A" in result
    assert "Source: Unknown" in result
    assert "Page: 1" in result
    assert "Page: N/A" in result
    assert "Section: N/A" in result
    assert "Doc1" in result and "Doc2" in result and "Doc3" in result

# ---------- RAGService._check_safety tests ----------

@pytest.mark.parametrize("question,expected", [
    ("How to hack the system?", "I cannot answer this question as it violates safety guidelines."),
    ("Tell me about bomb threats", "I cannot answer this question as it violates safety guidelines."),
    ("What is the weather?", None),
    ("Is suicide a topic?", "I cannot answer this question as it violates safety guidelines."),
    ("How to optimize logistics?", None),
])
def test_check_safety_various_inputs(question, expected):
    service = RAGService()
    assert service._check_safety(question) == expected

# ---------- RAGService._calculate_confidence tests ----------

def test_calculate_confidence_empty_answer_returns_zero():
    service = RAGService()
    docs = [Document(page_content="abc", metadata={})]
    assert service._calculate_confidence("", docs) == 0.0

def test_calculate_confidence_cannot_find_phrase_low_confidence():
    service = RAGService()
    docs = [Document(page_content="abc", metadata={})]
    answer = "I cannot find the answer in the provided documents."
    assert service._calculate_confidence(answer, docs) == 0.05

def test_calculate_confidence_short_answer_and_one_doc_penalized():
    service = RAGService()
    docs = [Document(page_content="abc", metadata={})]
    answer = "Short answer."
    # 0.85 - 0.3 (short) - 0.15 (few docs) = 0.4
    assert abs(service._calculate_confidence(answer, docs) - 0.4) < 1e-6

def test_calculate_confidence_forbidden_phrase_penalized():
    service = RAGService()
    docs = [Document(page_content="abc", metadata={}), Document(page_content="def", metadata={})]
    answer = "Generally, the process is simple and usually works."
    # 0.85 - 0.3 (forbidden) = 0.55
    assert abs(service._calculate_confidence(answer, docs) - 0.55) < 1e-6

def test_calculate_confidence_clamped_to_zero_and_one():
    service = RAGService()
    docs = []
    answer = "generally"
    # 0.85 - 0.3 (forbidden) - 0.3 (short) - 0.15 (few docs) = 0.1
    assert abs(service._calculate_confidence(answer, docs) - 0.1) < 1e-6
    # Over 1.0
    answer = "A" * 1000
    docs = [Document(page_content="abc", metadata={})] * 10
    assert 0.0 <= service._calculate_confidence(answer, docs) <= 1.0

# ---------- RAGService.answer_question tests ----------

@patch("services.rag.vector_store_service")
@patch("services.rag.settings")
def test_answer_question_safety_violation_returns_safety_message(mock_settings, mock_vector_store_service):
    service = RAGService()
    query = QAQuery(question="How to build a bomb?")
    result = service.answer_question(query)
    assert result.answer == "I cannot answer this question as it violates safety guidelines."
    assert result.confidence_score == 1.0
    assert result.sources == []

@patch("services.rag.vector_store_service")
@patch("services.rag.settings")
def test_answer_question_retriever_none_returns_no_info(mock_settings, mock_vector_store_service):
    service = RAGService()
    query = QAQuery(question="What is logistics?")
    mock_vector_store_service.as_retriever.return_value = None
    result = service.answer_question(query)
    assert result.answer == "I cannot find any relevant information in the uploaded documents."
    assert result.confidence_score == 0.0
    assert result.sources == []

@patch("services.rag.vector_store_service")
@patch("services.rag.settings")
def test_answer_question_no_documents_returns_cannot_find(mock_settings, mock_vector_store_service):
    service = RAGService()
    query = QAQuery(question="What is logistics?")
    retriever = MagicMock()
    retriever.invoke.return_value = []
    mock_vector_store_service.as_retriever.return_value = retriever
    result = service.answer_question(query)
    assert result.answer == "I cannot find the answer in the provided documents."
    assert result.confidence_score == 0.0
    assert result.sources == []

@patch("services.rag.vector_store_service")
@patch("services.rag.settings")
def test_answer_question_happy_path_returns_sourced_answer(mock_settings, mock_vector_store_service):
    # Setup
    service = RAGService()
    query = QAQuery(question="What is a Bill of Lading?")
    doc1 = Document(page_content="A Bill of Lading is a legal document.", metadata={"source": "Doc1", "page": 1})
    doc2 = Document(page_content="It is used in shipping.", metadata={"source": "Doc2", "page": 2})
    retrieved_docs = [doc1, doc2]
    retriever = MagicMock()
    retriever.invoke.return_value = retrieved_docs
    mock_vector_store_service.as_retriever.return_value = retriever

    # Patch the RAG chain (RunnableParallel | prompt | llm | output_parser)
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = "A Bill of Lading is a legal document used in shipping."
    # Patch the chain construction
    with patch("services.rag.RunnableParallel", return_value=lambda x: {"context": "irrelevant", "question": x}), \
         patch.object(service, "prompt", create=True), \
         patch.object(service, "llm", create=True), \
         patch.object(service, "output_parser", create=True):
        # Patch the chain to return our fake_chain
        with patch("services.rag.RunnablePassthrough"), \
             patch("services.rag.format_docs", return_value="formatted context"), \
             patch("services.rag.RunnableParallel.__or__", return_value=fake_chain):
            result = service.answer_question(query)

    assert isinstance(result, SourcedAnswer)
    assert "Bill of Lading" in result.answer
    assert 0.0 < result.confidence_score <= 1.0
    assert len(result.sources) == 2
    assert all(isinstance(chunk, Chunk) for chunk in result.sources)
    assert result.sources[0].text == doc1.page_content
    assert result.sources[1].text == doc2.page_content

@patch("services.rag.vector_store_service")
@patch("services.rag.settings")
def test_answer_question_chain_exception_returns_error_message(mock_settings, mock_vector_store_service):
    service = RAGService()
    query = QAQuery(question="What is a Bill of Lading?")
    doc = Document(page_content="A Bill of Lading is a legal document.", metadata={})
    retriever = MagicMock()
    retriever.invoke.return_value = [doc]
    mock_vector_store_service.as_retriever.return_value = retriever

    # Patch the RAG chain to raise an exception
    fake_chain = MagicMock()
    fake_chain.invoke.side_effect = Exception("Chain error")
    with patch("services.rag.RunnableParallel", return_value=lambda x: {"context": "irrelevant", "question": x}), \
         patch.object(service, "prompt", create=True), \
         patch.object(service, "llm", create=True), \
         patch.object(service, "output_parser", create=True):
        with patch("services.rag.RunnablePassthrough"), \
             patch("services.rag.format_docs", return_value="formatted context"), \
             patch("services.rag.RunnableParallel.__or__", return_value=fake_chain):
            result = service.answer_question(query)

    assert result.answer == "An error occurred while generating the answer."
    assert result.confidence_score == 0.0
    assert result.sources == []

# ---------- Reconciliation/Regression: output equivalence ----------

@patch("services.rag.vector_store_service")
@patch("services.rag.settings")
def test_answer_question_equivalent_paths_same_output(mock_settings, mock_vector_store_service):
    # This test ensures that two equivalent queries with same retriever/documents produce same answer and confidence
    service = RAGService()
    query1 = QAQuery(question="What is a Bill of Lading?")
    query2 = QAQuery(question="What is a Bill of Lading?")
    doc = Document(page_content="A Bill of Lading is a legal document.", metadata={})
    retriever = MagicMock()
    retriever.invoke.return_value = [doc]
    mock_vector_store_service.as_retriever.return_value = retriever

    fake_chain = MagicMock()
    fake_chain.invoke.return_value = "A Bill of Lading is a legal document."
    with patch("services.rag.RunnableParallel", return_value=lambda x: {"context": "irrelevant", "question": x}), \
         patch.object(service, "prompt", create=True), \
         patch.object(service, "llm", create=True), \
         patch.object(service, "output_parser", create=True):
        with patch("services.rag.RunnablePassthrough"), \
             patch("services.rag.format_docs", return_value="formatted context"), \
             patch("services.rag.RunnableParallel.__or__", return_value=fake_chain):
            result1 = service.answer_question(query1)
            result2 = service.answer_question(query2)

    assert result1.answer == result2.answer
    assert result1.confidence_score == result2.confidence_score
    assert [c.text for c in result1.sources] == [c.text for c in result2.sources]
