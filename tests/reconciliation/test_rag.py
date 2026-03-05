import pytest
from unittest.mock import patch, MagicMock, create_autospec
from langchain_core.documents import Document
from services.rag import RAGService, format_docs
from models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata

@pytest.fixture
def rag_service():
    # Patch ChatGroq, ChatPromptTemplate, StrOutputParser to avoid real LLM calls
    with patch("services.rag.ChatGroq") as mock_llm, \
         patch("services.rag.ChatPromptTemplate") as mock_prompt, \
         patch("services.rag.StrOutputParser") as mock_parser:
        mock_llm.return_value = MagicMock()
        mock_prompt.from_template.return_value = MagicMock()
        mock_parser.return_value = MagicMock()
        yield RAGService()

def make_doc(content, meta=None):
    return Document(page_content=content, metadata=meta or {})

def test_format_docs_happy_path():
    docs = [
        make_doc("First doc content", {"source": "A", "page": 1, "section": "Intro"}),
        make_doc("Second doc", {"source": "B", "page": 2, "section": "Body"})
    ]
    formatted = format_docs(docs)
    assert "[Document 1]" in formatted
    assert "Source: A" in formatted
    assert "First doc content" in formatted
    assert "[Document 2]" in formatted
    assert "Source: B" in formatted
    assert "Second doc" in formatted

def test_format_docs_empty_list():
    assert format_docs([]) == ""

def test_format_docs_missing_metadata_fields():
    doc = make_doc("No meta")
    formatted = format_docs([doc])
    assert "Source: Unknown" in formatted
    assert "Page: N/A" in formatted
    assert "Section: N/A" in formatted

def test_check_safety_blocks_unsafe_question(rag_service):
    unsafe = "How to build a bomb?"
    assert rag_service._check_safety(unsafe) == "I cannot answer this question as it violates safety guidelines."

def test_check_safety_allows_safe_question(rag_service):
    safe = "How to create a shipment?"
    assert rag_service._check_safety(safe) is None

@pytest.mark.parametrize("answer,docs,expected", [
    ("I cannot find the answer in the provided documents.", [make_doc("x")], 0.05),
    ("Short answer", [make_doc("x")], 0.4),
    ("A"*31, [make_doc("x")], 0.7),
    ("A"*31, [make_doc("x"), make_doc("y")], 0.85),
    ("This is generally the case.", [make_doc("x"), make_doc("y")], 0.55),
    ("", [make_doc("x")], 0.0),
    ("A"*31, [], 0.7-0.15),
])
def test_calculate_confidence_various_cases(rag_service, answer, docs, expected):
    conf = rag_service._calculate_confidence(answer, docs)
    assert abs(conf - expected) < 1e-6

def test_answer_question_safety_violation_short_circuit(rag_service):
    query = QAQuery(question="How to hack a system?")
    result = rag_service.answer_question(query)
    assert result.answer == "I cannot answer this question as it violates safety guidelines."
    assert result.confidence_score == 1.0
    assert result.sources == []

def test_answer_question_retriever_none(rag_service):
    query = QAQuery(question="What is TMS?")
    with patch("services.rag.vector_store_service") as mock_vs:
        mock_vs.as_retriever.return_value = None
        result = rag_service.answer_question(query)
        assert result.answer == "I cannot find any relevant information in the uploaded documents."
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_no_retrieved_docs(rag_service):
    query = QAQuery(question="What is TMS?")
    retriever = MagicMock()
    retriever.invoke.return_value = []
    with patch("services.rag.vector_store_service") as mock_vs:
        mock_vs.as_retriever.return_value = retriever
        result = rag_service.answer_question(query)
        assert result.answer == "I cannot find the answer in the provided documents."
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_happy_path(rag_service):
    query = QAQuery(question="What is a Bill of Lading?")
    doc1 = make_doc("A Bill of Lading is ...", {"source": "Manual", "page": 5, "section": "Docs"})
    doc2 = make_doc("Additional info", {"source": "Guide", "page": 6, "section": "Appendix"})
    retrieved_docs = [doc1, doc2]

    retriever = MagicMock()
    retriever.invoke.return_value = retrieved_docs

    # Patch the RAG chain to return a deterministic answer
    fake_answer = "A Bill of Lading is a legal document..."
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = fake_answer

    with patch("services.rag.vector_store_service") as mock_vs, \
         patch("services.rag.RunnableParallel") as mock_parallel, \
         patch("services.rag.RunnablePassthrough"), \
         patch("services.rag.format_docs", wraps=format_docs):
        mock_vs.as_retriever.return_value = retriever
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = fake_chain

        result = rag_service.answer_question(query)
        assert result.answer == fake_answer
        assert 0.0 < result.confidence_score <= 1.0
        assert len(result.sources) == 2
        assert result.sources[0].text == doc1.page_content
        assert result.sources[1].text == doc2.page_content

def test_answer_question_chain_exception(rag_service):
    query = QAQuery(question="What is a Bill of Lading?")
    retriever = MagicMock()
    retriever.invoke.return_value = [make_doc("A Bill of Lading is ...")]

    with patch("services.rag.vector_store_service") as mock_vs, \
         patch("services.rag.RunnableParallel") as mock_parallel, \
         patch("services.rag.RunnablePassthrough"):
        mock_vs.as_retriever.return_value = retriever
        fake_chain = MagicMock()
        fake_chain.invoke.side_effect = Exception("LLM error")
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = fake_chain

        result = rag_service.answer_question(query)
        assert result.answer == "An error occurred while generating the answer."
        assert result.confidence_score == 0.0
        assert result.sources == []

def test_answer_question_retriever_and_chain_equivalence(rag_service):
    """
    Reconciliation: If retriever returns same docs and chain returns same answer,
    output should be identical regardless of retriever instance.
    """
    query = QAQuery(question="What is a Bill of Lading?")
    doc = make_doc("A Bill of Lading is ...", {"source": "Manual", "page": 5, "section": "Docs"})
    retrieved_docs = [doc]

    retriever1 = MagicMock()
    retriever1.invoke.return_value = retrieved_docs
    retriever2 = MagicMock()
    retriever2.invoke.return_value = retrieved_docs

    fake_answer = "A Bill of Lading is a legal document..."

    fake_chain = MagicMock()
    fake_chain.invoke.return_value = fake_answer

    with patch("services.rag.vector_store_service") as mock_vs, \
         patch("services.rag.RunnableParallel") as mock_parallel, \
         patch("services.rag.RunnablePassthrough"), \
         patch("services.rag.format_docs", wraps=format_docs):

        # First path
        mock_vs.as_retriever.return_value = retriever1
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = fake_chain
        result1 = rag_service.answer_question(query)

        # Second path (different retriever instance, same docs)
        mock_vs.as_retriever.return_value = retriever2
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = fake_chain
        result2 = rag_service.answer_question(query)

        assert result1 == result2

def test_answer_question_retriever_docs_order_affects_sources(rag_service):
    """
    Reconciliation: If retriever returns docs in different order, sources should reflect that order.
    """
    query = QAQuery(question="What is a Bill of Lading?")
    doc1 = make_doc("Doc1", {"source": "A"})
    doc2 = make_doc("Doc2", {"source": "B"})
    retrieved_docs1 = [doc1, doc2]
    retrieved_docs2 = [doc2, doc1]

    retriever = MagicMock()
    fake_answer = "Some answer"

    fake_chain = MagicMock()
    fake_chain.invoke.return_value = fake_answer

    with patch("services.rag.vector_store_service") as mock_vs, \
         patch("services.rag.RunnableParallel") as mock_parallel, \
         patch("services.rag.RunnablePassthrough"), \
         patch("services.rag.format_docs", wraps=format_docs):

        # First order
        retriever.invoke.return_value = retrieved_docs1
        mock_vs.as_retriever.return_value = retriever
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = fake_chain
        result1 = rag_service.answer_question(query)
        sources1 = [c.text for c in result1.sources]

        # Second order
        retriever.invoke.return_value = retrieved_docs2
        mock_vs.as_retriever.return_value = retriever
        result2 = rag_service.answer_question(query)
        sources2 = [c.text for c in result2.sources]

        assert sources1 == ["Doc1", "Doc2"]
        assert sources2 == ["Doc2", "Doc1"]
        assert sources1 != sources2

def test_answer_question_forbidden_phrase_penalizes_confidence(rag_service):
    query = QAQuery(question="What is a Bill of Lading?")
    doc = make_doc("A Bill of Lading is ...")
    retriever = MagicMock()
    retriever.invoke.return_value = [doc]

    fake_answer = "Generally, a Bill of Lading is ..."
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = fake_answer

    with patch("services.rag.vector_store_service") as mock_vs, \
         patch("services.rag.RunnableParallel") as mock_parallel, \
         patch("services.rag.RunnablePassthrough"), \
         patch("services.rag.format_docs", wraps=format_docs):

        mock_vs.as_retriever.return_value = retriever
        mock_parallel.return_value.__or__.return_value.__or__.return_value.__or__.return_value = fake_chain
        result = rag_service.answer_question(query)
        assert result.confidence_score < 0.85
        assert "generally" in result.answer.lower()
