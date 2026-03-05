from typing import List, Optional
import logging

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document

from models.schemas import QAQuery, SourcedAnswer, Chunk, DocumentMetadata
from services.vector_store import vector_store_service
from core.config import settings

logger = logging.getLogger(__name__)


# -----------------------------
# Context Formatter
# -----------------------------
def format_docs(docs: List[Document]) -> str:
    """
    Format retrieved documents with metadata for stronger grounding.
    """
    if not docs:
        return ""

    formatted_blocks = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        source = meta.get("source", "Unknown")
        page = meta.get("page", "N/A")
        section = meta.get("section", "N/A")

        block = (
            f"[Document {i}]\n"
            f"Source: {source} | Page: {page} | Section: {section}\n"
            f"{doc.page_content}"
        )
        formatted_blocks.append(block)

    return "\n\n".join(formatted_blocks)


# -----------------------------
# RAG Service
# -----------------------------
class RAGService:
    def __init__(self):
        # Deterministic LLM for enterprise RAG
        self.llm = ChatGroq(
            temperature=0,
            model_name=settings.QA_MODEL,
            api_key=settings.GROQ_API_KEY,
            max_tokens=1024,
            top_p=0.9
        )

        # Hardened Logistics / TMS Prompt
        self.rag_template = """
You are an AI assistant specialized in Logistics and Transportation Management Systems (TMS).

You MUST answer ONLY using the information explicitly present in the provided context.
The context contains official logistics documents such as SOPs, user manuals, API guides, rate cards, shipment workflows, compliance policies, and configuration guides.

STRICT RULES:
- Do NOT use outside knowledge.
- Do NOT infer missing details.
- Do NOT assume workflows or configurations.
- Do NOT generate examples unless present in context.
- If the answer is not explicitly stated, respond EXACTLY with:
"I cannot find the answer in the provided documents."
- Accuracy is more important than completeness.
- Preserve terminology exactly as written.

CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER (With proper english sentence matching the question):
"""

        self.prompt = ChatPromptTemplate.from_template(self.rag_template)
        self.output_parser = StrOutputParser()

        # Hallucination phrase guard list
        self.forbidden_phrases = [
            "generally",
            "usually",
            "in most cases",
            "best practice",
            "typically",
            "commonly"
        ]

    # -----------------------------
    # Safety Filter
    # -----------------------------
    def _check_safety(self, question: str) -> Optional[str]:
        unsafe_keywords = ["bomb", "kill", "suicide", "hack", "exploit", "weapon"]
        q_lower = question.lower()

        if any(k in q_lower for k in unsafe_keywords):
            return "I cannot answer this question as it violates safety guidelines."

        return None

    # -----------------------------
    # Confidence Scoring
    # -----------------------------
    def _calculate_confidence(self, answer: str, retrieved_docs: List[Document]) -> float:
        if not answer:
            return 0.0

        answer_lower = answer.lower()
        confidence = 0.85

        if "cannot find the answer" in answer_lower:
            return 0.05

        if len(answer.strip()) < 30:
            confidence -= 0.3

        if len(retrieved_docs) < 2:
            confidence -= 0.15

        if any(p in answer_lower for p in self.forbidden_phrases):
            confidence -= 0.3

        return max(0.0, min(confidence, 1.0))

    # -----------------------------
    # Main QA Method
    # -----------------------------
    def answer_question(self, query: QAQuery) -> SourcedAnswer:
        # 1. Safety Check
        safety_error = self._check_safety(query.question)
        if safety_error:
            return SourcedAnswer(
                answer=safety_error,
                confidence_score=1.0,
                sources=[]
            )

        # 2. Retriever (MMR for diversity)
        retriever = vector_store_service.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": settings.TOP_K,
                "fetch_k": settings.TOP_K * 3,
                "lambda_mult": 0.7
            }
        )

        if retriever is None:
            return SourcedAnswer(
                answer="I cannot find any relevant information in the uploaded documents.",
                confidence_score=0.0,
                sources=[]
            )

        try:
            # 3. Retrieve documents first (empty guard)
            retrieved_docs = retriever.invoke(query.question)

            if not retrieved_docs:
                return SourcedAnswer(
                    answer="I cannot find the answer in the provided documents.",
                    confidence_score=0.0,
                    sources=[]
                )

            logger.info(f"Retrieved {len(retrieved_docs)} documents")

            # 4. Build LCEL RAG Chain
            rag_chain = (
                RunnableParallel(
                    {
                        "context": retriever | format_docs,
                        "question": RunnablePassthrough()
                    }
                )
                | self.prompt
                | self.llm
                | self.output_parser
            )

            # 5. Invoke Chain
            answer = rag_chain.invoke(query.question)
            logger.debug(f"Generated answer preview: {answer[:120]}")

            # 6. Convert sources
            source_chunks = [
                Chunk(
                    text=doc.page_content,
                    metadata=DocumentMetadata(**(doc.metadata or {}))
                )
                for doc in retrieved_docs
            ]

            # 7. Confidence
            confidence = self._calculate_confidence(answer, retrieved_docs)
            logger.info(f"Answer confidence: {confidence:.2f}")

            return SourcedAnswer(
                answer=answer,
                confidence_score=confidence,
                sources=source_chunks
            )

        except Exception as e:
            logger.error(f"Error in RAG chain: {e}", exc_info=True)
            return SourcedAnswer(
                answer="An error occurred while generating the answer.",
                confidence_score=0.0,
                sources=[]
            )


# Singleton instance
rag_service = RAGService()
