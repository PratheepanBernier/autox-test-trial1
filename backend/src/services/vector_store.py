import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever
from models.schemas import Chunk, DocumentMetadata
from core.config import settings
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class VectorStoreService:
    def __init__(self):
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
            logger.info(f"Initialized HuggingFace embeddings with model: {settings.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}", exc_info=True)
            raise
            
        self.vector_store = None

    def add_documents(self, chunks: List[Chunk]):
        """
        Add chunks to the vector store.
        """
        try:
            documents = [
                Document(page_content=chunk.text, metadata=chunk.metadata.model_dump())
                for chunk in chunks
            ]
            
            if self.vector_store is None:
                logger.info("Initializing new FAISS vector store.")
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                self.vector_store.add_documents(documents)
            
            logger.info(f"Added {len(documents)} documents to vector store.")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}", exc_info=True)
            raise

    def as_retriever(self, search_type: str = "similarity", search_kwargs: Optional[dict] = None) -> Optional[BaseRetriever]:
        """
        Return a Retriever interface for the vector store.
        This follows LangChain best practices for RAG chains.
        
        Args:
            search_type: Type of search ("similarity", "mmr", "similarity_score_threshold")
            search_kwargs: Additional search parameters (e.g., {"k": 4})
        
        Returns:
            BaseRetriever instance or None if vector store is empty
        """
        if self.vector_store is None:
            logger.warning("Vector store is empty, cannot create retriever.")
            return None
        
        if search_kwargs is None:
            search_kwargs = {"k": settings.TOP_K}
        
        try:
            retriever = self.vector_store.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )
            logger.debug(f"Created retriever with search_type={search_type}, kwargs={search_kwargs}")
            return retriever
        except Exception as e:
            logger.error(f"Error creating retriever: {str(e)}", exc_info=True)
            return None

    def similarity_search(self, query: str, k: int = 4) -> List[Chunk]:
        """
        Search for similar chunks (legacy method, kept for backwards compatibility).
        Prefer using as_retriever() for new code.
        """
        if self.vector_store is None:
            logger.warning("Vector store is empty, returning no results.")
            return []
            
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            chunks = []
            for doc in docs:
                metadata = DocumentMetadata(**doc.metadata)
                chunks.append(Chunk(text=doc.page_content, metadata=metadata))
            return chunks
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}", exc_info=True)
            return []

vector_store_service = VectorStoreService()
