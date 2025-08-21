"""
Retriever Implementation for RAG System
Handles document retrieval from vector database
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from ..vectordb import ChromaVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """Retrieved document with metadata"""
    content: str
    instruction: str
    response: str
    score: float
    metadata: Dict[str, Any]
    
    def format_context(self) -> str:
        """Format document for context"""
        return f"Q: {self.instruction}\nA: {self.response}"


class Retriever:
    """Abstract base class for retrievers"""
    
    def retrieve(self, query: str, k: int = 5) -> List[RetrievedDocument]:
        """Retrieve relevant documents"""
        raise NotImplementedError
        
    def rerank(self, query: str, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """Rerank retrieved documents"""
        return documents


class DenseRetriever(Retriever):
    """Dense retrieval using vector similarity"""
    
    def __init__(
        self,
        vector_store: ChromaVectorStore = None,
        collection_name: str = "customer_support_rag",
        persist_dir: str = "./data/chromadb",
        rerank: bool = False
    ):
        """
        Initialize dense retriever
        
        Args:
            vector_store: Pre-initialized vector store
            collection_name: Name of ChromaDB collection
            persist_dir: Directory with persisted database
            rerank: Whether to apply reranking
        """
        if vector_store:
            self.vector_store = vector_store
        else:
            self.vector_store = ChromaVectorStore(
                collection_name=collection_name,
                persist_directory=persist_dir,
                embedding_model="BAAI/bge-m3"
            )
        
        self.rerank_enabled = rerank
        logger.info(f"Initialized DenseRetriever with collection: {collection_name}")
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[RetrievedDocument]:
        """
        Retrieve relevant documents using vector similarity
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            filter_dict: Optional metadata filters
            
        Returns:
            List of retrieved documents
        """
        # Search in vector store
        results = self.vector_store.search(query, k=k, filter_dict=filter_dict)
        
        # Convert to RetrievedDocument objects
        documents = []
        for result in results:
            doc = RetrievedDocument(
                content=result.get('document', ''),
                instruction=result['metadata'].get('instruction', ''),
                response=result['metadata'].get('response', ''),
                score=result['score'],
                metadata=result['metadata']
            )
            documents.append(doc)
        
        # Apply reranking if enabled
        if self.rerank_enabled and len(documents) > 0:
            documents = self.rerank(query, documents)
        
        return documents
    
    def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        method: str = "similarity"
    ) -> List[RetrievedDocument]:
        """
        Rerank retrieved documents
        
        Args:
            query: Original query
            documents: Retrieved documents
            method: Reranking method
            
        Returns:
            Reranked documents
        """
        if method == "similarity":
            # Already sorted by similarity
            return documents
        elif method == "diversity":
            # MMR (Maximal Marginal Relevance) for diversity
            return self._mmr_rerank(query, documents)
        else:
            return documents
    
    def _mmr_rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        lambda_param: float = 0.7
    ) -> List[RetrievedDocument]:
        """
        Maximal Marginal Relevance reranking for diversity
        
        Args:
            query: Original query
            documents: Retrieved documents
            lambda_param: Trade-off parameter (0-1)
            
        Returns:
            Reranked documents with diversity
        """
        if len(documents) <= 1:
            return documents
        
        # Start with highest scoring document
        reranked = [documents[0]]
        remaining = documents[1:]
        
        while remaining and len(reranked) < len(documents):
            mmr_scores = []
            
            for doc in remaining:
                # Relevance to query (already computed)
                relevance = doc.score
                
                # Max similarity to already selected documents
                max_sim = max([self._text_similarity(
                    doc.format_context(),
                    selected.format_context()
                ) for selected in reranked])
                
                # MMR score
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                mmr_scores.append(mmr)
            
            # Select document with highest MMR score
            best_idx = np.argmax(mmr_scores)
            reranked.append(remaining[best_idx])
            remaining.pop(best_idx)
        
        return reranked
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Simple text similarity using Jaccard similarity
        (For production, use embedding similarity)
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # Tokenize
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        # Jaccard similarity
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / len(union)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        return {
            "vector_store_stats": self.vector_store.get_collection_stats(),
            "rerank_enabled": self.rerank_enabled
        }


class HybridRetriever(Retriever):
    """
    Hybrid retrieval combining dense and sparse methods
    (Optional - for future implementation)
    """
    
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_weight: float = 0.3
    ):
        """
        Initialize hybrid retriever
        
        Args:
            dense_retriever: Dense retriever instance
            sparse_weight: Weight for sparse retrieval (0-1)
        """
        self.dense_retriever = dense_retriever
        self.sparse_weight = sparse_weight
        
    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[RetrievedDocument]:
        """
        Hybrid retrieval combining dense and sparse methods
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            filter_dict: Optional metadata filters
            
        Returns:
            List of retrieved documents
        """
        # Get dense retrieval results
        dense_docs = self.dense_retriever.retrieve(query, k=k*2, filter_dict=filter_dict)
        
        # TODO: Implement sparse retrieval (BM25, TF-IDF)
        # sparse_docs = self.sparse_retrieve(query, k=k*2)
        
        # TODO: Combine and rerank
        # combined_docs = self._combine_results(dense_docs, sparse_docs)
        
        return dense_docs[:k]


if __name__ == "__main__":
    # Test retriever
    logging.basicConfig(level=logging.INFO)
    
    retriever = DenseRetriever(
        collection_name="customer_support_rag",
        persist_dir="./data/chromadb"
    )
    
    # Test retrieval
    query = "How can I reset my password?"
    documents = retriever.retrieve(query, k=3)
    
    print(f"\nQuery: {query}")
    print(f"Retrieved {len(documents)} documents:")
    
    for i, doc in enumerate(documents, 1):
        print(f"\n{i}. Score: {doc.score:.4f}")
        print(f"   Instruction: {doc.instruction[:100]}...")
        print(f"   Response: {doc.response[:100]}...")