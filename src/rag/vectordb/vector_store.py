"""
Vector Store Implementation for RAG System
Using BGE-M3 embedding model with ChromaDB
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm
import hashlib

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)


class VectorStore:
    """Abstract base class for vector stores"""
    
    def __init__(self, collection_name: str, persist_directory: str = None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
    def add_documents(self, documents: List[Dict], batch_size: int = 100):
        """Add documents to the vector store"""
        raise NotImplementedError
        
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant documents"""
        raise NotImplementedError
        
    def delete_collection(self):
        """Delete the entire collection"""
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of vector store"""
    
    def __init__(
        self,
        collection_name: str = "customer_support",
        persist_directory: str = "./data/chromadb",
        embedding_model: str = "BAAI/bge-m3"
    ):
        """
        Initialize ChromaDB vector store
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
            embedding_model: Name of the embedding model
        """
        super().__init__(collection_name, persist_directory)
        
        # Create persist directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding function using SentenceTransformer
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def _generate_id(self, text: str, index: int) -> str:
        """Generate unique ID for document"""
        return hashlib.md5(f"{text}_{index}".encode()).hexdigest()
    
    def add_documents(
        self,
        documents: List[Dict],
        batch_size: int = 100,
        show_progress: bool = True
    ):
        """
        Add documents to ChromaDB
        
        Args:
            documents: List of documents with 'instruction', 'response', etc.
            batch_size: Batch size for adding documents
            show_progress: Show progress bar
        """
        logger.info(f"Adding {len(documents)} documents to collection")
        
        # Prepare documents for insertion
        all_texts = []
        all_metadatas = []
        all_ids = []
        
        for idx, doc in enumerate(documents):
            # Combine instruction and response for embedding
            text = f"질문: {doc.get('instruction', '')}\n답변: {doc.get('response', '')}"
            
            # Prepare metadata
            metadata = {
                "instruction": doc.get('instruction', ''),
                "response": doc.get('response', ''),
                "source": doc.get('source', 'unknown'),
                "domain": doc.get('domain', 'customer_support'),
                "has_context": doc.get('has_context', False)
            }
            
            all_texts.append(text)
            all_metadatas.append(metadata)
            all_ids.append(self._generate_id(text, idx))
        
        # Add documents in batches
        iterator = range(0, len(all_texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Adding documents")
        
        for i in iterator:
            batch_texts = all_texts[i:i+batch_size]
            batch_metadatas = all_metadatas[i:i+batch_size]
            batch_ids = all_ids[i:i+batch_size]
            
            self.collection.add(
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
        
        logger.info(f"Successfully added {len(documents)} documents")
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of relevant documents with scores
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=filter_dict
        )
        
        # Format results
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                doc = {
                    'id': results['ids'][0][i],
                    'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'metadata': results['metadatas'][0][i],
                    'document': results['documents'][0][i] if 'documents' in results else None
                }
                formatted_results.append(doc)
        
        return formatted_results
    
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        keyword_weight: float = 0.3
    ) -> List[Dict]:
        """
        Hybrid search combining vector and keyword search
        (Optional - to be implemented later)
        
        Args:
            query: Search query
            k: Number of results
            keyword_weight: Weight for keyword search (0-1)
            
        Returns:
            List of relevant documents
        """
        # Vector search
        vector_results = self.search(query, k=k*2)
        
        # TODO: Implement BM25 or keyword search
        # keyword_results = self.keyword_search(query, k=k*2)
        
        # TODO: Combine and rerank results
        # combined_results = self._combine_results(vector_results, keyword_results, keyword_weight)
        
        return vector_results[:k]
    
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": self.persist_directory
        }


if __name__ == "__main__":
    # Test the vector store
    logging.basicConfig(level=logging.INFO)
    
    # Initialize vector store
    vector_store = ChromaVectorStore(
        collection_name="test_collection",
        persist_directory="./data/chromadb_test"
    )
    
    # Test data
    test_docs = [
        {
            "instruction": "How do I reset my password?",
            "response": "To reset your password, go to the login page and click 'Forgot Password'.",
            "source": "test",
            "domain": "customer_support"
        },
        {
            "instruction": "What are your business hours?",
            "response": "Our business hours are Monday to Friday, 9 AM to 5 PM EST.",
            "source": "test",
            "domain": "customer_support"
        }
    ]
    
    # Add documents
    vector_store.add_documents(test_docs)
    
    # Test search
    results = vector_store.search("password reset", k=2)
    print("\nSearch results:")
    for result in results:
        print(f"Score: {result['score']:.4f}")
        print(f"Response: {result['metadata']['response']}")
        print()
    
    # Get stats
    stats = vector_store.get_collection_stats()
    print(f"Collection stats: {stats}")