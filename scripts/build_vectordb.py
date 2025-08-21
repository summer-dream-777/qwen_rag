#!/usr/bin/env python3
"""
Build Vector Database for RAG System
Indexes customer support data into ChromaDB using BGE-M3 embeddings
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.vectordb import ChromaVectorStore

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_path: str, max_samples: int = None) -> List[Dict]:
    """
    Load customer support data
    
    Args:
        data_path: Path to data file
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        List of documents
    """
    logger.info(f"Loading data from: {data_path}")
    
    documents = []
    
    if data_path.endswith('.json'):
        # Load JSON or JSONL file
        try:
            # Try JSONL format first
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        documents.append(json.loads(line))
        except:
            # Try regular JSON
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    documents = data
                else:
                    documents = [data]
    
    elif data_path.endswith('.csv'):
        # Load CSV file
        df = pd.read_csv(data_path)
        documents = df.to_dict('records')
    
    else:
        # Try loading as HuggingFace dataset
        from datasets import load_from_disk
        dataset = load_from_disk(data_path)
        documents = [item for item in dataset]
    
    # Limit samples if specified
    if max_samples and max_samples > 0:
        documents = documents[:max_samples]
    
    logger.info(f"Loaded {len(documents)} documents")
    return documents


def preprocess_documents(documents: List[Dict]) -> List[Dict]:
    """
    Preprocess documents for indexing
    
    Args:
        documents: Raw documents
        
    Returns:
        Preprocessed documents
    """
    logger.info("Preprocessing documents...")
    
    processed_docs = []
    for doc in tqdm(documents, desc="Preprocessing"):
        # Ensure required fields exist
        if 'instruction' in doc and 'response' in doc:
            # Clean text
            instruction = doc['instruction'].strip()
            response = doc['response'].strip()
            
            # Skip empty documents
            if not instruction or not response:
                continue
            
            # Remove "Context:" prefix if it exists
            if instruction.startswith("Context :"):
                parts = instruction.split("\n\nQuestion :", 1)
                if len(parts) == 2:
                    context = parts[0].replace("Context :", "").strip()
                    question = parts[1].strip()
                    instruction = f"{question}\n\nContext: {context[:500]}..."  # Limit context length
                else:
                    instruction = instruction[:1000]  # Limit length
            
            processed_doc = {
                'instruction': instruction[:1000],  # Limit length
                'response': response[:2000],  # Limit length
                'source': doc.get('source', 'unknown'),
                'domain': doc.get('domain', 'customer_support'),
                'has_context': doc.get('has_context', False)
            }
            processed_docs.append(processed_doc)
    
    logger.info(f"Preprocessed {len(processed_docs)} valid documents")
    return processed_docs


def build_vectordb(
    data_path: str,
    collection_name: str = "customer_support_rag",
    persist_dir: str = "./data/chromadb",
    max_samples: int = None,
    batch_size: int = 100,
    reset_collection: bool = False
):
    """
    Build vector database from customer support data
    
    Args:
        data_path: Path to data file
        collection_name: Name of ChromaDB collection
        persist_dir: Directory to persist database
        max_samples: Maximum samples to index
        batch_size: Batch size for indexing
        reset_collection: Whether to reset existing collection
    """
    # Load data
    documents = load_data(data_path, max_samples)
    
    # Preprocess documents
    documents = preprocess_documents(documents)
    
    if not documents:
        logger.error("No valid documents to index")
        return
    
    # Initialize vector store
    logger.info(f"Initializing vector store: {collection_name}")
    vector_store = ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_model="BAAI/bge-m3"
    )
    
    # Reset collection if requested
    if reset_collection:
        logger.info("Resetting existing collection...")
        vector_store.delete_collection()
        vector_store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_model="BAAI/bge-m3"
        )
    
    # Check current collection stats
    stats = vector_store.get_collection_stats()
    logger.info(f"Current collection stats: {stats}")
    
    # Add documents to vector store
    logger.info("Building vector database...")
    vector_store.add_documents(
        documents,
        batch_size=batch_size,
        show_progress=True
    )
    
    # Get final stats
    final_stats = vector_store.get_collection_stats()
    logger.info(f"Final collection stats: {final_stats}")
    
    # Test search
    logger.info("\nTesting search functionality...")
    test_queries = [
        "How do I reset my password?",
        "What are the refund policies?",
        "How can I track my order?"
    ]
    
    for query in test_queries[:1]:  # Test with first query
        logger.info(f"\nQuery: {query}")
        results = vector_store.search(query, k=3)
        for i, result in enumerate(results):
            logger.info(f"Result {i+1} (Score: {result['score']:.4f}):")
            logger.info(f"  Instruction: {result['metadata']['instruction'][:100]}...")
            logger.info(f"  Response: {result['metadata']['response'][:100]}...")
    
    logger.info("\nVector database built successfully!")
    return vector_store


def main():
    parser = argparse.ArgumentParser(description="Build vector database for RAG system")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/unified_customer_support.json",
        help="Path to data file"
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="customer_support_rag",
        help="ChromaDB collection name"
    )
    parser.add_argument(
        "--persist_dir",
        type=str,
        default="./data/chromadb",
        help="Directory to persist vector database"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to index"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for indexing"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset existing collection"
    )
    
    args = parser.parse_args()
    
    build_vectordb(
        data_path=args.data_path,
        collection_name=args.collection_name,
        persist_dir=args.persist_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        reset_collection=args.reset
    )


if __name__ == "__main__":
    main()