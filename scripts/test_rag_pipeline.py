#!/usr/bin/env python3
"""
Test RAG Pipeline
Quick test script to verify RAG system functionality
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag import RAGPipeline, RAGConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_rag_pipeline():
    """Test RAG pipeline with sample queries"""
    
    # Initialize RAG pipeline
    logger.info("Initializing RAG pipeline...")
    
    config = RAGConfig(
        collection_name="customer_support_rag",
        persist_dir="./data/chromadb",
        retrieval_top_k=3,
        model_name="Qwen/Qwen3-0.6B",
        load_in_4bit=True,
        temperature=0.7,
        max_new_tokens=256,
        use_rag=True,
        verbose=True
    )
    
    pipeline = RAGPipeline(config)
    
    # Test queries
    test_queries = [
        "How do I reset my password?",
        "What is your refund policy?",
        "How can I contact customer support?",
        "My order hasn't arrived yet",
        "How do I cancel my subscription?"
    ]
    
    logger.info(f"\nTesting with {len(test_queries)} queries...")
    
    for i, query in enumerate(test_queries[:2], 1):  # Test first 2 queries
        print(f"\n{'='*80}")
        print(f"Test {i}/{len(test_queries[:2])}")
        print(f"{'='*80}")
        print(f"Query: {query}")
        
        # Test with RAG
        result = pipeline.query(query)
        
        print(f"\n📚 RAG Response:")
        print(f"{result['response']}")
        
        print(f"\n📊 Metadata:")
        print(f"  - Total time: {result['metadata']['total_time']:.2f}s")
        print(f"  - Retrieval time: {result['metadata']['retrieval_time']:.2f}s")
        print(f"  - Generation time: {result['metadata']['generation_time']:.2f}s")
        print(f"  - Retrieved docs: {result['metadata']['num_retrieved_docs']}")
        
        if 'retrieved_context' in result:
            print(f"\n🔍 Retrieved Context (Top 3):")
            for j, doc in enumerate(result['retrieved_context'][:3], 1):
                print(f"\n  Document {j} (Score: {doc['score']:.3f}):")
                print(f"  Q: {doc['instruction'][:100]}...")
                print(f"  A: {doc['response'][:100]}...")
    
    # Compare with non-RAG
    logger.info("\n\nComparing with non-RAG generation...")
    pipeline.toggle_rag(False)
    
    query = test_queries[0]
    print(f"\n{'='*80}")
    print(f"Comparison: RAG vs Non-RAG")
    print(f"{'='*80}")
    print(f"Query: {query}")
    
    # Non-RAG response
    result_no_rag = pipeline.query(query)
    print(f"\n❌ Without RAG:")
    print(f"{result_no_rag['response']}")
    
    # Re-enable RAG
    pipeline.toggle_rag(True)
    result_with_rag = pipeline.query(query)
    print(f"\n✅ With RAG:")
    print(f"{result_with_rag['response']}")
    
    # Statistics
    print(f"\n📈 Pipeline Statistics:")
    stats = pipeline.get_statistics()
    print(f"  - Model: {stats['config']['model']}")
    print(f"  - Collection: {stats['retriever']['vector_store_stats']['collection_name']}")
    print(f"  - Documents: {stats['retriever']['vector_store_stats']['document_count']}")
    
    logger.info("\nRAG pipeline test completed successfully!")


if __name__ == "__main__":
    test_rag_pipeline()