#!/usr/bin/env python3
"""Simple RAG test"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag import RAGPipeline, RAGConfig

# Simple test
print("Initializing RAG pipeline...")
config = RAGConfig(
    load_in_4bit=True,
    retrieval_top_k=2,
    max_new_tokens=100,
    temperature=0.7
)

pipeline = RAGPipeline(config)

# Test query
query = "How can I reset my password?"
print(f"\nQuery: {query}")

result = pipeline.query(query)

print(f"\nResponse: {result['response'][:500]}")
print(f"\nRetrieved {result['metadata']['num_retrieved_docs']} documents")
print(f"Total time: {result['metadata']['total_time']:.2f}s")