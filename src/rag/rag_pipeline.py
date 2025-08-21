"""
RAG Pipeline Implementation
Integrates retrieval and generation for complete RAG system
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
from pathlib import Path

from .retriever import DenseRetriever
from .generator import QwenGenerator
from .vectordb import ChromaVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline"""
    # Retriever config
    collection_name: str = "customer_support_rag"
    persist_dir: str = "./data/chromadb"
    retrieval_top_k: int = 5
    rerank: bool = False
    
    # Generator config
    model_name: str = "Qwen/Qwen3-0.6B"
    adapter_path: Optional[str] = None
    load_in_4bit: bool = False
    max_context_length: int = 2048
    max_new_tokens: int = 512
    
    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    
    # Pipeline config
    use_rag: bool = True
    include_scores: bool = False
    verbose: bool = True


class RAGPipeline:
    """Complete RAG pipeline integrating retrieval and generation"""
    
    def __init__(self, config: RAGConfig = None):
        """
        Initialize RAG pipeline
        
        Args:
            config: RAG configuration
        """
        self.config = config or RAGConfig()
        
        # Initialize retriever
        if self.config.use_rag:
            logger.info("Initializing retriever...")
            self.retriever = DenseRetriever(
                collection_name=self.config.collection_name,
                persist_dir=self.config.persist_dir,
                rerank=self.config.rerank
            )
        else:
            self.retriever = None
        
        # Initialize generator
        logger.info("Initializing generator...")
        self.generator = QwenGenerator(
            model_name=self.config.model_name,
            adapter_path=self.config.adapter_path,
            load_in_4bit=self.config.load_in_4bit,
            max_context_length=self.config.max_context_length,
            max_new_tokens=self.config.max_new_tokens
        )
        
        logger.info("RAG pipeline initialized successfully")
    
    def query(
        self,
        query: str,
        retrieval_k: Optional[int] = None,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline
        
        Args:
            query: User query
            retrieval_k: Number of documents to retrieve
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        
        # Retrieve context if RAG is enabled
        context = None
        retrieval_time = 0
        
        if self.config.use_rag and self.retriever:
            retrieval_start = time.time()
            k = retrieval_k or self.config.retrieval_top_k
            context = self.retriever.retrieve(query, k=k)
            retrieval_time = time.time() - retrieval_start
            
            if self.config.verbose:
                logger.info(f"Retrieved {len(context)} documents in {retrieval_time:.2f}s")
        
        # Generate response
        generation_start = time.time()
        
        # Merge generation parameters
        gen_params = {
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'top_k': self.config.top_k,
            'do_sample': self.config.do_sample
        }
        gen_params.update(generation_kwargs)
        
        response, gen_metadata = self.generator.generate(
            query=query,
            context=context,
            **gen_params
        )
        
        generation_time = time.time() - generation_start
        
        if self.config.verbose:
            logger.info(f"Generated response in {generation_time:.2f}s")
        
        # Compile results
        total_time = time.time() - start_time
        
        result = {
            'query': query,
            'response': response,
            'metadata': {
                'total_time': total_time,
                'retrieval_time': retrieval_time,
                'generation_time': generation_time,
                'num_retrieved_docs': len(context) if context else 0,
                'used_rag': self.config.use_rag,
                **gen_metadata
            }
        }
        
        # Add retrieved context if requested
        if context and self.config.verbose:
            result['retrieved_context'] = [
                {
                    'instruction': doc.instruction[:200],
                    'response': doc.response[:200],
                    'score': doc.score
                }
                for doc in context
            ]
        
        return result
    
    def batch_query(
        self,
        queries: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries
        
        Args:
            queries: List of queries
            **kwargs: Additional parameters
            
        Returns:
            List of results
        """
        results = []
        for query in queries:
            result = self.query(query, **kwargs)
            results.append(result)
        return results
    
    def evaluate_response(
        self,
        query: str,
        response: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate generated response
        (Placeholder for evaluation metrics)
        
        Args:
            query: Original query
            response: Generated response
            ground_truth: Expected response (if available)
            
        Returns:
            Evaluation metrics
        """
        metrics = {
            'response_length': len(response.split()),
            'has_answer': len(response) > 0
        }
        
        # TODO: Add more sophisticated evaluation metrics
        # - BLEU, ROUGE scores if ground_truth available
        # - Semantic similarity
        # - Factual accuracy check
        
        return metrics
    
    def update_model(self, adapter_path: str):
        """
        Update the generator model with new adapter
        
        Args:
            adapter_path: Path to new adapter (SFT/DPO checkpoint)
        """
        logger.info(f"Updating model with adapter: {adapter_path}")
        
        # Reinitialize generator with new adapter
        self.generator = QwenGenerator(
            model_name=self.config.model_name,
            adapter_path=adapter_path,
            load_in_4bit=self.config.load_in_4bit,
            max_context_length=self.config.max_context_length,
            max_new_tokens=self.config.max_new_tokens
        )
        
        self.config.adapter_path = adapter_path
        logger.info("Model updated successfully")
    
    def toggle_rag(self, use_rag: bool):
        """
        Toggle RAG on/off
        
        Args:
            use_rag: Whether to use RAG
        """
        self.config.use_rag = use_rag
        logger.info(f"RAG {'enabled' if use_rag else 'disabled'}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        stats = {
            'config': {
                'model': self.config.model_name,
                'adapter': self.config.adapter_path,
                'use_rag': self.config.use_rag,
                'retrieval_k': self.config.retrieval_top_k
            }
        }
        
        if self.retriever:
            stats['retriever'] = self.retriever.get_statistics()
        
        return stats


def create_rag_pipeline(
    model_type: str = "base",
    collection_name: str = "customer_support_rag",
    **kwargs
) -> RAGPipeline:
    """
    Factory function to create RAG pipeline
    
    Args:
        model_type: Type of model ("base", "sft", "dpo")
        collection_name: Name of vector database collection
        **kwargs: Additional configuration
        
    Returns:
        Configured RAG pipeline
    """
    # Model paths based on type
    model_configs = {
        "base": {
            "model_name": "Qwen/Qwen3-0.6B",
            "adapter_path": None
        },
        "sft": {
            "model_name": "Qwen/Qwen3-0.6B",
            "adapter_path": "experiments/latest/sft/final_model"  # Update with actual path
        },
        "dpo": {
            "model_name": "Qwen/Qwen3-0.6B",
            "adapter_path": "experiments/latest/dpo/final_model"  # Update with actual path
        }
    }
    
    if model_type not in model_configs:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create config
    config = RAGConfig(
        collection_name=collection_name,
        **model_configs[model_type],
        **kwargs
    )
    
    return RAGPipeline(config)


if __name__ == "__main__":
    # Test RAG pipeline
    logging.basicConfig(level=logging.INFO)
    
    # Create pipeline
    pipeline = create_rag_pipeline(
        model_type="base",
        use_rag=True,
        retrieval_top_k=3,
        load_in_4bit=True
    )
    
    # Test queries
    test_queries = [
        "How do I reset my password?",
        "What is your refund policy?",
        "How can I track my order?"
    ]
    
    for query in test_queries[:1]:  # Test with first query
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        result = pipeline.query(query)
        
        print(f"\nResponse: {result['response']}")
        print(f"\nMetadata:")
        for key, value in result['metadata'].items():
            print(f"  {key}: {value}")
        
        if 'retrieved_context' in result:
            print(f"\nRetrieved Context:")
            for i, doc in enumerate(result['retrieved_context'], 1):
                print(f"  {i}. Score: {doc['score']:.4f}")
                print(f"     Q: {doc['instruction'][:100]}...")
                print(f"     A: {doc['response'][:100]}...")