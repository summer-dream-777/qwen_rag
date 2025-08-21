"""
RAGAS-based Evaluation Framework for RAG System
Implements comprehensive evaluation metrics for RAG pipelines
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm
import pandas as pd
import json
from pathlib import Path
import time

# Embedding imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import Dataset

# Local imports
from ..rag_pipeline import RAGPipeline, RAGConfig

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    # RAG-specific metrics
    context_relevancy: float = 0.0  # How relevant is the retrieved context
    answer_relevancy: float = 0.0   # How relevant is the answer to the question
    faithfulness: float = 0.0        # Is the answer faithful to the context
    context_recall: float = 0.0      # Does context contain all required info
    context_precision: float = 0.0   # Precision of retrieved context
    
    # Response quality metrics
    answer_similarity: float = 0.0    # Similarity to ground truth
    answer_correctness: float = 0.0   # Correctness of the answer
    
    # Efficiency metrics
    latency: float = 0.0             # Response time
    tokens_generated: int = 0         # Number of tokens generated
    
    # Aggregate scores
    rag_score: float = 0.0           # Overall RAG performance
    quality_score: float = 0.0       # Overall quality score
    
    def calculate_aggregate_scores(self):
        """Calculate aggregate scores"""
        # RAG score (average of RAG-specific metrics)
        rag_metrics = [
            self.context_relevancy,
            self.answer_relevancy,
            self.faithfulness,
            self.context_recall,
            self.context_precision
        ]
        self.rag_score = np.mean([m for m in rag_metrics if m > 0])
        
        # Quality score (average of response quality metrics)
        quality_metrics = [
            self.answer_similarity,
            self.answer_correctness
        ]
        self.quality_score = np.mean([m for m in quality_metrics if m > 0])
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'context_relevancy': self.context_relevancy,
            'answer_relevancy': self.answer_relevancy,
            'faithfulness': self.faithfulness,
            'context_recall': self.context_recall,
            'context_precision': self.context_precision,
            'answer_similarity': self.answer_similarity,
            'answer_correctness': self.answer_correctness,
            'latency': self.latency,
            'tokens_generated': self.tokens_generated,
            'rag_score': self.rag_score,
            'quality_score': self.quality_score
        }


class RAGASEvaluator:
    """RAGAS-based evaluator for RAG systems"""
    
    def __init__(
        self,
        pipeline: RAGPipeline,
        use_openai: bool = False,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "BAAI/bge-m3"
    ):
        """
        Initialize RAGAS evaluator
        
        Args:
            pipeline: RAG pipeline to evaluate
            use_openai: Whether to use OpenAI for evaluation
            openai_api_key: OpenAI API key
            embedding_model: Embedding model for similarity
        """
        self.pipeline = pipeline
        self.use_openai = use_openai
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize LLM for evaluation (if using OpenAI)
        self.eval_llm = None
        if use_openai and openai_api_key:
            self.eval_llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                openai_api_key=openai_api_key,
                temperature=0
            )
        
        logger.info("RAGAS Evaluator initialized")
    
    def evaluate_single(
        self,
        question: str,
        ground_truth: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, EvaluationMetrics]:
        """
        Evaluate a single query
        
        Args:
            question: Query to evaluate
            ground_truth: Expected answer (optional)
            
        Returns:
            Tuple of (generated_answer, metrics)
        """
        # Generate response
        start_time = time.time()
        result = self.pipeline.query(question, **kwargs)
        latency = time.time() - start_time
        
        generated_answer = result['response']
        retrieved_contexts = []
        
        if 'retrieved_context' in result:
            retrieved_contexts = [
                f"Q: {doc['instruction']}\nA: {doc['response']}"
                for doc in result['retrieved_context']
            ]
        
        # Initialize metrics
        metrics = EvaluationMetrics(latency=latency)
        
        # Calculate basic metrics
        metrics.tokens_generated = len(generated_answer.split())
        
        # Calculate embedding-based similarity if ground truth available
        if ground_truth:
            metrics.answer_similarity = self._calculate_similarity(
                generated_answer, ground_truth
            )
        
        # Calculate RAGAS metrics if contexts available
        if retrieved_contexts:
            metrics.context_relevancy = self._calculate_context_relevancy(
                question, retrieved_contexts
            )
            metrics.answer_relevancy = self._calculate_answer_relevancy(
                question, generated_answer
            )
            
            if ground_truth:
                metrics.faithfulness = self._calculate_faithfulness(
                    generated_answer, retrieved_contexts
                )
                metrics.context_recall = self._calculate_context_recall(
                    ground_truth, retrieved_contexts
                )
        
        # Calculate aggregate scores
        metrics.calculate_aggregate_scores()
        
        return generated_answer, metrics
    
    def evaluate_dataset(
        self,
        test_data: List[Dict[str, str]],
        save_results: bool = True,
        output_dir: str = "./evaluation_results"
    ) -> Dict[str, Any]:
        """
        Evaluate on a dataset
        
        Args:
            test_data: List of test examples with 'question' and optionally 'answer'
            save_results: Whether to save results
            output_dir: Directory to save results
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating {len(test_data)} examples...")
        
        all_metrics = []
        all_responses = []
        
        for example in tqdm(test_data, desc="Evaluating"):
            question = example['question']
            ground_truth = example.get('answer', None)
            
            answer, metrics = self.evaluate_single(question, ground_truth)
            
            all_metrics.append(metrics.to_dict())
            all_responses.append({
                'question': question,
                'generated_answer': answer,
                'ground_truth': ground_truth,
                **metrics.to_dict()
            })
        
        # Calculate average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if m[key] > 0]
            if values:
                avg_metrics[f'avg_{key}'] = np.mean(values)
                avg_metrics[f'std_{key}'] = np.std(values)
        
        # Compile results
        results = {
            'total_examples': len(test_data),
            'average_metrics': avg_metrics,
            'detailed_results': all_responses,
            'model_config': {
                'model': self.pipeline.config.model_name,
                'adapter': self.pipeline.config.adapter_path,
                'use_rag': self.pipeline.config.use_rag,
                'retrieval_k': self.pipeline.config.retrieval_top_k
            }
        }
        
        # Save results if requested
        if save_results:
            self._save_results(results, output_dir)
        
        return results
    
    def compare_models(
        self,
        test_data: List[Dict[str, str]],
        model_configs: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Compare multiple model configurations
        
        Args:
            test_data: Test dataset
            model_configs: Dictionary of model configurations
            
        Returns:
            Comparison results as DataFrame
        """
        comparison_results = []
        
        for model_name, config in model_configs.items():
            logger.info(f"Evaluating {model_name}...")
            
            # Update pipeline configuration
            if 'adapter_path' in config:
                self.pipeline.update_model(config['adapter_path'])
            
            if 'use_rag' in config:
                self.pipeline.toggle_rag(config['use_rag'])
            
            # Evaluate
            results = self.evaluate_dataset(test_data, save_results=False)
            
            # Extract key metrics
            avg_metrics = results['average_metrics']
            comparison_results.append({
                'model': model_name,
                'rag_score': avg_metrics.get('avg_rag_score', 0),
                'quality_score': avg_metrics.get('avg_quality_score', 0),
                'latency': avg_metrics.get('avg_latency', 0),
                'context_relevancy': avg_metrics.get('avg_context_relevancy', 0),
                'answer_relevancy': avg_metrics.get('avg_answer_relevancy', 0),
                'faithfulness': avg_metrics.get('avg_faithfulness', 0)
            })
        
        return pd.DataFrame(comparison_results)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        try:
            emb1 = self.embeddings.embed_query(text1)
            emb2 = self.embeddings.embed_query(text2)
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def _calculate_context_relevancy(
        self,
        question: str,
        contexts: List[str]
    ) -> float:
        """Calculate relevancy of retrieved contexts"""
        if not contexts:
            return 0.0
        
        relevancy_scores = []
        for context in contexts:
            score = self._calculate_similarity(question, context)
            relevancy_scores.append(score)
        
        return np.mean(relevancy_scores)
    
    def _calculate_answer_relevancy(
        self,
        question: str,
        answer: str
    ) -> float:
        """Calculate relevancy of answer to question"""
        return self._calculate_similarity(question, answer)
    
    def _calculate_faithfulness(
        self,
        answer: str,
        contexts: List[str]
    ) -> float:
        """Calculate faithfulness of answer to context"""
        if not contexts:
            return 0.0
        
        # Check if answer content is grounded in context
        context_combined = " ".join(contexts)
        return self._calculate_similarity(answer, context_combined)
    
    def _calculate_context_recall(
        self,
        ground_truth: str,
        contexts: List[str]
    ) -> float:
        """Calculate recall of required information in context"""
        if not contexts:
            return 0.0
        
        context_combined = " ".join(contexts)
        return self._calculate_similarity(ground_truth, context_combined)
    
    def _save_results(self, results: Dict[str, Any], output_dir: str):
        """Save evaluation results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_path = output_path / f"evaluation_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save CSV for easy analysis
        df = pd.DataFrame(results['detailed_results'])
        csv_path = output_path / f"evaluation_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        # Save summary
        summary_path = output_path / f"summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write("=== Evaluation Summary ===\n\n")
            f.write(f"Total Examples: {results['total_examples']}\n")
            f.write(f"Model: {results['model_config']['model']}\n")
            f.write(f"Adapter: {results['model_config']['adapter']}\n")
            f.write(f"RAG Enabled: {results['model_config']['use_rag']}\n\n")
            
            f.write("Average Metrics:\n")
            for key, value in results['average_metrics'].items():
                if 'avg_' in key:
                    f.write(f"  {key.replace('avg_', '')}: {value:.4f}\n")
        
        logger.info(f"Results saved to {output_path}")


def create_test_dataset(data_path: str = None, max_samples: int = 100) -> List[Dict[str, str]]:
    """
    Create test dataset for evaluation
    
    Args:
        data_path: Path to data file
        max_samples: Maximum number of samples
        
    Returns:
        List of test examples
    """
    if data_path and Path(data_path).exists():
        # Load from file
        with open(data_path, 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    data.append({
                        'question': item.get('instruction', ''),
                        'answer': item.get('response', '')
                    })
                    if len(data) >= max_samples:
                        break
        return data
    else:
        # Default test questions
        return [
            {
                'question': "How do I reset my password?",
                'answer': "To reset your password, go to the login page and click 'Forgot Password'. Enter your email address and follow the instructions sent to your email."
            },
            {
                'question': "What is your refund policy?",
                'answer': "We offer a 30-day money-back guarantee on all purchases. To request a refund, contact our support team with your order number."
            },
            {
                'question': "How can I track my order?",
                'answer': "You can track your order by logging into your account and visiting the 'My Orders' section. You'll find tracking information for all your recent orders there."
            },
            {
                'question': "What payment methods do you accept?",
                'answer': "We accept all major credit cards (Visa, MasterCard, American Express), PayPal, and bank transfers."
            },
            {
                'question': "How do I contact customer support?",
                'answer': "You can contact our customer support team via email at support@company.com, through our live chat on the website, or by calling 1-800-XXX-XXXX."
            }
        ]


if __name__ == "__main__":
    # Test evaluator
    logging.basicConfig(level=logging.INFO)
    
    # Create pipeline
    from ..rag_pipeline import create_rag_pipeline
    
    pipeline = create_rag_pipeline(
        model_type="base",
        use_rag=True,
        load_in_4bit=True
    )
    
    # Create evaluator
    evaluator = RAGASEvaluator(pipeline)
    
    # Create test dataset
    test_data = create_test_dataset(max_samples=5)
    
    # Evaluate
    results = evaluator.evaluate_dataset(test_data)
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    for key, value in results['average_metrics'].items():
        if 'avg_' in key:
            print(f"{key}: {value:.4f}")