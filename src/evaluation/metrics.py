"""
Evaluation metrics for customer support models
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
import evaluate
from sentence_transformers import SentenceTransformer
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    automatic_metrics: Dict[str, float]
    task_specific_metrics: Dict[str, float]
    sample_outputs: List[Dict[str, str]]
    aggregate_scores: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return {
            "automatic_metrics": self.automatic_metrics,
            "task_specific_metrics": self.task_specific_metrics,
            "aggregate_scores": self.aggregate_scores,
            "sample_outputs": self.sample_outputs[:10]  # Limit samples
        }
    
    def save(self, path: str):
        """Save results to JSON"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class CustomerSupportEvaluator:
    """Comprehensive evaluator for customer support models"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Load evaluation metrics
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")
        
        # Load sentence transformer for semantic similarity
        self.sentence_model = SentenceTransformer(model_name)
        
        # Keywords for customer support quality
        self.helpfulness_keywords = [
            "help", "assist", "support", "solution", "resolve", "answer",
            "guide", "explain", "clarify", "understand"
        ]
        
        self.professionalism_keywords = [
            "thank", "appreciate", "sorry", "apologize", "please",
            "certainly", "happy to", "glad to"
        ]
        
    def compute_automatic_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Compute automatic evaluation metrics"""
        metrics = {}
        
        # BLEU Score
        try:
            bleu_results = self.bleu.compute(
                predictions=predictions,
                references=[[ref] for ref in references]
            )
            metrics['bleu'] = bleu_results['bleu']
        except Exception as e:
            logger.error(f"Error computing BLEU: {e}")
            metrics['bleu'] = 0.0
            
        # ROUGE Scores
        try:
            rouge_results = self.rouge.compute(
                predictions=predictions,
                references=references
            )
            metrics['rouge1'] = rouge_results['rouge1']
            metrics['rouge2'] = rouge_results['rouge2']
            metrics['rougeL'] = rouge_results['rougeL']
        except Exception as e:
            logger.error(f"Error computing ROUGE: {e}")
            
        # BERTScore
        try:
            bertscore_results = self.bertscore.compute(
                predictions=predictions,
                references=references,
                lang="en",
                model_type="microsoft/deberta-xlarge-mnli"
            )
            metrics['bertscore_precision'] = np.mean(bertscore_results['precision'])
            metrics['bertscore_recall'] = np.mean(bertscore_results['recall'])
            metrics['bertscore_f1'] = np.mean(bertscore_results['f1'])
        except Exception as e:
            logger.error(f"Error computing BERTScore: {e}")
            
        return metrics
    
    def compute_semantic_similarity(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """Compute semantic similarity using sentence embeddings"""
        # Encode sentences
        pred_embeddings = self.sentence_model.encode(predictions)
        ref_embeddings = self.sentence_model.encode(references)
        
        # Compute cosine similarity
        similarities = []
        for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
            similarity = np.dot(pred_emb, ref_emb) / (
                np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb)
            )
            similarities.append(similarity)
            
        return np.mean(similarities)
    
    def compute_helpfulness_score(self, texts: List[str]) -> float:
        """Compute helpfulness score based on keyword presence"""
        scores = []
        
        for text in texts:
            text_lower = text.lower()
            keyword_count = sum(
                keyword in text_lower 
                for keyword in self.helpfulness_keywords
            )
            # Normalize by text length
            score = keyword_count / (len(text.split()) + 1)
            scores.append(score)
            
        return np.mean(scores)
    
    def compute_professionalism_score(self, texts: List[str]) -> float:
        """Compute professionalism score"""
        scores = []
        
        for text in texts:
            text_lower = text.lower()
            keyword_count = sum(
                keyword in text_lower 
                for keyword in self.professionalism_keywords
            )
            score = keyword_count / (len(text.split()) + 1)
            scores.append(score)
            
        return np.mean(scores)
    
    def compute_response_length_ratio(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """Compute ratio of response lengths"""
        ratios = []
        
        for pred, ref in zip(predictions, references):
            pred_len = len(pred.split())
            ref_len = len(ref.split())
            
            if ref_len > 0:
                ratio = min(pred_len / ref_len, ref_len / pred_len)
            else:
                ratio = 0.0
                
            ratios.append(ratio)
            
        return np.mean(ratios)
    
    def evaluate_safety(self, texts: List[str]) -> Dict[str, float]:
        """Basic safety evaluation"""
        unsafe_keywords = [
            "kill", "die", "suicide", "murder", "violence",
            "hate", "racist", "sexist"
        ]
        
        safe_count = 0
        flagged_indices = []
        
        for idx, text in enumerate(texts):
            text_lower = text.lower()
            is_safe = not any(keyword in text_lower for keyword in unsafe_keywords)
            
            if is_safe:
                safe_count += 1
            else:
                flagged_indices.append(idx)
                
        return {
            "safety_score": safe_count / len(texts),
            "flagged_count": len(flagged_indices),
            "flagged_indices": flagged_indices[:10]  # Limit for logging
        }
    
    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        prompts: Optional[List[str]] = None
    ) -> EvaluationResult:
        """Run comprehensive evaluation"""
        
        # Ensure equal lengths
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]
        
        # Compute automatic metrics
        logger.info("Computing automatic metrics...")
        automatic_metrics = self.compute_automatic_metrics(predictions, references)
        
        # Compute task-specific metrics
        logger.info("Computing task-specific metrics...")
        task_specific_metrics = {
            "semantic_similarity": self.compute_semantic_similarity(predictions, references),
            "helpfulness_score": self.compute_helpfulness_score(predictions),
            "professionalism_score": self.compute_professionalism_score(predictions),
            "response_length_ratio": self.compute_response_length_ratio(predictions, references),
        }
        
        # Safety evaluation
        safety_results = self.evaluate_safety(predictions)
        task_specific_metrics.update(safety_results)
        
        # Aggregate scores
        aggregate_scores = {
            "overall_quality": np.mean([
                automatic_metrics.get('bertscore_f1', 0),
                task_specific_metrics['semantic_similarity'],
                task_specific_metrics['helpfulness_score'],
                task_specific_metrics['professionalism_score']
            ]),
            "relevance_score": np.mean([
                automatic_metrics.get('bertscore_f1', 0),
                task_specific_metrics['semantic_similarity']
            ]),
            "style_score": np.mean([
                task_specific_metrics['helpfulness_score'],
                task_specific_metrics['professionalism_score']
            ])
        }
        
        # Sample outputs for inspection
        sample_outputs = []
        num_samples = min(20, len(predictions))
        indices = np.random.choice(len(predictions), num_samples, replace=False)
        
        for idx in indices:
            sample = {
                "index": int(idx),
                "prompt": prompts[idx] if prompts else "N/A",
                "prediction": predictions[idx],
                "reference": references[idx]
            }
            sample_outputs.append(sample)
            
        return EvaluationResult(
            automatic_metrics=automatic_metrics,
            task_specific_metrics=task_specific_metrics,
            sample_outputs=sample_outputs,
            aggregate_scores=aggregate_scores
        )


class ModelComparator:
    """Compare multiple models"""
    
    def __init__(self, evaluator: CustomerSupportEvaluator):
        self.evaluator = evaluator
        
    def compare_models(
        self,
        model_outputs: Dict[str, List[str]],
        references: List[str],
        prompts: Optional[List[str]] = None
    ) -> Dict[str, EvaluationResult]:
        """Compare multiple models"""
        results = {}
        
        for model_name, predictions in model_outputs.items():
            logger.info(f"Evaluating {model_name}...")
            results[model_name] = self.evaluator.evaluate(
                predictions, references, prompts
            )
            
        return results
    
    def generate_comparison_report(
        self,
        comparison_results: Dict[str, EvaluationResult]
    ) -> Dict:
        """Generate comparison report"""
        report = {
            "model_scores": {},
            "best_models": {},
            "detailed_comparison": {}
        }
        
        # Collect all metrics
        all_metrics = set()
        for result in comparison_results.values():
            all_metrics.update(result.automatic_metrics.keys())
            all_metrics.update(result.task_specific_metrics.keys())
            all_metrics.update(result.aggregate_scores.keys())
            
        # Compare each metric
        for metric in all_metrics:
            metric_scores = {}
            
            for model_name, result in comparison_results.items():
                score = None
                if metric in result.automatic_metrics:
                    score = result.automatic_metrics[metric]
                elif metric in result.task_specific_metrics:
                    score = result.task_specific_metrics[metric]
                elif metric in result.aggregate_scores:
                    score = result.aggregate_scores[metric]
                    
                if score is not None:
                    metric_scores[model_name] = score
                    
            if metric_scores:
                report["detailed_comparison"][metric] = metric_scores
                best_model = max(metric_scores, key=metric_scores.get)
                report["best_models"][metric] = best_model
                
        # Overall scores
        for model_name, result in comparison_results.items():
            report["model_scores"][model_name] = result.aggregate_scores
            
        return report