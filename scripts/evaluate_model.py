#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import json
from typing import List, Dict
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.metrics import CustomerSupportEvaluator, ModelComparator
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelInference:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model from {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        batch_size: int = 8
    ) -> List[str]:
        responses = []
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating responses"):
            batch_prompts = prompts[i:i + batch_size]
            
            formatted_prompts = [
                f"### User:\n{prompt}\n\n### Assistant:\n" 
                for prompt in batch_prompts
            ]
            
            inputs = self.tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            for j, output in enumerate(outputs):
                full_response = self.tokenizer.decode(output, skip_special_tokens=True)
                response = full_response[len(formatted_prompts[j]):]
                responses.append(response)
                
        return responses


def load_test_data(data_source: str, max_samples: int = None) -> Dict[str, List[str]]:
    if data_source.endswith('.json'):
        with open(data_source, 'r') as f:
            data = json.load(f)
    elif data_source.endswith('.csv'):
        df = pd.read_csv(data_source)
        data = df.to_dict('list')
    else:
        dataset = load_dataset(data_source, split='test')
        data = {col: dataset[col] for col in dataset.column_names}
    
    prompts = data.get('instruction', data.get('prompt', data.get('question', [])))
    references = data.get('response', data.get('answer', data.get('completion', [])))
    
    if not prompts or not references:
        raise ValueError("Could not find prompt/response columns in the data")
    
    if max_samples:
        prompts = prompts[:max_samples]
        references = references[:max_samples]
    
    logger.info(f"Loaded {len(prompts)} test examples")
    
    return {
        'prompts': prompts,
        'references': references
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate customer support models")
    parser.add_argument(
        "--model_paths",
        type=str,
        nargs='+',
        required=True,
        help="Paths to model checkpoints to evaluate"
    )
    parser.add_argument(
        "--model_names",
        type=str,
        nargs='+',
        help="Names for the models (defaults to path basenames)"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test data (JSON, CSV, or HuggingFace dataset)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple models"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading test data...")
    test_data = load_test_data(args.test_data, args.max_samples)
    prompts = test_data['prompts']
    references = test_data['references']
    
    logger.info("Initializing evaluator...")
    evaluator = CustomerSupportEvaluator()
    
    all_results = {}
    model_outputs = {}
    
    model_names = args.model_names or [Path(p).name for p in args.model_paths]
    
    for model_path, model_name in zip(args.model_paths, model_names):
        logger.info(f"\nEvaluating model: {model_name}")
        
        model_inference = ModelInference(model_path)
        
        logger.info("Generating responses...")
        predictions = model_inference.generate(
            prompts,
            batch_size=args.batch_size
        )
        
        predictions_file = output_dir / f"{model_name}_predictions.json"
        with open(predictions_file, 'w') as f:
            json.dump({
                'prompts': prompts,
                'predictions': predictions,
                'references': references
            }, f, indent=2)
        logger.info(f"Predictions saved to {predictions_file}")
        
        logger.info("Running evaluation...")
        result = evaluator.evaluate(predictions, references, prompts)
        all_results[model_name] = result
        model_outputs[model_name] = predictions
        
        result_file = output_dir / f"{model_name}_results.json"
        result.save(str(result_file))
        logger.info(f"Results saved to {result_file}")
        
        logger.info(f"\n{model_name} Results Summary:")
        logger.info(f"  Automatic Metrics:")
        for metric, value in result.automatic_metrics.items():
            logger.info(f"    {metric}: {value:.4f}")
        logger.info(f"  Task-Specific Metrics:")
        for metric, value in result.task_specific_metrics.items():
            if isinstance(value, float):
                logger.info(f"    {metric}: {value:.4f}")
        logger.info(f"  Aggregate Scores:")
        for metric, value in result.aggregate_scores.items():
            logger.info(f"    {metric}: {value:.4f}")
    
    if args.compare and len(model_paths) > 1:
        logger.info("\nComparing models...")
        comparator = ModelComparator(evaluator)
        
        comparison_report = comparator.generate_comparison_report(all_results)
        
        comparison_file = output_dir / "model_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_report, f, indent=2)
        logger.info(f"Comparison report saved to {comparison_file}")
        
        logger.info("\nModel Comparison Summary:")
        logger.info("\nBest model for each metric:")
        for metric, model in comparison_report['best_models'].items():
            logger.info(f"  {metric}: {model}")
        
        logger.info("\nOverall scores:")
        for model, scores in comparison_report['model_scores'].items():
            logger.info(f"\n  {model}:")
            for metric, value in scores.items():
                logger.info(f"    {metric}: {value:.4f}")
    
    logger.info("\nEvaluation completed!")


if __name__ == "__main__":
    main()