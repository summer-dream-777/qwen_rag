#!/usr/bin/env python3
"""
Model Evaluation Script
Evaluates Base, SFT, and DPO models with and without RAG
"""

import sys
import logging
import argparse
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag import RAGPipeline, RAGConfig
from src.rag.evaluator import RAGASEvaluator, create_test_dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_all_models(
    test_data_path: str = None,
    max_samples: int = 50,
    output_dir: str = "./evaluation_results"
):
    """
    Evaluate all model configurations
    
    Args:
        test_data_path: Path to test data
        max_samples: Maximum test samples
        output_dir: Output directory for results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create test dataset
    logger.info("Creating test dataset...")
    test_data = create_test_dataset(test_data_path, max_samples)
    logger.info(f"Loaded {len(test_data)} test examples")
    
    # Model configurations to test
    model_configs = {
        "Base_NoRAG": {
            "model_name": "Qwen/Qwen3-0.6B",
            "adapter_path": None,
            "use_rag": False,
            "load_in_4bit": True
        },
        "Base_WithRAG": {
            "model_name": "Qwen/Qwen3-0.6B",
            "adapter_path": None,
            "use_rag": True,
            "load_in_4bit": True,
            "retrieval_top_k": 3
        }
    }
    
    # Check for SFT model
    sft_path = Path("experiments/20250821_064206/sft_debug/final_model")
    if sft_path.exists():
        model_configs["SFT_NoRAG"] = {
            "model_name": "Qwen/Qwen3-0.6B",
            "adapter_path": str(sft_path),
            "use_rag": False,
            "load_in_4bit": True
        }
        model_configs["SFT_WithRAG"] = {
            "model_name": "Qwen/Qwen3-0.6B",
            "adapter_path": str(sft_path),
            "use_rag": True,
            "load_in_4bit": True,
            "retrieval_top_k": 3
        }
        logger.info(f"Found SFT model at: {sft_path}")
    
    # Check for DPO model (if exists)
    # dpo_path = Path("experiments/latest/dpo/final_model")
    # if dpo_path.exists():
    #     model_configs["DPO_WithRAG"] = {...}
    
    all_results = {}
    comparison_data = []
    
    # Evaluate each configuration
    for config_name, config in model_configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {config_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Create pipeline
            pipeline_config = RAGConfig(
                **config,
                max_new_tokens=256,
                temperature=0.7,
                verbose=False
            )
            pipeline = RAGPipeline(pipeline_config)
            
            # Create evaluator
            evaluator = RAGASEvaluator(pipeline)
            
            # Evaluate
            results = evaluator.evaluate_dataset(
                test_data[:10],  # Use subset for testing
                save_results=True,
                output_dir=str(output_path / config_name)
            )
            
            all_results[config_name] = results
            
            # Extract key metrics for comparison
            avg_metrics = results['average_metrics']
            comparison_data.append({
                'Model': config_name,
                'RAG Score': avg_metrics.get('avg_rag_score', 0),
                'Quality Score': avg_metrics.get('avg_quality_score', 0),
                'Context Relevancy': avg_metrics.get('avg_context_relevancy', 0),
                'Answer Relevancy': avg_metrics.get('avg_answer_relevancy', 0),
                'Faithfulness': avg_metrics.get('avg_faithfulness', 0),
                'Latency (s)': avg_metrics.get('avg_latency', 0)
            })
            
            logger.info(f"Completed evaluation for {config_name}")
            logger.info(f"RAG Score: {avg_metrics.get('avg_rag_score', 0):.4f}")
            logger.info(f"Quality Score: {avg_metrics.get('avg_quality_score', 0):.4f}")
            
        except Exception as e:
            logger.error(f"Error evaluating {config_name}: {e}")
            continue
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as CSV
    csv_path = output_path / f"model_comparison_{timestamp}.csv"
    comparison_df.to_csv(csv_path, index=False)
    
    # Save detailed results as JSON
    json_path = output_path / f"all_results_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Print comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    # Calculate improvements
    if "Base_WithRAG" in comparison_data[0]['Model'] and len(comparison_data) > 1:
        print("\n" + "="*80)
        print("IMPROVEMENTS FROM BASELINE")
        print("="*80)
        
        baseline_rag = next((d for d in comparison_data if d['Model'] == 'Base_WithRAG'), None)
        if baseline_rag:
            for data in comparison_data:
                if data['Model'] != 'Base_WithRAG':
                    print(f"\n{data['Model']} vs Base_WithRAG:")
                    for metric in ['RAG Score', 'Quality Score', 'Answer Relevancy']:
                        if metric in data and metric in baseline_rag:
                            improvement = ((data[metric] - baseline_rag[metric]) / baseline_rag[metric]) * 100
                            print(f"  {metric}: {improvement:+.2f}%")
    
    logger.info(f"\nAll results saved to: {output_path}")
    
    return comparison_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG models")
    parser.add_argument(
        "--test_data",
        type=str,
        default="./data/test_dataset.json",
        help="Path to test data"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50,
        help="Maximum test samples"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_all_models(
        test_data_path=args.test_data,
        max_samples=args.max_samples,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()