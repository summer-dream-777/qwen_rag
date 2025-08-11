#!/usr/bin/env python3
"""
SFT Training Script for Customer Support Model
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.sft_dataset import create_sft_datasets, SFTDataConfig
from src.training.sft_trainer import SFTTrainer
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_data(data_path: str, test_size: float = 0.1, random_state: int = 42):
    """Prepare train/eval split from unified dataset"""
    # Load data
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    elif data_path.endswith('.json'):
        # Try different JSON reading methods
        try:
            # First try reading as JSON lines (JSONL)
            data = pd.read_json(data_path, lines=True)
        except:
            try:
                # Try reading as regular JSON
                data = pd.read_json(data_path)
            except:
                # Finally, try loading with datasets library
                from datasets import load_dataset
                dataset = load_dataset('json', data_files=data_path, split='train')
                data = pd.DataFrame(dataset)
    else:
        # Assume it's a huggingface dataset saved locally
        from datasets import load_from_disk
        dataset = load_from_disk(data_path)
        data = pd.DataFrame(dataset)
    
    logger.info(f"Loaded {len(data)} examples from {data_path}")
    
    # Split data
    train_data, eval_data = train_test_split(
        data, 
        test_size=test_size, 
        random_state=random_state,
        stratify=data['source'] if 'source' in data.columns else None
    )
    
    logger.info(f"Train set: {len(train_data)} examples")
    logger.info(f"Eval set: {len(eval_data)} examples")
    
    return train_data, eval_data


def main():
    parser = argparse.ArgumentParser(description="Train SFT model for customer support")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sft_config.yaml",
        help="Path to training config file"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to unified dataset"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Test set size for train/eval split"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for debugging)"
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful customer support assistant. Provide clear, concise, and helpful responses.",
        help="System prompt for training"
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    logger.info("Initializing SFT trainer...")
    trainer = SFTTrainer(args.config)
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    trainer.load_model_and_tokenizer()
    
    # Prepare data
    logger.info("Preparing data...")
    train_data, eval_data = prepare_data(args.data_path, args.test_size)
    
    # Limit samples if specified
    if args.max_samples:
        train_data = train_data.head(args.max_samples)
        eval_data = eval_data.head(int(args.max_samples * args.test_size))
        logger.info(f"Limited to {len(train_data)} train and {len(eval_data)} eval samples")
    
    # Create datasets
    data_config = SFTDataConfig(
        max_length=trainer.config['data']['max_length'],
        system_prompt=args.system_prompt
    )
    
    train_dataset, eval_dataset = create_sft_datasets(
        train_data=train_data,
        eval_data=eval_data,
        tokenizer=trainer.tokenizer,
        config=data_config
    )
    
    # Start training
    logger.info("Starting training...")
    train_results, eval_results = trainer.train(train_dataset, eval_dataset)
    
    logger.info("Training completed!")
    logger.info(f"Final eval loss: {eval_results.get('eval_loss', 'N/A')}")
    logger.info(f"Results saved to: {trainer.output_dir}")
    
    # Generate sample outputs
    logger.info("Generating sample outputs...")
    sample_prompts = [
        "### User:\nMy order hasn't arrived yet. It's been 10 days.\n\n### Assistant:\n",
        "### User:\nHow do I reset my password?\n\n### Assistant:\n",
        "### User:\nI want to cancel my subscription.\n\n### Assistant:\n"
    ]
    
    sample_outputs = trainer.generate_samples(sample_prompts)
    
    # Save samples
    with open(trainer.output_dir / "sample_outputs.txt", "w") as f:
        for prompt, output in zip(sample_prompts, sample_outputs):
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Output: {output}\n")
            f.write("-" * 80 + "\n")
    
    logger.info("Sample outputs saved!")


if __name__ == "__main__":
    main()