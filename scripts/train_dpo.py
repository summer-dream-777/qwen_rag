#!/usr/bin/env python3
"""
DPO Training Script for Preference Learning
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dpo_dataset import create_dpo_datasets
from src.training.dpo_trainer import DPOTrainer
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train DPO model for preference learning")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dpo_config.yaml",
        help="Path to DPO training config file"
    )
    parser.add_argument(
        "--sft_checkpoint",
        type=str,
        required=True,
        help="Path to SFT checkpoint directory"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="SAGI-1/ultrafeedback_binarized_dpo",
        help="Name of preference dataset"
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train_prefs",
        help="Training split name"
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default=None,
        help="Evaluation split name (if None, will use 10% of train)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for debugging)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for datasets"
    )
    
    args = parser.parse_args()
    
    # Validate SFT checkpoint
    sft_checkpoint = Path(args.sft_checkpoint)
    if not sft_checkpoint.exists():
        raise ValueError(f"SFT checkpoint not found: {sft_checkpoint}")
    
    # Initialize trainer
    logger.info("Initializing DPO trainer...")
    trainer = DPOTrainer(args.config, str(sft_checkpoint))
    
    # Load models and tokenizer
    logger.info("Loading models and tokenizer...")
    trainer.load_models_and_tokenizer()
    
    # Create datasets
    logger.info(f"Loading preference dataset: {args.dataset_name}")
    train_dataset, eval_dataset = create_dpo_datasets(
        tokenizer=trainer.tokenizer,
        dataset_name=args.dataset_name,
        train_split=args.train_split,
        eval_split=args.eval_split,
        max_length=trainer.config['data']['max_length'],
        max_prompt_length=trainer.config['data']['max_prompt_length'],
        cache_dir=args.cache_dir
    )
    
    # Limit samples if specified (for debugging)
    if args.max_samples:
        logger.info(f"Limiting to {args.max_samples} samples for debugging")
        train_dataset.dataset = train_dataset.dataset.select(range(min(args.max_samples, len(train_dataset))))
        if eval_dataset:
            eval_size = min(int(args.max_samples * 0.1), len(eval_dataset))
            eval_dataset.dataset = eval_dataset.dataset.select(range(eval_size))
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # Start training
    logger.info("Starting DPO training...")
    trainer.train(train_dataset, eval_dataset)
    
    logger.info("DPO training completed!")
    logger.info(f"Results saved to: {trainer.output_dir}")
    
    # Test generation with the final model
    logger.info("Testing generation with DPO model...")
    test_prompts = [
        "What's the best way to handle an angry customer?",
        "How should I respond to a refund request?",
        "Can you help me track my order?"
    ]
    
    trainer.model.eval()
    with torch.no_grad():
        for prompt in test_prompts:
            formatted_prompt = f"### User:\n{prompt}\n\n### Assistant:\n"
            inputs = trainer.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(trainer.device)
            
            outputs = trainer.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=trainer.tokenizer.pad_token_id,
                eos_token_id=trainer.tokenizer.eos_token_id
            )
            
            response = trainer.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"\nPrompt: {prompt}")
            logger.info(f"Response: {response[len(formatted_prompt):]}\n")


if __name__ == "__main__":
    main()