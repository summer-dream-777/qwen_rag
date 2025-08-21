#!/usr/bin/env python3
"""
Prepare DPO (Direct Preference Optimization) Dataset
Creates preference pairs from customer support data
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import random
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_preference_pairs(
    data: List[Dict],
    num_pairs: int = 1000
) -> List[Dict]:
    """
    Create preference pairs for DPO training
    
    Args:
        data: Original training data
        num_pairs: Number of preference pairs to create
        
    Returns:
        List of preference pairs
    """
    preference_pairs = []
    
    # Group data by similar questions (simplified approach)
    question_groups = {}
    for item in data:
        # Use first 50 chars of instruction as key
        key = item['instruction'][:50] if len(item['instruction']) > 50 else item['instruction']
        if key not in question_groups:
            question_groups[key] = []
        question_groups[key].append(item)
    
    # Create pairs from groups
    for key, items in tqdm(question_groups.items(), desc="Creating pairs"):
        if len(items) >= 2:
            # Sort by response length/quality (heuristic)
            items_sorted = sorted(items, key=lambda x: len(x['response']), reverse=True)
            
            # Create preference pair
            chosen = items_sorted[0]  # Longer/better response
            rejected = items_sorted[-1]  # Shorter/worse response
            
            # Only create pair if there's meaningful difference
            if len(chosen['response']) > len(rejected['response']) * 1.2:
                pair = {
                    'prompt': chosen['instruction'],
                    'chosen': chosen['response'],
                    'rejected': rejected['response'],
                    'source': 'customer_support'
                }
                preference_pairs.append(pair)
        
        if len(preference_pairs) >= num_pairs:
            break
    
    # If not enough pairs, create synthetic ones
    if len(preference_pairs) < num_pairs:
        logger.info(f"Creating synthetic pairs to reach {num_pairs} total")
        
        for _ in range(num_pairs - len(preference_pairs)):
            # Random sample
            item = random.choice(data)
            
            # Create worse version (truncated, generic)
            rejected_response = create_worse_response(item['response'])
            
            pair = {
                'prompt': item['instruction'],
                'chosen': item['response'],
                'rejected': rejected_response,
                'source': 'synthetic'
            }
            preference_pairs.append(pair)
    
    return preference_pairs[:num_pairs]


def create_worse_response(good_response: str) -> str:
    """
    Create a worse version of a response for training
    
    Args:
        good_response: Good response to degrade
        
    Returns:
        Worse response
    """
    worse_templates = [
        "I don't have enough information to help with that.",
        "Please contact our support team for assistance.",
        "I'm not sure about that.",
        "Sorry, I cannot help with this request.",
        "This is beyond my capabilities."
    ]
    
    # Strategy 1: Truncate response
    if len(good_response) > 100 and random.random() > 0.5:
        return good_response[:50] + "..."
    
    # Strategy 2: Generic response
    elif random.random() > 0.3:
        return random.choice(worse_templates)
    
    # Strategy 3: Remove details
    else:
        words = good_response.split()
        if len(words) > 10:
            return " ".join(words[:5]) + " Please refer to our documentation."
        else:
            return "I understand your concern. " + random.choice(worse_templates)


def prepare_dpo_dataset(
    input_path: str,
    output_dir: str,
    num_pairs: int = 2000,
    test_split: float = 0.1,
    max_samples: int = None
):
    """
    Prepare DPO dataset from customer support data
    
    Args:
        input_path: Path to input data
        output_dir: Output directory
        num_pairs: Number of preference pairs
        test_split: Test split ratio
        max_samples: Maximum samples to load
    """
    logger.info(f"Loading data from {input_path}")
    
    # Load data
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.strip():
                data.append(json.loads(line))
                if max_samples and i >= max_samples:
                    break
    
    logger.info(f"Loaded {len(data)} samples")
    
    # Create preference pairs
    logger.info(f"Creating {num_pairs} preference pairs...")
    preference_pairs = create_preference_pairs(data, num_pairs)
    
    # Split into train/test
    random.shuffle(preference_pairs)
    split_idx = int(len(preference_pairs) * (1 - test_split))
    train_pairs = preference_pairs[:split_idx]
    test_pairs = preference_pairs[split_idx:]
    
    logger.info(f"Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save datasets
    train_path = output_path / "dpo_train.json"
    test_path = output_path / "dpo_test.json"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for pair in train_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    with open(test_path, 'w', encoding='utf-8') as f:
        for pair in test_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    # Save statistics
    stats = {
        'total_pairs': len(preference_pairs),
        'train_pairs': len(train_pairs),
        'test_pairs': len(test_pairs),
        'avg_chosen_length': np.mean([len(p['chosen'].split()) for p in preference_pairs]),
        'avg_rejected_length': np.mean([len(p['rejected'].split()) for p in preference_pairs]),
        'synthetic_pairs': len([p for p in preference_pairs if p['source'] == 'synthetic'])
    }
    
    stats_path = output_path / "dpo_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"DPO dataset saved to {output_path}")
    logger.info(f"Statistics: {stats}")
    
    # Show sample pairs
    logger.info("\nSample preference pairs:")
    for i, pair in enumerate(train_pairs[:3]):
        print(f"\n--- Pair {i+1} ---")
        print(f"Prompt: {pair['prompt'][:100]}...")
        print(f"Chosen: {pair['chosen'][:150]}...")
        print(f"Rejected: {pair['rejected'][:150]}...")
    
    return train_path, test_path


def main():
    parser = argparse.ArgumentParser(description="Prepare DPO dataset")
    parser.add_argument(
        "--input_path",
        type=str,
        default="./data/unified_customer_support.json",
        help="Input data path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/dpo_dataset",
        help="Output directory"
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=2000,
        help="Number of preference pairs"
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="Test split ratio"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to load"
    )
    
    args = parser.parse_args()
    
    # Import numpy here to avoid issues
    global np
    import numpy as np
    
    prepare_dpo_dataset(
        input_path=args.input_path,
        output_dir=args.output_dir,
        num_pairs=args.num_pairs,
        test_split=args.test_split,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()