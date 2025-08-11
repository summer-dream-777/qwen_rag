#!/bin/bash
# Quick test script for SFT -> DPO -> Evaluation pipeline

set -e  # Exit on error

echo "========================================="
echo "Quick Pipeline Test with Small Dataset"
echo "========================================="

# Configuration
MAX_SAMPLES=50  # Use only 50 samples for quick testing
TEST_SIZE=0.2

# Step 1: Run SFT training with debug config
echo ""
echo "Step 1: Running SFT training with $MAX_SAMPLES samples..."
echo "-----------------------------------------"
python scripts/train_sft.py \
    --config configs/sft_config_debug.yaml \
    --data_path ./data/unified_customer_support.json \
    --test_size $TEST_SIZE \
    --max_samples $MAX_SAMPLES

# Get the latest SFT checkpoint directory
SFT_DIR=$(ls -td experiments/*/sft_debug 2>/dev/null | head -1)
if [ -z "$SFT_DIR" ]; then
    echo "Error: No SFT checkpoint found!"
    exit 1
fi

echo ""
echo "SFT training completed! Checkpoint saved at: $SFT_DIR"

# Find the best checkpoint (usually the last one)
CHECKPOINT=$(ls -td $SFT_DIR/checkpoint-* 2>/dev/null | head -1)
if [ -z "$CHECKPOINT" ]; then
    echo "Warning: No checkpoint found, using base model directory"
    CHECKPOINT=$SFT_DIR
fi

echo "Using checkpoint: $CHECKPOINT"

# Step 2: Run DPO training (if script exists)
if [ -f "scripts/train_dpo.py" ]; then
    echo ""
    echo "Step 2: Running DPO training..."
    echo "-----------------------------------------"
    
    # Create DPO debug config if not exists
    if [ ! -f "configs/dpo_config_debug.yaml" ]; then
        echo "Creating DPO debug config..."
        python -c "
import yaml
import sys

config = {
    'model': {
        'name_or_path': '$CHECKPOINT',
        'use_lora': True,
        'lora_config': {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            'task_type': 'CAUSAL_LM'
        }
    },
    'data': {
        'max_length': 512,
        'train_batch_size': 2,
        'eval_batch_size': 4
    },
    'training': {
        'num_epochs': 1,
        'learning_rate': 1e-5,
        'beta': 0.1,
        'label_smoothing': 0.0,
        'warmup_ratio': 0.1,
        'gradient_accumulation_steps': 1,
        'fp16': True,
        'save_steps': 10,
        'eval_steps': 10,
        'logging_steps': 1
    },
    'output_dir': 'experiments/{timestamp}/dpo_debug'
}

with open('configs/dpo_config_debug.yaml', 'w') as f:
    yaml.dump(config, f)
print('DPO debug config created!')
"
    fi
    
    # Run DPO training with correct arguments
    python scripts/train_dpo.py \
        --config configs/dpo_config_debug.yaml \
        --sft_checkpoint $CHECKPOINT \
        --max_samples $MAX_SAMPLES || echo "DPO training skipped or failed"
else
    echo "Step 2: DPO training script not found, skipping..."
fi

# Step 3: Run evaluation
if [ -f "scripts/evaluate_model.py" ]; then
    echo ""
    echo "Step 3: Running evaluation..."
    echo "-----------------------------------------"
    # Try different test data formats
    if [ -d "./data/test_dataset" ]; then
        # Use HF dataset format if directory exists
        python scripts/evaluate_model.py \
            --model_paths $CHECKPOINT \
            --test_data ./data/test_dataset \
            --max_samples 10 || echo "Evaluation with HF dataset failed, trying JSON..."
    fi
    
    # If that fails or doesn't exist, try the JSON file
    if [ -f "./data/test_dataset.json" ]; then
        # Convert JSON Lines to proper format or use as-is
        python scripts/evaluate_model.py \
            --model_paths $CHECKPOINT \
            --test_data ./data/unified_customer_support.json \
            --max_samples 10 || echo "Evaluation skipped or failed"
    fi
else
    echo "Step 3: Evaluation script not found, skipping..."
fi

echo ""
echo "========================================="
echo "Pipeline test completed!"
echo "========================================="
echo ""
echo "Results:"
echo "- SFT checkpoint: $CHECKPOINT"
echo ""
echo "To run full training, use:"
echo "  python scripts/train_sft.py --config configs/sft_config.yaml --data_path ./data/unified_customer_support.json"