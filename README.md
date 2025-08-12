# RAG Customer Support Chatbot

A fine-tuned customer support chatbot using Qwen models with SFT (Supervised Fine-Tuning) and DPO (Direct Preference Optimization).

## Features

- **SFT Training**: Supervised fine-tuning on customer support conversations
- **DPO Training**: Preference learning for improved response quality
- **LoRA/QLoRA**: Efficient fine-tuning with reduced memory requirements
- **Evaluation**: Comprehensive metrics including BERT score and ROUGE
- **Experiment Tracking**: Weights & Biases integration for monitoring

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- 32GB+ system RAM

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag_chatbot
```

2. Create virtual environment:
```bash
python -m venv qrag_venv
source qrag_venv/bin/activate  # On Windows: qrag_venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Setup Weights & Biases:
```bash
wandb login
```

## Project Structure

```
rat_chatbot/
├── configs/                # Training configurations
│   ├── sft_config.yaml    # SFT configuration
│   ├── dpo_config.yaml    # DPO configuration
│   └── eval_config.yaml   # Evaluation configuration
├── data/                   # Dataset files
│   ├── unified_customer_support.json
│   ├── train_dataset/
│   └── test_dataset/
├── scripts/                # Training and evaluation scripts
│   ├── train_sft.py       # SFT training script
│   ├── train_dpo.py       # DPO training script
│   ├── evaluate_model.py  # Model evaluation
│   └── prepare_data.py    # Data preprocessing
├── src/                    # Source code
│   ├── data/              # Dataset classes
│   ├── evaluation/        # Evaluation metrics
│   ├── models/            # Model utilities
│   └── training/          # Training logic
└── experiments/            # Training outputs and checkpoints
```

## Quick Start

### 1. Prepare Data

The repository includes a pre-processed customer support dataset with ~10,000 samples. To use your own data:

```bash
python scripts/prepare_data.py --input_file your_data.json --output_dir ./data/
```

### 2. Train SFT Model

```bash
# Full training
python scripts/train_sft.py \
    --config configs/sft_config.yaml \
    --data_path ./data/unified_customer_support.json \
    --test_size 0.1

# Quick test with small dataset
python scripts/train_sft.py \
    --config configs/sft_config_debug.yaml \
    --data_path ./data/unified_customer_support.json \
    --max_samples 50 \
    --test_size 0.2
```

### 3. Train DPO Model (Optional)

After SFT training, you can further improve the model with DPO:

```bash
# Find your SFT checkpoint
ls experiments/*/sft/checkpoint-*

# Run DPO training
python scripts/train_dpo.py \
    --config configs/dpo_config.yaml \
    --sft_checkpoint experiments/[timestamp]/sft/checkpoint-[step] \
    --max_samples 1000  # Optional: limit samples for testing
```

### 4. Evaluate Model

```bash
python scripts/evaluate_model.py \
    --model_paths experiments/[timestamp]/sft/checkpoint-[step] \
    --test_data ./data/test_dataset.json \
    --max_samples 100
```

## Configuration

### Model Configuration

Edit `configs/sft_config.yaml` to adjust model parameters:

```yaml
model:
  name_or_path: "Qwen/Qwen3-0.6B"  # Base model
  use_lora: true                    # Enable LoRA
  lora_config:
    r: 64                          # LoRA rank
    lora_alpha: 128                # LoRA alpha
```

### Training Configuration

```yaml
training:
  num_epochs: 3
  learning_rate: 2e-4
  train_batch_size: 4
  gradient_accumulation_steps: 4
```

## Memory Optimization

For limited GPU memory:

1. **Use smaller model**: Change to `Qwen/Qwen3-0.6B`
2. **Enable 4-bit quantization**: Set `load_in_4bit: true` in config
3. **Reduce batch size**: Set `train_batch_size: 1`
4. **Increase gradient accumulation**: Set `gradient_accumulation_steps: 16`

## Common Issues

### Learning Rate Type Error
**Error**: `TypeError: '<=' not supported between instances of 'float' and 'str'`
**Solution**: Already fixed in the code. Ensure you're using the latest version.

### Gradient Error with LoRA
**Error**: `RuntimeError: element 0 of tensors does not require grad`
**Solution**: Already fixed. Gradient checkpointing is now properly configured.

### Out of Memory
**Solution**: 
- Reduce `max_length` in config
- Use smaller batch size
- Enable 4-bit quantization
- Use smaller model (Qwen3-0.6B instead of Qwen3-4B)

## Dataset Format

Training data should be in JSON Lines format with the following structure:

```json
{
  "instruction": "Customer query",
  "response": "Support agent response",
  "source": "data_source",
  "domain": "customer_support"
}
```

## Results

Training metrics are logged to:
- **Console**: Real-time training progress
- **Weights & Biases**: Detailed metrics and visualizations
- **Local files**: `experiments/[timestamp]/*/training_log.txt`

## Citation

If you use this code in your research, please cite:

```bibtex
@software{rag_chatbot_2025,
  title = {RAG Customer Support Chatbot},
  year = {2025},
  url = {https://github.com/yourusername/rat_chatbot}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Qwen team for the base models
- Hugging Face for the transformers library
- PEFT library for efficient fine-tuning
