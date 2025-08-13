"""
DPO Dataset for Preference Learning
"""
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DPODataset(Dataset):
    """Dataset for Direct Preference Optimization"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_name: str = "SAGI-1/ultrafeedback_binarized_dpo",
        split: str = "train_prefs",
        max_length: int = 1024,
        max_prompt_length: int = 512,
        cache_dir: Optional[str] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load dataset
        logger.info(f"Loading dataset {dataset_name} split {split}")
        self.dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        logger.info(f"Loaded {len(self.dataset)} preference pairs")

        # Validate dataset format
        self._validate_dataset()

    def _validate_dataset(self):
        """Validate dataset has required fields"""
        required_fields = ['prompt', 'chosen', 'rejected']
        sample = self.dataset[0]

        for field in required_fields:
            if field not in sample:
                raise ValueError(f"Dataset missing required field: {field}")

        logger.info("Dataset validation passed")

    def __len__(self) -> int:
        return len(self.dataset)

    def _tokenize_pair(
        self,
        prompt: str,
        chosen: str,
        rejected: str
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a preference pair"""

        # Format prompt + response
        def format_sample(prompt: str, response: str) -> str:
            return f"### User:\n{prompt}\n\n### Assistant:\n{response}"

        chosen_text = format_sample(prompt, chosen)
        rejected_text = format_sample(prompt, rejected)

        # Tokenize chosen response
        chosen_encoding = self.tokenizer(
            chosen_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Tokenize rejected response
        rejected_encoding = self.tokenizer(
            rejected_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Tokenize prompt separately for masking
        prompt_encoding = self.tokenizer(
            f"### User:\n{prompt}\n\n### Assistant:\n",
            truncation=True,
            max_length=self.max_prompt_length,
            add_special_tokens=False
        )
        prompt_length = len(prompt_encoding['input_ids'])

        return {
            'chosen_input_ids': chosen_encoding['input_ids'].squeeze(),
            'chosen_attention_mask': chosen_encoding['attention_mask'].squeeze(),
            'rejected_input_ids': rejected_encoding['input_ids'].squeeze(),
            'rejected_attention_mask': rejected_encoding['attention_mask'].squeeze(),
            'prompt_length': prompt_length
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a tokenized preference pair"""
        item = self.dataset[idx]

        # Extract prompt and responses
        prompt = item['prompt'].strip()

        # Handle different formats of chosen/rejected
        if isinstance(item['chosen'], dict):
            chosen = item['chosen'].get('content', item['chosen'].get('text', ''))
            rejected = item['rejected'].get('content', item['rejected'].get('text', ''))
        else:
            chosen = str(item['chosen']).strip()
            rejected = str(item['rejected']).strip()

        # Tokenize
        tokenized = self._tokenize_pair(prompt, chosen, rejected)

        # Add metadata
        tokenized['idx'] = idx

        return tokenized


class DPODataCollator:
    """Custom data collator for DPO training"""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch for DPO training"""

        # Separate chosen and rejected
        chosen_features = []
        rejected_features = []
        prompt_lengths = []

        for f in features:
            chosen_features.append({
                'input_ids': f['chosen_input_ids'],
                'attention_mask': f['chosen_attention_mask']
            })
            rejected_features.append({
                'input_ids': f['rejected_input_ids'],
                'attention_mask': f['rejected_attention_mask']
            })
            prompt_lengths.append(f['prompt_length'])

        # Stack tensors
        batch = {
            'chosen_input_ids': torch.stack([f['input_ids'] for f in chosen_features]),
            'chosen_attention_mask': torch.stack([f['attention_mask'] for f in chosen_features]),
            'rejected_input_ids': torch.stack([f['input_ids'] for f in rejected_features]),
            'rejected_attention_mask': torch.stack([f['attention_mask'] for f in rejected_features]),
            'prompt_lengths': torch.tensor(prompt_lengths)
        }

        # Create labels (mask prompt tokens)
        batch['chosen_labels'] = batch['chosen_input_ids'].clone()
        batch['rejected_labels'] = batch['rejected_input_ids'].clone()

        # Mask padding tokens
        batch['chosen_labels'][batch['chosen_labels'] == self.tokenizer.pad_token_id] = -100
        batch['rejected_labels'][batch['rejected_labels'] == self.tokenizer.pad_token_id] = -100

        return batch


def create_dpo_datasets(
    tokenizer: PreTrainedTokenizer,
    dataset_name: str = "SAGI-1/ultrafeedback_binarized_dpo",
    train_split: str = "train_prefs",
    eval_split: str = "test_prefs",
    max_length: int = 1024,
    max_prompt_length: int = 512,
    cache_dir: Optional[str] = None
) -> Tuple[DPODataset, Optional[DPODataset]]:
    """Create train and eval DPO datasets"""

    train_dataset = DPODataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        split=train_split,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        cache_dir=cache_dir
    )

    # Try to load eval split
    eval_dataset = None
    try:
        eval_dataset = DPODataset(
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            split=eval_split,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            cache_dir=cache_dir
        )
    except Exception as e:
        logger.warning(f"Could not load eval split {eval_split}: {e}")
        logger.info("Creating eval dataset from train split (10%)")
        # Create eval from train
        eval_size = int(0.1 * len(train_dataset))
        train_dataset.dataset = train_dataset.dataset.train_test_split(
            test_size=eval_size,
            seed=42
        )['train']

    return train_dataset, eval_dataset