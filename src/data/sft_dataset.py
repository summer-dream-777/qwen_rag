"""
SFT Dataset for Customer Support Training
"""
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Dict, List, Optional, Union
import pandas as pd
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SFTDataConfig:
    """Configuration for SFT dataset"""
    max_length: int = 2048
    system_prompt: str = "You are a helpful customer support assistant. Provide clear, concise, and helpful responses."
    instruction_template: str = "### User:\n{instruction}\n\n### Assistant:\n{response}"
    add_eos_token: bool = True


class CustomerSupportDataset(Dataset):
    """Dataset for customer support SFT training"""

    def __init__(
        self,
        data: Union[pd.DataFrame, List[Dict]],
        tokenizer: PreTrainedTokenizer,
        config: Optional[SFTDataConfig] = None
    ):
        self.tokenizer = tokenizer
        self.config = config or SFTDataConfig()

        # Convert to list of dicts if DataFrame
        if isinstance(data, pd.DataFrame):
            self.data = data.to_dict('records')
        else:
            self.data = data

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Loaded {len(self.data)} examples for training")

        # Preprocess and filter valid examples
        self.processed_data = self._preprocess_data()
        logger.info(f"After preprocessing: {len(self.processed_data)} valid examples")

    def _preprocess_data(self) -> List[Dict]:
        """Preprocess and validate data"""
        processed = []

        for idx, item in enumerate(self.data):
            try:
                # Validate required fields
                if not item.get('instruction') or not item.get('response'):
                    logger.warning(f"Skipping item {idx}: missing instruction or response")
                    continue

                # Clean text
                instruction = item['instruction'].strip()
                response = item['response'].strip()

                if not instruction or not response:
                    continue

                # Format with template
                formatted_text = self.config.instruction_template.format(
                    instruction=instruction,
                    response=response
                )

                # Add system prompt if specified
                if self.config.system_prompt:
                    formatted_text = f"{self.config.system_prompt}\n\n{formatted_text}"

                processed_item = {
                    'text': formatted_text,
                    'instruction': instruction,
                    'response': response,
                    **{k: v for k, v in item.items() if k not in ['instruction', 'response']}
                }

                processed.append(processed_item)

            except Exception as e:
                logger.error(f"Error processing item {idx}: {e}")
                continue

        return processed

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized example"""
        item = self.processed_data[idx]

        # Tokenize
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )

        # Create labels (same as input_ids for causal LM)
        labels = encoding['input_ids'].clone()

        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

    def get_metadata(self, idx: int) -> Dict:
        """Get metadata for a specific example"""
        return {k: v for k, v in self.processed_data[idx].items() if k != 'text'}


class DataCollatorForSFT:
    """Custom data collator for SFT training"""

    def __init__(self, tokenizer: PreTrainedTokenizer, padding: bool = True):
        self.tokenizer = tokenizer
        self.padding = padding

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch of examples"""
        # Stack tensors
        batch = {
            key: torch.stack([f[key] for f in features])
            for key in features[0].keys()
        }

        return batch


def create_sft_datasets(
    train_data: Union[pd.DataFrame, List[Dict]],
    eval_data: Union[pd.DataFrame, List[Dict]],
    tokenizer: PreTrainedTokenizer,
    config: Optional[SFTDataConfig] = None
) -> tuple:
    """Create train and eval datasets"""

    train_dataset = CustomerSupportDataset(train_data, tokenizer, config)
    eval_dataset = CustomerSupportDataset(eval_data, tokenizer, config)

    return train_dataset, eval_dataset