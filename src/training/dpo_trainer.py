"""
DPO Trainer for Preference Learning
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from typing import Dict, Optional, Any, List, Tuple
import logging
from datetime import datetime
import json
import wandb
from pathlib import Path
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DPOLoss(nn.Module):
    """Direct Preference Optimization Loss"""
    
    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        loss_type: str = "sigmoid"
    ):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        
    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute DPO loss
        
        Args:
            policy_chosen_logps: Log probabilities of chosen responses under policy model
            policy_rejected_logps: Log probabilities of rejected responses under policy model
            reference_chosen_logps: Log probabilities of chosen responses under reference model
            reference_rejected_logps: Log probabilities of rejected responses under reference model
        """
        # Compute log ratios
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)
        
        # Compute loss based on type
        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(chosen_rewards - rejected_rewards)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - (chosen_rewards - rejected_rewards))
        elif self.loss_type == "ipo":  # Identity Preference Optimization
            losses = (chosen_rewards - rejected_rewards - 1) ** 2
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
            
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            losses = losses * (1 - self.label_smoothing) + self.label_smoothing * 0.5
            
        loss = losses.mean()
        
        # Compute metrics
        rewards_accuracy = (chosen_rewards > rejected_rewards).float().mean()
        rewards_margin = (chosen_rewards - rejected_rewards).mean()
        
        metrics = {
            "loss": loss,
            "rewards_accuracy": rewards_accuracy,
            "rewards_margin": rewards_margin,
            "chosen_rewards": chosen_rewards.mean(),
            "rejected_rewards": rejected_rewards.mean(),
        }
        
        return loss, metrics


class DPOTrainingCallback(TrainerCallback):
    """Custom callback for DPO training"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and wandb.run:
            # Extract DPO specific metrics
            dpo_metrics = {k: v for k, v in logs.items() if 'reward' in k or 'accuracy' in k}
            if dpo_metrics:
                wandb.log(dpo_metrics, step=state.global_step)


class DPOTrainer:
    """Trainer for Direct Preference Optimization"""
    
    def __init__(self, config_path: str, sft_checkpoint_path: str):
        # Load config
        with open(config_path, 'r') as f:
            import yaml
            self.config = yaml.safe_load(f)
            
        # Update model path to SFT checkpoint
        self.config['model']['name_or_path'] = sft_checkpoint_path
        
        # Setup paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config['output_dir'] = self.config['output_dir'].format(timestamp=timestamp)
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
            
        # Initialize wandb
        if self.config.get('wandb'):
            wandb.init(
                project=self.config['wandb']['project'],
                name=f"{self.config['wandb']['name']}_{timestamp}",
                tags=self.config['wandb']['tags'],
                config=self.config
            )
            
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize loss
        self.dpo_loss = DPOLoss(
            beta=self.config['training']['beta'],
            label_smoothing=self.config['training']['label_smoothing'],
            loss_type=self.config['training']['loss_type']
        )
        
    def load_models_and_tokenizer(self):
        """Load policy and reference models"""
        logger.info(f"Loading SFT checkpoint from: {self.config['model']['name_or_path']}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['name_or_path'],
            trust_remote_code=True
        )
        
        # Load reference model (frozen SFT model)
        logger.info("Loading reference model...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['name_or_path'],
            torch_dtype=torch.float16 if self.config['training']['fp16'] else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        self.ref_model.eval()
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        # Load policy model (trainable)
        logger.info("Loading policy model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['name_or_path'],
            torch_dtype=torch.float16 if self.config['training']['fp16'] else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Apply LoRA to policy model
        if self.config['model'].get('use_lora'):
            logger.info("Applying LoRA to policy model")
            lora_config = LoraConfig(
                r=self.config['model']['lora_config']['r'],
                lora_alpha=self.config['model']['lora_config']['lora_alpha'],
                target_modules=self.config['model']['lora_config']['target_modules'],
                lora_dropout=self.config['model']['lora_config']['lora_dropout'],
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
        # Enable gradient checkpointing
        if self.config['training']['gradient_checkpointing']:
            self.model.gradient_checkpointing_enable()
            
    def get_batch_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probabilities for a batch"""
        with torch.no_grad() if model == self.ref_model else torch.enable_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            logits = outputs.logits
            
        # Shift for next token prediction
        logits = logits[:, :-1, :]
        labels = input_ids[:, 1:]
        attention_mask = attention_mask[:, 1:]
        
        # Compute per-token log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask prompt tokens and padding
        batch_size = input_ids.shape[0]
        for i in range(batch_size):
            prompt_len = prompt_lengths[i].item()
            selected_log_probs[i, :prompt_len-1] = 0
            
        selected_log_probs = selected_log_probs * attention_mask
        
        # Sum over sequence length
        return selected_log_probs.sum(dim=-1)
        
    def compute_dpo_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute DPO loss for a batch"""
        # Get log probabilities from policy model
        policy_chosen_logps = self.get_batch_logps(
            self.model,
            batch['chosen_input_ids'],
            batch['chosen_attention_mask'],
            batch['prompt_lengths']
        )
        policy_rejected_logps = self.get_batch_logps(
            self.model,
            batch['rejected_input_ids'],
            batch['rejected_attention_mask'],
            batch['prompt_lengths']
        )
        
        # Get log probabilities from reference model
        with torch.no_grad():
            ref_chosen_logps = self.get_batch_logps(
                self.ref_model,
                batch['chosen_input_ids'],
                batch['chosen_attention_mask'],
                batch['prompt_lengths']
            )
            ref_rejected_logps = self.get_batch_logps(
                self.ref_model,
                batch['rejected_input_ids'],
                batch['rejected_attention_mask'],
                batch['prompt_lengths']
            )
        
        # Compute DPO loss
        loss, metrics = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps
        )
        
        return loss, metrics
        
    def train(self, train_dataset, eval_dataset=None):
        """Run DPO training"""
        from src.data.dpo_dataset import DPODataCollator
        
        # Create data collator
        data_collator = DPODataCollator(self.tokenizer)
        
        # Create data loaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config['data']['train_batch_size'],
            shuffle=True,
            collate_fn=data_collator,
            num_workers=self.config['data']['num_workers']
        )
        
        eval_dataloader = None
        if eval_dataset:
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=self.config['data']['eval_batch_size'],
                shuffle=False,
                collate_fn=data_collator,
                num_workers=self.config['data']['num_workers']
            )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate'])
        )
        
        # Training loop
        num_epochs = self.config['training']['num_epochs']
        global_step = 0
        best_eval_loss = float('inf')
        
        logger.info(f"Starting DPO training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_losses = []
            train_metrics = {
                'rewards_accuracy': [],
                'rewards_margin': [],
                'chosen_rewards': [],
                'rejected_rewards': []
            }
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Compute loss
                loss, metrics = self.compute_dpo_loss(batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Log metrics
                train_losses.append(loss.item())
                for k, v in metrics.items():
                    if k != 'loss':
                        train_metrics[k].append(v.item())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{metrics['rewards_accuracy'].item():.2%}"
                })
                
                # Log to wandb
                if global_step % self.config['training']['logging_steps'] == 0:
                    log_metrics = {
                        'train/loss': loss.item(),
                        'train/rewards_accuracy': metrics['rewards_accuracy'].item(),
                        'train/rewards_margin': metrics['rewards_margin'].item(),
                    }
                    if wandb.run:
                        wandb.log(log_metrics, step=global_step)
                        
                global_step += 1
                
                # Evaluation
                if eval_dataloader and global_step % self.config['training']['eval_steps'] == 0:
                    eval_loss, eval_metrics = self.evaluate(eval_dataloader)
                    
                    logger.info(f"Step {global_step} - Eval loss: {eval_loss:.4f}, "
                              f"Eval accuracy: {eval_metrics['rewards_accuracy']:.2%}")
                    
                    # Save best model
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        self.save_model(self.output_dir / "best_model")
                        
            # End of epoch summary
            avg_train_loss = np.mean(train_losses)
            avg_train_acc = np.mean(train_metrics['rewards_accuracy'])
            
            logger.info(f"Epoch {epoch+1} - Avg train loss: {avg_train_loss:.4f}, "
                      f"Avg train accuracy: {avg_train_acc:.2%}")
            
        # Save final model
        self.save_model(self.output_dir / "final_model")
        
        # Final evaluation
        if eval_dataloader:
            final_eval_loss, final_eval_metrics = self.evaluate(eval_dataloader)
            
            # Save results
            results = {
                'final_eval_loss': final_eval_loss,
                'final_eval_metrics': final_eval_metrics,
                'best_eval_loss': best_eval_loss,
                'training_config': self.config
            }
            
            with open(self.output_dir / "training_results.json", "w") as f:
                json.dump(results, f, indent=2)
                
        if wandb.run:
            wandb.finish()
            
    def evaluate(self, eval_dataloader) -> Tuple[float, Dict[str, float]]:
        """Evaluate model on eval set"""
        self.model.eval()
        eval_losses = []
        eval_metrics = {
            'rewards_accuracy': [],
            'rewards_margin': [],
            'chosen_rewards': [],
            'rejected_rewards': []
        }
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss, metrics = self.compute_dpo_loss(batch)
                
                eval_losses.append(loss.item())
                for k, v in metrics.items():
                    if k != 'loss':
                        eval_metrics[k].append(v.item())
                        
        # Compute averages
        avg_eval_loss = np.mean(eval_losses)
        avg_eval_metrics = {k: np.mean(v) for k, v in eval_metrics.items()}
        
        return avg_eval_loss, avg_eval_metrics
        
    def save_model(self, save_path: Path):
        """Save model and tokenizer"""
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(save_path)
        else:
            torch.save(self.model.state_dict(), save_path / "pytorch_model.bin")
            
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save config
        with open(save_path / "dpo_config.yaml", "w") as f:
            import yaml
            yaml.dump(self.config, f)
            
        logger.info(f"Model saved to {save_path}")