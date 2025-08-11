"""
SFT Trainer for Customer Support Model
"""
import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from typing import Dict, Optional, Any, List
import logging
from datetime import datetime
import json
import wandb
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsCallback(TrainerCallback):
    """Custom callback for logging additional metrics"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Log to wandb if available
            if wandb.run:
                wandb.log(logs, step=state.global_step)
                
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            logger.info(f"Evaluation metrics at step {state.global_step}: {metrics}")


class SFTTrainer:
    """Trainer class for SFT"""
    
    def __init__(self, config_path: str):
        # Load config
        with open(config_path, 'r') as f:
            import yaml
            self.config = yaml.safe_load(f)
            
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
        
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with LoRA if specified"""
        logger.info(f"Loading model: {self.config['model']['name_or_path']}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['name_or_path'],
            trust_remote_code=True
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Model loading arguments
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.config['training']['fp16'] else torch.float32,
            "device_map": "auto"
        }
        
        # Add quantization if specified
        if self.config['model'].get('load_in_8bit'):
            model_kwargs['load_in_8bit'] = True
        elif self.config['model'].get('load_in_4bit'):
            model_kwargs['load_in_4bit'] = True
            model_kwargs['bnb_4bit_compute_dtype'] = torch.float16
            model_kwargs['bnb_4bit_quant_type'] = "nf4"
            
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['name_or_path'],
            **model_kwargs
        )
        
        # Prepare for training
        if self.config['model'].get('load_in_8bit') or self.config['model'].get('load_in_4bit'):
            self.model = prepare_model_for_kbit_training(self.model)
            
        # Enable gradient checkpointing before LoRA
        if self.config['training']['gradient_checkpointing']:
            self.model.gradient_checkpointing_enable()
            
        # Apply LoRA if specified
        if self.config['model'].get('use_lora'):
            logger.info("Applying LoRA configuration")
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
            
        # Ensure model is in training mode
        self.model.train()
            
    def create_training_arguments(self) -> TrainingArguments:
        """Create training arguments"""
        return TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config['training']['num_epochs'],
            per_device_train_batch_size=self.config['data']['train_batch_size'],
            per_device_eval_batch_size=self.config['data']['eval_batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=float(self.config['training']['learning_rate']),
            warmup_ratio=self.config['training']['warmup_ratio'],
            lr_scheduler_type=self.config['training']['lr_scheduler_type'],
            
            # Optimization
            optim=self.config['training']['optim'],
            fp16=self.config['training']['fp16'],
            gradient_checkpointing=self.config['training']['gradient_checkpointing'],
            
            # Logging and saving
            logging_steps=self.config['training']['logging_steps'],
            save_strategy=self.config['training']['save_strategy'],
            save_steps=self.config['training']['save_steps'],
            eval_strategy=self.config['training']['eval_strategy'],
            eval_steps=self.config['training']['eval_steps'],
            save_total_limit=self.config['training']['save_total_limit'],
            
            # Best model
            load_best_model_at_end=self.config['training']['load_best_model_at_end'],
            metric_for_best_model=self.config['training']['metric_for_best_model'],
            greater_is_better=self.config['training']['greater_is_better'],
            
            # Other
            dataloader_num_workers=self.config['data']['num_workers'],
            remove_unused_columns=False,
            report_to=["wandb"] if self.config.get('wandb') else ["none"],
            run_name=f"sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        
        # Calculate perplexity
        loss = torch.nn.functional.cross_entropy(
            torch.tensor(predictions).view(-1, predictions.shape[-1]),
            torch.tensor(labels).view(-1),
            reduction='mean'
        )
        perplexity = torch.exp(loss).item()
        
        return {
            "perplexity": perplexity,
        }
        
    def train(self, train_dataset, eval_dataset):
        """Run training"""
        # Create training arguments
        training_args = self.create_training_arguments()
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[
                MetricsCallback(),
                EarlyStoppingCallback(
                    early_stopping_patience=self.config['training']['early_stopping_patience'],
                    early_stopping_threshold=self.config['training']['early_stopping_threshold']
                )
            ]
        )
        
        # Train
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save final model
        logger.info("Saving final model...")
        trainer.save_model(str(self.output_dir / "final_model"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final_model"))
        
        # Save training results
        with open(self.output_dir / "train_results.json", "w") as f:
            json.dump(train_result.metrics, f, indent=2)
            
        # Evaluate final model
        logger.info("Running final evaluation...")
        eval_results = trainer.evaluate()
        
        with open(self.output_dir / "eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)
            
        # Log to wandb
        if wandb.run:
            wandb.log({"final_eval": eval_results})
            wandb.finish()
            
        return train_result, eval_results
    
    def generate_samples(self, prompts: List[str], max_length: int = 256) -> List[str]:
        """Generate sample outputs for inspection"""
        self.model.eval()
        generated = []
        
        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated.append(generated_text)
                
        return generated