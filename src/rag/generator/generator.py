"""
Generator Implementation for RAG System
Handles response generation using retrieved context
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import PeftModel

from ..retriever import RetrievedDocument

logger = logging.getLogger(__name__)


class Generator:
    """Abstract base class for generators"""
    
    def generate(
        self,
        query: str,
        context: List[RetrievedDocument],
        **kwargs
    ) -> str:
        """Generate response given query and context"""
        raise NotImplementedError


class QwenGenerator(Generator):
    """Qwen model-based generator for RAG"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        adapter_path: Optional[str] = None,
        device: str = None,
        load_in_4bit: bool = False,
        max_context_length: int = 2048,
        max_new_tokens: int = 512
    ):
        """
        Initialize Qwen generator
        
        Args:
            model_name: Base model name or path
            adapter_path: Path to LoRA adapter (SFT/DPO checkpoint)
            device: Device to use
            load_in_4bit: Whether to use 4-bit quantization
            max_context_length: Maximum context length
            max_new_tokens: Maximum tokens to generate
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.max_context_length = max_context_length
        self.max_new_tokens = max_new_tokens
        
        # Load model and tokenizer
        self._load_model(load_in_4bit)
        
    def _load_model(self, load_in_4bit: bool):
        """Load model and tokenizer"""
        logger.info(f"Loading model: {self.model_name}")
        
        # Quantization config
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Load adapter if provided
        if self.adapter_path and Path(self.adapter_path).exists():
            logger.info(f"Loading adapter from: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.adapter_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            # Merge adapter for faster inference
            self.model = self.model.merge_and_unload()
        
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def format_prompt(
        self,
        query: str,
        context: List[RetrievedDocument],
        include_scores: bool = False
    ) -> str:
        """
        Format prompt with query and retrieved context
        
        Args:
            query: User query
            context: Retrieved documents
            include_scores: Whether to include relevance scores
            
        Returns:
            Formatted prompt
        """
        # Build context string
        context_str = "참고 문서:\n"
        for i, doc in enumerate(context, 1):
            if include_scores:
                context_str += f"\n[문서 {i} - 관련도: {doc.score:.2f}]\n"
            else:
                context_str += f"\n[문서 {i}]\n"
            context_str += f"Q: {doc.instruction}\n"
            context_str += f"A: {doc.response}\n"
        
        # Format final prompt
        prompt = f"""다음 참고 문서를 바탕으로 사용자의 질문에 답변해주세요.

{context_str}

사용자 질문: {query}

답변:"""
        
        return prompt
    
    def generate(
        self,
        query: str,
        context: List[RetrievedDocument] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate response using retrieved context
        
        Args:
            query: User query
            context: Retrieved documents
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Tuple of (generated response, metadata)
        """
        # Format prompt
        if context:
            prompt = self.format_prompt(query, context)
        else:
            # No RAG, direct generation
            prompt = f"### User:\n{query}\n\n### Assistant:\n"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_length
        )
        
        # Move inputs to model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Metadata
        metadata = {
            "model": self.model_name,
            "adapter": self.adapter_path,
            "context_docs": len(context) if context else 0,
            "prompt_length": inputs['input_ids'].shape[1],
            "response_length": outputs.shape[1] - inputs['input_ids'].shape[1],
            "temperature": temperature,
            "top_p": top_p
        }
        
        return response.strip(), metadata
    
    def batch_generate(
        self,
        queries: List[str],
        contexts: List[List[RetrievedDocument]] = None,
        batch_size: int = 4,
        **kwargs
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Batch generation for multiple queries
        
        Args:
            queries: List of queries
            contexts: List of context documents for each query
            batch_size: Batch size for generation
            
        Returns:
            List of (response, metadata) tuples
        """
        results = []
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i+batch_size]
            batch_contexts = contexts[i:i+batch_size] if contexts else [None] * len(batch_queries)
            
            for query, context in zip(batch_queries, batch_contexts):
                response, metadata = self.generate(query, context, **kwargs)
                results.append((response, metadata))
        
        return results


class LangChainGenerator(Generator):
    """
    LangChain-based generator for advanced RAG pipelines
    (Optional - for future implementation)
    """
    
    def __init__(self, model_name: str, **kwargs):
        """Initialize LangChain generator"""
        # TODO: Implement LangChain integration
        pass
    
    def generate(
        self,
        query: str,
        context: List[RetrievedDocument],
        **kwargs
    ) -> str:
        """Generate using LangChain"""
        # TODO: Implement LangChain generation
        pass


if __name__ == "__main__":
    # Test generator
    logging.basicConfig(level=logging.INFO)
    
    # Initialize generator
    generator = QwenGenerator(
        model_name="Qwen/Qwen3-0.6B",
        load_in_4bit=True
    )
    
    # Test generation without context
    query = "What is the capital of France?"
    response, metadata = generator.generate(query)
    
    print(f"Query: {query}")
    print(f"Response: {response}")
    print(f"Metadata: {metadata}")