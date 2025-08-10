"""Main Constitutional AI model implementation."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from ..utils.config import Config
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GenerationOutput:
    """Output from model generation."""
    text: str
    tokens: List[int]
    logprobs: Optional[List[float]] = None
    scores: Optional[List[float]] = None


class ConstitutionalAIModel(nn.Module):
    """Main Constitutional AI model class."""
    
    def __init__(self, config: Config, pretrained_path: Optional[str] = None):
        super().__init__()
        self.config = config
        self.model_name = config.model.model_name
        self.device = config.model.device
        
        # Load tokenizer and model (optionally from a pretrained path)
        load_source = pretrained_path or self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(load_source)
        # Resolve dtype safely for the selected device
        requested_dtype_name = getattr(config.model, "dtype", "float32")
        requested_dtype = getattr(torch, requested_dtype_name, torch.float32)
        if self.device == "cpu" and requested_dtype in (torch.float16, torch.bfloat16):
            warnings.warn(
                f"Requested dtype {requested_dtype_name} is not ideal on CPU; falling back to float32.")
            requested_dtype = torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            load_source,
            torch_dtype=requested_dtype,
        )
        self.model.to(self.device)
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Set up generation config
        self.generation_config = GenerationConfig(
            max_length=config.model.max_length,
            temperature=config.model.temperature,
            top_p=config.model.top_p,
            do_sample=config.model.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        logger.info(f"Loaded Constitutional AI model: {self.model_name}")
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        num_return_sequences: int = 1,
        return_full_text: bool = False,
        **kwargs
    ) -> List[GenerationOutput]:
        """Generate text from prompt."""
        # Optionally allow principle tokens conditioning
        if self.config.model.use_principle_tokens and kwargs.get("principles"):
            principles: List[str] = kwargs.pop("principles")
            prefix = "\n".join([f"[PRINCIPLE] {p}" for p in principles]) + "\n"
            prompt = prefix + prompt
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.model.max_length
        ).to(self.device)
        
        # Update generation config
        gen_config = GenerationConfig(
            max_length=max_length or self.generation_config.max_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature or self.generation_config.temperature,
            top_p=top_p or self.generation_config.top_p,
            do_sample=do_sample if do_sample is not None else self.generation_config.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config,
            )
            
        # Process outputs
        results = []
        input_length = inputs.input_ids.shape[1]
        
        for i in range(num_return_sequences):
            tokens = outputs.sequences[i]
            
            # Extract only the generated part if not returning full text
            if not return_full_text:
                tokens = tokens[input_length:]
                
            # Decode text
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            
            # Extract scores if available
            scores = None
            if hasattr(outputs, 'scores') and outputs.scores:
                scores = [score[i].cpu().numpy() for score in outputs.scores]
                
            results.append(GenerationOutput(
                text=text,
                tokens=tokens.cpu().tolist(),
                scores=scores
            ))
            
        return results
        
    def get_logprobs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get log probabilities for input tokens."""
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # Get log probabilities for actual tokens
            token_log_probs = torch.gather(log_probs[:, :-1], dim=-1, index=input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            
            return token_log_probs
            
    def compute_loss(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute language modeling loss."""
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss
        
    def save_pretrained(self, save_directory: str) -> None:
        """Save model and tokenizer."""
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        logger.info(f"Model saved to {save_directory}")
        
    @classmethod
    def from_pretrained(cls, model_path: str, config: Config) -> "ConstitutionalAIModel":
        """Load model from pretrained checkpoint."""
        instance = cls(config, pretrained_path=model_path)
        logger.info(f"Model loaded from {model_path}")
        return instance
        
    def to(self, device: str) -> "ConstitutionalAIModel":
        """Move model to device."""
        self.model.to(device)
        self.device = device
        return self
        
    def eval(self) -> "ConstitutionalAIModel":
        """Set model to evaluation mode."""
        self.model.eval()
        return self
        
    def train(self) -> "ConstitutionalAIModel":
        """Set model to training mode."""
        self.model.train()
        return self
        
    def parameters(self):
        """Get model parameters."""
        return self.model.parameters()
        
    def named_parameters(self):
        """Get named model parameters."""
        return self.model.named_parameters()
        
    def state_dict(self):
        """Get model state dict."""
        return self.model.state_dict()
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load model state dict."""
        self.model.load_state_dict(state_dict)