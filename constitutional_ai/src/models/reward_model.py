"""Reward model for Constitutional AI RLAIF training."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

from ..utils.config import Config
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RewardOutput:
    """Output from reward model."""
    rewards: torch.Tensor
    logits: torch.Tensor
    values: torch.Tensor


class RewardModel(nn.Module):
    """Reward model for constitutional AI training."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model_name = config.model.model_name
        
        # Load base model
        self.base_model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Reward head
        self.reward_head = nn.Linear(self.base_model.config.hidden_size, 1)
        self.dropout = nn.Dropout(0.1)
        
        # Value head for PPO
        self.value_head = nn.Linear(self.base_model.config.hidden_size, 1)
        
        logger.info(f"Initialized RewardModel with {self.model_name}")
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = True
    ) -> RewardOutput:
        """Forward pass through reward model."""
        
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get last hidden state
        last_hidden_state = outputs.last_hidden_state
        
        # Apply attention mask to get sequence representation
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_state.size(0)
        
        # Get the last token representation for each sequence
        last_token_hidden = last_hidden_state[torch.arange(batch_size), sequence_lengths]
        
        # Apply dropout
        hidden = self.dropout(last_token_hidden)
        
        # Compute rewards and values
        rewards = self.reward_head(hidden).squeeze(-1)
        values = self.value_head(hidden).squeeze(-1)
        
        if return_dict:
            return RewardOutput(
                rewards=rewards,
                logits=rewards,  # For compatibility
                values=values
            )
        else:
            return rewards, values
            
    def compute_reward(self, texts: List[str]) -> torch.Tensor:
        """Compute rewards for a batch of texts."""
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.model.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        device = next(self.parameters()).device if any(self.parameters()) else torch.device('cpu')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(**inputs)
            
        return outputs.rewards
        
    def compute_pairwise_loss(
        self,
        chosen_texts: List[str],
        rejected_texts: List[str]
    ) -> torch.Tensor:
        """Compute pairwise ranking loss."""
        
        # Tokenize chosen and rejected responses
        chosen_inputs = self.tokenizer(
            chosen_texts,
            padding=True,
            truncation=True,
            max_length=self.config.model.max_length,
            return_tensors="pt"
        )
        
        rejected_inputs = self.tokenizer(
            rejected_texts,
            padding=True,
            truncation=True,
            max_length=self.config.model.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        device = next(self.parameters()).device
        chosen_inputs = {k: v.to(device) for k, v in chosen_inputs.items()}
        rejected_inputs = {k: v.to(device) for k, v in rejected_inputs.items()}
        
        # Forward pass
        chosen_outputs = self.forward(**chosen_inputs)
        rejected_outputs = self.forward(**rejected_inputs)
        
        # Compute pairwise loss
        loss = -torch.log(torch.sigmoid(chosen_outputs.rewards - rejected_outputs.rewards)).mean()
        
        return loss
        
    def save_pretrained(self, save_directory: str) -> None:
        """Save reward model."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), os.path.join(save_directory, "reward_model.pt"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)
        
        logger.info(f"Reward model saved to {save_directory}")
        
    @classmethod
    def from_pretrained(cls, model_path: str, config: Config) -> "RewardModel":
        """Load reward model from checkpoint."""
        import os
        
        model = cls(config)
        
        # Load model state
        state_dict_path = os.path.join(model_path, "reward_model.pt")
        if os.path.exists(state_dict_path):
            model.load_state_dict(torch.load(state_dict_path))
            
        # Load tokenizer
        model.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        logger.info(f"Reward model loaded from {model_path}")
        return model