"""Critic model for PPO value estimation.
Produces per-token scalar values using a linear head over transformer hidden states.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from ..utils.config import Config
from ..utils.logging import get_logger

logger = get_logger(__name__)


class CriticModel(nn.Module):
    """Value function approximator for PPO (critic)."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model_name = config.model.model_name
        self.device = config.model.device

        self.base_model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        hidden_size = self.base_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)

        self.to(self.device)
        logger.info(f"Initialized CriticModel with base {self.model_name}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return per-token values [B, T]."""
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = outputs.last_hidden_state  # [B, T, H]
        values = self.value_head(hidden).squeeze(-1)  # [B, T]
        return values

    def to(self, device: str):  # type: ignore[override]
        super().to(device)
        self.device = device
        return self
