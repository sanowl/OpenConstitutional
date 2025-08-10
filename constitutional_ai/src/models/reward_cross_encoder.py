"""
Cross-encoder pairwise reward model: jointly encodes (question, response A, response B)
using a transformer backbone and predicts preference logits (A > B).
Also exposes per-principle heads for multi-task training (optional).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from ..utils.config import Config
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CrossEncoderOutput:
    margin_logits: torch.Tensor  # [B]
    per_principle: Optional[Dict[str, torch.Tensor]] = None  # each [B]


class CrossEncoderRewardModel(nn.Module):
    """Pairwise cross-encoder for preference modeling (A vs B)."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model_name = config.model.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.backbone = AutoModel.from_pretrained(self.model_name)
        hidden = self.backbone.config.hidden_size
        # Binary head for A vs B
        self.classifier = nn.Linear(hidden, 1)
        # Optional per-principle heads
        self.head_helpfulness = nn.Linear(hidden, 1)
        self.head_harmlessness = nn.Linear(hidden, 1)
        self.head_honesty = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(0.1)

    def _encode_pair(self, questions: List[str], a_list: List[str], b_list: List[str]) -> torch.Tensor:
        """Encode [CLS] question [SEP] A [SEP] B and pool CLS hidden state."""
        inputs = self.tokenizer(
            [f"{q} [SEP] A: {a} [SEP] B: {b}" for q, a, b in zip(questions, a_list, b_list)],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.model.max_length,
        )
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        outputs = self.backbone(**inputs, return_dict=True)
        cls = outputs.last_hidden_state[:, 0]  # [B, H]
        cls = self.dropout(cls)
        return cls

    def forward(self, questions: List[str], responses_a: List[str], responses_b: List[str]) -> CrossEncoderOutput:
        cls = self._encode_pair(questions, responses_a, responses_b)
        margin = self.classifier(cls).squeeze(-1)
        per = {
            "helpfulness": self.head_helpfulness(cls).squeeze(-1),
            "harmlessness": self.head_harmlessness(cls).squeeze(-1),
            "honesty": self.head_honesty(cls).squeeze(-1),
        }
        return CrossEncoderOutput(margin_logits=margin, per_principle=per)

    @torch.no_grad()
    def compute_margin(self, question: str, a: str, b: str) -> float:
        out = self.forward([question], [a], [b])
        return float(out.margin_logits[0].item())

    def compute_pairwise_loss(self, questions: List[str], responses_a: List[str], responses_b: List[str], labels: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy on margin logits vs labels (1 for A preferred, 0 for B preferred)."""
        logits = self.forward(questions, responses_a, responses_b).margin_logits
        return nn.functional.binary_cross_entropy_with_logits(logits, labels.float().to(logits.device))
