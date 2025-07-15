"""
Constitutional AI - End-to-end implementation of Anthropic's Constitutional AI methodology.

This package provides tools for training helpful, harmless, and honest AI assistants
using AI feedback rather than human labels.
"""

__version__ = "0.1.0"
__author__ = "Constitutional AI Research"

from .training import ConstitutionalTrainer
from .models import ConstitutionalAIModel
from .data_processing import ConstitutionalDataset
from .evaluation import ConstitutionalEvaluator

__all__ = [
    "ConstitutionalTrainer",
    "ConstitutionalAIModel", 
    "ConstitutionalDataset",
    "ConstitutionalEvaluator",
]