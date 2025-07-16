"""
Phase 1 trainer for Constitutional AI.
Handles constitutional fine-tuning with critique and revision generation.
"""

from .constitutional_trainer import ConstitutionalTrainer
from ..utils.logging import get_logger

logger = get_logger(__name__)


class Phase1Trainer(ConstitutionalTrainer):
    """Specialized trainer for Phase 1 constitutional fine-tuning."""
    
    def __init__(self, config):
        super().__init__(config)
        logger.info("Initialized Phase 1 trainer")
        
    def train_phase1(self, train_dataset, eval_dataset=None):
        """Run Phase 1 training with critique and revision generation."""
        
        logger.info("Starting Phase 1: Constitutional Fine-tuning")
        
        # Generate critiques and revisions
        train_dataset = self.generate_critiques_and_revisions(train_dataset)
        
        if eval_dataset:
            eval_dataset = self.generate_critiques_and_revisions(eval_dataset)
        
        # Train on revised responses
        self.train(train_dataset, eval_dataset)
        
        logger.info("Phase 1 training completed")
        
        return train_dataset, eval_dataset