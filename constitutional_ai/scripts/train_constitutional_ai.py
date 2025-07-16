#!/usr/bin/env python3
"""
Main training script for Constitutional AI.
Implements the end-to-end training pipeline including:
1. Phase 1: Constitutional Fine-tuning (SFT)
2. Phase 2: Reinforcement Learning from AI Feedback (RLAIF)
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoTokenizer

from constitutional_ai.training.constitutional_trainer import ConstitutionalTrainer
from constitutional_ai.data_processing.constitutional_dataset import ConstitutionalDataset
from constitutional_ai.utils.config import Config, load_config, save_config
from constitutional_ai.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Constitutional AI system")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.json",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for models and logs"
    )
    
    parser.add_argument(
        "--phase",
        type=str,
        choices=["1", "2", "all"],
        default="all",
        help="Training phase to run (1=SFT, 2=RLAIF, all=both)"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Override model name from config"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of training samples"
    )
    
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(log_level="INFO")
    logger.info("Starting Constitutional AI training")
    
    # Set seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line args
    if args.model_name:
        config.model.model_name = args.model_name
    if args.max_samples:
        config.data.max_samples = args.max_samples
    if args.use_wandb:
        config.logging.use_wandb = True
    
    config.logging.output_dir = args.output_dir
    config.training.seed = args.seed
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    save_config(config, os.path.join(args.output_dir, "config.json"))
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = ConstitutionalDataset(
        config=config,
        tokenizer=tokenizer,
        split="train",
        max_samples=config.data.max_samples
    )
    
    eval_dataset = ConstitutionalDataset(
        config=config,
        tokenizer=tokenizer,
        split="test",
        max_samples=min(config.data.max_samples // 10, 100) if config.data.max_samples > 0 else None
    )
    
    # Initialize trainer
    trainer = ConstitutionalTrainer(config)
    
    # Phase 1: Constitutional Fine-tuning
    if args.phase in ["1", "all"]:
        logger.info("=" * 50)
        logger.info("PHASE 1: Constitutional Fine-tuning")
        logger.info("=" * 50)
        
        # Generate critiques and revisions
        logger.info("Generating critiques and revisions for training data...")
        train_dataset = trainer.generate_critiques_and_revisions(train_dataset)
        
        logger.info("Generating critiques and revisions for evaluation data...")
        eval_dataset = trainer.generate_critiques_and_revisions(eval_dataset)
        
        # Save processed datasets
        train_dataset.save_to_disk(os.path.join(args.output_dir, "train_dataset.json"))
        eval_dataset.save_to_disk(os.path.join(args.output_dir, "eval_dataset.json"))
        
        # Train model
        logger.info("Starting Phase 1 training...")
        trainer.train(train_dataset, eval_dataset)
        
        logger.info("Phase 1 completed successfully!")
    
    # Phase 2: RLAIF (Reinforcement Learning from AI Feedback)
    if args.phase in ["2", "all"]:
        logger.info("=" * 50)
        logger.info("PHASE 2: Reinforcement Learning from AI Feedback")
        logger.info("=" * 50)
        
        # Import Phase 2 trainer
        try:
            from constitutional_ai.training.phase2_trainer import Phase2Trainer
        except ImportError as e:
            logger.error(f"Failed to import Phase2Trainer: {e}")
            logger.error("Please ensure all training components are properly installed")
            return
        
        # Initialize Phase 2 trainer with Phase 1 model
        phase1_model_path = os.path.join(args.output_dir, "final_model")
        
        if not os.path.exists(phase1_model_path):
            logger.error(f"Phase 1 model not found at {phase1_model_path}")
            logger.error("Please run Phase 1 training first or provide a valid model path")
            return
            
        try:
            phase2_trainer = Phase2Trainer(config, phase1_model_path)
        except Exception as e:
            logger.error(f"Failed to initialize Phase2Trainer: {e}")
            return
        
        # Train Phase 2
        logger.info("Starting Phase 2 training...")
        phase2_stats = phase2_trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            num_preference_samples=min(config.data.max_samples, 5000),
            num_ppo_epochs=3
        )
        
        # Save Phase 2 statistics
        with open(os.path.join(args.output_dir, "phase2_stats.json"), 'w') as f:
            import json
            json.dump(phase2_stats, f, indent=2)
            
        logger.info("Phase 2 completed successfully!")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()