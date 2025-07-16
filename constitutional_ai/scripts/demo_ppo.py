#!/usr/bin/env python3
"""
Demo script for PPO training in Constitutional AI.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from constitutional_ai.models.constitutional_model import ConstitutionalAIModel
from constitutional_ai.models.reward_model import RewardModel
from constitutional_ai.training.ppo_trainer import PPOTrainer
from constitutional_ai.utils.config import Config

def main():
    """Run PPO training demo."""
    
    print("Constitutional AI PPO Training Demo")
    print("=" * 50)
    
    # Create config
    config = Config()
    config.model.model_name = "microsoft/DialoGPT-small"  # Small model for demo
    config.model.device = "cpu"  # Use CPU for demo
    config.ppo.batch_size = 2
    config.ppo.mini_batch_size = 1
    config.ppo.ppo_epochs = 2
    
    print("Loading models...")
    
    # Initialize models
    policy_model = ConstitutionalAIModel(config)
    ref_model = ConstitutionalAIModel(config)  # Reference model (frozen)
    reward_model = RewardModel(config)
    
    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
        config=config
    )
    
    # Example training queries
    training_queries = [
        "How can I help someone who is feeling anxious?",
        "What's the best way to resolve conflicts?",
        "How do I stay motivated when facing challenges?",
        "What should I do if I make a mistake at work?",
        "How can I be more empathetic to others?"
    ]
    
    print(f"Training on {len(training_queries)} queries...")
    print("Sample queries:")
    for i, query in enumerate(training_queries[:3], 1):
        print(f"  {i}. {query}")
    
    # Train with PPO
    print("\nStarting PPO training...")
    try:
        training_stats = ppo_trainer.train(
            queries=training_queries,
            num_epochs=1,  # Just 1 epoch for demo
            batch_size=config.ppo.batch_size
        )
        
        print("\nTraining completed! Statistics:")
        for key, values in training_stats.items():
            if values:
                avg_value = sum(values) / len(values)
                print(f"  {key}: {avg_value:.4f}")
                
    except Exception as e:
        print(f"\nTraining failed: {e}")
        print("This is expected in demo mode without proper model weights.")
        
    print("\nDemo completed!")
    print("Key PPO Components Demonstrated:")
    print("  1. Policy model for generating responses")
    print("  2. Reference model for KL divergence constraint")
    print("  3. Reward model for constitutional scoring")
    print("  4. PPO algorithm for policy optimization")
    print("  5. Adaptive KL controller for stable training")

if __name__ == "__main__":
    main()