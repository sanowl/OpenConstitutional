"""
Phase 2 Trainer for Constitutional AI RLAIF.
Integrates preference collection, reward model training, and PPO optimization.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple
import os
import json
from tqdm import tqdm
import numpy as np

from .ppo_trainer import PPOTrainer
from .reward_trainer import RewardTrainer
from ..models.constitutional_model import ConstitutionalAIModel
from ..models.preference_model import PreferenceModel
from ..models.reward_model import RewardModel
from ..models.reward_cross_encoder import CrossEncoderRewardModel
from ..data_processing.constitutional_dataset import ConstitutionalDataset
from ..data_processing.preference_dataset import PreferenceDataset
from ..utils.config import Config
from ..utils.logging import get_logger, WandBLogger

logger = get_logger(__name__)


class Phase2Trainer:
    """Phase 2 trainer for Constitutional AI RLAIF."""
    
    def __init__(self, config: Config, phase1_model_path: str):
        self.config = config
        self.device = torch.device(config.model.device)
        
        # Load Phase 1 model as starting point
        self.policy_model = ConstitutionalAIModel.from_pretrained(
            phase1_model_path, config
        ).to(self.device)
        
        # Create reference model (frozen copy of initial policy)
        self.ref_model = ConstitutionalAIModel.from_pretrained(
            phase1_model_path, config
        ).to(self.device)
        self.ref_model.eval()
        
        # Initialize preference model for collecting AI feedback
        self.preference_model = PreferenceModel(config).to(self.device)
        
        # Initialize reward model (cross-encoder optional)
        if getattr(config.training, "use_cross_encoder_reward", False):
            self.reward_model = CrossEncoderRewardModel(config).to(self.device)
        else:
            self.reward_model = RewardModel(config).to(self.device)
        
        # Initialize sub-trainers
        self.reward_trainer = RewardTrainer(self.reward_model, config)
        self.ppo_trainer = None  # Will be initialized after reward model training
        
        # Initialize logging
        self.wandb_logger = WandBLogger(
            project=config.logging.wandb_project,
            entity=config.logging.wandb_entity,
            name=f"{config.logging.run_name}_phase2" if config.logging.run_name else None,
            config=config.__dict__
        ) if config.logging.use_wandb else None
        
        logger.info("Initialized Phase 2 trainer")
        
    def train(
        self,
        train_dataset: ConstitutionalDataset,
        eval_dataset: Optional[ConstitutionalDataset] = None,
        num_preference_samples: int = 10000,
        num_ppo_epochs: int = 3
    ) -> Dict[str, Any]:
        """Complete Phase 2 training pipeline."""
        
        if self.wandb_logger:
            self.wandb_logger.init()
            
        logger.info("Starting Phase 2 Constitutional AI training")
        
        # Step 1: Collect preference data using AI feedback
        logger.info("=" * 50)
        logger.info("Step 1: Collecting AI Preference Data")
        logger.info("=" * 50)
        
        preference_dataset = self._collect_preference_data(
            train_dataset, num_preference_samples
        )
        
        # Step 2: Train reward model on preference data
        logger.info("=" * 50)
        logger.info("Step 2: Training Reward Model")
        logger.info("=" * 50)
        
        reward_training_stats = self._train_reward_model(
            preference_dataset, eval_dataset
        )
        
        # Step 3: Optimize policy using PPO with trained reward model
        logger.info("=" * 50)
        logger.info("Step 3: PPO Policy Optimization")
        logger.info("=" * 50)
        
        ppo_training_stats = self._train_policy_with_ppo(
            train_dataset, num_ppo_epochs
        )
        
        # Step 4: Final evaluation
        logger.info("=" * 50)
        logger.info("Step 4: Final Evaluation")
        logger.info("=" * 50)
        
        evaluation_stats = self._evaluate_final_model(eval_dataset)
        
        # Combine all statistics
        training_stats = {
            "reward_training": reward_training_stats,
            "ppo_training": ppo_training_stats,
            "final_evaluation": evaluation_stats
        }
        
        # Save final model
        self._save_final_model()
        
        if self.wandb_logger:
            self.wandb_logger.finish()
            
        logger.info("Phase 2 training completed successfully!")
        
        return training_stats
        
    def _collect_preference_data(
        self, dataset: ConstitutionalDataset, num_samples: int
    ) -> PreferenceDataset:
        """Collect preference data using AI feedback."""
        
        logger.info(f"Collecting {num_samples} preference samples")
        
        # Get questions and responses from dataset
        questions, responses = dataset.get_questions_and_responses()
        
        # Generate response pairs for comparison
        response_pairs = []
        for i in range(min(num_samples, len(questions))):
            question = questions[i]
            
            # Generate two different responses for the same question using varied decoding
            response_a = self.policy_model.generate(
                prompt=question,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                max_length=self.config.model.max_length
            )[0].text
            response_b = self.policy_model.generate(
                prompt=question,
                temperature=1.0,
                top_p=0.7,
                do_sample=True,
                num_return_sequences=1,
                max_length=self.config.model.max_length
            )[0].text
            if response_b.strip() == response_a.strip():
                # simple perturbation: tweak temperature
                response_b = self.policy_model.generate(
                    prompt=question,
                    temperature=1.2,
                    top_p=0.9,
                    do_sample=True,
                    num_return_sequences=1,
                    max_length=self.config.model.max_length
                )[0].text
            
            response_pairs.append((response_a, response_b))
            
        # Collect AI preferences
        preference_data = self.preference_model.collect_preference_dataset(
            questions=questions[:num_samples],
            response_pairs=response_pairs,
            principles=self.config.constitutional_principles,
            num_samples=num_samples
        )
        
        # Create preference dataset
        preference_dataset = PreferenceDataset(preference_data, self.reward_model.tokenizer)
        
        # Save preference data
        self._save_preference_data(preference_data)
        
        logger.info(f"Collected {len(preference_data)} preference samples")
        
        return preference_dataset
        
    def _train_reward_model(
        self, 
        preference_dataset: PreferenceDataset,
        eval_dataset: Optional[ConstitutionalDataset] = None
    ) -> Dict[str, Any]:
        """Train reward model on preference data."""
        
        logger.info("Training reward model on preference data")
        
        # Train reward model
        training_stats = self.reward_trainer.train(
            preference_dataset,
            num_epochs=self.config.training.num_epochs,
            batch_size=self.config.training.batch_size
        )
        
        # Evaluate reward model if eval dataset is provided
        if eval_dataset:
            eval_stats = self.reward_trainer.evaluate(eval_dataset)
            training_stats.update(eval_stats)
            
        # Save trained reward model
        reward_model_path = os.path.join(
            self.config.logging.output_dir, "reward_model"
        )
        self.reward_model.save_pretrained(reward_model_path)
        
        logger.info("Reward model training completed")
        
        return training_stats
        
    def _train_policy_with_ppo(
        self, dataset: ConstitutionalDataset, num_epochs: int
    ) -> Dict[str, Any]:
        """Train policy using PPO with trained reward model."""
        
        logger.info("Starting PPO policy optimization")
        
        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            model=self.policy_model,
            ref_model=self.ref_model,
            reward_model=self.reward_model,
            config=self.config
        )
        
        # Get training queries
        questions, _ = dataset.get_questions_and_responses()
        
        # Train with PPO
        training_stats = self.ppo_trainer.train(
            queries=questions,
            num_epochs=num_epochs,
            batch_size=self.config.ppo.batch_size
        )
        
        # Save PPO checkpoint
        ppo_checkpoint_path = os.path.join(
            self.config.logging.output_dir, "ppo_checkpoint.pt"
        )
        self.ppo_trainer.save_checkpoint(ppo_checkpoint_path)
        
        logger.info("PPO training completed")
        
        return training_stats
        
    def _evaluate_final_model(
        self, eval_dataset: Optional[ConstitutionalDataset] = None
    ) -> Dict[str, Any]:
        """Evaluate the final trained model."""
        
        logger.info("Evaluating final model")
        
        if eval_dataset is None:
            logger.warning("No evaluation dataset provided")
            return {}
            
        # Evaluate model performance
        eval_stats = {}
        
        # Get evaluation questions
        questions, ground_truth_responses = eval_dataset.get_questions_and_responses()
        
        # Generate responses with final model
        generated_responses = []
        for question in tqdm(questions[:100], desc="Generating responses"):  # Limit for demo
            response = self.policy_model.generate(
                prompt=question,
                temperature=self.config.model.temperature,
                num_return_sequences=1
            )[0].text
            generated_responses.append(response)
            
        # Compute reward scores
        reward_scores = []
        for question, response in zip(questions[:100], generated_responses):
            full_text = f"{question} {response}"
            reward = self.reward_model.compute_reward([full_text])
            reward_scores.append(reward.item())
            
        # Basic statistics
        eval_stats.update({
            "avg_reward": np.mean(reward_scores),
            "std_reward": np.std(reward_scores),
            "min_reward": np.min(reward_scores),
            "max_reward": np.max(reward_scores),
            "num_eval_samples": len(generated_responses)
        })
        
        # Constitutional compliance evaluation
        constitutional_stats = self._evaluate_constitutional_compliance(
            questions[:100], generated_responses
        )
        eval_stats.update(constitutional_stats)
        
        logger.info(f"Final evaluation completed: avg_reward={eval_stats['avg_reward']:.4f}")
        
        return eval_stats
        
    def _evaluate_constitutional_compliance(
        self, questions: List[str], responses: List[str]
    ) -> Dict[str, Any]:
        """Evaluate constitutional compliance of generated responses."""
        
        compliance_stats = {}
        
        # Use preference model to evaluate constitutional compliance
        preferences = self.preference_model.generate_batch_preferences(
            questions=questions,
            responses_a=responses,
            responses_b=["I don't know."] * len(responses),  # Neutral baseline
            principles=self.config.constitutional_principles
        )
        
        # Compute compliance metrics
        high_confidence_preferred = sum(
            1 for p in preferences 
            if p.preferred_response == "A" and p.confidence > 0.7
        )
        
        compliance_stats.update({
            "constitutional_compliance_rate": high_confidence_preferred / len(preferences),
            "avg_preference_confidence": np.mean([p.confidence for p in preferences]),
            "preferred_responses": sum(1 for p in preferences if p.preferred_response == "A")
        })
        
        return compliance_stats
        
    def _save_preference_data(self, preference_data: List[Dict[str, Any]]):
        """Save preference data to disk."""
        
        save_path = os.path.join(
            self.config.logging.output_dir, "preference_data.json"
        )
        
        with open(save_path, 'w') as f:
            json.dump(preference_data, f, indent=2)
            
        logger.info(f"Preference data saved to {save_path}")
        
    def _save_final_model(self):
        """Save the final trained model."""
        
        final_model_path = os.path.join(
            self.config.logging.output_dir, "final_constitutional_model"
        )
        
        self.policy_model.save_pretrained(final_model_path)
        
        logger.info(f"Final model saved to {final_model_path}")
        
    def generate_response(self, question: str) -> str:
        """Generate a response using the trained model."""
        
        self.policy_model.eval()
        
        with torch.no_grad():
            response = self.policy_model.generate(
                prompt=question,
                temperature=self.config.model.temperature,
                num_return_sequences=1
            )[0].text
            
        return response
        
    def compare_models(
        self, questions: List[str], phase1_model_path: str
    ) -> Dict[str, Any]:
        """Compare Phase 2 model with Phase 1 model."""
        
        logger.info("Comparing Phase 2 model with Phase 1 model")
        
        # Load Phase 1 model
        phase1_model = ConstitutionalAIModel.from_pretrained(
            phase1_model_path, self.config
        ).to(self.device)
        
        # Generate responses with both models
        phase1_responses = []
        phase2_responses = []
        
        for question in tqdm(questions[:50], desc="Generating comparison responses"):
            # Phase 1 response
            phase1_response = phase1_model.generate(
                prompt=question,
                temperature=self.config.model.temperature,
                num_return_sequences=1
            )[0].text
            phase1_responses.append(phase1_response)
            
            # Phase 2 response
            phase2_response = self.generate_response(question)
            phase2_responses.append(phase2_response)
            
        # Compare using preference model
        preferences = self.preference_model.generate_batch_preferences(
            questions=questions[:50],
            responses_a=phase1_responses,
            responses_b=phase2_responses,
            principles=self.config.constitutional_principles
        )
        
        # Compute comparison statistics
        phase2_preferred = sum(1 for p in preferences if p.preferred_response == "B")
        phase1_preferred = sum(1 for p in preferences if p.preferred_response == "A")
        
        comparison_stats = {
            "phase1_preferred": phase1_preferred,
            "phase2_preferred": phase2_preferred,
            "phase2_win_rate": phase2_preferred / len(preferences),
            "avg_confidence": np.mean([p.confidence for p in preferences])
        }
        
        logger.info(f"Model comparison completed: Phase 2 win rate = {comparison_stats['phase2_win_rate']:.2%}")
        
        return comparison_stats