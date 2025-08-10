"""Main Constitutional AI trainer implementation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Optional, Any
import os
from tqdm import tqdm

from ..models.constitutional_model import ConstitutionalAIModel
from ..models.critique_model import CritiqueModel
from ..models.revision_model import RevisionModel
from ..data_processing.constitutional_dataset import ConstitutionalDataset
from ..utils.config import Config
from ..evaluation.safety_metrics import SafetyMetrics
from ..utils.logging import get_logger, WandBLogger, MetricsLogger

logger = get_logger(__name__)


class ConstitutionalTrainer:
    """Main trainer for Constitutional AI system."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.model.device)
        
        # Initialize models
        self.model = ConstitutionalAIModel(config).to(self.device)
        self.critique_model = CritiqueModel(config).to(self.device)
        self.revision_model = RevisionModel(config).to(self.device)
        
        # Initialize logging
        self.wandb_logger = WandBLogger(
            project=config.logging.wandb_project,
            entity=config.logging.wandb_entity,
            name=config.logging.run_name,
            config=config.__dict__
        ) if config.logging.use_wandb else None
        
        self.metrics_logger = MetricsLogger(config.logging.output_dir)
        
        # Training state
        self.current_step = 0
        self.current_epoch = 0
        # Initialize safety metrics for adherence scoring
        self.safety = SafetyMetrics(config)
        
        logger.info("Initialized ConstitutionalTrainer")
        
    def setup_training(self, train_dataset: ConstitutionalDataset, eval_dataset: Optional[ConstitutionalDataset] = None):
        """Set up training components."""
        
        # Create data loaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )
        
        self.eval_dataloader = None
        if eval_dataset:
            self.eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=self.config.evaluation.eval_batch_size,
                shuffle=False,
                num_workers=self.config.data.num_workers,
                pin_memory=True
            )
            
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            eps=self.config.training.adam_epsilon
        )
        
        # Setup scheduler
        total_steps = max(1, len(self.train_dataloader)) * max(1, self.config.training.num_epochs)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Initialize W&B
        if self.wandb_logger:
            self.wandb_logger.init()
            self.wandb_logger.watch(self.model)
            
        logger.info("Training setup completed")
        
    def train(self, train_dataset: ConstitutionalDataset, eval_dataset: Optional[ConstitutionalDataset] = None):
        """Main training loop."""
        
        self.setup_training(train_dataset, eval_dataset)
        
        logger.info("Starting Constitutional AI training")
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch()
            
            # Evaluation phase (step-based or end-of-epoch)
            if self.eval_dataloader and (
                (self.current_step % max(1, self.config.evaluation.eval_steps) == 0)
                or (epoch == self.config.training.num_epochs - 1)
            ):
                eval_metrics = self._evaluate()
                
                # Log metrics
                metrics = {**train_metrics, **eval_metrics}
                self._log_metrics(metrics, epoch)
                
            # Save checkpoint (step-based or end-of-epoch)
            if (
                self.current_step % max(1, self.config.training.save_steps) == 0
                or (epoch == self.config.training.num_epochs - 1)
            ):
                self._save_checkpoint(epoch)
                
        logger.info("Training completed")
        
        # Final evaluation
        if self.eval_dataloader:
            final_metrics = self._evaluate()
            self._log_metrics(final_metrics, self.config.training.num_epochs)
            
        # Save final model
        self._save_model()
        
        if self.wandb_logger:
            self.wandb_logger.finish()
            
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            loss = self._compute_loss(batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm
            )
            
            # Update parameters
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.current_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]})
            
            # Log step metrics
            if self.current_step % self.config.training.logging_steps == 0:
                step_metrics = {
                    "train_loss": loss.item(),
                    "learning_rate": self.scheduler.get_last_lr()[0],
                    "step": self.current_step
                }
                
                if self.wandb_logger:
                    self.wandb_logger.log(step_metrics, step=self.current_step)
                    
                self.metrics_logger.log_metrics(step_metrics, self.current_step)
                
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "train_loss": avg_loss,
            "epoch": self.current_epoch
        }
        
    def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute training loss with constitutional weighting (optional)."""
        revised_input_ids = batch["revised_input_ids"]
        revised_attention_mask = batch["revised_attention_mask"]
        # Standard LM loss
        labels = revised_input_ids.clone()
        labels[labels == self.model.tokenizer.pad_token_id] = -100
        outputs = self.model(
            input_ids=revised_input_ids,
            attention_mask=revised_attention_mask,
            labels=labels
        )
        loss = outputs.loss

        # Constitutional adherence penalty (heuristic, no placeholders)
        penalty_weight = float(getattr(self.config.training, "constitutional_penalty_weight", 0.0))
        if penalty_weight > 0.0:
            # Decode current targets roughly and compute safety/compliance proxies
            with torch.no_grad():
                # Build decoded text for simple scoring. We only decode labels where not -100.
                decoded_texts = []
                for i in range(revised_input_ids.size(0)):
                    mask_valid = labels[i] != -100
                    token_ids = revised_input_ids[i][mask_valid]
                    text = self.model.tokenizer.decode(token_ids, skip_special_tokens=True)
                    decoded_texts.append(text)
            # Compute penalties: sum of (1 - safety_score) plus missing keyword adherence
            total_penalty = 0.0
            keywords = self.config.training.adherence_keywords or [
                "helpful", "harmless", "honest"
            ]
            for text in decoded_texts:
                safety_score = self.safety.evaluate_response(question="", response=text)
                lack_keywords = sum(1 for kw in keywords if kw.lower() not in text.lower())
                # Normalize: keywords fraction missing
                keyword_penalty = lack_keywords / max(1, len(keywords))
                total_penalty += (1.0 - safety_score) * 0.7 + keyword_penalty * 0.3
            avg_penalty = total_penalty / max(1, len(decoded_texts))
            loss = loss + penalty_weight * torch.tensor(avg_penalty, device=loss.device, dtype=loss.dtype)

        return loss
        
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate model."""
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                loss = self._compute_loss(batch)
                
                total_loss += loss.item()
                num_batches += 1
                
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "eval_loss": avg_loss,
            "eval_perplexity": torch.exp(torch.tensor(avg_loss)).item()
        }
        
    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """Log training metrics."""
        
        if self.wandb_logger:
            self.wandb_logger.log(metrics, step=step)
            
        self.metrics_logger.log_metrics(metrics, step)
        
        # Log to console
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Step {step} - {metric_str}")
        
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        
        checkpoint_dir = os.path.join(self.config.logging.output_dir, f"checkpoint-{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        
        # Save training state
        torch.save({
            "epoch": epoch,
            "step": self.current_step,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.config
        }, os.path.join(checkpoint_dir, "training_state.pt"))
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
        
    def _save_model(self):
        """Save final model."""
        
        model_dir = os.path.join(self.config.logging.output_dir, "final_model")
        os.makedirs(model_dir, exist_ok=True)
        
        self.model.save_pretrained(model_dir)
        
        logger.info(f"Final model saved to {model_dir}")
        
    def generate_critiques_and_revisions(self, dataset: ConstitutionalDataset) -> ConstitutionalDataset:
        """Generate critiques and revisions for dataset."""
        
        logger.info("Generating critiques and revisions")
        
        questions, responses = dataset.get_questions_and_responses()
        
        # Generate critiques
        critiques = []
        for question, response in tqdm(zip(questions, responses), desc="Generating critiques"):
            critique_output = self.critique_model.generate_critique(
                question=question,
                response=response,
                critique_type="constitutional"
            )
            critiques.append(critique_output.critique)
            
        # Generate revisions
        revisions = []
        for question, response, critique in tqdm(zip(questions, responses, critiques), desc="Generating revisions"):
            revision_output = self.revision_model.generate_revision(
                question=question,
                original_response=response,
                critique=critique,
                revision_type="constitutional"
            )
            revisions.append(revision_output.revised_response)
            
        # Add to dataset
        dataset.add_critiques_and_revisions(critiques, revisions)
        
        logger.info("Critiques and revisions generated")
        
        return dataset