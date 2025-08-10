"""
Reward model trainer for Constitutional AI.
Trains reward models on AI preference data.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from tqdm import tqdm
import os

from ..models.reward_model import RewardModel
from ..models.reward_cross_encoder import CrossEncoderRewardModel
import torch.nn.functional as F
from ..data_processing.preference_dataset import PreferenceDataset
from ..utils.config import Config
from ..utils.logging import get_logger, MetricsLogger
from scipy.stats import spearmanr

logger = get_logger(__name__)


class RewardTrainer:
    """Trainer for reward models."""
    
    def __init__(self, reward_model: RewardModel, config: Config):
        self.reward_model = reward_model
        self.config = config
        self.device = torch.device(config.model.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.reward_model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            eps=config.training.adam_epsilon
        )
        
        # Initialize metrics logger
        self.metrics_logger = MetricsLogger(
            os.path.join(config.logging.output_dir, "reward_training_metrics")
        )
        
        # Training state
        self.step = 0
        self.epoch = 0
        
        logger.info("Initialized Reward Trainer")
        
    def train(
        self,
        train_dataset: PreferenceDataset,
        num_epochs: int = 3,
        batch_size: int = 4,
        eval_dataset: Optional[PreferenceDataset] = None
    ) -> Dict[str, Any]:
        """Train reward model on preference data."""
        
        # Create data loader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )
        
        eval_dataloader = None
        if eval_dataset:
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.config.data.num_workers,
                pin_memory=True
            )
            
        # Setup scheduler
        total_steps = len(train_dataloader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Starting reward model training for {num_epochs} epochs")
        
        training_stats = {
            "train_losses": [],
            "eval_losses": [],
            "accuracies": [],
            "learning_rates": [],
            "per_principle_margins_mean": {"helpfulness": [], "harmlessness": [], "honesty": []},
        }
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training phase
            train_loss = self._train_epoch(train_dataloader)
            training_stats["train_losses"].append(train_loss)
            
            # Evaluation phase and support metrics
            if eval_dataloader:
                eval_loss, accuracy = self._evaluate_epoch(eval_dataloader)
                training_stats["eval_losses"].append(eval_loss)
                training_stats["accuracies"].append(accuracy)
                
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                           f"eval_loss={eval_loss:.4f}, accuracy={accuracy:.4f}")
            else:
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}")
                
            # Log metrics
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": self.scheduler.get_last_lr()[0]
            }
            
            if eval_dataloader:
                epoch_metrics.update({
                    "eval_loss": eval_loss,
                    "accuracy": accuracy
                })
                
            self.metrics_logger.log_metrics(epoch_metrics, self.step)
            
            # Save checkpoint (step-based or end-of-epoch)
            if (
                self.step % max(1, self.config.training.save_steps) == 0
                or (epoch == num_epochs - 1)
            ):
                self._save_checkpoint(epoch)
                
        logger.info("Reward model training completed")
        
        return training_stats
        
    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        
        self.reward_model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {self.epoch}")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            loss = self._compute_loss(batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.reward_model.parameters(),
                self.config.training.max_grad_norm
            )
            
            # Update parameters
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss.item(),
                "lr": self.scheduler.get_last_lr()[0]
            })
            
            # Log step metrics
            if self.step % self.config.training.logging_steps == 0:
                step_metrics = {
                    "step_loss": loss.item(),
                    "learning_rate": self.scheduler.get_last_lr()[0],
                    "step": self.step
                }
                self.metrics_logger.log_metrics(step_metrics, self.step)
                
        return total_loss / num_batches if num_batches > 0 else 0.0
        
    def _evaluate_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate for one epoch."""
        
        self.reward_model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        # Aggregate per-principle margins if available
        per_principle_margins = {"helpfulness": [], "harmlessness": [], "honesty": []}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating Epoch {self.epoch}"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                loss = self._compute_loss(batch)
                total_loss += loss.item()
                
                # Compute accuracy
                accuracy_batch = self._compute_accuracy(batch)
                correct_predictions += accuracy_batch * batch["chosen_input_ids"].size(0)
                total_predictions += batch["chosen_input_ids"].size(0)

                # Per-principle margins for separate-encoder path
                if not isinstance(self.reward_model, CrossEncoderRewardModel):
                    chosen_inputs = {
                        "input_ids": batch["chosen_input_ids"],
                        "attention_mask": batch["chosen_attention_mask"]
                    }
                    rejected_inputs = {
                        "input_ids": batch["rejected_input_ids"],
                        "attention_mask": batch["rejected_attention_mask"]
                    }
                    out_c = self.reward_model(**chosen_inputs)
                    out_r = self.reward_model(**rejected_inputs)
                    for key, tensor in ("helpfulness", out_c.helpfulness), ("harmlessness", out_c.harmlessness), ("honesty", out_c.honesty):
                        if tensor is not None:
                            margin = (tensor - getattr(out_r, key)).detach().cpu().tolist()
                            per_principle_margins[key].extend(margin)
                
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        # Log mean per-principle margins if collected
        for key, arr in per_principle_margins.items():
            if arr:
                self.metrics_logger.log_metric(f"eval_margin_{key}_mean", float(np.mean(arr)), self.step)
        
        return avg_loss, accuracy
        
    def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute pairwise ranking loss."""
        # Cross-encoder path
        if isinstance(self.reward_model, CrossEncoderRewardModel):
            # Use raw strings provided by PreferenceDataset
            questions = batch.get("question")
            if questions is None:
                raise ValueError("PreferenceDataset batch missing 'question' strings needed for cross-encoder.")
            responses_a = batch.get("chosen_response")
            responses_b = batch.get("rejected_response")
            if responses_a is None or responses_b is None:
                raise ValueError("PreferenceDataset batch missing response strings for cross-encoder.")
            labels = torch.ones(len(responses_a), device=self.device)
            loss = self.reward_model.compute_pairwise_loss(questions, responses_a, responses_b, labels)
            return loss
        
        # Default separate-encoder path
        chosen_inputs = {
            "input_ids": batch["chosen_input_ids"],
            "attention_mask": batch["chosen_attention_mask"]
        }
        rejected_inputs = {
            "input_ids": batch["rejected_input_ids"],
            "attention_mask": batch["rejected_attention_mask"]
        }
        chosen_outputs = self.reward_model(**chosen_inputs)
        rejected_outputs = self.reward_model(**rejected_inputs)
        margin = chosen_outputs.rewards - rejected_outputs.rewards
        loss = F.binary_cross_entropy_with_logits(margin, torch.ones_like(margin))
        return loss
        
    def _compute_accuracy(self, batch: Dict[str, Any]) -> float:
        """Compute accuracy of preference predictions."""
        # Cross-encoder path
        if isinstance(self.reward_model, CrossEncoderRewardModel):
            questions = batch.get("question")
            responses_a = batch.get("chosen_response")
            responses_b = batch.get("rejected_response")
            if questions is None or responses_a is None or responses_b is None:
                raise ValueError("PreferenceDataset batch missing required string fields for cross-encoder accuracy computation.")
            logits = self.reward_model.forward(questions, responses_a, responses_b).margin_logits
            preds = (logits > 0).float()
            # Label is 1 for chosen preferred (A)
            labels = torch.ones_like(preds)
            return (preds == labels).float().mean().item()

        # Separate-encoder path
        chosen_inputs = {
            "input_ids": batch["chosen_input_ids"],
            "attention_mask": batch["chosen_attention_mask"]
        }
        rejected_inputs = {
            "input_ids": batch["rejected_input_ids"],
            "attention_mask": batch["rejected_attention_mask"]
        }
        chosen_outputs = self.reward_model(**chosen_inputs)
        rejected_outputs = self.reward_model(**rejected_inputs)
        correct = (chosen_outputs.rewards > rejected_outputs.rewards).float()
        return correct.mean().item()
        
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        
        checkpoint_dir = os.path.join(
            self.config.logging.output_dir, f"reward_checkpoint_{epoch}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        self.reward_model.save_pretrained(checkpoint_dir)
        
        # Save training state
        torch.save({
            "epoch": epoch,
            "step": self.step,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.config
        }, os.path.join(checkpoint_dir, "training_state.pt"))
        
        logger.info(f"Reward model checkpoint saved to {checkpoint_dir}")
        
    def evaluate(self, eval_dataset: PreferenceDataset) -> Dict[str, float]:
        """Evaluate reward model on a validation preference dataset with support metrics."""
        logger.info("Evaluating reward model")
        dataloader = DataLoader(
            eval_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
        )

        self.reward_model.eval()
        all_margins: List[float] = []
        all_conf: List[float] = []
        correct = 0
        total = 0
        per_principle_margins = {"helpfulness": [], "harmlessness": [], "honesty": []}

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating RewardModel"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                if isinstance(self.reward_model, CrossEncoderRewardModel):
                    questions = batch.get("question")
                    responses_a = batch.get("chosen_response")
                    responses_b = batch.get("rejected_response")
                    logits = self.reward_model.forward(questions, responses_a, responses_b).margin_logits
                    margins = logits.detach().cpu().tolist()
                else:
                    chosen_inputs = {"input_ids": batch["chosen_input_ids"], "attention_mask": batch["chosen_attention_mask"]}
                    rejected_inputs = {"input_ids": batch["rejected_input_ids"], "attention_mask": batch["rejected_attention_mask"]}
                    out_c = self.reward_model(**chosen_inputs)
                    out_r = self.reward_model(**rejected_inputs)
                    margins = (out_c.rewards - out_r.rewards).detach().cpu().tolist()
                    # collect per-principle margins if exposed
                    for key, tensor in ("helpfulness", out_c.helpfulness), ("harmlessness", out_c.harmlessness), ("honesty", out_c.honesty):
                        if tensor is not None:
                            per_principle_margins[key].extend((tensor - getattr(out_r, key)).detach().cpu().tolist())

                all_margins.extend(margins)
                # chosen preferred label is 1, so correct when margin>0
                preds = (torch.tensor(margins) > 0).float()
                correct += preds.sum().item()
                total += preds.numel()
                # confidence correlation support
                conf = batch.get("confidence")
                if isinstance(conf, torch.Tensor):
                    all_conf.extend(conf.detach().cpu().tolist())

        accuracy = float(correct / total) if total else 0.0
        mean_margin = float(np.mean(all_margins)) if all_margins else 0.0
        std_margin = float(np.std(all_margins)) if all_margins else 0.0
        # Spearman correlation between margins and provided confidences (if any)
        if all_margins and all_conf and len(all_margins) == len(all_conf) and len(all_margins) > 1:
            try:
                spearman = spearmanr(all_margins, all_conf).correlation
                spearman = float(spearman) if spearman is not None else None
            except Exception:
                spearman = None
        else:
            spearman = None

        metrics: Dict[str, Any] = {
            "model_parameters": sum(p.numel() for p in self.reward_model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.reward_model.parameters() if p.requires_grad),
            "accuracy": accuracy,
            "mean_margin": mean_margin,
            "std_margin": std_margin,
            "spearman_margin_confidence": spearman,
        }
        # Per-principle margin means when available
        for key, arr in per_principle_margins.items():
            if arr:
                metrics[f"mean_margin_{key}"] = float(np.mean(arr))

        logger.info(f"Reward eval: acc={accuracy:.4f}, margin={mean_margin:.4f}±{std_margin:.4f}")
        return metrics