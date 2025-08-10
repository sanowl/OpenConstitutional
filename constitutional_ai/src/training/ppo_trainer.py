"""
PPO Trainer for Constitutional AI RLAIF (Phase 2).
Implements Proximal Policy Optimization with constitutional reward models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from tqdm import tqdm
import os
from dataclasses import dataclass
from collections import defaultdict

from ..models.constitutional_model import ConstitutionalAIModel
from ..models.reward_model import RewardModel
from ..models.critic_model import CriticModel
from ..utils.config import Config, PPOConfig
from ..utils.logging import get_logger, WandBLogger, MetricsLogger

logger = get_logger(__name__)


@dataclass
class PPOBatch:
    """Batch data for PPO training."""
    queries: List[str]
    responses: List[str]
    input_ids: torch.Tensor          # [B, T]
    attention_mask: torch.Tensor     # [B, T]
    gen_mask: torch.Tensor           # [B, T-1] masks target positions belonging to generated region
    old_token_logprobs: torch.Tensor # [B, T-1]
    advantages_t: torch.Tensor       # [B, T-1]
    returns_t: torch.Tensor          # [B, T-1]


@dataclass
class PPOStats:
    """Statistics from PPO training step."""
    policy_loss: float
    value_loss: float
    entropy_loss: float
    total_loss: float
    kl_divergence: float
    clipfrac: float
    explained_variance: float
    approx_kl: float
    ratio_mean: float
    ratio_std: float


class PPODataset(Dataset):
    """Dataset for PPO training."""
    
    def __init__(self, queries: List[str], responses: List[str], rewards: List[float]):
        self.queries = queries
        self.responses = responses
        self.rewards = rewards
        
    def __len__(self):
        return len(self.queries)
        
    def __getitem__(self, idx):
        return {
            'query': self.queries[idx],
            'response': self.responses[idx],
            'reward': self.rewards[idx]
        }


class PPOTrainer:
    """PPO Trainer for Constitutional AI."""
    
    def __init__(
        self,
        model: ConstitutionalAIModel,
        ref_model: ConstitutionalAIModel,
        reward_model: RewardModel,
        config: Config
    ):
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.config = config
        self.ppo_config = config.ppo
        
        self.device = torch.device(config.model.device)
        
        # Move models to device
        self.model.to(self.device)
        self.ref_model.to(self.device)
        self.reward_model.to(self.device)
        self.critic = CriticModel(config).to(self.device)
        
        # Set reference and reward models to eval; critic is trainable
        self.ref_model.eval()
        self.reward_model.eval()
        for param in self.reward_model.parameters():
            param.requires_grad = False
        
        # Initialize optimizers
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.ppo_config.learning_rate,
            eps=1e-5
        )
        self.value_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.ppo_config.learning_rate,
            eps=1e-5
        )
        
        # KL controller for adaptive KL penalty
        self.kl_ctl = AdaptiveKLController(
            init_kl_coef=self.ppo_config.init_kl_coef,
            target_kl=self.ppo_config.target_kl,
            horizon=10000
        )
        
        # Initialize logging
        self.wandb_logger = WandBLogger(
            project=config.logging.wandb_project,
            entity=config.logging.wandb_entity,
            name=f"{config.logging.run_name}_ppo" if config.logging.run_name else None,
            config=config.__dict__
        ) if config.logging.use_wandb else None
        
        self.metrics_logger = MetricsLogger(
            os.path.join(config.logging.output_dir, "ppo_metrics")
        )
        
        # Training state
        self.step = 0
        self.epoch = 0
        
        logger.info("Initialized PPO Trainer")
        
    def train(
        self,
        queries: List[str],
        num_epochs: int = 1,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Main PPO training loop."""
        
        if batch_size is None:
            batch_size = self.ppo_config.batch_size
            
        if self.wandb_logger:
            self.wandb_logger.init()
            
        logger.info(f"Starting PPO training for {num_epochs} epochs")
        
        training_stats = defaultdict(list)
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_stats = self._train_epoch(queries, batch_size)
            
            # Log epoch statistics
            for key, value in epoch_stats.items():
                training_stats[key].append(value)
                
            # Log to wandb and metrics logger
            if self.wandb_logger:
                self.wandb_logger.log(epoch_stats, step=self.step)
            self.metrics_logger.log_metrics(epoch_stats, self.step)
            
            logger.info(f"Epoch {epoch} completed: "
                       f"policy_loss={epoch_stats['policy_loss']:.4f}, "
                       f"value_loss={epoch_stats['value_loss']:.4f}, "
                       f"kl_div={epoch_stats['kl_divergence']:.4f}")
                       
        if self.wandb_logger:
            self.wandb_logger.finish()
            
        return dict(training_stats)
        
    def _train_epoch(self, queries: List[str], batch_size: int) -> Dict[str, float]:
        """Train for one epoch."""
        
        # Generate responses and collect rollouts
        rollouts = self._collect_rollouts(queries, batch_size)
        
        # Train on collected rollouts
        epoch_stats = defaultdict(list)
        
        for ppo_epoch in range(self.ppo_config.ppo_epochs):
            # Shuffle rollouts
            indices = torch.randperm(len(rollouts['queries']))
            
            # Process mini-batches
            for i in range(0, len(rollouts['queries']), self.ppo_config.mini_batch_size):
                end_idx = min(i + self.ppo_config.mini_batch_size, len(rollouts['queries']))
                mini_batch_indices = indices[i:end_idx]
                
                # Create mini-batch
                mini_batch = self._create_mini_batch(rollouts, mini_batch_indices)
                
                # PPO update step
                stats = self._ppo_step(mini_batch)
                
                # Accumulate statistics
                for key, value in stats.__dict__.items():
                    epoch_stats[key].append(value)
                    
                self.step += 1
                
        # Average statistics over mini-batches
        return {key: np.mean(values) for key, values in epoch_stats.items()}
        
    def _collect_rollouts(self, queries: List[str], batch_size: int) -> Dict[str, List]:
        """Collect rollouts from the current policy."""
        
        rollouts = {
            'queries': [],
            'responses': [],
            'input_ids': [],
            'attention_mask': [],
            'gen_mask': [],
            'old_token_logprobs': [],
            'advantages_t': [],
            'returns_t': [],
        }
        
        self.model.eval()
        
        with torch.no_grad():
            for i in tqdm(range(0, len(queries), batch_size), desc="Collecting rollouts"):
                batch_queries = queries[i:i + batch_size]
                
                # Generate responses
                responses = self._generate_responses(batch_queries)
                
                # Tokenize full texts
                full_texts = [f"{q} {r}" for q, r in zip(batch_queries, responses)]
                tok = self.model.tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.config.model.max_length).to(self.device)
                q_tok = self.model.tokenizer(batch_queries, return_tensors="pt", padding=True, truncation=True, max_length=self.config.model.max_length).to(self.device)
                q_lens = q_tok.attention_mask.sum(dim=1)
                B, T = tok.input_ids.shape
                # Build gen_mask for token predictions (aligns with positions 1..T-1)
                gen_mask = torch.zeros((B, T-1), device=self.device, dtype=tok.attention_mask.dtype)
                for i in range(B):
                    start = int(torch.clamp(q_lens[i]-1, min=0, max=T-2))
                    gen_mask[i, start:] = 1
                
                # Old token logprobs
                with torch.no_grad():
                    outputs = self.model(**tok)
                    logits = outputs.logits
                    log_probs = F.log_softmax(logits, dim=-1)
                    token_log_probs = torch.gather(log_probs[:, :-1], dim=-1, index=tok.input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                old_token_logprobs = token_log_probs
                
                # Sequence-level reward from reward model
                with torch.no_grad():
                    rewards_seq = self._compute_rewards(batch_queries, responses).float()  # [B]
                # Per-token rewards: terminal-only shaping at last generated token
                rewards_t = torch.zeros_like(gen_mask, dtype=torch.float32)
                for i in range(B):
                    # last generated index where gen_mask is 1
                    gen_positions = (gen_mask[i] > 0).nonzero(as_tuple=False)
                    if gen_positions.numel() > 0:
                        last_idx = int(gen_positions[-1].item())
                        rewards_t[i, last_idx] = rewards_seq[i]
                
                # Values from critic per-token (aligned to next-token predictions)
                with torch.no_grad():
                    token_values = self.critic(tok.input_ids, tok.attention_mask)  # [B, T]
                    values_t = token_values[:, 1:]  # [B, T-1]
                
                # Compute GAE advantages and returns over generated region
                advantages_t, returns_t = self._compute_gae(values_t, rewards_t, gen_mask)
                
                # Store rollouts
                rollouts['queries'].extend(batch_queries)
                rollouts['responses'].extend(responses)
                rollouts['input_ids'].append(tok.input_ids.cpu())
                rollouts['attention_mask'].append(tok.attention_mask.cpu())
                rollouts['gen_mask'].append(gen_mask.cpu())
                rollouts['old_token_logprobs'].append(old_token_logprobs.cpu())
                rollouts['advantages_t'].append(advantages_t.cpu())
                rollouts['returns_t'].append(returns_t.cpu())
                
        return rollouts

    def _compute_gae(
        self,
        values_t: torch.Tensor,         # [B, T-1]
        rewards_t: torch.Tensor,        # [B, T-1]
        gen_mask: torch.Tensor,         # [B, T-1] (0/1)
        gamma: Optional[float] = None,
        lam: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute token-level GAE(λ) advantages and returns over generated region.
        values_t are V(s_t) aligned to predicting token at t+1; bootstrap next value from V(s_{t+1}).
        """
        gamma = self.ppo_config.gamma if gamma is None else gamma
        lam = self.ppo_config.lam if lam is None else lam
        B, Tm1 = values_t.shape
        advantages = torch.zeros_like(values_t)
        last_gae = torch.zeros((B,), device=values_t.device, dtype=values_t.dtype)
        # Append zero at end for V_{T}
        next_values = torch.cat([values_t[:, 1:], torch.zeros(B, 1, device=values_t.device, dtype=values_t.dtype)], dim=1)
        for t in reversed(range(Tm1)):
            mask_t = gen_mask[:, t].float()
            delta = rewards_t[:, t] + gamma * next_values[:, t] * mask_t - values_t[:, t]
            last_gae = delta + gamma * lam * mask_t * last_gae
            advantages[:, t] = last_gae * mask_t
        returns = advantages + values_t
        # Normalize advantages across valid tokens
        valid = gen_mask.bool()
        if valid.any():
            mean = advantages[valid].mean()
            std = advantages[valid].std(unbiased=False).clamp_min(1e-8)
            advantages = (advantages - mean) / std
        return advantages, returns
        
    def _generate_responses(self, queries: List[str]) -> List[str]:
        """Generate responses for queries."""
        
        responses = []
        for query in queries:
            # Generate response using the model
            outputs = self.model.generate(
                prompt=query,
                max_length=self.config.model.max_length,
                temperature=self.config.model.temperature,
                top_p=self.config.model.top_p,
                do_sample=self.config.model.do_sample,
                num_return_sequences=1
            )
            responses.append(outputs[0].text)
            
        return responses
        
    def _compute_rewards(self, queries: List[str], responses: List[str]) -> torch.Tensor:
        """Compute rewards using the reward model."""
        
        # Combine queries and responses
        full_texts = [f"{query} {response}" for query, response in zip(queries, responses)]
        
        # Get rewards from reward model
        rewards = self.reward_model.compute_reward(full_texts)
        
        return rewards
        
    def _compute_logprobs_and_values(
        self, queries: List[str], responses: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities and values for query-response pairs."""
        
        # Tokenize inputs
        full_texts = [f"{query} {response}" for query, response in zip(queries, responses)]
        
        inputs = self.model.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.model.max_length
        ).to(self.device)

        # Tokenize queries separately to locate generated region
        q_tok = self.model.tokenizer(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.model.max_length
        ).to(self.device)
        q_lens = q_tok.attention_mask.sum(dim=1)  # [B]
        
        # Forward pass through model
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Compute log probabilities
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get log probabilities for actual tokens
        token_log_probs = torch.gather(
            log_probs[:, :-1], 
            dim=-1, 
            index=inputs.input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        # Sum log probabilities over generated region only, mask pads
        attention_mask = inputs.attention_mask[:, 1:]
        B, Tm1 = attention_mask.shape
        gen_mask = torch.zeros_like(attention_mask)
        for i in range(B):
            # positions >= q_lens[i] are generated (token_log_probs aligns to positions 1..T-1)
            start = int(torch.clamp(q_lens[i]-1, min=0, max=Tm1-1))
            gen_mask[i, start:] = 1
        eff_mask = attention_mask * gen_mask
        sequence_log_probs = (token_log_probs * eff_mask).sum(dim=-1)
        
        # Compute sequence-level values from critic by averaging per-token values over generated tokens
        token_values = self.critic(inputs.input_ids, inputs.attention_mask)  # [B, T]
        gen_mask_full = torch.zeros_like(inputs.attention_mask)
        for i in range(B):
            start = int(q_lens[i])
            gen_mask_full[i, start:] = 1
        vmask = inputs.attention_mask * gen_mask_full
        values = (token_values * vmask).sum(dim=1) / torch.clamp_min(vmask.sum(dim=1), 1)
        
        return sequence_log_probs, values
        
    def _compute_advantages_and_returns(
        self, rewards: torch.Tensor, values: torch.Tensor, gamma: Optional[float] = None, lam: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages and returns using GAE(lambda) over sequence-level rewards.
        Note: current pipeline produces one reward/value per query-response pair (no per-timestep rollout),
        so we fall back to delta-based advantages while keeping normalization.
        """
        gamma = self.ppo_config.gamma if gamma is None else gamma
        lam = self.ppo_config.lam if lam is None else lam
        # With one-step episodes, GAE reduces to A = r - V(s)
        returns = rewards
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return advantages, returns
        
    def _create_mini_batch(
        self, rollouts: Dict[str, List], indices: torch.Tensor
    ) -> PPOBatch:
        """Create a mini-batch from rollouts."""
        
        # Extract data for selected indices
        queries = [rollouts['queries'][int(i)] for i in indices]
        responses = [rollouts['responses'][int(i)] for i in indices]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [rollouts['input_ids'][int(i)].squeeze(0) for i in indices], batch_first=True, padding_value=self.model.tokenizer.pad_token_id
        ).to(self.device)
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [rollouts['attention_mask'][int(i)].squeeze(0) for i in indices], batch_first=True, padding_value=0
        ).to(self.device)
        # T-1 aligned tensors
        old_token_logprobs = torch.nn.utils.rnn.pad_sequence(
            [rollouts['old_token_logprobs'][int(i)] for i in indices], batch_first=True, padding_value=0.0
        ).to(self.device)
        advantages_t = torch.nn.utils.rnn.pad_sequence(
            [rollouts['advantages_t'][int(i)] for i in indices], batch_first=True, padding_value=0.0
        ).to(self.device)
        returns_t = torch.nn.utils.rnn.pad_sequence(
            [rollouts['returns_t'][int(i)] for i in indices], batch_first=True, padding_value=0.0
        ).to(self.device)
        gen_mask = torch.nn.utils.rnn.pad_sequence(
            [rollouts['gen_mask'][int(i)] for i in indices], batch_first=True, padding_value=0
        ).to(self.device)

        return PPOBatch(
            queries=queries,
            responses=responses,
            input_ids=input_ids,
            attention_mask=attention_mask,
            gen_mask=gen_mask,
            old_token_logprobs=old_token_logprobs,
            advantages_t=advantages_t,
            returns_t=returns_t,
        )
        
    def _ppo_step(self, batch: PPOBatch) -> PPOStats:
        """Perform one PPO update step."""
        
        self.model.train()
        
        # Forward pass
        outputs = self.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask
        )
        
        # Compute current log probabilities
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get log probabilities for actual tokens
        token_log_probs = torch.gather(
            log_probs[:, :-1], 
            dim=-1, 
            index=batch.input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        # Per-token PPO: restrict to generated region
        attention_mask = batch.attention_mask[:, 1:]
        eff_mask = attention_mask * batch.gen_mask
        current_logprobs_t = token_log_probs * eff_mask
        old_logprobs_t = batch.old_token_logprobs * eff_mask
        # Ratio per token
        ratio_t = torch.exp(current_logprobs_t - old_logprobs_t.detach()) * eff_mask
        # Policy loss per token (clipped)
        policy_loss_1 = batch.advantages_t * ratio_t
        policy_loss_2 = batch.advantages_t * torch.clamp(
            ratio_t, 1 - self.ppo_config.cliprange, 1 + self.ppo_config.cliprange
        )
        valid = eff_mask.bool()
        if valid.any():
            policy_loss = -torch.mean(torch.where(valid, torch.min(policy_loss_1, policy_loss_2), torch.zeros_like(policy_loss_1)))
        else:
            policy_loss = torch.tensor(0.0, device=self.device)
        
        # Compute values using critic (average over generated tokens)
        token_values = self.critic(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask
        )  # [B, T]
        vmask = batch.attention_mask * torch.cat([batch.gen_mask[:, :1]*0, batch.gen_mask], dim=1)
        values = (token_values * vmask).sum(dim=1) / torch.clamp_min(vmask.sum(dim=1), 1)
        
        # Compute value loss
        returns_seq = (batch.returns_t * eff_mask).sum(dim=1) / torch.clamp_min(eff_mask.sum(dim=1), 1)
        value_loss = F.mse_loss(values, returns_seq)
        
        # Compute entropy loss
        # Entropy over next-token distribution; average over non-pad tokens
        next_log_probs = log_probs[:, :-1]
        entropy_tok = -(next_log_probs * torch.exp(next_log_probs)).sum(dim=-1)
        nonpad = eff_mask
        entropy = (entropy_tok * nonpad).sum() / torch.clamp_min(nonpad.sum(), 1)
        entropy_loss = -entropy  # We want to maximize entropy
        
        # KL divergence with reference model (mask padded positions)
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask
            )
            ref_log_probs = F.log_softmax(ref_outputs.logits, dim=-1)
        # compute token-wise kl on next-token positions
        logp = log_probs[:, :-1]
        logp_ref = ref_log_probs[:, :-1]
        token_kl = torch.exp(logp) * (logp - logp_ref)
        # mask pad of target tokens
        nonpad = eff_mask.unsqueeze(-1)
        token_kl = token_kl * nonpad
        denom = nonpad.sum()
        kl_div = token_kl.sum() / torch.clamp_min(denom, 1)
        
        # Separate updates: policy (actor) and value (critic)
        policy_total = policy_loss + 0.01 * entropy_loss + self.kl_ctl.value * kl_div
        self.optimizer.zero_grad()
        policy_total.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.ppo_config.max_grad_norm)
        self.optimizer.step()

        # Critic update (MSE toward returns)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.ppo_config.max_grad_norm)
        self.value_optimizer.step()
        
        # Update KL controller
        self.kl_ctl.update(kl_div.item(), batch.input_ids.size(0))
        
        # Compute statistics
        with torch.no_grad():
            clipfrac = ((ratio_t[valid] - 1.0).abs() > self.ppo_config.cliprange).float().mean() if valid.any() else torch.tensor(0.0)
            var_returns = returns_seq.var(unbiased=False)
            explained_variance = 1 - (returns_seq - values).var(unbiased=False) / torch.clamp_min(var_returns, 1e-8)
            approx_kl = (old_logprobs_t[valid] - current_logprobs_t[valid]).mean() if valid.any() else torch.tensor(0.0)
            
        return PPOStats(
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            entropy_loss=entropy_loss.item(),
            total_loss=(policy_total + self.ppo_config.vf_coef * value_loss).item(),
            kl_divergence=kl_div.item(),
            clipfrac=clipfrac.item(),
            explained_variance=explained_variance.item(),
            approx_kl=approx_kl.item(),
            ratio_mean=(ratio_t[valid].mean().item() if valid.any() else 0.0),
            ratio_std=(ratio_t[valid].std().item() if valid.any() else 0.0)
        )
        
    def save_checkpoint(self, save_path: str):
        """Save PPO training checkpoint."""
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'kl_ctl_state': self.kl_ctl.__dict__,
            'step': self.step,
            'epoch': self.epoch,
            'config': self.config
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"PPO checkpoint saved to {save_path}")
        
    def load_checkpoint(self, load_path: str):
        """Load PPO training checkpoint."""
        
        checkpoint = torch.load(load_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.kl_ctl.__dict__.update(checkpoint['kl_ctl_state'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        
        logger.info(f"PPO checkpoint loaded from {load_path}")


class AdaptiveKLController:
    """Adaptive KL divergence controller for PPO."""
    
    def __init__(self, init_kl_coef: float, target_kl: float, horizon: int):
        self.value = init_kl_coef
        self.target_kl = target_kl
        self.horizon = horizon
        self.proportional_error = 0.0
        self.integral_error = 0.0
        
    def update(self, current_kl: float, batch_size: int):
        """Update KL coefficient based on current KL divergence."""
        
        # Proportional error
        self.proportional_error = current_kl - self.target_kl
        
        # Integral error
        self.integral_error += self.proportional_error * batch_size / self.horizon
        
        # Update KL coefficient
        self.value = max(
            1e-6,  # Minimum value
            self.value + self.proportional_error * 0.1 + self.integral_error * 0.01
        )
        
    def reset(self):
        """Reset the controller."""
        self.proportional_error = 0.0
        self.integral_error = 0.0