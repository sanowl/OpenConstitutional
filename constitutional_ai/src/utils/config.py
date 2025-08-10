"""Configuration utilities for Constitutional AI."""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model settings."""
    model_name: str = "microsoft/DialoGPT-medium"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    pad_token_id: int = 50256
    eos_token_id: int = 50256
    device: str = "cuda"
    dtype: str = "float16"  # will fallback to float32 on CPU
    use_principle_tokens: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training settings."""
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    max_steps: int = -1
    warmup_steps: int = 100
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    seed: int = 42
    fp16: bool = True
    deepspeed_config: Optional[str] = None
    use_peft: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    # Constitutional adherence regularization
    constitutional_penalty_weight: float = 0.2
    adherence_keywords: Optional[List[str]] = None


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    batch_size: int = 4
    mini_batch_size: int = 1
    ppo_epochs: int = 4
    learning_rate: float = 1.4e-5
    init_kl_coef: float = 0.02
    target_kl: float = 0.1
    adap_kl_ctrl: bool = True
    gamma: float = 1.0
    lam: float = 0.95
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    max_grad_norm: float = 0.5
    forward_batch_size: int = 4
    gradient_accumulation_steps: int = 1


@dataclass
class DataConfig:
    """Configuration for data processing."""
    dataset_name: str = "Anthropic/hh-rlhf"
    dataset_subset: Optional[str] = None
    train_split: str = "train"
    test_split: str = "test"
    max_samples: int = -1
    max_input_length: int = 512
    max_output_length: int = 512
    shuffle: bool = True
    num_workers: int = 4
    preprocessing_num_workers: int = 4
    allow_dummy_data: bool = False


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    eval_batch_size: int = 8
    eval_steps: int = 100
    save_eval_results: bool = True
    eval_output_dir: str = "eval_results"
    metrics: List[str] = field(default_factory=lambda: ["helpfulness", "harmlessness", "honesty"])
    human_eval: bool = False
    safety_eval: bool = True
    use_real_metrics: bool = False
    bert_score_model_type: str = "roberta-large"
    perplexity_model_name: str = "gpt2"


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    project_name: str = "constitutional-ai"
    run_name: Optional[str] = None
    output_dir: str = "outputs"
    logging_dir: str = "logs"
    log_level: str = "INFO"
    use_wandb: bool = False
    wandb_project: str = "constitutional-ai"
    wandb_entity: Optional[str] = None
    save_model: bool = True
    save_optimizer: bool = False


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Constitutional AI specific settings
    constitutional_principles: List[str] = field(default_factory=lambda: [
        "Choose the response that is more helpful, honest, and harmless."
    ])
    critique_model: str = "microsoft/DialoGPT-medium"
    revision_model: str = "microsoft/DialoGPT-medium"
    preference_model: str = "microsoft/DialoGPT-medium"
    num_critique_rounds: int = 1
    num_revision_rounds: int = 1
    use_self_critique: bool = True
    critique_temperature: float = 0.7
    revision_temperature: float = 0.7
    preference_temperature: float = 0.7
    # Constitutional weighting
    principle_weights: Dict[str, float] = field(default_factory=lambda: {
        "helpfulness": 1.0,
        "harmlessness": 1.0,
        "honesty": 1.0
    })


def load_config(config_path: str) -> Config:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Convert nested dictionaries to dataclass instances
    model_config = ModelConfig(**config_dict.get('model', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    ppo_config = PPOConfig(**config_dict.get('ppo', {}))
    data_config = DataConfig(**config_dict.get('data', {}))
    evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
    logging_config = LoggingConfig(**config_dict.get('logging', {}))
    
    # Extract other fields
    other_fields = {k: v for k, v in config_dict.items() 
                   if k not in ['model', 'training', 'ppo', 'data', 'evaluation', 'logging']}
    
    return Config(
        model=model_config,
        training=training_config,
        ppo=ppo_config,
        data=data_config,
        evaluation=evaluation_config,
        logging=logging_config,
        **other_fields
    )


def save_config(config: Config, config_path: str) -> None:
    """Save configuration to JSON file."""
    config_dict = {
        'model': config.model.__dict__,
        'training': config.training.__dict__,
        'ppo': config.ppo.__dict__,
        'data': config.data.__dict__,
        'evaluation': config.evaluation.__dict__,
        'logging': config.logging.__dict__,
        'constitutional_principles': config.constitutional_principles,
        'critique_model': config.critique_model,
        'revision_model': config.revision_model,
        'preference_model': config.preference_model,
        'num_critique_rounds': config.num_critique_rounds,
        'num_revision_rounds': config.num_revision_rounds,
        'use_self_critique': config.use_self_critique,
        'critique_temperature': config.critique_temperature,
        'revision_temperature': config.revision_temperature,
        'preference_temperature': config.preference_temperature,
    }
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()