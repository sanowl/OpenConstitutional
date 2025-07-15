"""Logging utilities for Constitutional AI."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import wandb


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs"
) -> None:
    """Set up logging configuration."""
    
    # Create log directory if it doesn't exist
    if log_file:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)
    else:
        log_path = None
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([] if log_path is None else [logging.FileHandler(log_path)])
        ]
    )


def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)


class WandBLogger:
    """Weights & Biases logger wrapper."""
    
    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[dict] = None,
        tags: Optional[list] = None,
        group: Optional[str] = None
    ):
        self.project = project
        self.entity = entity
        self.name = name or f"constitutional-ai-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.config = config or {}
        self.tags = tags or []
        self.group = group
        self.run = None
        
    def init(self) -> None:
        """Initialize W&B run."""
        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.name,
            config=self.config,
            tags=self.tags,
            group=self.group
        )
        
    def log(self, data: dict, step: Optional[int] = None) -> None:
        """Log data to W&B."""
        if self.run:
            wandb.log(data, step=step)
            
    def log_artifact(self, artifact_path: str, artifact_type: str, name: str) -> None:
        """Log artifact to W&B."""
        if self.run:
            artifact = wandb.Artifact(name, type=artifact_type)
            artifact.add_file(artifact_path)
            self.run.log_artifact(artifact)
            
    def finish(self) -> None:
        """Finish W&B run."""
        if self.run:
            wandb.finish()
            
    def watch(self, model) -> None:
        """Watch model with W&B."""
        if self.run:
            wandb.watch(model)


class MetricsLogger:
    """Custom metrics logger."""
    
    def __init__(self, log_dir: str = "metrics"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.metrics = {}
        
    def log_metric(self, name: str, value: float, step: int) -> None:
        """Log a single metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({"step": step, "value": value})
        
    def log_metrics(self, metrics: dict, step: int) -> None:
        """Log multiple metrics."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)
            
    def save_metrics(self, filename: str = "metrics.json") -> None:
        """Save metrics to file."""
        import json
        with open(self.log_dir / filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
    def get_metric(self, name: str) -> list:
        """Get metric history."""
        return self.metrics.get(name, [])
        
    def get_latest_metric(self, name: str) -> Optional[float]:
        """Get latest value of a metric."""
        metric_history = self.get_metric(name)
        return metric_history[-1]["value"] if metric_history else None