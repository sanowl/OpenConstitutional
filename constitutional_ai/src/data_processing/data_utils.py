"""
Data utilities for Constitutional AI.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DataCollatorConfig:
    """Configuration for data collation."""
    pad_token_id: int = 0
    max_length: int = 512
    return_tensors: str = "pt"


class DataCollator:
    """Data collator for Constitutional AI training."""
    
    def __init__(self, config: DataCollatorConfig):
        self.config = config
        
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of examples."""
        
        # Extract different types of data
        input_ids = []
        attention_masks = []
        labels = []
        
        for example in batch:
            if "input_ids" in example:
                input_ids.append(example["input_ids"])
            if "attention_mask" in example:
                attention_masks.append(example["attention_mask"])
            if "labels" in example:
                labels.append(example["labels"])
                
        # Pad sequences
        collated_batch = {}
        
        if input_ids:
            collated_batch["input_ids"] = pad_sequence(
                input_ids, 
                batch_first=True, 
                padding_value=self.config.pad_token_id
            )
            
        if attention_masks:
            collated_batch["attention_mask"] = pad_sequence(
                attention_masks,
                batch_first=True,
                padding_value=0
            )
            
        if labels:
            collated_batch["labels"] = pad_sequence(
                labels,
                batch_first=True,
                padding_value=-100
            )
            
        return collated_batch


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn: Optional[Any] = None,
    pin_memory: bool = False,
    drop_last: bool = False
) -> DataLoader:
    """Create a DataLoader with common settings."""
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last
    )


def compute_batch_statistics(batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Compute statistics for a batch."""
    
    stats = {}
    
    if "input_ids" in batch:
        input_ids = batch["input_ids"]
        stats["batch_size"] = input_ids.size(0)
        stats["seq_length"] = input_ids.size(1)
        stats["num_tokens"] = input_ids.numel()
        
    if "attention_mask" in batch:
        attention_mask = batch["attention_mask"]
        stats["avg_seq_length"] = attention_mask.sum(dim=1).float().mean().item()
        stats["padding_ratio"] = (attention_mask == 0).float().mean().item()
        
    return stats


def filter_examples_by_length(
    examples: List[Dict[str, Any]],
    min_length: int = 10,
    max_length: int = 512,
    length_key: str = "input_ids"
) -> List[Dict[str, Any]]:
    """Filter examples by sequence length."""
    
    filtered_examples = []
    
    for example in examples:
        if length_key in example:
            length = len(example[length_key])
            if min_length <= length <= max_length:
                filtered_examples.append(example)
                
    logger.info(f"Filtered {len(examples)} examples to {len(filtered_examples)}")
    
    return filtered_examples


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> tuple[Dataset, Dataset, Dataset]:
    """Split dataset into train/val/test."""
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    import random
    random.seed(seed)
    
    # Get indices
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    # Calculate split points
    train_end = int(len(indices) * train_ratio)
    val_end = train_end + int(len(indices) * val_ratio)
    
    # Split indices
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Create subset datasets
    from torch.utils.data import Subset
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    logger.info(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    return train_dataset, val_dataset, test_dataset


def create_balanced_dataset(
    dataset: Dataset,
    balance_key: str,
    max_samples_per_class: Optional[int] = None,
    seed: int = 42
) -> Dataset:
    """Create a balanced dataset by sampling equal numbers from each class."""
    
    import random
    from collections import defaultdict
    
    random.seed(seed)
    
    # Group examples by class
    class_examples = defaultdict(list)
    
    for i, example in enumerate(dataset):
        if balance_key in example:
            class_value = example[balance_key]
            class_examples[class_value].append(i)
            
    # Find minimum class size
    min_class_size = min(len(indices) for indices in class_examples.values())
    
    if max_samples_per_class:
        min_class_size = min(min_class_size, max_samples_per_class)
        
    # Sample equal numbers from each class
    balanced_indices = []
    for class_value, indices in class_examples.items():
        sampled_indices = random.sample(indices, min_class_size)
        balanced_indices.extend(sampled_indices)
        
    # Create balanced dataset
    from torch.utils.data import Subset
    balanced_dataset = Subset(dataset, balanced_indices)
    
    logger.info(f"Created balanced dataset with {len(balanced_dataset)} examples")
    
    return balanced_dataset