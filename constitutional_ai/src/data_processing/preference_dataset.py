"""
Preference dataset for reward model training.
Handles AI-generated preference data for Constitutional AI.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PreferenceExample:
    """Single preference example."""
    question: str
    chosen_response: str
    rejected_response: str
    reasoning: str
    confidence: float
    principles: List[str]
    metadata: Dict[str, Any]


class PreferenceDataset(Dataset):
    """Dataset for preference-based reward model training."""
    
    def __init__(
        self,
        preference_data: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Process preference data
        self.examples = self._process_preference_data(preference_data)
        
        logger.info(f"Loaded {len(self.examples)} preference examples")
        
    def _process_preference_data(
        self, preference_data: List[Dict[str, Any]]
    ) -> List[PreferenceExample]:
        """Process raw preference data into examples."""
        
        examples = []
        
        for item in preference_data:
            try:
                # Determine chosen and rejected responses
                if item["preferred"] == "A":
                    chosen_response = item["response_a"]
                    rejected_response = item["response_b"]
                else:
                    chosen_response = item["response_b"]
                    rejected_response = item["response_a"]
                    
                example = PreferenceExample(
                    question=item["question"],
                    chosen_response=chosen_response,
                    rejected_response=rejected_response,
                    reasoning=item["reasoning"],
                    confidence=item["confidence"],
                    principles=item["principles"],
                    metadata=item.get("metadata", {})
                )
                
                examples.append(example)
                
            except Exception as e:
                logger.warning(f"Error processing preference item: {e}")
                continue
                
        return examples
        
    def __len__(self) -> int:
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single preference example."""
        
        example = self.examples[idx]
        
        # Create full text for chosen and rejected responses
        chosen_text = f"{example.question} {example.chosen_response}"
        rejected_text = f"{example.question} {example.rejected_response}"
        
        # Tokenize chosen response
        chosen_tokens = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize rejected response
        rejected_tokens = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "question": example.question,
            "chosen_response": example.chosen_response,
            "rejected_response": example.rejected_response,
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(),
            "reasoning": example.reasoning,
            "confidence": example.confidence,
            "principles": example.principles,
            "metadata": example.metadata
        }
        
    def get_preference_pairs(self) -> List[tuple]:
        """Get all preference pairs."""
        
        pairs = []
        for example in self.examples:
            pairs.append((
                example.question,
                example.chosen_response,
                example.rejected_response,
                example.confidence
            ))
            
        return pairs
        
    def filter_by_confidence(self, min_confidence: float = 0.7) -> "PreferenceDataset":
        """Filter dataset by minimum confidence threshold."""
        
        filtered_data = []
        for example in self.examples:
            if example.confidence >= min_confidence:
                # Convert back to dict format
                item = {
                    "question": example.question,
                    "response_a": example.chosen_response,
                    "response_b": example.rejected_response,
                    "preferred": "A",  # Always A since we already determined chosen
                    "reasoning": example.reasoning,
                    "confidence": example.confidence,
                    "principles": example.principles,
                    "metadata": example.metadata
                }
                filtered_data.append(item)
                
        logger.info(f"Filtered dataset from {len(self.examples)} to {len(filtered_data)} examples")
        
        return PreferenceDataset(filtered_data, self.tokenizer, self.max_length)
        
    def balance_preferences(self) -> "PreferenceDataset":
        """Balance the dataset to have equal A/B preferences using multiple strategies."""
        
        import random
        from collections import defaultdict
        
        # Analyze current preference distribution
        current_preferences = defaultdict(int)
        for example in self.examples:
            # Since we always convert to chosen/rejected, all are effectively "A" preferred
            current_preferences["A"] += 1
            
        logger.info(f"Current preference distribution: {dict(current_preferences)}")
        
        # Strategy 1: Create balanced pairs by flipping half the examples
        balanced_data = []
        shuffled_examples = self.examples.copy()
        random.shuffle(shuffled_examples)
        
        # Calculate target counts for perfect balance
        total_examples = len(shuffled_examples)
        target_a_count = total_examples // 2
        target_b_count = total_examples - target_a_count
        
        a_count = 0
        b_count = 0
        
        for example in shuffled_examples:
            # Decide whether to keep as A or flip to B based on current counts
            if a_count < target_a_count and (b_count >= target_b_count or random.random() < 0.5):
                # Keep as A (chosen response wins)
                item = {
                    "question": example.question,
                    "response_a": example.chosen_response,
                    "response_b": example.rejected_response,
                    "preferred": "A",
                    "reasoning": example.reasoning,
                    "confidence": example.confidence,
                    "principles": example.principles,
                    "metadata": {**example.metadata, "balanced_as": "A"}
                }
                a_count += 1
            else:
                # Flip to B (rejected response wins)
                item = {
                    "question": example.question,
                    "response_a": example.rejected_response,
                    "response_b": example.chosen_response,
                    "preferred": "B",
                    "reasoning": f"Flipped preference: {example.reasoning}",
                    "confidence": example.confidence,
                    "principles": example.principles,
                    "metadata": {**example.metadata, "balanced_as": "B"}
                }
                b_count += 1
                
            balanced_data.append(item)
        
        # Strategy 2: Add synthetic balanced pairs for better balance
        if abs(a_count - b_count) > 1:
            # If still imbalanced, create additional synthetic pairs
            examples_to_duplicate = shuffled_examples[:abs(a_count - b_count)]
            
            for example in examples_to_duplicate:
                if a_count < b_count:
                    # Need more A preferences
                    synthetic_item = {
                        "question": example.question,
                        "response_a": example.chosen_response,
                        "response_b": example.rejected_response,
                        "preferred": "A",
                        "reasoning": f"Synthetic A preference: {example.reasoning}",
                        "confidence": max(0.1, example.confidence - 0.2),  # Lower confidence for synthetic
                        "principles": example.principles,
                        "metadata": {**example.metadata, "balanced_as": "A", "synthetic": True}
                    }
                    balanced_data.append(synthetic_item)
                    a_count += 1
                else:
                    # Need more B preferences
                    synthetic_item = {
                        "question": example.question,
                        "response_a": example.rejected_response,
                        "response_b": example.chosen_response,
                        "preferred": "B",
                        "reasoning": f"Synthetic B preference: {example.reasoning}",
                        "confidence": max(0.1, example.confidence - 0.2),  # Lower confidence for synthetic
                        "principles": example.principles,
                        "metadata": {**example.metadata, "balanced_as": "B", "synthetic": True}
                    }
                    balanced_data.append(synthetic_item)
                    b_count += 1
        
        # Strategy 3: Quality-based balancing
        # Group by confidence levels for more sophisticated balancing
        confidence_groups = defaultdict(list)
        for item in balanced_data:
            conf_level = "high" if item["confidence"] > 0.7 else "medium" if item["confidence"] > 0.4 else "low"
            confidence_groups[conf_level].append(item)
        
        # Ensure each confidence group has balanced A/B preferences
        final_balanced_data = []
        for conf_level, items in confidence_groups.items():
            a_items = [item for item in items if item["preferred"] == "A"]
            b_items = [item for item in items if item["preferred"] == "B"]
            
            # Balance within confidence group
            min_count = min(len(a_items), len(b_items))
            if min_count > 0:
                # Take equal numbers from each preference
                final_balanced_data.extend(a_items[:min_count])
                final_balanced_data.extend(b_items[:min_count])
                
                # Add remaining items with preference flipping if needed
                remaining_a = a_items[min_count:]
                remaining_b = b_items[min_count:]
                
                if len(remaining_a) > len(remaining_b):
                    # Too many A preferences, flip some to B
                    flip_count = (len(remaining_a) - len(remaining_b)) // 2
                    for i in range(flip_count):
                        flipped_item = remaining_a[i].copy()
                        flipped_item["response_a"], flipped_item["response_b"] = flipped_item["response_b"], flipped_item["response_a"]
                        flipped_item["preferred"] = "B"
                        flipped_item["reasoning"] = f"Confidence-balanced flip: {flipped_item['reasoning']}"
                        flipped_item["metadata"]["confidence_balanced"] = True
                        final_balanced_data.append(flipped_item)
                    
                    # Add remaining unflipped items
                    final_balanced_data.extend(remaining_a[flip_count:])
                    final_balanced_data.extend(remaining_b)
                    
                elif len(remaining_b) > len(remaining_a):
                    # Too many B preferences, flip some to A
                    flip_count = (len(remaining_b) - len(remaining_a)) // 2
                    for i in range(flip_count):
                        flipped_item = remaining_b[i].copy()
                        flipped_item["response_a"], flipped_item["response_b"] = flipped_item["response_b"], flipped_item["response_a"]
                        flipped_item["preferred"] = "A"
                        flipped_item["reasoning"] = f"Confidence-balanced flip: {flipped_item['reasoning']}"
                        flipped_item["metadata"]["confidence_balanced"] = True
                        final_balanced_data.append(flipped_item)
                    
                    # Add remaining unflipped items
                    final_balanced_data.extend(remaining_b[flip_count:])
                    final_balanced_data.extend(remaining_a)
                else:
                    # Equal remaining, add all
                    final_balanced_data.extend(remaining_a)
                    final_balanced_data.extend(remaining_b)
            else:
                # No items to balance in this confidence group
                final_balanced_data.extend(items)
        
        # Final verification and statistics
        final_a_count = sum(1 for item in final_balanced_data if item["preferred"] == "A")
        final_b_count = sum(1 for item in final_balanced_data if item["preferred"] == "B")
        
        # Calculate balance metrics
        total_final = len(final_balanced_data)
        balance_ratio = min(final_a_count, final_b_count) / max(final_a_count, final_b_count) if max(final_a_count, final_b_count) > 0 else 1.0
        
        logger.info(f"Balanced dataset created:")
        logger.info(f"  Total examples: {total_final}")
        logger.info(f"  A preferences: {final_a_count} ({final_a_count/total_final:.1%})")
        logger.info(f"  B preferences: {final_b_count} ({final_b_count/total_final:.1%})")
        logger.info(f"  Balance ratio: {balance_ratio:.3f}")
        logger.info(f"  Synthetic examples: {sum(1 for item in final_balanced_data if item['metadata'].get('synthetic', False))}")
        
        return PreferenceDataset(final_balanced_data, self.tokenizer, self.max_length)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        
        stats = {
            "total_examples": len(self.examples),
            "avg_confidence": sum(ex.confidence for ex in self.examples) / len(self.examples),
            "confidence_distribution": {},
            "principles_coverage": {},
            "question_length_stats": {},
            "response_length_stats": {}
        }
        
        # Confidence distribution
        confidence_bins = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
        for i in range(len(confidence_bins) - 1):
            bin_min, bin_max = confidence_bins[i], confidence_bins[i + 1]
            count = sum(1 for ex in self.examples if bin_min <= ex.confidence < bin_max)
            stats["confidence_distribution"][f"{bin_min}-{bin_max}"] = count
            
        # Principles coverage
        all_principles = set()
        for example in self.examples:
            all_principles.update(example.principles)
            
        for principle in all_principles:
            count = sum(1 for ex in self.examples if principle in ex.principles)
            stats["principles_coverage"][principle] = count
            
        # Length statistics
        question_lengths = [len(ex.question.split()) for ex in self.examples]
        chosen_lengths = [len(ex.chosen_response.split()) for ex in self.examples]
        rejected_lengths = [len(ex.rejected_response.split()) for ex in self.examples]
        
        stats["question_length_stats"] = {
            "avg": sum(question_lengths) / len(question_lengths),
            "min": min(question_lengths),
            "max": max(question_lengths)
        }
        
        stats["response_length_stats"] = {
            "chosen_avg": sum(chosen_lengths) / len(chosen_lengths),
            "rejected_avg": sum(rejected_lengths) / len(rejected_lengths),
            "chosen_min": min(chosen_lengths),
            "chosen_max": max(chosen_lengths),
            "rejected_min": min(rejected_lengths),
            "rejected_max": max(rejected_lengths)
        }
        
        return stats
        
    def save_to_disk(self, save_path: str):
        """Save preference dataset to disk."""
        
        data = []
        for example in self.examples:
            data.append({
                "question": example.question,
                "chosen_response": example.chosen_response,
                "rejected_response": example.rejected_response,
                "reasoning": example.reasoning,
                "confidence": example.confidence,
                "principles": example.principles,
                "metadata": example.metadata
            })
            
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Preference dataset saved to {save_path}")
        
    @classmethod
    def load_from_disk(cls, load_path: str, tokenizer, max_length: int = 512) -> "PreferenceDataset":
        """Load preference dataset from disk."""
        
        with open(load_path, 'r') as f:
            data = json.load(f)
            
        # Convert to preference data format
        preference_data = []
        for item in data:
            preference_item = {
                "question": item["question"],
                "response_a": item["chosen_response"],
                "response_b": item["rejected_response"],
                "preferred": "A",  # Always A since chosen is preferred
                "reasoning": item["reasoning"],
                "confidence": item["confidence"],
                "principles": item["principles"],
                "metadata": item.get("metadata", {})
            }
            preference_data.append(preference_item)
            
        logger.info(f"Preference dataset loaded from {load_path}")
        
        return cls(preference_data, tokenizer, max_length)
        
    def split_dataset(self, train_ratio: float = 0.8) -> tuple["PreferenceDataset", "PreferenceDataset"]:
        """Split dataset into train and validation sets."""
        
        import random
        
        # Shuffle examples
        shuffled_examples = self.examples.copy()
        random.shuffle(shuffled_examples)
        
        # Split
        split_idx = int(len(shuffled_examples) * train_ratio)
        train_examples = shuffled_examples[:split_idx]
        val_examples = shuffled_examples[split_idx:]
        
        # Convert back to preference data format
        def examples_to_preference_data(examples):
            preference_data = []
            for ex in examples:
                item = {
                    "question": ex.question,
                    "response_a": ex.chosen_response,
                    "response_b": ex.rejected_response,
                    "preferred": "A",
                    "reasoning": ex.reasoning,
                    "confidence": ex.confidence,
                    "principles": ex.principles,
                    "metadata": ex.metadata
                }
                preference_data.append(item)
            return preference_data
            
        train_data = examples_to_preference_data(train_examples)
        val_data = examples_to_preference_data(val_examples)
        
        train_dataset = PreferenceDataset(train_data, self.tokenizer, self.max_length)
        val_dataset = PreferenceDataset(val_data, self.tokenizer, self.max_length)
        
        logger.info(f"Split dataset into {len(train_dataset)} train and {len(val_dataset)} validation examples")
        
        return train_dataset, val_dataset