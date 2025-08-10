"""
HH-RLHF dataset processor for Constitutional AI.
Processes Anthropic's HH-RLHF dataset for constitutional training.
"""

import json
from typing import Dict, List, Any, Optional
from datasets import load_dataset
from dataclasses import dataclass

from ..utils.config import Config
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HHRLHFExample:
    """Single example from HH-RLHF dataset."""
    question: str
    chosen_response: str
    rejected_response: str
    metadata: Dict[str, Any]


class HHRLHFProcessor:
    """Processor for Anthropic's HH-RLHF dataset."""
    
    def __init__(self, config: Config):
        self.config = config
        logger.info("Initialized HH-RLHF processor")
        
    def load_dataset(self, split: str = "train", max_samples: Optional[int] = None) -> List[HHRLHFExample]:
        """Load HH-RLHF dataset."""
        
        try:
            dataset = load_dataset("Anthropic/hh-rlhf", split=split)
            
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
                
            examples = []
            for item in dataset:
                example = self._process_item(item)
                if example:
                    examples.append(example)
                    
            logger.info(f"Loaded {len(examples)} examples from HH-RLHF {split} split")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to load HH-RLHF dataset: {e}")
            return self._create_dummy_examples()
            
    def _process_item(self, item: Dict[str, Any]) -> Optional[HHRLHFExample]:
        """Process a single HH-RLHF item."""
        
        try:
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")
            
            # Extract question and responses
            question = self._extract_question(chosen)
            chosen_response = self._extract_response(chosen)
            rejected_response = self._extract_response(rejected)
            
            return HHRLHFExample(
                question=question,
                chosen_response=chosen_response,
                rejected_response=rejected_response,
                metadata={"source": "hh-rlhf", "original_item": item}
            )
            
        except Exception as e:
            logger.warning(f"Failed to process HH-RLHF item: {e}")
            return None
            
    def _extract_question(self, conversation: str) -> str:
        """Extract first human turn as question, supporting multi-turn."""
        lines = [l.strip() for l in conversation.split('\n') if l.strip()]
        # Prefer the first Human: line
        for line in lines:
            if line.lower().startswith("human:"):
                return line.split(":", 1)[1].strip()
        # Fallback to first non-empty line
        return lines[0] if lines else conversation[:200].strip()
        
    def _extract_response(self, conversation: str) -> str:
        """Extract the first assistant turn following a human turn."""
        lines = [l.strip() for l in conversation.split('\n') if l.strip()]
        for i, line in enumerate(lines):
            if line.lower().startswith("human:"):
                # Look ahead for the next assistant line
                for j in range(i + 1, len(lines)):
                    if lines[j].lower().startswith("assistant:"):
                        return lines[j].split(":", 1)[1].strip()
                break
        # Fallback to first Assistant line anywhere
        for line in lines:
            if line.lower().startswith("assistant:"):
                return line.split(":", 1)[1].strip()
        return ""
        
    def _create_dummy_examples(self) -> List[HHRLHFExample]:
        """Create dummy examples when dataset loading fails."""
        
        logger.info("Creating dummy HH-RLHF examples")
        
        dummy_data = [
            {
                "question": "How should I respond to online criticism?",
                "chosen": "Consider the feedback objectively and respond professionally if needed.",
                "rejected": "Just ignore the haters and block everyone who disagrees with you."
            },
            {
                "question": "What's the best way to handle workplace stress?",
                "chosen": "Take regular breaks, practice stress management techniques, and communicate with your supervisor.",
                "rejected": "Just power through it and work longer hours to prove yourself."
            },
            {
                "question": "How do I deal with difficult family members?",
                "chosen": "Set healthy boundaries, practice patience, and focus on constructive communication.",
                "rejected": "Cut them out of your life completely and don't look back."
            }
        ]
        
        examples = []
        for data in dummy_data:
            example = HHRLHFExample(
                question=data["question"],
                chosen_response=data["chosen"],
                rejected_response=data["rejected"],
                metadata={"source": "dummy"}
            )
            examples.append(example)
            
        return examples
        
    def convert_to_constitutional_format(self, examples: List[HHRLHFExample]) -> List[Dict[str, Any]]:
        """Convert HH-RLHF examples to constitutional training format."""
        
        constitutional_examples = []
        
        for example in examples:
            # Use chosen response as the base, rejected as comparison
            constitutional_example = {
                "question": example.question,
                "original_response": example.chosen_response,
                "critique": "",  # To be generated
                "revised_response": "",  # To be generated
                "principles": self.config.constitutional_principles,
                "metadata": {
                    **example.metadata,
                    "rejected_response": example.rejected_response
                }
            }
            constitutional_examples.append(constitutional_example)
            
        return constitutional_examples
        
    def save_processed_data(self, examples: List[HHRLHFExample], save_path: str) -> None:
        """Save processed HH-RLHF data."""
        
        data = []
        for example in examples:
            data.append({
                "question": example.question,
                "chosen_response": example.chosen_response,
                "rejected_response": example.rejected_response,
                "metadata": example.metadata
            })
            
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved {len(examples)} processed examples to {save_path}")
        
    def load_processed_data(self, load_path: str) -> List[HHRLHFExample]:
        """Load processed HH-RLHF data."""
        
        with open(load_path, 'r') as f:
            data = json.load(f)
            
        examples = []
        for item in data:
            example = HHRLHFExample(
                question=item["question"],
                chosen_response=item["chosen_response"],
                rejected_response=item["rejected_response"],
                metadata=item["metadata"]
            )
            examples.append(example)
            
        logger.info(f"Loaded {len(examples)} processed examples from {load_path}")
        return examples