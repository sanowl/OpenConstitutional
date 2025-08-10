"""Constitutional AI dataset implementation."""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset as HFDataset
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from ..utils.config import Config
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConstitutionalExample:
    """Single example for constitutional training."""
    question: str
    original_response: str
    critique: str
    revised_response: str
    principles: List[str]
    metadata: Dict[str, Any]


class ConstitutionalDataset(Dataset):
    """Dataset for Constitutional AI training."""
    
    def __init__(
        self,
        config: Config,
        tokenizer,
        split: str = "train",
        max_samples: Optional[int] = None
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.max_samples = max_samples
        
        # Load and process data
        self.examples = self._load_examples()
        
        logger.info(f"Loaded {len(self.examples)} examples for {split} split")
        
    def _load_examples(self) -> List[ConstitutionalExample]:
        """Load constitutional examples from dataset."""
        examples = []
        
        try:
            # Load HH-RLHF dataset
            dataset = load_dataset(
                self.config.data.dataset_name,
                split=self.split,
                streaming=False
            )
            
            if self.max_samples:
                dataset = dataset.select(range(min(self.max_samples, len(dataset))))
                
            for item in dataset:
                example = self._process_item(item)
                if example:
                    examples.append(example)
                    
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            # Create dummy examples for testing
            examples = self._create_dummy_examples()
            
        return examples
        
    def _process_item(self, item: Dict[str, Any]) -> Optional[ConstitutionalExample]:
        """Process a single item from the dataset."""
        try:
            # Extract question and response from HH-RLHF format
            if "chosen" in item and "rejected" in item:
                # Use chosen response as original
                conversation = item["chosen"]
                question = self._extract_question(conversation)
                original_response = self._extract_response(conversation)
                
                # Create example
                example = ConstitutionalExample(
                    question=question,
                    original_response=original_response,
                    critique="",  # Will be generated
                    revised_response="",  # Will be generated
                    principles=self.config.constitutional_principles,
                    metadata={"source": "hh-rlhf", "original_item": item}
                )
                
                return example
                
        except Exception as e:
            logger.warning(f"Error processing item: {e}")
            return None
            
    def _extract_question(self, conversation: str) -> str:
        """Extract question from conversation."""
        # Simple heuristic for HH-RLHF format
        lines = conversation.split('\n')
        for line in lines:
            if line.startswith("Human:"):
                return line.replace("Human:", "").strip()
        return conversation[:200]  # fallback
        
    def _extract_response(self, conversation: str) -> str:
        """Extract response from conversation."""
        # Simple heuristic for HH-RLHF format
        lines = conversation.split('\n')
        for line in lines:
            if line.startswith("Assistant:"):
                return line.replace("Assistant:", "").strip()
        return conversation[200:]  # fallback
        
    def _create_dummy_examples(self) -> List[ConstitutionalExample]:
        """Create dummy examples for testing."""
        examples = []
        
        dummy_data = [
            {
                "question": "How can I help someone who is feeling sad?",
                "original_response": "Just tell them to get over it and move on.",
                "critique": "This response lacks empathy and is not helpful.",
                "revised_response": "Listen to them with empathy and offer emotional support."
            },
            {
                "question": "What should I do if I find a wallet on the street?",
                "original_response": "Keep it, finders keepers!",
                "critique": "This response encourages dishonest behavior.",
                "revised_response": "Try to return it to the owner or turn it in to authorities."
            }
        ]
        
        for data in dummy_data:
            example = ConstitutionalExample(
                question=data["question"],
                original_response=data["original_response"],
                critique=data["critique"],
                revised_response=data["revised_response"],
                principles=self.config.constitutional_principles,
                metadata={"source": "dummy"}
            )
            examples.append(example)
            
        return examples
        
    def __len__(self) -> int:
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single example."""
        example = self.examples[idx]
        
        # Tokenize inputs
        question_tokens = self.tokenizer(
            example.question,
            max_length=self.config.data.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        original_response_tokens = self.tokenizer(
            example.original_response,
            max_length=self.config.data.max_output_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Ensure revised_response has content; if empty, fallback to original to avoid empty labels
        revised_text = example.revised_response if example.revised_response else example.original_response
        revised_response_tokens = self.tokenizer(
            revised_text,
            max_length=self.config.data.max_output_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "question": example.question,
            "original_response": example.original_response,
            "critique": example.critique,
            "revised_response": example.revised_response,
            "principles": example.principles,
            "question_input_ids": question_tokens["input_ids"].squeeze(),
            "question_attention_mask": question_tokens["attention_mask"].squeeze(),
            "original_input_ids": original_response_tokens["input_ids"].squeeze(),
            "original_attention_mask": original_response_tokens["attention_mask"].squeeze(),
            "revised_input_ids": revised_response_tokens["input_ids"].squeeze(),
            "revised_attention_mask": revised_response_tokens["attention_mask"].squeeze(),
            "metadata": example.metadata
        }
        
    def add_critiques_and_revisions(self, critiques: List[str], revisions: List[str]) -> None:
        """Add generated critiques and revisions to examples."""
        if len(critiques) != len(self.examples):
            raise ValueError(f"Number of critiques ({len(critiques)}) must match number of examples ({len(self.examples)})")
        if len(revisions) != len(self.examples):
            raise ValueError(f"Number of revisions ({len(revisions)}) must match number of examples ({len(self.examples)})")
            
        for i, (critique, revision) in enumerate(zip(critiques, revisions)):
            if critique is None or revision is None:
                logger.warning(f"Skipping example {i} due to None critique or revision")
                continue
            self.examples[i].critique = critique
            self.examples[i].revised_response = revision
            
        logger.info("Added critiques and revisions to dataset")
        
    def get_questions_and_responses(self) -> Tuple[List[str], List[str]]:
        """Get lists of questions and original responses."""
        questions = [ex.question for ex in self.examples]
        responses = [ex.original_response for ex in self.examples]
        return questions, responses
        
    def get_revised_responses(self) -> List[str]:
        """Get list of revised responses."""
        return [ex.revised_response for ex in self.examples]
        
    def save_to_disk(self, save_path: str) -> None:
        """Save dataset to disk."""
        import json
        
        data = []
        for example in self.examples:
            data.append({
                "question": example.question,
                "original_response": example.original_response,
                "critique": example.critique,
                "revised_response": example.revised_response,
                "principles": example.principles,
                "metadata": example.metadata
            })
            
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Dataset saved to {save_path}")
        
    @classmethod
    def load_from_disk(cls, load_path: str, config: Config, tokenizer) -> "ConstitutionalDataset":
        """Load dataset from disk."""
        import json
        
        with open(load_path, 'r') as f:
            data = json.load(f)
            
        dataset = cls(config, tokenizer, split="custom", max_samples=0)
        dataset.examples = []
        
        for item in data:
            example = ConstitutionalExample(
                question=item["question"],
                original_response=item["original_response"],
                critique=item["critique"],
                revised_response=item["revised_response"],
                principles=item["principles"],
                metadata=item.get("metadata", {})
            )
            dataset.examples.append(example)
            
        logger.info(f"Dataset loaded from {load_path}")
        return dataset