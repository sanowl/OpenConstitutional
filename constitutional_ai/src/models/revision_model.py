"""Constitutional revision model implementation."""

import torch
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from .constitutional_model import ConstitutionalAIModel, GenerationOutput
from .critique_model import CritiqueOutput
from ..utils.config import Config
from ..utils.constants import REVISION_TEMPLATES, CONSTITUTIONAL_PRINCIPLES
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RevisionOutput:
    """Output from revision generation."""
    revised_response: str
    improvements: List[str]
    quality_score: float
    original_response: str


class RevisionModel(ConstitutionalAIModel):
    """Model for generating constitutional revisions."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.constitutional_principles = config.constitutional_principles
        self.revision_temperature = config.revision_temperature
        
        logger.info("Initialized RevisionModel")
        
    def generate_revision(
        self,
        question: str,
        original_response: str,
        critique: str,
        revision_type: str = "constitutional",
        principles: Optional[List[str]] = None,
        temperature: Optional[float] = None
    ) -> RevisionOutput:
        """Generate a revision for a given question-response-critique triplet."""
        
        # Use provided principles or default ones
        if principles is None:
            principles = self.constitutional_principles
            
        # Get revision template
        if revision_type not in REVISION_TEMPLATES:
            raise ValueError(f"Unknown revision type: {revision_type}")
            
        template = REVISION_TEMPLATES[revision_type]
        
        # Format prompt
        if revision_type == "constitutional":
            principles_text = "\n".join(f"- {p}" for p in principles)
            prompt = template.format(
                question=question,
                response=original_response,
                critique=critique,
                principles=principles_text
            )
        else:
            prompt = template.format(
                question=question,
                response=original_response,
                critique=critique
            )
            
        # Generate revision
        outputs = self.generate(
            prompt=prompt,
            temperature=temperature or self.revision_temperature,
            max_length=self.config.model.max_length,
            num_return_sequences=1,
            return_full_text=False
        )
        
        revised_text = outputs[0].text.strip()
        
        # Parse revision components
        revision_output = self._parse_revision(revised_text, original_response)
        
        logger.debug(f"Generated revision for {revision_type}: {revision_output.revised_response[:100]}...")
        
        return revision_output
        
    def generate_batch_revisions(
        self,
        questions: List[str],
        original_responses: List[str],
        critiques: List[str],
        revision_type: str = "constitutional",
        principles: Optional[List[str]] = None,
        temperature: Optional[float] = None
    ) -> List[RevisionOutput]:
        """Generate revisions for multiple question-response-critique triplets."""
        
        revisions = []
        for question, response, critique in zip(questions, original_responses, critiques):
            revision = self.generate_revision(
                question=question,
                original_response=response,
                critique=critique,
                revision_type=revision_type,
                principles=principles,
                temperature=temperature
            )
            revisions.append(revision)
            
        return revisions
        
    def _parse_revision(self, revised_text: str, original_response: str) -> RevisionOutput:
        """Parse revision text into structured output."""
        
        # Extract improvements made
        improvements = []
        improvement_keywords = ["improved", "enhanced", "corrected", "fixed", "addressed"]
        
        lines = revised_text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in improvement_keywords):
                improvements.append(line.strip())
                
        # Calculate quality score (simple heuristic based on length and improvements)
        length_ratio = len(revised_text) / max(len(original_response), 1)
        improvement_bonus = min(len(improvements) * 0.1, 0.3)
        quality_score = min(0.5 + length_ratio * 0.3 + improvement_bonus, 1.0)
        
        return RevisionOutput(
            revised_response=revised_text,
            improvements=improvements,
            quality_score=quality_score,
            original_response=original_response
        )
        
    def iterative_revision(
        self,
        question: str,
        original_response: str,
        critique: str,
        num_iterations: int = 3,
        improvement_threshold: float = 0.1
    ) -> List[RevisionOutput]:
        """Perform iterative revision until improvement threshold is met."""
        
        revisions = []
        current_response = original_response
        current_critique = critique
        
        for iteration in range(num_iterations):
            revision = self.generate_revision(
                question=question,
                original_response=current_response,
                critique=current_critique,
                revision_type="constitutional"
            )
            
            revisions.append(revision)
            
            # Check if improvement threshold is met
            if len(revisions) > 1:
                improvement = revision.quality_score - revisions[-2].quality_score
                if improvement < improvement_threshold:
                    logger.info(f"Revision converged after {iteration + 1} iterations")
                    break
                    
            # Update for next iteration
            current_response = revision.revised_response
            # Generate new critique for the revised response (would need critique model)
            
            logger.debug(f"Completed revision iteration {iteration + 1}")
            
        return revisions
        
    def evaluate_revision_quality(
        self,
        revision: RevisionOutput,
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Evaluate the quality of a generated revision."""
        
        metrics = {}
        
        # Length-based metrics
        original_length = len(revision.original_response.split())
        revised_length = len(revision.revised_response.split())
        
        metrics["original_length"] = original_length
        metrics["revised_length"] = revised_length
        metrics["length_ratio"] = revised_length / max(original_length, 1)
        
        # Improvement metrics
        metrics["num_improvements"] = len(revision.improvements)
        metrics["quality_score"] = revision.quality_score
        
        # Text similarity metrics (simple overlap)
        original_words = set(revision.original_response.lower().split())
        revised_words = set(revision.revised_response.lower().split())
        
        if original_words:
            jaccard_similarity = len(original_words & revised_words) / len(original_words | revised_words)
            metrics["jaccard_similarity"] = jaccard_similarity
            
        # If ground truth is provided, compute accuracy metrics
        if ground_truth:
            if "target_improvements" in ground_truth:
                target_improvements = set(ground_truth["target_improvements"])
                actual_improvements = set(revision.improvements)
                
                if target_improvements:
                    precision = len(actual_improvements & target_improvements) / len(actual_improvements) if actual_improvements else 0
                    recall = len(actual_improvements & target_improvements) / len(target_improvements)
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    metrics["improvement_precision"] = precision
                    metrics["improvement_recall"] = recall
                    metrics["improvement_f1"] = f1
                    
            if "target_quality" in ground_truth:
                quality_error = abs(revision.quality_score - ground_truth["target_quality"])
                metrics["quality_error"] = quality_error
                
        return metrics
        
    def compare_revisions(
        self,
        revision1: RevisionOutput,
        revision2: RevisionOutput,
        criteria: List[str] = ["quality", "length", "improvements"]
    ) -> Dict[str, Any]:
        """Compare two revisions across multiple criteria."""
        
        comparison = {}
        
        if "quality" in criteria:
            comparison["quality_winner"] = "revision1" if revision1.quality_score > revision2.quality_score else "revision2"
            comparison["quality_diff"] = revision1.quality_score - revision2.quality_score
            
        if "length" in criteria:
            len1 = len(revision1.revised_response.split())
            len2 = len(revision2.revised_response.split())
            comparison["length_winner"] = "revision1" if len1 > len2 else "revision2"
            comparison["length_diff"] = len1 - len2
            
        if "improvements" in criteria:
            imp1 = len(revision1.improvements)
            imp2 = len(revision2.improvements)
            comparison["improvements_winner"] = "revision1" if imp1 > imp2 else "revision2"
            comparison["improvements_diff"] = imp1 - imp2
            
        return comparison