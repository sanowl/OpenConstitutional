"""Constitutional critique model implementation."""

import torch
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from .constitutional_model import ConstitutionalAIModel, GenerationOutput
from ..utils.config import Config
from ..utils.constants import CRITIQUE_TEMPLATES, CONSTITUTIONAL_PRINCIPLES
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CritiqueOutput:
    """Output from critique generation."""
    critique: str
    principle_violations: List[str]
    severity_score: float
    suggestions: List[str]


class CritiqueModel(ConstitutionalAIModel):
    """Model for generating constitutional critiques."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.constitutional_principles = config.constitutional_principles
        self.critique_temperature = config.critique_temperature
        
        logger.info("Initialized CritiqueModel")
        
    def generate_critique(
        self,
        question: str,
        response: str,
        critique_type: str = "constitutional",
        principles: Optional[List[str]] = None,
        temperature: Optional[float] = None
    ) -> CritiqueOutput:
        """Generate a critique for a given question-response pair."""
        
        # Use provided principles or default ones
        if principles is None:
            principles = self.constitutional_principles
            
        # Get critique template
        if critique_type not in CRITIQUE_TEMPLATES:
            raise ValueError(f"Unknown critique type: {critique_type}")
            
        template = CRITIQUE_TEMPLATES[critique_type]
        
        # Format prompt
        if critique_type == "constitutional":
            principles_text = "\n".join(f"- {p}" for p in principles)
            prompt = template.format(
                question=question,
                response=response,
                principles=principles_text
            )
        else:
            prompt = template.format(question=question, response=response)
            
        # Generate critique aiming for JSON output
        outputs = self.generate(
            prompt=prompt,
            temperature=temperature or self.critique_temperature,
            max_length=self.config.model.max_length,
            do_sample=True,
            num_return_sequences=1,
            return_full_text=False
        )
        critique_text = outputs[0].text.strip()
        
        # Try to parse JSON first, fallback to heuristic
        parsed = self._parse_json_critique(critique_text)
        if parsed is not None:
            critique_output = CritiqueOutput(
                critique=parsed.get("critique", critique_text),
                principle_violations=parsed.get("violations", []),
                severity_score=float(parsed.get("severity", 0.5)),
                suggestions=parsed.get("suggestions", [])
            )
        else:
            critique_output = self._parse_critique(critique_text, principles)
        
        logger.debug(f"Generated critique for {critique_type}: {critique_output.critique[:100]}...")
        
        return critique_output
        
    def generate_batch_critiques(
        self,
        questions: List[str],
        responses: List[str],
        critique_type: str = "constitutional",
        principles: Optional[List[str]] = None,
        temperature: Optional[float] = None
    ) -> List[CritiqueOutput]:
        """Generate critiques for multiple question-response pairs."""
        
        critiques = []
        for question, response in zip(questions, responses):
            critique = self.generate_critique(
                question=question,
                response=response,
                critique_type=critique_type,
                principles=principles,
                temperature=temperature
            )
            critiques.append(critique)
            
        return critiques
        
    def _parse_critique(self, critique_text: str, principles: List[str]) -> CritiqueOutput:
        """Parse critique text into structured output."""
        
        # Extract principle violations
        violations = []
        for principle in principles:
            if any(keyword in critique_text.lower() for keyword in ["violates", "fails", "problematic"]):
                violations.append(principle)
                
        # Extract severity score (simple heuristic)
        severity_keywords = {
            "severe": 0.9,
            "serious": 0.8,
            "significant": 0.7,
            "moderate": 0.5,
            "minor": 0.3,
            "slight": 0.2
        }
        
        severity_score = 0.5  # default
        for keyword, score in severity_keywords.items():
            if keyword in critique_text.lower():
                severity_score = score
                break
                
        # Extract suggestions (simple heuristic)
        suggestions = []
        lines = critique_text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ["should", "could", "suggest", "recommend"]):
                suggestions.append(line.strip())
                
        return CritiqueOutput(
            critique=critique_text,
            principle_violations=violations,
            severity_score=severity_score,
            suggestions=suggestions
        )

    def _parse_json_critique(self, text: str) -> Optional[Dict[str, Any]]:
        """Attempt to parse a compact JSON critique."""
        import json
        # Find JSON substring if extra text exists
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = text[start:end+1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
        return None
        
    def evaluate_critique_quality(self, critique: CritiqueOutput, ground_truth: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Evaluate the quality of a generated critique."""
        
        metrics = {}
        
        # Length-based metrics
        critique_length = len(critique.critique.split())
        metrics["critique_length"] = critique_length
        metrics["has_suggestions"] = float(len(critique.suggestions) > 0)
        metrics["has_violations"] = float(len(critique.principle_violations) > 0)
        
        # Severity score reasonableness
        metrics["severity_score"] = critique.severity_score
        
        # If ground truth is provided, compute accuracy metrics
        if ground_truth:
            if "violations" in ground_truth:
                true_violations = set(ground_truth["violations"])
                pred_violations = set(critique.principle_violations)
                
                if len(true_violations) > 0:
                    precision = len(pred_violations & true_violations) / len(pred_violations) if pred_violations else 0
                    recall = len(pred_violations & true_violations) / len(true_violations)
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    metrics["violation_precision"] = precision
                    metrics["violation_recall"] = recall
                    metrics["violation_f1"] = f1
                    
            if "severity" in ground_truth:
                severity_error = abs(critique.severity_score - ground_truth["severity"])
                metrics["severity_error"] = severity_error
                
        return metrics
        
    def self_critique(self, question: str, response: str, num_rounds: int = 1) -> List[CritiqueOutput]:
        """Perform multiple rounds of self-critique."""
        
        critiques = []
        current_response = response
        
        for round_num in range(num_rounds):
            critique = self.generate_critique(
                question=question,
                response=current_response,
                critique_type="constitutional"
            )
            
            critiques.append(critique)
            
            # Use critique as context for next round
            current_response = f"{response}\n\nPrevious critique: {critique.critique}"
            
            logger.debug(f"Completed self-critique round {round_num + 1}")
            
        return critiques