"""Preference model for AI feedback collection."""

import torch
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

from .constitutional_model import ConstitutionalAIModel, GenerationOutput
from ..utils.config import Config
from ..utils.constants import PREFERENCE_TEMPLATES, CONSTITUTIONAL_PRINCIPLES
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PreferenceOutput:
    """Output from preference comparison."""
    preferred_response: str  # "A" or "B"
    reasoning: str
    confidence: float
    criteria_scores: Dict[str, float]


class PreferenceModel(ConstitutionalAIModel):
    """Model for generating AI preferences between responses."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.constitutional_principles = config.constitutional_principles
        self.preference_temperature = config.preference_temperature
        
        logger.info("Initialized PreferenceModel")
        
    def compare_responses(
        self,
        question: str,
        response_a: str,
        response_b: str,
        principles: Optional[List[str]] = None,
        temperature: Optional[float] = None
    ) -> PreferenceOutput:
        """Compare two responses and return preference."""
        
        # Use provided principles or default ones
        if principles is None:
            principles = self.constitutional_principles
            
        # Format prompt
        principles_text = "\n".join(f"- {p}" for p in principles)
        prompt = PREFERENCE_TEMPLATES["comparison"].format(
            question=question,
            response_a=response_a,
            response_b=response_b,
            principles=principles_text
        )
        
        # Generate preference
        outputs = self.generate(
            prompt=prompt,
            temperature=temperature or self.preference_temperature,
            max_length=self.config.model.max_length,
            num_return_sequences=1,
            return_full_text=False
        )
        
        preference_text = outputs[0].text.strip()
        
        # Parse preference components
        preference_output = self._parse_preference(preference_text, response_a, response_b)
        
        logger.debug(f"Generated preference: {preference_output.preferred_response}")
        
        return preference_output
        
    def generate_batch_preferences(
        self,
        questions: List[str],
        responses_a: List[str],
        responses_b: List[str],
        principles: Optional[List[str]] = None,
        temperature: Optional[float] = None
    ) -> List[PreferenceOutput]:
        """Generate preferences for multiple response pairs."""
        
        preferences = []
        for question, response_a, response_b in zip(questions, responses_a, responses_b):
            preference = self.compare_responses(
                question=question,
                response_a=response_a,
                response_b=response_b,
                principles=principles,
                temperature=temperature
            )
            preferences.append(preference)
            
        return preferences
        
    def _parse_preference(self, preference_text: str, response_a: str, response_b: str) -> PreferenceOutput:
        """Parse preference text into structured output."""
        
        # Extract preferred response
        preferred_response = "A"  # default
        if any(phrase in preference_text.lower() for phrase in ["response b", "option b", "b is better", "choose b"]):
            preferred_response = "B"
        elif any(phrase in preference_text.lower() for phrase in ["response a", "option a", "a is better", "choose a"]):
            preferred_response = "A"
            
        # Extract reasoning
        reasoning = preference_text
        
        # Extract confidence (simple heuristic)
        confidence_keywords = {
            "clearly": 0.9,
            "definitely": 0.9,
            "obviously": 0.8,
            "likely": 0.7,
            "probably": 0.6,
            "somewhat": 0.5,
            "slightly": 0.4
        }
        
        confidence = 0.5  # default
        for keyword, conf in confidence_keywords.items():
            if keyword in preference_text.lower():
                confidence = conf
                break
                
        # Extract criteria scores (simple heuristic)
        criteria_scores = {}
        criteria = ["helpful", "harmless", "honest", "accurate", "respectful"]
        
        for criterion in criteria:
            if criterion in preference_text.lower():
                criteria_scores[criterion] = 0.8 if preferred_response == "A" else 0.2
            else:
                criteria_scores[criterion] = 0.5
                
        return PreferenceOutput(
            preferred_response=preferred_response,
            reasoning=reasoning,
            confidence=confidence,
            criteria_scores=criteria_scores
        )
        
    def collect_preference_dataset(
        self,
        questions: List[str],
        response_pairs: List[Tuple[str, str]],
        principles: Optional[List[str]] = None,
        num_samples: int = 1000
    ) -> List[Dict[str, Any]]:
        """Collect a dataset of AI preferences."""
        
        preference_dataset = []
        
        for i, (question, (response_a, response_b)) in enumerate(zip(questions, response_pairs)):
            if i >= num_samples:
                break
                
            preference = self.compare_responses(
                question=question,
                response_a=response_a,
                response_b=response_b,
                principles=principles
            )
            
            # Create dataset entry
            entry = {
                "question": question,
                "response_a": response_a,
                "response_b": response_b,
                "preferred": preference.preferred_response,
                "reasoning": preference.reasoning,
                "confidence": preference.confidence,
                "criteria_scores": preference.criteria_scores,
                "principles": principles or self.constitutional_principles
            }
            
            preference_dataset.append(entry)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Collected {i + 1} preference samples")
                
        logger.info(f"Collected {len(preference_dataset)} preference samples total")
        
        return preference_dataset
        
    def evaluate_preference_consistency(
        self,
        preferences: List[PreferenceOutput],
        ground_truth: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate consistency of preference predictions."""
        
        metrics = {}
        
        # Basic statistics
        a_count = sum(1 for p in preferences if p.preferred_response == "A")
        b_count = sum(1 for p in preferences if p.preferred_response == "B")
        
        metrics["preference_balance"] = min(a_count, b_count) / max(a_count, b_count) if max(a_count, b_count) > 0 else 0
        metrics["avg_confidence"] = sum(p.confidence for p in preferences) / len(preferences) if preferences else 0
        
        # Confidence distribution
        high_confidence = sum(1 for p in preferences if p.confidence > 0.7)
        low_confidence = sum(1 for p in preferences if p.confidence < 0.3)
        
        metrics["high_confidence_ratio"] = high_confidence / len(preferences) if preferences else 0
        metrics["low_confidence_ratio"] = low_confidence / len(preferences) if preferences else 0
        
        # If ground truth is provided, compute accuracy
        if ground_truth:
            if len(ground_truth) == len(preferences):
                correct = sum(1 for p, gt in zip(preferences, ground_truth) if p.preferred_response == gt)
                metrics["accuracy"] = correct / len(preferences)
                
                # Confidence-weighted accuracy
                weighted_correct = sum(p.confidence for p, gt in zip(preferences, ground_truth) if p.preferred_response == gt)
                total_confidence = sum(p.confidence for p in preferences)
                metrics["confidence_weighted_accuracy"] = weighted_correct / total_confidence if total_confidence > 0 else 0
                
        return metrics
        
    def analyze_preference_reasons(self, preferences: List[PreferenceOutput]) -> Dict[str, Any]:
        """Analyze common reasons for preferences."""
        
        analysis = {
            "common_reasons": {},
            "criteria_importance": {},
            "avg_reasoning_length": 0
        }
        
        # Extract common phrases from reasoning
        all_reasoning = " ".join(p.reasoning for p in preferences)
        words = all_reasoning.lower().split()
        
        # Count word frequencies (simple approach)
        word_counts = {}
        for word in words:
            if len(word) > 3:  # ignore short words
                word_counts[word] = word_counts.get(word, 0) + 1
                
        # Get top reasons
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        analysis["common_reasons"] = dict(sorted_words[:10])
        
        # Analyze criteria importance
        criteria_totals = {}
        for preference in preferences:
            for criterion, score in preference.criteria_scores.items():
                if criterion not in criteria_totals:
                    criteria_totals[criterion] = []
                criteria_totals[criterion].append(score)
                
        for criterion, scores in criteria_totals.items():
            analysis["criteria_importance"][criterion] = sum(scores) / len(scores)
            
        # Average reasoning length
        if preferences:
            analysis["avg_reasoning_length"] = sum(len(p.reasoning.split()) for p in preferences) / len(preferences)
            
        return analysis