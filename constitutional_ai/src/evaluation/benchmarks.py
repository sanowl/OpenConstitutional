"""
Benchmark evaluation for Constitutional AI.
Implements standard NLP benchmarks and metrics.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
import re
from collections import Counter
from dataclasses import dataclass
import math

from ..utils.config import Config
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Result from benchmark evaluation."""
    metric_name: str
    score: float
    details: Dict[str, Any]


class BenchmarkEvaluator:
    """Evaluator for standard NLP benchmarks."""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize metrics
        self.metrics = {
            "bleu": self._compute_bleu_score,
            "rouge": self._compute_rouge_score,
            "bert_score": self._compute_bert_score,
            "perplexity": self._compute_perplexity,
            "diversity": self._compute_diversity_metrics,
            "coherence": self._compute_coherence_metrics,
            "relevance": self._compute_relevance_metrics
        }
        
        logger.info("Initialized Benchmark Evaluator")
        
    def evaluate(
        self,
        questions: List[str],
        responses: List[str],
        references: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate responses using specified metrics."""
        
        if metrics is None:
            metrics = list(self.metrics.keys())
            
        results = {}
        
        for metric_name in metrics:
            if metric_name in self.metrics:
                try:
                    logger.info(f"Computing {metric_name} metric...")
                    
                    if metric_name in ["bleu", "rouge", "bert_score"] and references is None:
                        logger.warning(f"Skipping {metric_name} - requires references")
                        continue
                        
                    metric_func = self.metrics[metric_name]
                    
                    if metric_name in ["bleu", "rouge", "bert_score"]:
                        score = metric_func(responses, references)
                    else:
                        score = metric_func(questions, responses)
                        
                    results[metric_name] = score
                    
                except Exception as e:
                    logger.error(f"Error computing {metric_name}: {e}")
                    results[metric_name] = {"error": str(e)}
                    
        return results
        
    def _compute_bleu_score(
        self, responses: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Compute BLEU score."""
        
        bleu_scores = []
        
        for response, reference in zip(responses, references):
            score = self._bleu_sentence(response, reference)
            bleu_scores.append(score)
            
        return {
            "bleu_1": np.mean([s[0] for s in bleu_scores]),
            "bleu_2": np.mean([s[1] for s in bleu_scores]),
            "bleu_3": np.mean([s[2] for s in bleu_scores]),
            "bleu_4": np.mean([s[3] for s in bleu_scores]),
            "bleu_avg": np.mean([np.mean(s) for s in bleu_scores])
        }
        
    def _bleu_sentence(self, candidate: str, reference: str) -> List[float]:
        """Compute BLEU score for a single sentence."""
        
        candidate_tokens = candidate.lower().split()
        reference_tokens = reference.lower().split()
        
        if not candidate_tokens or not reference_tokens:
            return [0.0, 0.0, 0.0, 0.0]
            
        bleu_scores = []
        
        for n in range(1, 5):  # BLEU-1 to BLEU-4
            candidate_ngrams = self._get_ngrams(candidate_tokens, n)
            reference_ngrams = self._get_ngrams(reference_tokens, n)
            
            if not candidate_ngrams:
                bleu_scores.append(0.0)
                continue
                
            matches = 0
            for ngram in candidate_ngrams:
                if ngram in reference_ngrams:
                    matches += min(candidate_ngrams[ngram], reference_ngrams[ngram])
                    
            precision = matches / sum(candidate_ngrams.values())
            
            # Brevity penalty
            if len(candidate_tokens) > len(reference_tokens):
                bp = 1.0
            else:
                bp = math.exp(1 - len(reference_tokens) / len(candidate_tokens))
                
            bleu_scores.append(precision * bp)
            
        return bleu_scores
        
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-grams from tokens."""
        
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams[ngram] += 1
            
        return ngrams
        
    def _compute_rouge_score(
        self, responses: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Compute ROUGE score."""
        
        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []
        
        for response, reference in zip(responses, references):
            r1 = self._rouge_n(response, reference, 1)
            r2 = self._rouge_n(response, reference, 2)
            rl = self._rouge_l(response, reference)
            
            rouge_1_scores.append(r1)
            rouge_2_scores.append(r2)
            rouge_l_scores.append(rl)
            
        return {
            "rouge_1": np.mean(rouge_1_scores),
            "rouge_2": np.mean(rouge_2_scores),
            "rouge_l": np.mean(rouge_l_scores),
            "rouge_avg": np.mean([
                np.mean(rouge_1_scores),
                np.mean(rouge_2_scores),
                np.mean(rouge_l_scores)
            ])
        }
        
    def _rouge_n(self, candidate: str, reference: str, n: int) -> float:
        """Compute ROUGE-N score."""
        
        candidate_tokens = candidate.lower().split()
        reference_tokens = reference.lower().split()
        
        candidate_ngrams = self._get_ngrams(candidate_tokens, n)
        reference_ngrams = self._get_ngrams(reference_tokens, n)
        
        if not reference_ngrams:
            return 0.0
            
        matches = 0
        for ngram in reference_ngrams:
            if ngram in candidate_ngrams:
                matches += min(candidate_ngrams[ngram], reference_ngrams[ngram])
                
        return matches / sum(reference_ngrams.values())
        
    def _rouge_l(self, candidate: str, reference: str) -> float:
        """Compute ROUGE-L score (longest common subsequence)."""
        
        candidate_tokens = candidate.lower().split()
        reference_tokens = reference.lower().split()
        
        lcs_length = self._lcs_length(candidate_tokens, reference_tokens)
        
        if not candidate_tokens or not reference_tokens:
            return 0.0
            
        precision = lcs_length / len(candidate_tokens)
        recall = lcs_length / len(reference_tokens)
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * precision * recall / (precision + recall)
        
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute longest common subsequence length."""
        
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                    
        return dp[m][n]
        
    def _compute_bert_score(
        self, responses: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Compute BERTScore (simplified version)."""
        
        # This is a simplified version - in practice, you'd use the bert_score library
        # For now, we'll compute a simple embedding-based similarity
        
        similarities = []
        
        for response, reference in zip(responses, references):
            # Simple word overlap similarity as proxy
            response_words = set(response.lower().split())
            reference_words = set(reference.lower().split())
            
            if not response_words or not reference_words:
                similarity = 0.0
            else:
                intersection = response_words & reference_words
                union = response_words | reference_words
                similarity = len(intersection) / len(union)
                
            similarities.append(similarity)
            
        return {
            "bert_score_precision": np.mean(similarities),
            "bert_score_recall": np.mean(similarities),
            "bert_score_f1": np.mean(similarities)
        }
        
    def _compute_perplexity(
        self, questions: List[str], responses: List[str]
    ) -> Dict[str, float]:
        """Compute perplexity metrics."""
        
        # Simplified perplexity computation
        perplexities = []
        
        for response in responses:
            words = response.split()
            if not words:
                perplexities.append(float('inf'))
                continue
                
            # Simple character-based perplexity estimation
            unique_chars = len(set(response.lower()))
            total_chars = len(response)
            
            if unique_chars == 0:
                perplexity = float('inf')
            else:
                # Simplified perplexity calculation
                perplexity = total_chars / unique_chars
                
            perplexities.append(perplexity)
            
        # Filter out infinite values for mean calculation
        finite_perplexities = [p for p in perplexities if p != float('inf')]
        
        return {
            "perplexity_mean": np.mean(finite_perplexities) if finite_perplexities else float('inf'),
            "perplexity_std": np.std(finite_perplexities) if finite_perplexities else 0.0,
            "perplexity_min": np.min(finite_perplexities) if finite_perplexities else float('inf'),
            "perplexity_max": np.max(finite_perplexities) if finite_perplexities else float('inf')
        }
        
    def _compute_diversity_metrics(
        self, questions: List[str], responses: List[str]
    ) -> Dict[str, float]:
        """Compute diversity metrics."""
        
        # Type-Token Ratio (TTR)
        all_tokens = []
        for response in responses:
            tokens = response.lower().split()
            all_tokens.extend(tokens)
            
        unique_tokens = len(set(all_tokens))
        total_tokens = len(all_tokens)
        ttr = unique_tokens / total_tokens if total_tokens > 0 else 0.0
        
        # Distinct n-grams
        distinct_1 = set()
        distinct_2 = set()
        distinct_3 = set()
        total_1 = 0
        total_2 = 0
        total_3 = 0
        
        for response in responses:
            tokens = response.lower().split()
            
            # Unigrams
            for token in tokens:
                distinct_1.add(token)
                total_1 += 1
                
            # Bigrams
            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i + 1])
                distinct_2.add(bigram)
                total_2 += 1
                
            # Trigrams
            for i in range(len(tokens) - 2):
                trigram = (tokens[i], tokens[i + 1], tokens[i + 2])
                distinct_3.add(trigram)
                total_3 += 1
                
        return {
            "ttr": ttr,
            "distinct_1": len(distinct_1) / total_1 if total_1 > 0 else 0.0,
            "distinct_2": len(distinct_2) / total_2 if total_2 > 0 else 0.0,
            "distinct_3": len(distinct_3) / total_3 if total_3 > 0 else 0.0,
            "vocab_size": unique_tokens,
            "avg_response_length": np.mean([len(r.split()) for r in responses])
        }
        
    def _compute_coherence_metrics(
        self, questions: List[str], responses: List[str]
    ) -> Dict[str, float]:
        """Compute coherence metrics."""
        
        coherence_scores = []
        
        for response in responses:
            sentences = [s.strip() for s in response.split('.') if s.strip()]
            
            if len(sentences) < 2:
                coherence_scores.append(0.5)  # Neutral score for single sentence
                continue
                
            # Simple coherence based on word overlap between sentences
            overlaps = []
            for i in range(len(sentences) - 1):
                words1 = set(sentences[i].lower().split())
                words2 = set(sentences[i + 1].lower().split())
                
                if words1 and words2:
                    overlap = len(words1 & words2) / len(words1 | words2)
                    overlaps.append(overlap)
                    
            coherence = np.mean(overlaps) if overlaps else 0.0
            coherence_scores.append(coherence)
            
        return {
            "coherence_mean": np.mean(coherence_scores),
            "coherence_std": np.std(coherence_scores),
            "coherence_min": np.min(coherence_scores),
            "coherence_max": np.max(coherence_scores)
        }
        
    def _compute_relevance_metrics(
        self, questions: List[str], responses: List[str]
    ) -> Dict[str, float]:
        """Compute relevance metrics."""
        
        relevance_scores = []
        
        for question, response in zip(questions, responses):
            question_words = set(question.lower().split())
            response_words = set(response.lower().split())
            
            if not question_words or not response_words:
                relevance_scores.append(0.0)
                continue
                
            # Simple relevance based on word overlap
            overlap = len(question_words & response_words)
            relevance = overlap / len(question_words)
            relevance_scores.append(relevance)
            
        return {
            "relevance_mean": np.mean(relevance_scores),
            "relevance_std": np.std(relevance_scores),
            "relevance_min": np.min(relevance_scores),
            "relevance_max": np.max(relevance_scores)
        }
        
    def generate_benchmark_report(
        self,
        questions: List[str],
        responses: List[str],
        references: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        
        logger.info("Generating benchmark report...")
        
        # Evaluate all metrics
        results = self.evaluate(questions, responses, references)
        
        # Generate summary
        summary = {
            "total_samples": len(responses),
            "avg_response_length": np.mean([len(r.split()) for r in responses]),
            "std_response_length": np.std([len(r.split()) for r in responses]),
            "results": results
        }
        
        # Add interpretation
        summary["interpretation"] = self._interpret_results(results)
        
        # Save report if path provided
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Benchmark report saved to {save_path}")
            
        return summary
        
    def _interpret_results(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Interpret benchmark results."""
        
        interpretations = {}
        
        # BLEU interpretation
        if "bleu" in results:
            bleu_avg = results["bleu"].get("bleu_avg", 0)
            if bleu_avg > 0.4:
                interpretations["bleu"] = "Excellent similarity to references"
            elif bleu_avg > 0.2:
                interpretations["bleu"] = "Good similarity to references"
            elif bleu_avg > 0.1:
                interpretations["bleu"] = "Moderate similarity to references"
            else:
                interpretations["bleu"] = "Low similarity to references"
                
        # ROUGE interpretation
        if "rouge" in results:
            rouge_avg = results["rouge"].get("rouge_avg", 0)
            if rouge_avg > 0.5:
                interpretations["rouge"] = "High content overlap with references"
            elif rouge_avg > 0.3:
                interpretations["rouge"] = "Moderate content overlap with references"
            else:
                interpretations["rouge"] = "Low content overlap with references"
                
        # Diversity interpretation
        if "diversity" in results:
            ttr = results["diversity"].get("ttr", 0)
            if ttr > 0.7:
                interpretations["diversity"] = "High lexical diversity"
            elif ttr > 0.5:
                interpretations["diversity"] = "Moderate lexical diversity"
            else:
                interpretations["diversity"] = "Low lexical diversity"
                
        # Coherence interpretation
        if "coherence" in results:
            coherence_mean = results["coherence"].get("coherence_mean", 0)
            if coherence_mean > 0.6:
                interpretations["coherence"] = "High coherence between sentences"
            elif coherence_mean > 0.4:
                interpretations["coherence"] = "Moderate coherence between sentences"
            else:
                interpretations["coherence"] = "Low coherence between sentences"
                
        return interpretations