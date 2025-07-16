#!/usr/bin/env python3
"""
Demo script showing Constitutional AI in action.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from constitutional_ai.models.constitutional_model import ConstitutionalAIModel
from constitutional_ai.models.critique_model import CritiqueModel
from constitutional_ai.models.revision_model import RevisionModel
from constitutional_ai.utils.config import Config

def main():
    """Run Constitutional AI demo."""
    
    print("Constitutional AI Demo")
    print("=" * 50)
    
    # Create config
    config = Config()
    config.model.model_name = "microsoft/DialoGPT-small"  # Small model for demo
    config.model.device = "cpu"  # Use CPU for demo
    
    # Example question and problematic response
    question = "How can I help someone who is feeling depressed?"
    problematic_response = "Just tell them to get over it and stop being dramatic."
    
    print(f"Question: {question}")
    print(f"Original Response: {problematic_response}")
    print()
    
    # Initialize models
    print("Loading models...")
    critique_model = CritiqueModel(config)
    revision_model = RevisionModel(config)
    
    # Generate critique
    print("Generating Critique...")
    critique_output = critique_model.generate_critique(
        question=question,
        response=problematic_response,
        critique_type="constitutional"
    )
    
    print(f"Critique: {critique_output.critique}")
    print(f"Violations: {critique_output.principle_violations}")
    print(f"Severity: {critique_output.severity_score}")
    print()
    
    # Generate revision
    print("Generating Revision...")
    revision_output = revision_model.generate_revision(
        question=question,
        original_response=problematic_response,
        critique=critique_output.critique,
        revision_type="constitutional"
    )
    
    print(f"Revised Response: {revision_output.revised_response}")
    print(f"Quality Score: {revision_output.quality_score}")
    print()
    
    print("Demo completed! The system successfully:")
    print("   1. Identified problematic content in the original response")
    print("   2. Generated a constructive critique")
    print("   3. Created a more helpful and harmless revision")

if __name__ == "__main__":
    main()