# Constitutional AI Implementation

A research implementation of Constitutional AI methodology for training AI assistants to be helpful, harmless, and honest using AI feedback. This project reproduces key concepts from Anthropic's Constitutional AI paper for educational and research purposes.

## Overview

This implementation demonstrates Constitutional AI training through two phases:

1. **Phase 1: Constitutional Fine-tuning** - Models learn to critique and revise responses based on constitutional principles
2. **Phase 2: Reinforcement Learning from AI Feedback** - Policy optimization using AI-generated preferences and reward models

**Note**: This is a research implementation intended for learning and experimentation. Production deployment would require additional safety measures, extensive testing, and computational resources.

## Architecture

```
constitutional_ai/
├── src/
│   ├── models/                  # Model implementations
│   │   ├── constitutional_model.py
│   │   ├── critique_model.py
│   │   ├── revision_model.py
│   │   ├── preference_model.py
│   │   └── reward_model.py
│   ├── training/               # Training pipelines
│   │   ├── constitutional_trainer.py
│   │   ├── phase1_trainer.py
│   │   ├── phase2_trainer.py
│   │   └── ppo_trainer.py
│   ├── data_processing/        # Data handling
│   │   ├── constitutional_dataset.py
│   │   └── hh_rlhf_processor.py
│   ├── evaluation/            # Evaluation metrics
│   │   └── constitutional_evaluator.py
│   └── utils/                 # Utilities
│       ├── config.py
│       ├── logging.py
│       └── constants.py
├── configs/                   # Configuration files
├── scripts/                   # Training scripts
├── tests/                     # Unit tests
├── notebooks/                 # Jupyter notebooks
└── docs/                      # Documentation
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- At least 8GB RAM (16GB+ recommended for training)

### Installation

```bash
# Clone the repository
cd constitutional_ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```bash
# Run demo to see the system in action
python scripts/demo.py

# Train with default configuration (small scale)
python scripts/train_constitutional_ai.py --max_samples 100

# Train only Phase 1 with custom config
python scripts/train_constitutional_ai.py --phase 1 --config configs/default_config.json

# Evaluate a trained model
python scripts/evaluate_model.py --model_path outputs/final_model
```

### Configuration

The system uses JSON configuration files. Example:

```json
{
  "model": {
    "model_name": "microsoft/DialoGPT-medium",
    "max_length": 512,
    "temperature": 0.7
  },
  "training": {
    "batch_size": 4,
    "learning_rate": 5e-5,
    "num_epochs": 3
  },
  "constitutional_principles": [
    "Choose the response that is more helpful, honest, and harmless.",
    "Choose the response that avoids harmful or dangerous content."
  ]
}
```

## Constitutional Principles

The system uses a set of constitutional principles to guide AI behavior:

1. **Helpfulness** - Responses should be useful and address the user's needs
2. **Harmlessness** - Avoid harmful, dangerous, or illegal content
3. **Honesty** - Be truthful and acknowledge uncertainty
4. **Respect** - Consider different viewpoints respectfully
5. **Autonomy** - Promote human well-being and decision-making
6. **Balance** - Avoid extreme positions
7. **Privacy** - Protect user privacy and avoid surveillance
8. **Transparency** - Be clear about limitations
9. **Critical Thinking** - Encourage thoughtful analysis
10. **Constructiveness** - Focus on solutions and positive outcomes

## Training Process

### Phase 1: Constitutional Fine-tuning

1. **Critique Generation**: AI generates critiques of responses based on constitutional principles
2. **Revision Generation**: AI creates improved responses addressing the critiques
3. **Supervised Fine-tuning**: Model is trained on the revised responses

### Phase 2: Reinforcement Learning from AI Feedback

1. **Preference Collection**: AI compares response pairs and generates preferences
2. **Reward Model Training**: Train reward model on AI-generated preferences
3. **PPO Optimization**: Use PPO to optimize policy with constitutional rewards

## Features

- **Modular Architecture**: Components can be used independently or together
- **Multiple Model Support**: Compatible with various transformer models (GPT-2, DialoGPT, etc.)
- **Configurable Training**: JSON-based configuration for easy experimentation
- **Evaluation Tools**: Built-in metrics for safety, constitutional compliance, and quality
- **Extensible Design**: Easy to add new principles or modify existing behavior

## Evaluation

The system includes evaluation tools for:

- **Constitutional Compliance**: Measures adherence to defined principles
- **Safety Assessment**: Basic harmful content detection and filtering
- **Quality Metrics**: Response helpfulness, coherence, and relevance
- **Benchmark Comparison**: Standard NLP metrics (BLEU, ROUGE, etc.)

**Note**: Evaluation metrics are basic implementations suitable for research. Production systems would require more sophisticated safety measures.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black src/
isort src/

# Lint code
flake8 src/
```

### Adding New Principles

1. Add principle to `constitutional_principles` in config
2. Update critique and revision templates in `constants.py`
3. Add corresponding evaluation metrics if needed

## Limitations

- **Research Implementation**: This is a proof-of-concept, not production-ready
- **Limited Safety**: Basic safety measures only; extensive red-teaming needed for production
- **Computational Requirements**: Training requires significant computational resources
- **Model Dependencies**: Performance depends on the quality of the base language model
- **Evaluation Scope**: Evaluation metrics are simplified compared to production systems

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Anthropic for the Constitutional AI methodology and research
- HuggingFace for the transformers library
- The open-source AI research community

## Support

For questions and support:
- Open an issue on GitHub
- Check the documentation in the docs/ folder
- Review existing discussions and issues

---

**Important**: This implementation is for research and educational purposes. It demonstrates Constitutional AI concepts but should not be used in production without extensive additional safety measures, testing, and validation.