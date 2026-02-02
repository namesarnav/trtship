# RLHF for Code Generation

A complete implementation of Reinforcement Learning from Human Feedback (RLHF) for training code generation models. This project implements the full pipeline: supervised fine-tuning (SFT), reward model training, and PPO-based reinforcement learning.

## Overview

This project trains language models to generate better code through:
1. **Supervised Fine-Tuning (SFT)**: Initial fine-tuning on high-quality code examples
2. **Reward Model Training**: Learning to score code quality from human preferences
3. **PPO Training**: Optimizing the policy using the reward model with PPO

## Features

- RLHF pipeline from scratch
- Multi GPU training with DeepSpeed/FSDP
- Evaluation on HumanEval and MBPP
- Experiment tracking with Weights & Biases
- Modular, extensible architecture

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/code-rlhf.git
cd code-rlhf

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
# Or for development
pip install -e ".[dev]"
```

## Quick Start

### 1. Supervised Fine-Tuning

```bash
python data/prepare_sft_data.py --dataset CodeAlpaca --output data/sft_train.jsonl
bash scripts/train_sft.sh
```

### 2. Train Reward Model

```bash
python data/prepare_preference_data.py --dataset code-rm-bench --output data/preferences.jsonl
bash scripts/train_reward_model.sh
```

### 3. PPO Training

```bash
bash scripts/train_ppo.sh
```

### 4. Evaluation

```bash
bash scripts/evaluate.sh --checkpoint checkpoints/ppo_final
```

## Configuration

Edit `configs/training_config.yaml` to adjust:
- Model size and architecture
- PPO hyperparameters (learning rate, KL coefficient, etc.)
- Batch sizes and gradient accumulation
- Distributed training settings

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-6 | Policy network learning rate |
| `kl_coef` | 0.05 | KL penalty coefficient |
| `clip_range` | 0.2 | PPO clipping range |
| `ppo_epochs` | 4 | Optimization epochs per batch |
| `batch_size` | 32 | Training batch size |

## Evaluation Metrics

- **HumanEval Pass@1**: Percentage of problems solved on first attempt
- **MBPP Pass@1**: Performance on Mostly Basic Python Problems
- **Reward Model Score**: Average reward on generated samples
- **KL Divergence**: Distance from reference policy

## Results

| Model | HumanEval Pass@1 | MBPP Pass@1 |
|-------|------------------|-------------|
| Base Model | TBD | TBD |
| SFT Baseline | TBD | TBD |
| RLHF (PPO) | TBD | TBD |

## Development

```bash
# Run tests
pytest tests/

# Format code
black .
isort .

# Lint
flake8 .
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{code-rlhf-2024,
  author = {Your Name},
  title = {RLHF for Code Generation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/code-rlhf}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built on HuggingFace Transformers
- Inspired by InstructGPT and OpenAI's RLHF work
- Uses evaluation from HumanEval and MBPP benchmarks
