# Template ML Codebase

A machine learning research template for running systematic experiments with Hydra-based configuration management. This template is designed for reproducible ML research with clean experiment organization and tracking.

## Features

- **Hydra Configuration System**: Hierarchical, composable configurations for experiments
- **Flexible Training Pipeline**: Support for custom models and training configurations
- **Automated Experiment Tracking**: Integration with Weights & Biases (wandb)
- **Evaluation Framework**: Standalone evaluation with multiple metrics and modes
- **Reproducible Research**: Automatic directory management and config versioning

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- CUDA (optional, for GPU support)

### Setup

1. Clone the repository:
```bash
git clone git@github.com:DanielWurgaft/template-ml-codebase.git
cd template-ml-codebase
```

2. Install dependencies:
```bash
pip install torch hydra-core omegaconf wandb numpy pyprojroot python-dotenv safetensors transformers mup
```

3. Create a `.env` file in the root directory with the required environment variables (see [Environment Variables](#environment-variables) below).

## Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Required: Cache directory for experiments and data
CACHE_DIR=/path/to/your/cache/directory

# Optional: Weights & Biases configuration
WANDB_API_KEY=your_wandb_api_key_here
WANDB_DIR=/path/to/wandb/logs
WANDB_CACHE_DIR=/path/to/wandb/cache

# Optional: HuggingFace configuration (for pretrained models)
HF_TOKEN=your_huggingface_token_here
HF_HOME=~/.cache/huggingface
```

**Note**:
- `CACHE_DIR` is required for storing experiment outputs and datasets
- Weights & Biases variables are optional but recommended for experiment tracking
- HuggingFace token is only needed if you're using gated models (e.g., Llama)

## Repository Structure

```
template-ml-codebase/
├── conf/                          # Hydra configuration files
│   ├── config.yaml               # Main configuration
│   ├── experiment/               # Experiment templates
│   │   ├── llm_exp.yaml         # LLM evaluation experiments
│   │   ├── small_lm_exp.yaml    # Small model training experiments
│   │   └── bayes_eval.yaml      # Bayesian baseline experiments
│   ├── model/                    # Model architectures
│   │   ├── small_lm.yaml        # Small transformer config
│   │   ├── llm.yaml             # Pretrained LLM config
│   │   └── bayesian_model.yaml  # Bayesian baseline config
│   ├── setting/                  # Task/dataset configurations
│   │   └── balls_and_urns.yaml  # Example task setting
│   ├── training/                 # Training configurations
│   │   └── default.yaml         # Default training params
│   └── eval/                     # Evaluation configurations
│       └── small_lm_eval.yaml   # Evaluation settings
├── src/
│   ├── code/                     # Core implementation
│   │   ├── train.py             # Training pipeline
│   │   ├── eval.py              # Evaluation pipeline
│   │   ├── models.py            # Model architectures
│   │   ├── data.py              # Dataset classes
│   │   ├── config.py            # Configuration management
│   │   ├── analysis_utils.py    # Analysis tools
│   │   ├── plot_utils.py        # Plotting utilities
│   │   └── train_utils.py       # Training helpers
│   ├── notebooks/                # Jupyter notebooks for analysis
│   ├── scripts/                  # Utility scripts
│   └── tests/                    # Unit tests
├── .env                          # Environment variables (create this)
├── .gitignore                    # Git ignore rules
├── CLAUDE.md                     # Guidance for Claude Code
├── LICENSE                       # License file
└── README.md                     # This file
```

### Generated Directory Structure

When you run experiments, Hydra will automatically create organized output directories:

```
$CACHE_DIR/experiments/
└── {setting}/                    # e.g., balls_and_urns
    └── {experiment_name}/        # e.g., my_experiment
        ├── .hydra/              # Hydra config snapshots
        ├── data/                # Generated datasets (cached)
        ├── outputs/             # Model checkpoints and outputs
        │   └── {checkpoint}/    # e.g., 1000, 5000, 10000
        │       ├── model/       # Saved model weights
        │       └── metrics.pt   # Evaluation metrics
        └── wandb/               # W&B logs (if enabled)
```

## Usage

### Training

```bash
# Basic training with default configuration
python -m src.code.train

# Run specific experiment template
python -m src.code.train experiment=small_lm_exp

# Override parameters
python -m src.code.train model_kwargs.hidden_size=256 training_kwargs.learning_rate=1e-4

# Custom experiment name
python -m src.code.train name=my_experiment

# Parameter sweep with multirun
python -m src.code.train -m model_kwargs.hidden_size=128,256,512
```

### Evaluation

```bash
# Evaluate trained model (evaluates all checkpoints)
python -m src.code.eval experiment=small_lm_exp name=my_experiment

# Evaluate pretrained LLM
python -m src.code.eval experiment=llm_exp model_kwargs.model_path=meta-llama/Llama-3.1-8B

# Multirun evaluation across models
python -m src.code.eval -m experiment=llm_exp model_kwargs.model_path=model1,model2,model3

# Override evaluation settings
python -m src.code.eval experiment=my_experiment eval_kwargs.num_eval_batches=100

# Force overwrite existing results
python -m src.code.eval experiment=my_experiment overwrite=true
```

### Analysis

After training and evaluation, analyze results in Python:

```python
from src.code.analysis_utils import load_eval_results

# Load experiment results
df = load_eval_results(
    setting="balls_and_urns",
    experiment_name="my_experiment"
)

# Access metrics and configs
# df contains columns: 'config', 'metrics', 'checkpoint', and any sweep parameters
# For small_lm: one row per checkpoint
# For llm/bayes: one row per model configuration
```

## Configuration System

This template uses [Hydra](https://hydra.cc/) for hierarchical configuration management.

### Key Concepts

1. **Base Configuration** (`conf/config.yaml`): Default settings for all experiments
2. **Experiment Templates** (`conf/experiment/`): Predefined experiment configurations
3. **Composable Configs**: Mix and match model, training, and setting configs
4. **Parameter Overrides**: Override any parameter from the command line

### Creating New Experiments

1. Create a new experiment config in `conf/experiment/my_experiment.yaml`:

```yaml
# @package _global_

name: my_experiment

defaults:
  - /model: small_lm
  - /training: default
  - /setting: balls_and_urns

model_kwargs:
  hidden_size: 256
  num_layers: 4

training_kwargs:
  learning_rate: 1e-4
  num_train_steps: 10000
```

2. Run the experiment:

```bash
python -m src.code.train experiment=my_experiment
```

## Adding Custom Components

### Custom Model

1. Add model class to `src/code/models.py`
2. Create config in `conf/model/my_model.yaml`
3. Register in the model loading function

### Custom Dataset

1. Add dataset class to `src/code/data.py`
2. Create config in `conf/setting/my_setting.yaml`
3. Register in the dataset initialization function

## Best Practices

- **Use experiment templates** for reproducible experiments
- **Version your configs** by checking them into git
- **Name your experiments** meaningfully for easy identification
- **Use multirun** for parameter sweeps instead of manual loops
- **Check W&B** for real-time experiment tracking
- **Save configs** with experiments for full reproducibility

## Troubleshooting

### Common Issues

1. **CACHE_DIR not set**: Make sure `.env` file exists with `CACHE_DIR` defined
2. **Permission errors**: Ensure you have write access to `CACHE_DIR`
3. **Import errors**: Run commands as modules: `python -m src.code.train` (not `python src/code/train.py`)
4. **GPU out of memory**: Reduce `batch_size` in training config

## Contributing

When contributing, please:
- Follow the existing code style (see `CLAUDE.md` for style preferences)
- Add appropriate configs for new models/datasets
- Update documentation for new features
- Write clear commit messages

## License

See [LICENSE](LICENSE) file for details.

## Citation

If you use this template in your research, please cite:

```bibtex
@software{template_ml_codebase,
  author = {Wurgaft, Daniel},
  title = {Template ML Codebase},
  year = {2025},
  url = {https://github.com/DanielWurgaft/template-ml-codebase}
}
```

## Resources

- [Hydra Documentation](https://hydra.cc/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [See CLAUDE.md for detailed implementation notes](CLAUDE.md)
