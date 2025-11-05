# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a machine learning research project focused on normative scaling experiments studying in-context learning with structured regression tasks. The project uses **Hydra** for configuration management and experiment orchestration.

## Essential Commands

### Running Experiments

```bash
# Basic training run with default configuration
python -m src.code.train

# Run with specific experiment template
python -m src.code.train +experiment=default

# Parameter sweeps with multirun
python -m src.code.train -m model_kwargs.hidden_size=128,256,512,1024,2048

# Override specific parameters
python -m src.code.train model_kwargs.hidden_size=128 training_kwargs.learning_rate=0.001

# Custom experiment with name
python -m src.code.train name=my_experiment model_kwargs.hidden_size=128
```

### Running Evaluation

```bash
# Evaluate trained model checkpoints (iterates through all checkpoints)
python -m src.code.eval experiment=my_experiment

# Evaluate pretrained LLM on the task
python -m src.code.eval experiment=llm_eval model_kwargs.model_path=meta-llama/Llama-3.1-8B

# Multirun evaluation across models
python -m src.code.eval -m experiment=llm_eval model_kwargs.model_path=meta-llama/Llama-3.1-8B,meta-llama/Llama-3.1-70B

# Override evaluation settings
python -m src.code.eval experiment=my_experiment eval_kwargs.num_eval_batches=100

# Force overwrite existing evaluation results
python -m src.code.eval experiment=my_experiment overwrite=true
```

### Environment Setup

```bash
# Set required environment variables
export CACHE_DIR="/path/to/cache/directory"
export WANDB_API_KEY="your_wandb_key"  # Optional for experiment tracking

# Install dependencies (no requirements.txt - install manually)
pip install torch hydra-core omegaconf wandb numpy pyprojroot python-dotenv safetensors transformers mup
```

### Debug and Development

```bash
# Run with Hydra debug info
python -m src.code.train hydra.verbose=true

# Resume training (set overwrite=false)
python -m src.code.train name=my_experiment overwrite=false
```

## Architecture Overview

### Core Components

1. **Hydra Configuration System** (`conf/`): Hierarchical, composable configurations

   - `config.yaml`: Main config with defaults
   - `setting/`: Problem types
   - `model/`: Neural network architectures
   - `training/`: Optimization parameters
   - `experiment/`: Pre-defined experiment templates

2. **Training Pipeline** (`src/code/train.py`):

   - Hydra-decorated main function using HuggingFace Transformers
   - Automatic directory management and experiment tracking
   - Wandb integration for metrics logging

3. **Configuration Management** (`src/code/config.py`):

   - `Config` class wraps Hydra DictConfig
   - Handles checkpoint scheduling (sqrt_schedule, linear_schedule, end_only)
   - Automatic directory creation and path resolution

4. **Data Generation** (`src/code/dgp.py`):

   - Task priors (uniform, power_law, constant_freq)
   - MixtureDataset

5. **Model Implementation** (`src/code/models.py`):

   - MLP architecture
   - Transformer architecture

6. **Evaluation Pipeline** (`src/code/eval.py`):

   - `ModelEvaluator` class: Reusable evaluation with configurable metrics and modes
   - Evaluation modes: "standard" (mixed tasks) and "single_task" (per-task evaluation)
   - `run_evaluation()`: Hydra-decorated main function for standalone evaluation
   - Supports checkpoint-based evaluation for trained models and single-run evaluation for pretrained LLMs
   - Returns metrics, predictions, labels, and optionally hidden states

7. **Analysis Tools** (`src/code/analysis_utils.py`):

   - `load_eval_results()`: Load evaluation metrics and configs from completed experiments
   - Returns DataFrame with columns: 'config', 'metrics', 'checkpoint', and sweep parameters

8. **Plotting** (`src/code/plotting_utils.py`):

   - Plotting helpers

### Data Flow

1. **Experiment Launch**: Hydra manages configuration composition and multirun orchestration
2. **Directory Structure**: Automatic organization in `${CACHE_DIR}/experiments/${setting}/${name}/`
3. **Training**: HuggingFace Trainer with custom callbacks for checkpointing and metrics
4. **Analysis**: Load experiment results using `make_results_dfs_from_experiment_dirs()`

### Key Design Patterns

- **Hydra Integration**: All configuration via hierarchical YAML files, no hardcoded parameters
- **Automatic Path Management**: Hydra handles output directories, no manual path construction
- **Config-Driven Analysis**: Analysis functions work directly with Hydra experiment outputs

## Important Implementation Details

### Configuration System

- Uses Hydra's `@package` directives for hierarchical composition
- `Config` class processes save_steps into checkpoint schedules automatically
- Environment variable interpolation via `${oc.env:VARIABLE_NAME}`
- Multirun creates separate directories per parameter combination

### Training Architecture

- Uses HuggingFace Transformers framework with custom callbacks
- `CustomSaveCallback`: Handles model checkpointing at specified steps
- `StepMetricsCallback`: Computes and logs per-task metrics during training
- `PreBatchedDataCollator`: Manages pre-batched data efficiently

### Analysis Workflow

- **Post-experiment analysis**: Load completed experiments using `load_eval_results(setting, experiment_name)`
- **Automatic config loading**: Directly reads configs from `config_resolved.yaml` files
- **Checkpoint handling**: For small_lm, creates one DataFrame row per checkpoint; for LLM/Bayes, one row per config
- **Sweep parameter parsing**: Automatically extracts parameter values from directory names

### Data Generation

- Datasets cached in `${CACHE_DIR}/experiments/${setting}/${name}/data/`
- `MixtureDataset` base class with task prior sampling

### Evaluation System

- **ModelEvaluator class**: Handles batch-by-batch evaluation and metric aggregation
- **Evaluation modes**:
  - `standard`: Evaluate on mixed-task batches (default training distribution)
  - `single_task`: Evaluate performance on individual tasks separately
- **Checkpoint handling**: For trained models (`small_lm`), automatically iterates through all checkpoints
- **LLM evaluation**: For pretrained models, runs single evaluation and saves metrics to `metrics.pt`
- **Hidden states**: Can optionally extract and save hidden states for representational analysis
- **Metrics saved to**: `{checkpoint_dir}/metrics.pt` for trained models, `{output_dir}/metrics.pt` for LLMs

## Common Patterns

### Adding New Experiments

1. Create experiment template in `conf/experiment/my_experiment.yaml`
2. Use `@package _global_` to override base config
3. Define sweep parameters under `hydra.sweeper.params`

### Running Standalone Evaluation

```bash
# Evaluate trained model checkpoints
python -m src.code.eval experiment=my_experiment

# Evaluate pretrained LLM (single run)
python -m src.code.eval experiment=llm_eval

# Multirun evaluation across multiple LLMs
python -m src.code.eval -m experiment=llm_eval model_kwargs.model_path=meta-llama/Llama-3.1-8B,meta-llama/Llama-3.1-70B

# Override evaluation parameters
python -m src.code.eval experiment=my_experiment eval_kwargs.num_eval_batches=100 eval_kwargs.return_all=true

# Force re-evaluation (overwrite existing metrics)
python -m src.code.eval experiment=my_experiment overwrite=true
```

### Analysis After Training

```python
from src.code.analysis_utils import load_eval_results

# Load experiment results (automatically loads configs and metrics)
df = load_eval_results(
    setting="balls_and_urns",
    experiment_name="my_experiment"
)

# Access results
# df['config'] - Config objects for each run
# df['metrics'] - Metrics dict for each checkpoint/config
# df['checkpoint'] - Training step (None for LLM/Bayes)
# Additional columns for any sweep parameters
```

### Custom Model Architectures

- Add model class to `src/code/models.py`
- Create config in `conf/model/my_model.yaml`
- Register in `init_model()` function

## File Organization

- `conf/`: Hydra configuration hierarchy
- `src/code/`: Core implementation (training, models, data, analysis)
- `src/notebooks/`: Jupyter notebooks for analysis and visualization
- `figures/`: Generated plots and visualizations
- `.env`: Environment variables (CACHE_DIR, WANDB_API_KEY)

## Key Dependencies

- **Hydra**: Configuration management and multirun orchestration
- **HuggingFace Transformers**: Training loop and model management
- **PyTorch**: Neural network implementation and training
- **Wandb**: Experiment tracking and metrics logging
- **OmegaConf**: Configuration object manipulation

This codebase prioritizes reproducible research through systematic configuration management and automated experiment organization.

## Code Style Preferences

- **Write succinct, readable code** with comments only where needed for clarity
- **Avoid try-catch blocks** unless explicitly requested - let functions fail fast with clear error messages
- **Prefer explicit failures** over silent error handling - errors should surface immediately to aid debugging