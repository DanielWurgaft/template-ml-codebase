#########################
# imports
##########################

from .config import Config
import torch
import os
import pandas as pd

# set up tqdm
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from IPython import get_ipython
from transformers import AutoTokenizer
from pyprojroot import here
import sys
# add here to path
sys.path.append(str(here()))

# load environment variables
from dotenv import load_dotenv
load_dotenv()

#########################
# general helper functions
#########################

def tqdm_func():
    if get_ipython() is None:
        return tqdm  # Running in a script
    else:
        return tqdm_notebook  # Running in Jupyter Notebook
    
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


#########################
# Setup
#########################

def load_eval_results(setting, experiment_name):
    """Load evaluation results from cache directory."""
    def get_metrics(data_dir):
        if not os.path.exists(os.path.join(data_dir, "metrics.pt")):
            return None
        metrics_path = os.path.join(data_dir, "metrics.pt")
        metrics = torch.load(metrics_path, weights_only=False)
        return metrics
    
    experiment_dir = os.path.join(os.getenv("CACHE_DIR"), setting, experiment_name)    
    rows = []
    
    for subdir in os.listdir(experiment_dir):
        if not os.path.isdir(os.path.join(experiment_dir, subdir)):
            continue
        subdir_path = os.path.join(experiment_dir, subdir)            
        config = Config.from_yaml(os.path.join(subdir_path, 'config_resolved.yaml'))
        
        # Parse subdir_name into variables by splitting on ,
        subdir_vars = [subdir.split('=') for subdir in subdir.split(',')]
        subdir_vars = {(elem[0].split('.')[1] if len(elem[0].split('.')) > 1 else elem[0]): elem[1] for elem in subdir_vars}

        # Parse subdir_vars values into numerics if possible
        for key, value in subdir_vars.items():
            try:
                subdir_vars[key] = float(value)
            except (TypeError, ValueError):
                pass
        
        # Determine if this is an LLM or small LM experiment based on config
        if config.model_kwargs.model_type == "small_lm":
            # Small LM experiments: metrics stored in checkpoint subdirectories
            # Create one row per checkpoint
            checkpoints = sorted([int(item) for item in os.listdir(os.path.join(subdir_path, "checkpoints"))])
            if not checkpoints:
                print(f"No checkpoints found for {subdir_path}")
                continue
            for checkpoint in checkpoints:
                metrics = get_metrics(os.path.join(subdir_path, "checkpoints", str(checkpoint)))
                rows.append({
                    'config': config,
                    'metrics': metrics,
                    'checkpoint': checkpoint,
                    **subdir_vars
                })
        else:
            # LLM experiments or Bayesian models: metrics stored directly in experiment directory
            metrics = get_metrics(subdir_path)
            config.tokenizer = AutoTokenizer.from_pretrained(config.model_kwargs.model_path) if config.model_kwargs.model_type == "llm" else None
            rows.append({
                'config': config,
                'metrics': metrics,
                'checkpoint': None,
                **subdir_vars
            })

    df = pd.DataFrame(rows).reset_index(drop=True)
    return df