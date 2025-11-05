import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from pyprojroot import here
import sys
import os
# add here to path
sys.path.append(str(here()))

# load environment variables
from dotenv import load_dotenv
load_dotenv()

from .config import Config
from .models import load_model
from .data import init_dataset

#########################
# Model Evaluator Class
#########################

class ModelEvaluator:
    """model evaluation with incremental metrics computation."""
    
    def __init__(self, model, dataset, config, eval_modes = ["standard", "single_task"], metrics = ["standard"]):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = model.device
        self.model_output_key = model.output_key
        self.batch_size = config.training_kwargs.batch_size
        if hasattr(self.model, "eval"):
            self.model.eval()
        self.metrics = metrics
        self.eval_modes = eval_modes

    def _eval_batch(self, batch, output_hidden_states):
        """Evaluate a single batch and return outputs, labels, hidden_states."""
        
        # Forward pass
        all_outputs = self.model(batch["input_ids"].to(self.device), output_hidden_states=output_hidden_states)
        outputs = all_outputs[self.model_output_key].cpu()
        # filter output distribution
        if outputs.shape[-1] > 1 and outputs.shape[-1] != self.dataset.num_dims:
            outputs = outputs[..., self.dataset.item_tokens]

        labels = {"labels": batch["labels"].cpu(), "eval_labels": batch["eval_labels"].cpu() if "eval_labels" in batch else batch["labels"].cpu()}
        hidden_states = torch.stack([hidden_state.cpu() for hidden_state in all_outputs.get("hidden_states")], dim=1) if output_hidden_states else None
        
        return outputs, labels, hidden_states
    
    def _aggregate_metrics(self, batch_metrics_list):
        """aggregate metrics across batches."""
        aggregated_metrics = {}
        for key in batch_metrics_list[0].keys():
            # if numeric, average
            if isinstance(batch_metrics_list[0][key], (int, float)):
                aggregated_metrics[key] = sum(batch_metrics_list[i][key] for i in range(len(batch_metrics_list))) / len(batch_metrics_list)
            # if list or tensor, stack into a tensor
            elif isinstance(batch_metrics_list[0][key], (list, torch.Tensor)):
                aggregated_metrics[key] = torch.cat([batch_metrics_list[i][key] for i in range(len(batch_metrics_list))], dim=0)
            # if dict, recursively aggregate
            elif isinstance(batch_metrics_list[0][key], dict):
                aggregated_metrics[key] = self._aggregate_metrics([batch_metrics_list[i][key] for i in range(len(batch_metrics_list))])
        return aggregated_metrics
    
    def compute_dataset_metrics(self, return_all, output_hidden_states, num_batches):
        """Compute dataset-level metrics incrementally."""
        batch_metrics_list = []
        with torch.no_grad():
            for _ in range(num_batches):
                batch = self.dataset.__getitem__()
                outputs, labels, hidden_states = self._eval_batch(batch, output_hidden_states)
                for metric in self.metrics:
                    batch_metrics = {}
                    if metric == "standard":
                        batch_metrics.update(self.dataset.compute_metrics((outputs, labels["eval_labels"]), return_all=return_all))

                if return_all:
                    batch_metrics.update({"outputs": outputs, "labels": labels})
                    if output_hidden_states:
                        batch_metrics.update({"hidden_states": hidden_states})

                batch_metrics_list.append(batch_metrics)
        
        return self._aggregate_metrics(batch_metrics_list)

    def evaluate(self, return_all=True, output_hidden_states=True, num_batches=1):
        """Evaluate the model and return metrics."""
        metrics = {}
        for eval_mode in self.eval_modes:
            self.dataset.set_single_task_eval(enable=True if eval_mode == "single_task" else False)
            dataset_metrics = self.compute_dataset_metrics(num_batches=num_batches, return_all=return_all, 
                                output_hidden_states=False if "bayes" in self.config.model_kwargs.model_type else output_hidden_states)
            metrics[eval_mode] = dataset_metrics

        # set single task eval to False
        self.dataset.set_single_task_eval(enable=False)
        # set model to train mode if it has such a method
        if hasattr(self.model, "train"):
            self.model.train()
        return metrics


#########################
# Evaluation
#########################        

@hydra.main(version_base=None, config_path=str(here("conf")), config_name="config")
def run_evaluation(cfg: DictConfig) -> None:
    """Main evaluation function with Hydra configuration.
    
    Usage examples:
    1. Single run with experiment config:
       python -m src.code.analysis_utils experiment=llm_eval
    
    2. Multirun with parameter sweep:
       python -m src.code.analysis_utils experiment=llm_eval --multirun model_kwargs.model_path=meta-llama/Llama-3.1-8B-Instruct, meta-llama/Llama-3.1-405B-Instruct
       
    3. Override specific parameters:
       python -m src.code.analysis_utils experiment=llm_eval random_seed=42 num_tasks=128
    """
    def eval_model(config, model_path, save_path):    
        model, tokenizer = load_model(config, model_path, device="cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(model, "eval"):
            model.eval()
        if tokenizer is not None:
            config.tokenizer = tokenizer
        dataset = init_dataset(config=config, overwrite_data=cfg.overwrite)
        evaluator = ModelEvaluator(model, dataset, config)
        metrics = evaluator.evaluate(num_batches=config.eval_kwargs.num_eval_batches, return_all=config.eval_kwargs.return_all)
        torch.save(metrics, save_path)
        print(f"Metrics saved to {save_path}")

    config = Config(cfg, make_run_dirs=True)

    if config.model_kwargs.model_type == "small_lm": # for small lms, we want to run evaluation for each checkpoint
        checkpoints = sorted([int(item) for item in os.listdir(config.output_dir)])
        for checkpoint in tqdm(checkpoints, desc="Evaluating checkpoints"):   
            checkpoint_dir = os.path.join(config.output_dir, str(checkpoint))
            if len(os.listdir(checkpoint_dir)) > 0 and not cfg.overwrite:
                print(f"Checkpoint directory {checkpoint_dir} already exists and is not empty. Set overwrite=True to overwrite.")
                continue
            # run evaluation
            eval_model(config, os.path.join(checkpoint_dir, "model"), os.path.join(checkpoint_dir, f"metrics.pt"))
    else: 
        # check if output dir is empty
        if os.path.exists(config.output_dir) and len(os.listdir(config.output_dir)) > 0 and not cfg.overwrite:
            print(f"Output directory {config.output_dir} already exists and is not empty. Set overwrite=True to overwrite.")
            return
        # Save resolved config in run directory
        resolved_config_path = os.path.join(config.run_dir, "config_resolved.yaml")
        config.to_yaml(resolved_config_path)
        eval_model(config, config.model_kwargs.model_path if hasattr(config.model_kwargs, "model_path") else None, os.path.join(config.output_dir, f"metrics.pt"))

if __name__ == "__main__":
    run_evaluation()