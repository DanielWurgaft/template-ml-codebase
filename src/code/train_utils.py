import os
import torch
from transformers import TrainerCallback
import wandb
from .eval import ModelEvaluator

#########################
# helpers for training
#########################

class PreBatchedDataCollator:
    """
    Data collator for datasets that already return batched data from __getitem__.
    Simply extracts the batch from the single-item list returned by DataLoader.
    """
    
    def __call__(self, features):
        return features[0]

#########################
# helpers for saving 
#########################


class StepMetricsCallback(TrainerCallback):
    def __init__(self, step_interval, trainer, config):
        self.trainer = trainer
        self.config = config
        self.step_interval = step_interval  # Interval at which to execute

    def on_step_end(self, args, state, control, **kwargs):
        """
        Called at the end of each training step. Triggers metric computation
        every `step_interval` steps.
        """
        # Check if it's time to execute based on step interval
        if (
            state.global_step % self.step_interval == 0
            and self.trainer.compute_metrics is not None
        ):
            trainer = self.trainer

            trainer.model.eval()  # Set model to evaluation mode

            # Use helper function for consistent evaluation
            evaluator = ModelEvaluator(trainer.model, trainer.train_dataset, self.config)
            metrics = evaluator.evaluate()
            
            # Prepare all wandb metrics at once
            wandb_metrics = {}
            # Add metrics except for all_*
            wandb_metrics.update({f"eval/{k}": v for k, v in metrics.items() if "all" not in k})
            
            # Log everything at once
            wandb.log(wandb_metrics, step=state.global_step)            
            metrics["step"] = state.global_step

            if self.config.training_kwargs.save_online_eval:
                checkpoint_save_dir = f"{self.config.output_dir}/{state.global_step}"
                os.makedirs(checkpoint_save_dir, exist_ok=True)
                torch.save(metrics, os.path.join(checkpoint_save_dir, "metrics.pt"))

            trainer.model.train()  # Return model to training mode


class CustomSaveCallback(TrainerCallback):
    def __init__(self, steps_to_save, trainer):
        self.steps_to_save = set(steps_to_save)
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        """
        Save the model at specific intervals without saving unnecessary files.
        """
        trainer = self.trainer
        model = trainer.model
        if (
            state.global_step in self.steps_to_save
            or state.global_step == args.max_steps
        ):
            checkpoint_save_dir = f"{args.output_dir}/{state.global_step}"
            os.makedirs(checkpoint_save_dir, exist_ok=True)
            model.save_pretrained(os.path.join(checkpoint_save_dir, "model"))