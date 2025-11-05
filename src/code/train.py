import os
from transformers import TrainingArguments, Trainer, default_data_collator
import torch
import wandb
import hydra
from dotenv import load_dotenv
from pyprojroot import here
from omegaconf import DictConfig, OmegaConf
from .models import init_model
from .config import Config
from .data import init_dataset
from .train_utils import (
    CustomSaveCallback,
    StepMetricsCallback,
    PreBatchedDataCollator,
)

# load env variables
load_dotenv()

def print_current_gpus():
    if torch.cuda.is_available():
        print("Current GPUs:")
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPUs available.")

def wandb_init(config):
    run_name = f"{config.name}-{config.run_dir.split('/')[-1]}"
    config_dict = OmegaConf.to_container(config.cfg, resolve=True)
    
    wandb.init(
        entity="goodfire",
        project="icl-rep-dynamics",
        name=run_name,
        config=config_dict,
        save_code=False,  # Don't save code to wandb
    )
    
    # Ensure wandb doesn't watch the model (prevents weight logging)

    wandb.watch(models=[], log=None)
    
def train(config):
    # print current gpus
    print_current_gpus()

    # initialize wandb
    wandb_init(config)

    # initialize model
    model = init_model(config)

    # initialize dataset
    dataset = init_dataset(
        config,
    )

    # use pre-batched data collator for training
    data_collator = PreBatchedDataCollator() 
    train_batch_size = 1 # we use pre-batched data collator, so batch size is set to 1 in the dataloader
    
    config.process_save_steps() # process save steps 

    # Define training args
    training_args = TrainingArguments(
        optim=config.training_kwargs["optimizer"],
        output_dir=config.output_dir, # this is the directory where the checkpoints will be saved
        per_device_train_batch_size=train_batch_size,
        logging_steps=config.training_kwargs["logging_steps"],
        max_steps=config.training_kwargs["train_steps"],
        gradient_accumulation_steps=config.training_kwargs["gradient_accumulation_steps"],
        seed=config.random_seed,
        weight_decay=config.training_kwargs["weight_decay"],
        max_grad_norm=config.training_kwargs["max_grad_norm"],
        learning_rate=config.training_kwargs["learning_rate"],
        warmup_steps=config.training_kwargs["warmup_steps"],
        lr_scheduler_type=config.training_kwargs["lr_scheduler_type"],
        lr_scheduler_kwargs=config.training_kwargs["lr_scheduler_kwargs"],
        save_strategy="no",  # Disable default saving
        report_to="wandb",  # Enable wandb logging
        save_total_limit=0,  # Don't save any checkpoints for wandb
        logging_dir=None,  # Disable tensorboard logging
        dataloader_pin_memory=False, # disable pin_memory to avoid GPU tensor pinning issues with pre-batched data collator
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        compute_metrics=dataset.compute_metrics,
        data_collator=data_collator,
    )
    
    # Add saving callback to the trainer
    trainer.add_callback(
        CustomSaveCallback(
            config.training_kwargs["save_steps"],
            trainer=trainer,
        )
    )

    # Add the metrics callback to the Trainer
    trainer.add_callback(
        StepMetricsCallback(
            trainer=trainer, step_interval=config.training_kwargs["eval_steps"], config=config
        )
    )

    trainer.train()

    # Finish wandb run
    wandb.finish()

@hydra.main(version_base=None, config_path=str(here("conf")), config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration.
    
    Usage examples:
    1. Single run with experiment config:
       python -m code.train experiment=default
    
    2. Multirun with parameter sweep:
       python -m code.train experiment=default --multirun model_kwargs.hidden_size=128,256,512,1024
       
    3. Override specific parameters:
       python -m code.train experiment=default random_seed=42 num_tasks=128
    """    
    # initialize config
    config = Config(cfg, make_run_dirs=True)
    
    # Save resolved config in run directory
    resolved_config_path = os.path.join(config.run_dir, "config_resolved.yaml")
    config.to_yaml(resolved_config_path)
    
    # check if output dir is empty
    if os.path.exists(config.output_dir) and len(os.listdir(config.output_dir)) > 0 and not cfg.overwrite:
        print(f"Output directory {config.output_dir} already exists and is not empty. Set overwrite=True to overwrite.")
        return
    
    # run training
    train(config)


if __name__ == "__main__":
    main()