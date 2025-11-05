import os
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf

class Config:
    def __init__(self, cfg: DictConfig, make_run_dirs=False):
        """
        Initialize configuration from Hydra DictConfig.
        """        
        # Store the resolved config
        self.cfg = cfg

        # set the torch generator
        self.rng = torch.Generator()
        self.rng.manual_seed(cfg.random_seed)
                
        # Create directories if requested
        if make_run_dirs:
            self._make_directories()

    def _make_directories(self):
        """Create necessary directories based on config paths."""
        os.makedirs(self.cfg.run_dir, exist_ok=True)
        os.makedirs(self.cfg.output_dir, exist_ok=True)
                        
    def process_save_steps(self):
        """Process save_steps configuration with smart scheduling."""
        training_kwargs = self.cfg.training_kwargs
        save_steps_method = training_kwargs.get("save_steps_method", None)
        if save_steps_method in [None, "every_eval"]:
            training_kwargs.save_steps = np.arange(training_kwargs.eval_steps, training_kwargs.train_steps + 1, training_kwargs.eval_steps, dtype=int).tolist()
           
        elif save_steps_method == "linear_schedule":
            # Generate `num_checkpoints` evenly spaced points in linear space
            num_checkpoints = training_kwargs.num_save_checkpoints
            train_steps = training_kwargs.train_steps
            step_size = train_steps // num_checkpoints
            training_kwargs.save_steps = list(range(step_size, train_steps + 1, step_size))   
        
        elif save_steps_method == "sqrt_schedule":
            # Generate `num_checkpoints` evenly spaced points in square root space
            sqrt_space = np.linspace(0, 
                                     np.sqrt(training_kwargs.train_steps), 
                                     training_kwargs.num_save_checkpoints)
            training_kwargs.save_steps = [int(step) for step in (sqrt_space**2)]

        elif save_steps_method == "end_only":
            training_kwargs.save_steps = [training_kwargs.train_steps]

    def __getattr__(self, name):
        """Allow direct access to config parameters."""
        if hasattr(self.cfg, name):
            return getattr(self.cfg, name)
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def to_yaml(self, filepath):
        """Convert config to YAML file."""
        OmegaConf.save(self.cfg, filepath, resolve=True)

    @classmethod
    def from_yaml(cls, yaml_filepath, make_run_dirs=False):
        """Create Config from YAML file."""
        return cls(cfg=OmegaConf.load(yaml_filepath), make_run_dirs=make_run_dirs)