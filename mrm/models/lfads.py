# mrm/models/lfads.py
"""
LFADS model implementation using lfads-torch
"""

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import h5py
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import warnings

from mrm.dataset import NeuralDataset
from mrm.models.base import BaseModel

# LFADS-torch imports
try:
    from lfads_torch.model import LFADS
    from lfads_torch.datamodules import BasicDataModule
    from lfads_torch.modules import augmentations
    from lfads_torch.modules.priors import MultivariateNormal, AutoregressiveMultivariateNormal
    from lfads_torch.modules.recons import Poisson, Gaussian
    from lfads_torch.tuples import SessionBatch
    LFADS_AVAILABLE = True
except ImportError:
    LFADS_AVAILABLE = False
    warnings.warn("lfads-torch not available. Install with: pip install git+https://github.com/arsedler9/lfads-torch.git")


class LFADSModel(BaseModel):
    """LFADS model using lfads-torch with IBL data integration"""
    
    def __init__(self, 
             latent_dim: int = 32,
             generator_dim: int = 128,
             controller_dim: int = 64,
             ic_dim: int = 32,
             co_dim: int = 16,
             learning_rate: float = 0.007,
             max_epochs: int = 200,
             batch_size: int = 32,
             external_inputs: List[str] = None,
             reconstruction_type: str = 'poisson',
             dropout_rate: float = 0.1,
             kl_start_epoch: int = 0,
             kl_increase_epoch: int = 80,
             l2_start_epoch: int = 0,
             l2_increase_epoch: int = 80,
             use_cache: bool = True,
             
             # NEW: Additional configurable parameters
             variational: bool = True,
             ic_post_var_min: float = 1e-4,
             cell_clip: float = 5.0,
             loss_scale: float = 1.0,
             recon_reduce_mean: bool = True,
             lr_scheduler: bool = True,
             lr_stop: float = None,  # Will default to learning_rate/100
             lr_decay: float = 0.95,
             lr_patience: int = 6,
             lr_adam_beta1: float = 0.9,
             lr_adam_beta2: float = 0.999,
             lr_adam_epsilon: float = 1e-8,
             weight_decay: float = 0.0,
             l2_ic_enc_scale: float = 0.0,
             l2_ci_enc_scale: float = 0.0,
             l2_gen_scale: float = 0.0,
             l2_con_scale: float = 0.0,
             kl_ic_scale: float = 0.0,
             kl_co_scale: float = 0.0,
             ic_enc_dim: int = None,  # Will default to controller_dim
             ci_enc_dim: int = None,  # Will default to controller_dim
             ci_lag: int = 1,
             co_prior_tau: float = 10.0,
             co_prior_nvar: float = 0.1,
             ic_prior_mean: float = 0.0,
             ic_prior_variance: float = 0.1,
             **kwargs):
        """
        Initialize LFADS model with full parameter control
        """
        super().__init__()
        
        if not LFADS_AVAILABLE:
            raise ImportError("lfads-torch is required.")
            
        # Store all parameters
        self.latent_dim = latent_dim
        self.generator_dim = generator_dim
        self.controller_dim = controller_dim
        self.ic_dim = ic_dim
        self.co_dim = co_dim
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.external_inputs = external_inputs or []
        self.reconstruction_type = reconstruction_type
        self.dropout_rate = dropout_rate
        self.kl_start_epoch = kl_start_epoch
        self.kl_increase_epoch = kl_increase_epoch
        self.l2_start_epoch = l2_start_epoch
        self.l2_increase_epoch = l2_increase_epoch
        self.use_cache = use_cache
        
        # NEW: Store additional parameters
        self.variational = variational
        self.ic_post_var_min = ic_post_var_min
        self.cell_clip = cell_clip
        self.loss_scale = loss_scale
        self.recon_reduce_mean = recon_reduce_mean
        self.lr_scheduler = lr_scheduler
        self.lr_stop = lr_stop if lr_stop is not None else learning_rate / 100
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience
        self.lr_adam_beta1 = lr_adam_beta1
        self.lr_adam_beta2 = lr_adam_beta2
        self.lr_adam_epsilon = lr_adam_epsilon
        self.weight_decay = weight_decay
        self.l2_ic_enc_scale = l2_ic_enc_scale
        self.l2_ci_enc_scale = l2_ci_enc_scale
        self.l2_gen_scale = l2_gen_scale
        self.l2_con_scale = l2_con_scale
        self.kl_ic_scale = kl_ic_scale
        self.kl_co_scale = kl_co_scale
        self.ic_enc_dim = ic_enc_dim if ic_enc_dim is not None else controller_dim
        self.ci_enc_dim = ci_enc_dim if ci_enc_dim is not None else controller_dim
        self.ci_lag = ci_lag
        self.co_prior_tau = co_prior_tau
        self.co_prior_nvar = co_prior_nvar
        self.ic_prior_mean = ic_prior_mean
        self.ic_prior_variance = ic_prior_variance
        
        # Will be set during training
        self.lfads_model = None
        self.data_info = {}
        self.is_fitted = False
        
    def fit(self, dataset: NeuralDataset) -> 'LFADSModel':
        """Train LFADS model on dataset"""
        print("Training LFADS model...")
        
        # Convert dataset to LFADS format
        print("Converting dataset to LFADS format...")
        h5_path, data_info = self._convert_to_lfads_format(dataset)
        self.data_info = data_info
        
        # Create LFADS model
        print("Creating LFADS model...")
        self.lfads_model = self._create_lfads_model(data_info)
        
        # Create datamodule
        datamodule = BasicDataModule(
            datafile_pattern=h5_path,
            batch_size=self.batch_size,
        )
        
        # Setup trainer
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = CSVLogger(temp_dir, name="lfads_training")
            trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                logger=logger,
                enable_checkpointing=False,
                enable_progress_bar=True,
                log_every_n_steps=10,
                accelerator='auto',  # Use GPU if available
                devices=1
            )
            
            # Train model
            print(f"Training for {self.max_epochs} epochs...")
            trainer.fit(self.lfads_model, datamodule=datamodule)
            
            import pandas as pd
            metrics_file = os.path.join(temp_dir, "lfads_training", "version_0", "metrics.csv")
            if os.path.exists(metrics_file):
                df = pd.read_csv(metrics_file)
                # Store as instance variable
                self.training_history = {}
                for col in df.columns:
                    data = df[col].dropna()
                    if len(data) > 0:
                        self.training_history[col] = data.tolist()
                print(f"Captured {len(self.training_history)} training metrics")
            else:
                print("No training metrics found")
                self.training_history = {}
            
            # ===== END ADD =====
        
        self.is_fitted = True
        print("LFADS training completed!")
        return self
        
    def encode(self, neural_data: np.ndarray, 
           behavior_data: Dict[str, np.ndarray] = None) -> np.ndarray:
        """
        Encode neural data to latent factors
        
        Args:
            neural_data: Shape (n_trials, n_timepoints, n_neurons) or (n_timepoints, n_neurons)
            behavior_data: Dict containing behavioral signals (choice, stimulus_contrast_left, etc.)
                        Uses self.external_inputs to determine which signals to extract
            
        Returns:
            latent_factors: Shape (n_trials, n_timepoints, latent_dim) or (n_timepoints, latent_dim)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before encoding")
            
        # Handle single trial case
        single_trial = False
        if neural_data.ndim == 2:
            neural_data = neural_data[np.newaxis]  # Add trial dimension
            single_trial = True
            
        n_trials, n_timepoints, n_neurons = neural_data.shape
        
        # Process external inputs based on self.external_inputs
        if behavior_data is not None:
            # Compute external inputs from behavioral data using same logic as training
            external_inputs = self._extract_external_inputs(behavior_data, (n_trials, n_timepoints))
        else:
            # No behavioral data provided - create zeros
            n_ext_inputs = len(self.external_inputs)
            external_inputs = np.zeros((n_trials, n_timepoints, n_ext_inputs))
            if len(self.external_inputs) > 0:
                import warnings
                warnings.warn(f"No behavioral data provided. Using zeros for external inputs: {self.external_inputs}")
        
        # Convert to tensors
        neural_tensor = torch.FloatTensor(neural_data)
        ext_tensor = torch.FloatTensor(external_inputs)

        # Create session batch
        session_batch = SessionBatch(
            encod_data=neural_tensor,
            recon_data=neural_tensor,
            ext_input=ext_tensor,
            sv_mask=torch.ones_like(neural_tensor[..., 0:1]),  # No sample masking
            truth=torch.zeros_like(neural_tensor)  # Not used for encoding
        )
        
        # Get latent factors
        self.lfads_model.eval()
        with torch.no_grad():
            batch_dict = {0: session_batch}
            output = self.lfads_model(batch_dict)
            factors = output[0].factors  # Shape: (n_trials, n_timepoints, latent_dim)
            
        factors_np = factors.cpu().numpy()
        
        # Return single trial if input was single trial
        if single_trial:
            factors_np = factors_np[0]
            
        return factors_np
        
    def decode(self, latents: np.ndarray) -> np.ndarray:
        """
        Decode latent factors to reconstructed neural data

        Args:
            latents: Shape (n_trials, n_timepoints, latent_dim) or (n_timepoints, latent_dim)
            
        Returns:
            reconstructed_data: Shape (n_trials, n_timepoints, n_neurons) or (n_timepoints, n_neurons)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before decoding")
            
        # Handle single trial case
        single_trial = False
        if latents.ndim == 2:
            latents = latents[np.newaxis]
            single_trial = True
            
        n_trials, n_timepoints, latent_dim = latents.shape
        n_neurons = self.data_info['n_neurons']
        n_ext_inputs = len(self.external_inputs)

        # Convert latents to tensor
        latents_tensor = torch.FloatTensor(latents)

        # Create dummy external inputs (or use stored ones if available)
        ext_inputs = torch.zeros(n_trials, n_timepoints, n_ext_inputs)

        # Get device of the model
        device = next(self.lfads_model.parameters()).device
        latents_tensor = latents_tensor.to(device)
        ext_inputs = ext_inputs.to(device)

        self.lfads_model.eval()
        with torch.no_grad():
            # Access the generator directly to decode from latents
            # This requires understanding the LFADS model structure
            # Method 1: Try to use the readout layer directly
            readout = self.lfads_model.readout[0]  # First (and usually only) readout
            
            # Apply readout to latents to get rates
            rates = readout(latents_tensor)  # Shape: (n_trials, n_timepoints, n_neurons)
            
            # Convert rates to reconstruction based on distribution type
            if self.reconstruction_type == 'poisson':
                # For Poisson, rates are the natural parameters
                reconstructed = rates
            else:
                # For Gaussian, use rates directly
                reconstructed = rates
                
            
        # Convert back to numpy
        reconstructed_np = reconstructed.cpu().numpy()

        # Return single trial if input was single trial
        if single_trial:
            reconstructed_np = reconstructed_np[0]
            
        return reconstructed_np.astype(np.float32)
        
    def predict(self, neural_data: np.ndarray, steps_ahead: int = 1, 
                external_inputs: np.ndarray = None) -> np.ndarray:
        """Predict future neural activity"""
        # Get current latent state
        latents = self.encode(neural_data, external_inputs)
        
        # For now, return current reconstruction
        # True prediction would require implementing the generator forward pass
        return self.decode(latents)
        
    def save(self, filepath: str) -> None:
        """Save LFADS model"""
        save_data = {
            'model_type': 'lfads',
            'config': self.get_config(),
            'data_info': self.data_info,
            'is_fitted': self.is_fitted
        }
        
        # Save model state dict if fitted
        if self.is_fitted and self.lfads_model is not None:
            save_data['model_state_dict'] = self.lfads_model.state_dict()
            
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        # ===== ADD: Save loss curves =====
        if hasattr(self, 'training_history') and self.training_history:
            try:
                import matplotlib.pyplot as plt
                import json
                from pathlib import Path
                
                # Get directory from filepath
                save_dir = Path(filepath).parent
                
                # Save raw training data as JSON
                history_file = save_dir / "training_history.json"
                with open(history_file, 'w') as f:
                    json.dump(self.training_history, f, indent=2)
                
                # Create loss curve plots
                loss_cols = [col for col in self.training_history.keys() if 'loss' in col.lower()]
                if loss_cols:
                    n_plots = min(len(loss_cols), 4)
                    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                    axes = axes.flatten()
                    
                    for i, col in enumerate(loss_cols[:4]):
                        ax = axes[i]
                        data = self.training_history[col]
                        ax.plot(data)
                        ax.set_title(col)
                        ax.set_xlabel('Step')
                        ax.grid(True, alpha=0.3)
                        
                        # Log scale for loss curves
                        if len(data) > 0 and max(data) > 0:
                            ax.set_yscale('log')
                    
                    # Hide unused subplots
                    for i in range(len(loss_cols), 4):
                        axes[i].set_visible(False)
                    
                    plt.tight_layout()
                    curves_file = save_dir / "training_curves.png"
                    plt.savefig(curves_file, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    print(f"✓ Saved training curves: {curves_file}")
                
            except Exception as e:
                print(f"Warning: Could not save training curves: {e}")
        # ===== END ADD =====
            
    @classmethod
    def load(cls, filepath: str) -> 'LFADSModel':
        """Load LFADS model"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
            
        config = save_data['config']
        model = cls(**{k: v for k, v in config.items() if k != 'model_type'})
        model.data_info = save_data['data_info']
        model.is_fitted = save_data['is_fitted']
        
        # Restore model if fitted
        if model.is_fitted and 'model_state_dict' in save_data:
            model.lfads_model = model._create_lfads_model(model.data_info)
            model.lfads_model.load_state_dict(save_data['model_state_dict'])
            
        return model
        
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            'model_type': 'lfads',
            'latent_dim': self.latent_dim,
            'generator_dim': self.generator_dim,
            'controller_dim': self.controller_dim,
            'ic_dim': self.ic_dim,
            'co_dim': self.co_dim,
            'learning_rate': self.learning_rate,
            'max_epochs': self.max_epochs,
            'batch_size': self.batch_size,
            'external_inputs': self.external_inputs,
            'reconstruction_type': self.reconstruction_type,
            'dropout_rate': self.dropout_rate,
            'kl_start_epoch': self.kl_start_epoch,
            'kl_increase_epoch': self.kl_increase_epoch,
            'l2_start_epoch': self.l2_start_epoch,
            'l2_increase_epoch': self.l2_increase_epoch
        }
        
    def _convert_to_lfads_format(self, dataset: NeuralDataset) -> Tuple[str, Dict]:
        """Convert NeuralDataset to LFADS HDF5 format"""
        
        # Get all data splits
        train_neural = dataset.get_neural_data('train')
        val_neural = dataset.get_neural_data('val') 
        test_neural = dataset.get_neural_data('test')
        
        train_behavior = dataset.get_behavior_data('train')
        val_behavior = dataset.get_behavior_data('val')
        test_behavior = dataset.get_behavior_data('test')
        # Extract external inputs
        train_ext = self._extract_external_inputs(train_behavior, train_neural.shape[:2])
        val_ext = self._extract_external_inputs(val_behavior, val_neural.shape[:2])
        test_ext = self._extract_external_inputs(test_behavior, test_neural.shape[:2])

        # Store data info
        data_info = {
            'n_neurons': train_neural.shape[2],
            'n_timepoints': train_neural.shape[1],
            'n_ext_inputs': train_ext.shape[2],
            'n_train_trials': train_neural.shape[0],
            'n_val_trials': val_neural.shape[0],
            'n_test_trials': test_neural.shape[0]
        }
        
        # Create temporary HDF5 file
        temp_dir = tempfile.gettempdir()
        h5_path = os.path.join(temp_dir, f'lfads_data_{os.getpid()}.h5')
        with h5py.File(h5_path, 'w') as f:
            # Training data
            f['train_encod_data'] = train_neural
            f['train_recon_data'] = train_neural  # Same for autoencoder
            f['train_ext_input'] = train_ext
            
            # Validation data  
            f['valid_encod_data'] = val_neural
            f['valid_recon_data'] = val_neural
            f['valid_ext_input'] = val_ext
            
            # Test data (optional for LFADS)
            f['test_encod_data'] = test_neural
            f['test_recon_data'] = test_neural
            f['test_ext_input'] = test_ext
            
        print(f"Created LFADS data file: {h5_path}")
        print(f"Data shapes: Neural {train_neural.shape}, External {train_ext.shape}")
        
        return h5_path, data_info
        
    def _extract_external_inputs(self, behavior_data: Dict, data_shape: Tuple) -> np.ndarray:
        """Extract external inputs from behavioral data"""
        n_trials, n_timepoints = data_shape
        n_ext_inputs = len(self.external_inputs)
        ext_inputs = np.zeros((n_trials, n_timepoints, n_ext_inputs))
        
        for i, input_name in enumerate(self.external_inputs):
            if input_name == 'choice':
                # Map choice to -1, 0, 1
                choice = behavior_data.get('choice', np.zeros(n_trials))
                choice = np.nan_to_num(choice, nan=0.0)
                ext_inputs[:, :, i] = choice[:, np.newaxis]
                
            elif input_name == 'stimulus_contrast':
                # Combined stimulus contrast
                left_contrast = behavior_data.get('stimulus_contrast_left', np.zeros(n_trials))
                right_contrast = behavior_data.get('stimulus_contrast_right', np.zeros(n_trials))
                
                # Create signed contrast (negative for left, positive for right)
                contrast = np.zeros_like(left_contrast)
                left_mask = ~np.isnan(left_contrast) & (left_contrast > 0)
                right_mask = ~np.isnan(right_contrast) & (right_contrast > 0)
                
                contrast[left_mask] = -left_contrast[left_mask]
                contrast[right_mask] = right_contrast[right_mask]
                
                ext_inputs[:, :, i] = contrast[:, np.newaxis]
                
            elif input_name == 'cue':
                # Cue signal indicating stimulus side: -1 for left, +1 for right, 0 for no stimulus
                left_contrast = behavior_data.get('stimulus_contrast_left', np.zeros(n_trials))
                right_contrast = behavior_data.get('stimulus_contrast_right', np.zeros(n_trials))
                
                cue = np.zeros(n_trials)
                left_mask = ~np.isnan(left_contrast) & (left_contrast >= 0)  # Include 0 contrast
                right_mask = ~np.isnan(right_contrast) & (right_contrast >= 0)
                
                cue[left_mask] = -1.0
                cue[right_mask] = 1.0
                
                ext_inputs[:, :, i] = cue[:, np.newaxis]
                
            elif input_name == 'movement_onset':
                # Binary signal for movement onset (simplified)
                reaction_time = behavior_data.get('reaction_time', np.full(n_trials, np.nan))
                movement_signal = np.zeros_like(ext_inputs[:, :, i])
                
                # Set movement signal around expected movement time
                for trial in range(n_trials):
                    if not np.isnan(reaction_time[trial]) and reaction_time[trial] > 0:
                        # Assume stimulus onset at middle of trial
                        stim_onset_bin = n_timepoints // 2
                        movement_bin = min(n_timepoints - 1, 
                                        int(stim_onset_bin + reaction_time[trial] * 100))  # Assuming 10ms bins
                        movement_signal[trial, movement_bin] = 1.0
                        
                ext_inputs[:, :, i] = movement_signal
                
            else:
                # Unknown input type, leave as zeros
                pass
                
        return ext_inputs
        
    def _create_lfads_model(self, data_info: Dict) -> LFADS:
        """Create LFADS model with specified architecture using all config parameters"""
        
        # Choose reconstruction distribution
        if self.reconstruction_type == 'poisson':
            reconstruction = nn.ModuleList([Poisson()])
        else:
            reconstruction = nn.ModuleList([Gaussian()])
        
        # Get additional parameters with defaults
        variational = getattr(self, 'variational', True)
        ic_post_var_min = getattr(self, 'ic_post_var_min', 1e-4)
        cell_clip = getattr(self, 'cell_clip', 5.0)
        loss_scale = getattr(self, 'loss_scale', 1.0)
        recon_reduce_mean = getattr(self, 'recon_reduce_mean', True)
        
        # Learning rate scheduler parameters
        lr_scheduler = getattr(self, 'lr_scheduler', True)
        lr_stop = getattr(self, 'lr_stop', self.learning_rate / 100)
        lr_decay = getattr(self, 'lr_decay', 0.95)
        lr_patience = getattr(self, 'lr_patience', 6)
        
        # Adam optimizer parameters
        lr_adam_beta1 = getattr(self, 'lr_adam_beta1', 0.9)
        lr_adam_beta2 = getattr(self, 'lr_adam_beta2', 0.999)
        lr_adam_epsilon = getattr(self, 'lr_adam_epsilon', 1e-8)
        weight_decay = getattr(self, 'weight_decay', 0.0)
        
        # L2 regularization parameters
        l2_ic_enc_scale = getattr(self, 'l2_ic_enc_scale', 0.0)
        l2_ci_enc_scale = getattr(self, 'l2_ci_enc_scale', 0.0)
        l2_gen_scale = getattr(self, 'l2_gen_scale', 0.0)
        l2_con_scale = getattr(self, 'l2_con_scale', 0.0)
        
        # KL regularization parameters
        kl_ic_scale = getattr(self, 'kl_ic_scale', 0.0)
        kl_co_scale = getattr(self, 'kl_co_scale', 0.0)
        
        # Encoder parameters
        ic_enc_dim = getattr(self, 'ic_enc_dim', self.controller_dim)
        ci_enc_dim = getattr(self, 'ci_enc_dim', self.controller_dim)
        ci_lag = getattr(self, 'ci_lag', 1)
        
        # Prior parameters
        co_prior_tau = getattr(self, 'co_prior_tau', 10.0)
        co_prior_nvar = getattr(self, 'co_prior_nvar', 0.1)
        ic_prior_mean = getattr(self, 'ic_prior_mean', 0.0)
        ic_prior_variance = getattr(self, 'ic_prior_variance', 0.1)
            
        # Create LFADS model
        model = LFADS(
            # Data dimensions
            encod_data_dim=data_info['n_neurons'],
            encod_seq_len=data_info['n_timepoints'],
            recon_seq_len=data_info['n_timepoints'],
            ext_input_dim=data_info['n_ext_inputs'],
            
            # Encoder dimensions
            ic_enc_seq_len=min(20, data_info['n_timepoints'] // 4),
            ic_enc_dim=ic_enc_dim,
            ci_enc_dim=ci_enc_dim,
            ci_lag=ci_lag,
            
            # Core architecture  
            con_dim=self.controller_dim,
            co_dim=self.co_dim,
            ic_dim=self.ic_dim,
            gen_dim=self.generator_dim,
            fac_dim=self.latent_dim,
            
            # Training parameters
            dropout_rate=self.dropout_rate,
            variational=variational,
            ic_post_var_min=ic_post_var_min,
            cell_clip=cell_clip,
            
            # Reconstruction
            reconstruction=reconstruction,
            
            # Priors
            co_prior=AutoregressiveMultivariateNormal(tau=co_prior_tau, nvar=co_prior_nvar, shape=self.co_dim),
            ic_prior=MultivariateNormal(mean=ic_prior_mean, variance=ic_prior_variance, shape=self.ic_dim),
            
            # I/O modules
            train_aug_stack=augmentations.AugmentationStack([]),
            infer_aug_stack=augmentations.AugmentationStack([]),
            readin=nn.ModuleList([nn.Identity()]),
            readout=nn.ModuleList([nn.Linear(self.latent_dim, data_info['n_neurons'])]),
            
            # Loss scaling
            loss_scale=loss_scale,
            recon_reduce_mean=recon_reduce_mean,
            
            # Learning rate schedule
            lr_scheduler=lr_scheduler,
            lr_init=self.learning_rate,
            lr_stop=lr_stop,
            lr_decay=lr_decay,
            lr_patience=lr_patience,
            
            # Adam parameters
            lr_adam_beta1=lr_adam_beta1,
            lr_adam_beta2=lr_adam_beta2,
            lr_adam_epsilon=lr_adam_epsilon,
            weight_decay=weight_decay,
            
            # Regularization schedules
            l2_start_epoch=self.l2_start_epoch,
            l2_increase_epoch=self.l2_increase_epoch,
            l2_ic_enc_scale=l2_ic_enc_scale,
            l2_ci_enc_scale=l2_ci_enc_scale,
            l2_gen_scale=l2_gen_scale,
            l2_con_scale=l2_con_scale,
            
            # KL regularization
            kl_start_epoch=self.kl_start_epoch,
            kl_increase_epoch=self.kl_increase_epoch,
            kl_ic_scale=kl_ic_scale,
            kl_co_scale=kl_co_scale,
        )
        
        return model
