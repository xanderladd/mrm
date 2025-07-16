import pickle
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from sklearn.decomposition import PCA
from mrm.dataset import NeuralDataset

from mrm.models.base import BaseModel


class DummyModel(BaseModel):
    """Dummy model for testing"""
    
    def __init__(self, latent_dim: int = 10):
        super().__init__()
        self.latent_dim = latent_dim
        
    def fit(self, dataset: NeuralDataset) -> 'DummyModel':
        """Dummy fit - just store dimensions"""
        train_data = dataset.get_neural_data('train')
        self.input_shape = train_data.shape[1:]  # (n_timepoints, n_neurons)
        self.is_fitted = True
        print(f"Dummy model 'fitted' with input shape: {self.input_shape}")
        return self
        
    def encode(self, neural_data: np.ndarray) -> np.ndarray:
        """Dummy encoding - random projection"""
        if len(neural_data.shape) == 3:
            n_trials, n_timepoints, n_neurons = neural_data.shape
            return np.random.randn(n_trials, n_timepoints, self.latent_dim)
        else:
            n_timepoints, n_neurons = neural_data.shape
            return np.random.randn(n_timepoints, self.latent_dim)
            
    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Dummy decoding - random projection back"""
        if len(latents.shape) == 3:
            n_trials, n_timepoints, _ = latents.shape
            return np.random.randn(n_trials, n_timepoints, self.input_shape[1])
        else:
            n_timepoints, _ = latents.shape
            return np.random.randn(n_timepoints, self.input_shape[1])
            
    def save(self, filepath: str) -> None:
        """Save dummy model"""
        model_data = {
            'latent_dim': self.latent_dim,
            'input_shape': self.input_shape,
            'is_fitted': self.is_fitted,
            'model_type': 'dummy'
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
    @classmethod  
    def load(cls, filepath: str) -> 'DummyModel':
        """Load dummy model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        model = cls(latent_dim=model_data['latent_dim'])
        model.input_shape = model_data['input_shape']
        model.is_fitted = model_data['is_fitted']
        return model
        
    def get_config(self) -> Dict[str, Any]:
        """Get dummy model config"""
        return {
            'model_type': 'dummy',
            'latent_dim': self.latent_dim,
            'input_shape': self.input_shape
        }

