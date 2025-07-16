# models.py
"""
Base model classes and specific implementations for neural analysis
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pickle
import json
from pathlib import Path
from mrm.dataset import NeuralDataset


class BaseModel(ABC):
    """Base class for neural models with sklearn-style interface"""
    
    @abstractmethod
    def fit(self, dataset: NeuralDataset) -> 'BaseNeuralModel':
        """Train model on dataset"""
        pass
        
    @abstractmethod  
    def encode(self, neural_data: np.ndarray) -> np.ndarray:
        """Encode neural data to latent representation"""
        pass
        
    @abstractmethod
    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Decode latents to reconstructed neural data"""
        pass
        
    def predict(self, neural_data: np.ndarray, steps_ahead: int = 1) -> np.ndarray:
        """Predict future neural activity (optional)"""
        raise NotImplementedError("Prediction not implemented for this model")
        
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save model to disk"""
        pass
        
    @classmethod
    @abstractmethod  
    def load(cls, filepath: str) -> 'BaseNeuralModel':
        """Load model from disk"""
        pass
        
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration"""
        pass

