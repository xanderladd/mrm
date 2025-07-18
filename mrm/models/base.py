from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import numpy as np

class BaseModel(ABC):
    """Base class for all neural models with standardized interface"""
    
    @abstractmethod
    def fit(self, dataset: 'NeuralDataset') -> 'BaseModel':
        """Train the model on a dataset"""
        pass
    
    @abstractmethod
    def encode(self, neural_data: np.ndarray, 
               behavior_data: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """
        Encode neural data to latent representation
        
        Args:
            neural_data: Neural activity data
            behavior_data: Behavioral data dict (for computing external inputs)
            external_inputs: Pre-computed external inputs (overrides behavior_data)
            
        Returns:
            Latent representation
        """
        pass
    
    @abstractmethod
    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Decode latents back to neural data"""
        pass
    
    def predict(self, neural_data: np.ndarray, steps_ahead: int = 1,
                behavior_data: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """Predict future neural activity (optional to implement)"""
        # Default implementation: encode then decode
        latents = self.encode(neural_data, behavior_data, external_inputs)
        return self.decode(latents)
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save model to file"""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> 'BaseModel':
        """Load model from file"""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        pass