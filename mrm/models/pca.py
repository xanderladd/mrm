# models.py
"""
Neural models with standardized interface
"""

import pickle
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from sklearn.decomposition import PCA
from mrm.dataset import NeuralDataset
from mrm.models.base import BaseModel



class PCAModel(BaseModel):
    """PCA dimensionality reduction model"""
    
    def __init__(self, n_components: int = 10, **kwargs):
        super().__init__()
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, **kwargs)
        self.input_shape = None
        self.explained_variance_ratio_ = None
        
    def fit(self, dataset: NeuralDataset) -> 'PCAModel':
        """Fit PCA to training data"""
        print(f"Fitting PCA with {self.n_components} components...")
        
        # Get training data
        train_data = dataset.get_neural_data('train')  # Shape: (n_trials, n_timepoints, n_neurons)
        print(f"Training data shape: {train_data.shape}")
        
        # Reshape to (n_samples, n_features) for PCA
        # Concatenate all timepoints from all trials
        n_trials, n_timepoints, n_neurons = train_data.shape
        train_data_2d = train_data.reshape(-1, n_neurons)  # (n_trials * n_timepoints, n_neurons)
        
        print(f"Reshaped for PCA: {train_data_2d.shape}")
        
        # Store original shape info
        self.input_shape = (n_timepoints, n_neurons)
        
        # Fit PCA
        self.pca.fit(train_data_2d)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.is_fitted = True
        
        print(f"PCA fitted. Explained variance: {np.sum(self.explained_variance_ratio_):.3f}")
        print(f"First 5 components explain: {np.sum(self.explained_variance_ratio_[:5]):.3f}")
        
        return self
        
    def encode(self, neural_data: np.ndarray) -> np.ndarray:
        """Encode neural data to PCA latent space"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before encoding")
            
        original_shape = neural_data.shape
        
        if len(original_shape) == 3:
            # (n_trials, n_timepoints, n_neurons) -> (n_trials * n_timepoints, n_neurons)
            n_trials, n_timepoints, n_neurons = original_shape
            neural_data_2d = neural_data.reshape(-1, n_neurons)
        elif len(original_shape) == 2:
            # Already 2D
            neural_data_2d = neural_data
            n_trials = 1
            n_timepoints = original_shape[0]
        else:
            raise ValueError(f"Unsupported data shape: {original_shape}")
            
        # Apply PCA transform
        latents_2d = self.pca.transform(neural_data_2d)
        
        if len(original_shape) == 3:
            # Reshape back to (n_trials, n_timepoints, n_components)
            latents = latents_2d.reshape(n_trials, n_timepoints, self.n_components)
        else:
            latents = latents_2d
            
        return latents
        
    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Decode PCA latents back to neural space"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before decoding")
            
        original_shape = latents.shape
        
        if len(original_shape) == 3:
            # (n_trials, n_timepoints, n_components) -> (n_trials * n_timepoints, n_components)
            n_trials, n_timepoints, n_components = original_shape
            latents_2d = latents.reshape(-1, n_components)
        elif len(original_shape) == 2:
            # Already 2D
            latents_2d = latents
            n_trials = 1
            n_timepoints = original_shape[0]
        else:
            raise ValueError(f"Unsupported latent shape: {original_shape}")
            
        # Apply PCA inverse transform
        neural_data_2d = self.pca.inverse_transform(latents_2d)
        
        if len(original_shape) == 3:
            # Reshape back to (n_trials, n_timepoints, n_neurons)
            n_neurons = neural_data_2d.shape[1]
            neural_data = neural_data_2d.reshape(n_trials, n_timepoints, n_neurons)
        else:
            neural_data = neural_data_2d
            
        return neural_data
        
    def save(self, filepath: str) -> None:
        """Save PCA model to file"""
        model_data = {
            'pca': self.pca,
            'n_components': self.n_components,
            'input_shape': self.input_shape,
            'explained_variance_ratio_': self.explained_variance_ratio_,
            'is_fitted': self.is_fitted,
            'model_type': 'pca'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
    @classmethod
    def load(cls, filepath: str) -> 'PCAModel':
        """Load PCA model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        model = cls(n_components=model_data['n_components'])
        model.pca = model_data['pca']
        model.input_shape = model_data['input_shape']
        model.explained_variance_ratio_ = model_data['explained_variance_ratio_']
        model.is_fitted = model_data['is_fitted']
        
        return model
        
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            'model_type': 'pca',
            'n_components': self.n_components,
            'input_shape': self.input_shape,
            'explained_variance_ratio': self.explained_variance_ratio_.tolist() if self.explained_variance_ratio_ is not None else None,
            'total_explained_variance': float(np.sum(self.explained_variance_ratio_)) if self.explained_variance_ratio_ is not None else None
        }



if __name__ == "__main__":
    # Test PCA model with synthetic data
    print("Testing PCA model...")
    
    # Create synthetic neural data
    n_trials, n_timepoints, n_neurons = 100, 50, 200
    neural_data = np.random.randn(n_trials, n_timepoints, n_neurons)
    
    # Create a dummy dataset object for testing
    class TestDataset:
        def get_neural_data(self, split):
            return neural_data
    
    test_dataset = TestDataset()
    
    # Test PCA model
    pca_model = PCAModel(n_components=10)
    pca_model.fit(test_dataset)
    
    # Test encoding/decoding
    latents = pca_model.encode(neural_data)
    reconstructed = pca_model.decode(latents)
    
    print(f"Original shape: {neural_data.shape}")
    print(f"Latent shape: {latents.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Total explained variance: {np.sum(pca_model.explained_variance_ratio_):.3f}")
    
    # Test save/load
    pca_model.save("test_pca.pkl")
    loaded_model = PCAModel.load("test_pca.pkl")
    print(f"Loaded model config: {loaded_model.get_config()}")
    
    print("PCA model test completed!")