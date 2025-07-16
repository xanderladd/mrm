# models/pca.py
"""
PCA dimensionality reduction model with trial-averaged fitting option
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
    
    def __init__(self, n_components: int = 10, fit_method: str = None, **kwargs):
        super().__init__()
        self.n_components = n_components
        self.fit_method = fit_method  # None (concatenated) or "trial_averaged"
        self.pca = PCA(n_components=n_components, **kwargs)
        self.input_shape = None
        self.explained_variance_ratio_ = None
        
    def fit(self, dataset: NeuralDataset) -> 'PCAModel':
        """Fit PCA to training data"""
        print(f"Fitting PCA with {self.n_components} components using '{self.fit_method or 'concatenated'}' method...")
        
        # Get training data
        train_data = dataset.get_neural_data('train')  # Shape: (n_trials, n_timepoints, n_neurons)
        print(f"Training data shape: {train_data.shape}")
        
        # Store original shape info
        n_trials, n_timepoints, n_neurons = train_data.shape
        self.input_shape = (n_timepoints, n_neurons)
        
        # Choose fitting method
        if self.fit_method == "trial_averaged":
            train_data_2d = self._fit_trial_averaged(dataset, train_data)
        else:
            # Default: concatenate all timepoints from all trials
            train_data_2d = train_data.reshape(-1, n_neurons)  # (n_trials * n_timepoints, n_neurons)
            print(f"Using concatenated data for PCA: {train_data_2d.shape}")
        
        # Fit PCA
        self.pca.fit(train_data_2d)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.is_fitted = True
        
        print(f"PCA fitted. Explained variance: {np.sum(self.explained_variance_ratio_):.3f}")
        print(f"First 5 components explain: {np.sum(self.explained_variance_ratio_[:5]):.3f}")
        
        return self
        
    def _fit_trial_averaged(self, dataset: NeuralDataset, train_data: np.ndarray) -> np.ndarray:
        """Fit PCA on trial-averaged responses grouped by behavioral conditions"""
        
        # Get behavioral data
        behavior_data = dataset.get_behavior_data('train')
        
        print("Creating condition-averaged responses for PCA fitting...")
        
        # Extract behavioral variables
        choices = behavior_data.get('choice', np.full(train_data.shape[0], np.nan))
        feedback = behavior_data.get('feedback_type', np.full(train_data.shape[0], np.nan))
        contrast_left = behavior_data.get('stimulus_contrast_left', np.full(train_data.shape[0], np.nan))
        contrast_right = behavior_data.get('stimulus_contrast_right', np.full(train_data.shape[0], np.nan))
        
        # Create combined contrast measure (take max of left/right, preserve sign)
        contrast_combined = np.zeros_like(choices)
        for i in range(len(choices)):
            if not np.isnan(contrast_left[i]) and contrast_left[i] > 0:
                contrast_combined[i] = -contrast_left[i]  # Left stimulus = negative
            elif not np.isnan(contrast_right[i]) and contrast_right[i] > 0:
                contrast_combined[i] = contrast_right[i]   # Right stimulus = positive
            else:
                contrast_combined[i] = 0  # No stimulus or unclear
        
        # Bin contrast into categories for grouping
        contrast_abs = np.abs(contrast_combined)
        contrast_bins = np.digitize(contrast_abs, bins=[0.001, 0.06, 0.125, 1.0])  # 0, low, med, high
        
        # Collect condition averages
        condition_averages = []
        conditions_found = []
        
        # Iterate through behavioral conditions
        for choice_val in [-1, 0, 1]:  # Left, No-go, Right
            for feedback_val in [-1, 1]:  # Error, Correct
                for contrast_bin in range(4):  # 0, low, med, high contrast
                    
                    # Find trials matching this condition
                    choice_mask = (choices == choice_val) if not np.isnan(choice_val) else np.isnan(choices)
                    feedback_mask = (feedback == feedback_val) if not np.isnan(feedback_val) else np.isnan(feedback)
                    contrast_mask = (contrast_bins == contrast_bin)
                    
                    condition_mask = choice_mask & feedback_mask & contrast_mask
                    n_trials_condition = np.sum(condition_mask)
                    
                    if n_trials_condition >= 3:  # Minimum trials per condition
                        # Average neural activity across trials for this condition
                        condition_data = train_data[condition_mask]  # (n_condition_trials, n_timepoints, n_neurons)
                        condition_avg = np.mean(condition_data, axis=0)  # (n_timepoints, n_neurons)
                        
                        condition_averages.append(condition_avg)
                        conditions_found.append({
                            'choice': choice_val, 
                            'feedback': feedback_val, 
                            'contrast_bin': contrast_bin,
                            'n_trials': n_trials_condition
                        })
        
        if len(condition_averages) == 0:
            print("Warning: No valid conditions found, falling back to concatenated method")
            return train_data.reshape(-1, train_data.shape[-1])
        
        # Stack all condition averages for PCA fitting
        condition_data = np.vstack(condition_averages)  # (n_conditions * n_timepoints, n_neurons)
        
        print(f"Created {len(condition_averages)} condition averages")
        print(f"Conditions found: {len(conditions_found)}")
        print(f"Trial-averaged data for PCA: {condition_data.shape}")
        
        # Log some example conditions
        for i, cond in enumerate(conditions_found[:5]):
            print(f"  Condition {i+1}: choice={cond['choice']}, feedback={cond['feedback']}, "
                  f"contrast_bin={cond['contrast_bin']}, n_trials={cond['n_trials']}")
        
        return condition_data
        
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
            'fit_method': self.fit_method,
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
            
        model = cls(
            n_components=model_data['n_components'],
            fit_method=model_data.get('fit_method', None)  # Backwards compatibility
        )
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
            'fit_method': self.fit_method,
            'input_shape': self.input_shape,
            'explained_variance_ratio': self.explained_variance_ratio_.tolist() if self.explained_variance_ratio_ is not None else None,
            'total_explained_variance': float(np.sum(self.explained_variance_ratio_)) if self.explained_variance_ratio_ is not None else None
        }

