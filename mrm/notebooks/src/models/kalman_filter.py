"""Kalman Filter baseline for multi-region neural dynamics"""
import numpy as np
from pykalman import KalmanFilter
from sklearn.decomposition import PCA
import pickle

class KalmanFilterBaseline:
    """Kalman Filter baseline for bidirectional multi-region prediction"""
    
    def __init__(self, params):
        self.params = params
        self.n_latent = params.get('n_latent', 20)
        self.kf = None
        self.pca = None
        self.fitted = False
        self.n_region1 = None
        self.n_region2 = None
        self.data_mean = None
        self.data_std = None
        
    def fit(self, train_data, targets={}, **training_params):
        """Fit KF model to trajectory data with optional denoising
        
        Args:
            train_data: Dict with 'region_1' and 'region_2' arrays (potentially noisy)
                       Each shape: (n_trials, n_time, n_neurons)
            targets: Optional dict with clean targets for denoising
        """
        from utils import compute_r2
        
        # Use targets if provided (denoising), else standard reconstruction
        if len(targets.keys()):
            clean_data = targets
            print(f"  Training Kalman Filter with denoising")
        else:
            clean_data = train_data
            print(f"  Training Kalman Filter standard")
        
        input_data = train_data
        
        # Extract data
        region_1_input = input_data['region_1']
        region_2_input = input_data['region_2']
        region_1_clean = clean_data['region_1']
        region_2_clean = clean_data['region_2']
        
        # Store dimensions
        self.n_region1 = region_1_input.shape[2]
        self.n_region2 = region_2_input.shape[2]
        
        # Concatenate regions
        X_input = np.concatenate([region_1_input, region_2_input], axis=2)
        X_clean = np.concatenate([region_1_clean, region_2_clean], axis=2)
        n_trials, n_time, n_dims = X_input.shape
        
        # Normalize using clean data statistics
        X_clean_flat = X_clean.reshape(-1, n_dims)
        self.data_mean = X_clean_flat.mean(axis=0)
        self.data_std = X_clean_flat.std(axis=0) + 1e-8
        
        X_input_norm = (X_input - self.data_mean) / self.data_std
        X_clean_norm = (X_clean - self.data_mean) / self.data_std
        
        # Initialize latent space with PCA on clean data
        X_clean_norm_flat = X_clean_norm.reshape(-1, n_dims)
        X_input_norm_flat = X_input_norm.reshape(-1, n_dims)
        self.pca = PCA(n_components=min(self.n_latent, n_dims))
        latent_init = self.pca.fit_transform(X_clean_norm_flat)
        
        print(f"  PCA explained variance: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        # Setup Kalman Filter
        self.kf = KalmanFilter(
            n_dim_obs=n_dims,
            n_dim_state=self.n_latent,
            initial_state_mean=np.zeros(self.n_latent),
            initial_state_covariance=np.eye(self.n_latent),
            em_vars=['transition_matrices', 'observation_matrices',
                    'transition_covariance', 'observation_covariance',
                    'initial_state_mean', 'initial_state_covariance']
        )
        
        # Initialize matrices with PCA
        self.kf.observation_matrices = self.pca.components_.T
        self.kf.transition_matrices = 0.99 * np.eye(self.n_latent)  # Stable dynamics
        
        # For denoising: estimate observation noise from input-clean difference
        if len(targets.keys()):
            # Estimate observation noise from corruption
            noise_diff = X_input_norm_flat - X_clean_norm_flat
            obs_noise_cov = np.cov(noise_diff.T)
            # Regularize for stability
            self.kf.observation_covariance = obs_noise_cov + 0.01 * np.eye(n_dims)
        else:
            self.kf.observation_covariance = 0.1 * np.eye(n_dims)
        
        self.kf.transition_covariance = 0.01 * np.eye(self.n_latent)
        
        # Fit using EM on input data (learns to handle corrupted observations)
        X_input_concat = np.vstack([X_input_norm[i] for i in range(n_trials)])
        n_iter = training_params.get('n_iter', 30)
        
        print(f"  Fitting Kalman Filter with {n_iter} EM iterations...")
        
        # For denoising: EM learns dynamics from noisy observations
        # KF naturally handles noisy observations through its observation model
        self.kf = self.kf.em(X_input_concat, n_iter=n_iter)
        
        self.fitted = True
        
        # Compute training metrics: predict from input, compare to clean
        train_preds = self.predict([region_1_input, region_2_input])
        pred_r1, pred_r2 = train_preds
        
        # Compute MSE and R² against clean targets
        mse_r1 = np.mean((region_1_clean - pred_r1)**2)
        mse_r2 = np.mean((region_2_clean - pred_r2)**2)
        r2_r1 = compute_r2(pred_r1.flatten(), region_1_clean.flatten())
        r2_r2 = compute_r2(pred_r2.flatten(), region_2_clean.flatten())
        
        avg_mse = (mse_r1 + mse_r2) / 2
        avg_r2 = (r2_r1 + r2_r2) / 2
        
        print(f"  Train Region 1: MSE={mse_r1:.4f}, R²={r2_r1:.3f}")
        print(f"  Train Region 2: MSE={mse_r2:.4f}, R²={r2_r2:.3f}")
        print(f"  Train Average: MSE={avg_mse:.4f}, R²={avg_r2:.3f}")
        
        return {
            'mse': float(avg_mse),
            'r2': float(avg_r2),
            'mse_r1': float(mse_r1),
            'mse_r2': float(mse_r2),
            'r2_r1': float(r2_r1),
            'r2_r2': float(r2_r2),
            'model_type': 'kalman_filter',
            'n_latent': self.n_latent,
            'pca_variance_explained': float(self.pca.explained_variance_ratio_.sum())
        }
    
    def predict(self, test_data, region_ids=None):
        """Predict both regions using Kalman smoothing
        
        Args:
            test_data: List of [region_1, region_2] arrays or concatenated array
            region_ids: Optional list specifying region ordering
            
        Returns:
            List of [pred_region_1, pred_region_2] predictions
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Handle input format
        if isinstance(test_data, list):
            X = np.concatenate(test_data, axis=-1)
        else:
            X = test_data
        
        # Normalize
        X_norm = (X - self.data_mean) / self.data_std
        
        predictions = []
        for trial in X_norm:
            # Use Kalman smoothing (forward-backward pass)
            # This naturally denoises by using the learned dynamics model
            smoothed_states, _ = self.kf.smooth(trial)
            
            # Project back to observation space
            pred_norm = smoothed_states @ self.kf.observation_matrices.T
            predictions.append(pred_norm)
        
        predictions = np.array(predictions)
        
        # Denormalize
        predictions = predictions * self.data_std + self.data_mean
        
        # Split back into regions
        pred_r1 = predictions[:, :, :self.n_region1]
        pred_r2 = predictions[:, :, self.n_region1:]
        
        return [pred_r1, pred_r2]
    
    def save(self, path):
        """Save model to disk"""
        with open(f"{path}/model.pkl", 'wb') as f:
            pickle.dump({
                'kf': self.kf,
                'pca': self.pca,
                'params': self.params,
                'n_region1': self.n_region1,
                'n_region2': self.n_region2,
                'data_mean': self.data_mean,
                'data_std': self.data_std,
                'fitted': self.fitted
            }, f)
    
    def load(self, path):
        """Load model from disk"""
        with open(f"{path}/model.pkl", 'rb') as f:
            data = pickle.load(f)
            self.kf = data['kf']
            self.pca = data['pca']
            self.params = data['params']
            self.n_region1 = data['n_region1']
            self.n_region2 = data['n_region2']
            self.data_mean = data['data_mean']
            self.data_std = data['data_std']
            self.fitted = data['fitted']