"""CCA baseline model for multi-region neural dynamics"""
import os
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import torch

class CCABaseline:
    """Canonical Correlation Analysis baseline for multi-region modeling"""
    
    def __init__(self, params):
        self.n_components = params.get('n_components', 5)
        self.alpha = params.get('alpha', 0.0)  
        self.cca = CCA(n_components=self.n_components, scale=True)
        self.ridge = Ridge(alpha=.01)  # Linear mapping from X canonical to Y space
        self.fitted = False
        
    def fit(self, train_data, region_info=None, **training_params):
        """Fit single CCA model for bidirectional reconstruction"""
        from sklearn.metrics import r2_score
        
        region_1_data = train_data['region_1']  # [trials, time, region_dim]
        region_2_data = train_data['region_2']  # [trials, time, region_dim]
        
        print(f"  Training bidirectional CCA on shapes: {region_1_data.shape} ↔ {region_2_data.shape}")
        
        # Flatten for CCA
        X1 = region_1_data.reshape(-1, region_1_data.shape[-1])
        X2 = region_2_data.reshape(-1, region_2_data.shape[-1])
        
        # Center data
        self.X1_mean = np.mean(X1, axis=0)
        self.X2_mean = np.mean(X2, axis=0)
        X1_centered = X1 - self.X1_mean
        X2_centered = X2 - self.X2_mean


        # Fit single CCA model for bidirectional mapping
        self.cca.fit(X1_centered, X2_centered)
        
        # Get canonical components for both directions
        X1_c, X2_c = self.cca.transform(X1_centered, X2_centered)
        
        # Train ridge regressions for both directions
        print("  Training ridge regression: region_1_canonical → region_2_original")
        self.ridge_1to2 = Ridge(alpha=1.0)
        self.ridge_1to2.fit(X1_c, X2_centered)
        
        print("  Training ridge regression: region_2_canonical → region_1_original")
        self.ridge_2to1 = Ridge(alpha=1.0)  
        self.ridge_2to1.fit(X2_c, X1_centered)
        
        self.fitted = True
        
        # Store canonical correlations for analysis
        self.train_corr = [np.corrcoef(X1_c[:, i], X2_c[:, i])[0, 1] for i in range(self.n_components)]
        
        # Evaluate bidirectional reconstruction
        pred_list = self.predict([region_1_data, region_2_data], region_ids=['region_1', 'region_2'])
        region_1_pred, region_2_pred = pred_list
        
        # Compute metrics for both regions
        r1_flat = region_1_data.reshape(-1, region_1_data.shape[-1])
        r2_flat = region_2_data.reshape(-1, region_2_data.shape[-1])
        r1_pred_flat = region_1_pred.reshape(-1, region_1_pred.shape[-1])
        r2_pred_flat = region_2_pred.reshape(-1, region_2_pred.shape[-1])
        
        # Individual region metrics
        r1_mse = np.mean((r1_flat - r1_pred_flat)**2)
        r2_mse = np.mean((r2_flat - r2_pred_flat)**2)
        r1_r2 = r2_score(r1_flat, r1_pred_flat)
        r2_r2 = r2_score(r2_flat, r2_pred_flat)
        
        # Combined metrics (like MR-GNODE)
        target_combined = np.concatenate([r1_flat, r2_flat], axis=-1)
        pred_combined = np.concatenate([r1_pred_flat, r2_pred_flat], axis=-1)
        
        combined_mse = np.mean((target_combined - pred_combined)**2)
        combined_r2 = r2_score(target_combined, pred_combined)
        
        print(f"\n  Region 1 reconstruction: MSE={r1_mse:.4f}, R²={r1_r2:.3f}")
        print(f"  Region 2 reconstruction: MSE={r2_mse:.4f}, R²={r2_r2:.3f}")
        print(f"  Combined reconstruction: MSE={combined_mse:.4f}, R²={combined_r2:.3f}")
        print(f"  Mean Canonical Correlation: {np.mean(self.train_corr):.3f}")
        
        return {
            'mse': float(combined_mse),
            'r2': float(combined_r2),
            'region_1_mse': float(r1_mse),
            'region_1_r2': float(r1_r2),
            'region_2_mse': float(r2_mse), 
            'region_2_r2': float(r2_r2),
            'canonical_correlations': self.train_corr,
            'mean_correlation': float(np.mean(self.train_corr)),
            'model_type': 'cca',
            'data_shapes': {
                'region_1': region_1_data.shape,
                'region_2': region_2_data.shape
            }
        }

    def predict(self, data, region_ids=None):
        """Predict using single CCA model bidirectionally with direct weight multiplication"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
            
        if isinstance(data, list):
            region_1_data, region_2_data = data
            
            # Region 1 → Region 2
            X1 = region_1_data.reshape(-1, region_1_data.shape[-1])
            X1_centered = X1 - self.X1_mean
            X1_c = X1_centered @ self.cca.x_weights_  # Direct transformation to canonical space
            X2_pred_centered = self.ridge_1to2.predict(X1_c)
            region_2_pred = (X2_pred_centered + self.X2_mean).reshape(region_2_data.shape)
            
            # Region 2 → Region 1  
            X2 = region_2_data.reshape(-1, region_2_data.shape[-1])
            X2_centered = X2 - self.X2_mean
            X2_c = X2_centered @ self.cca.y_weights_  # Direct transformation to canonical space
            X1_pred_centered = self.ridge_2to1.predict(X2_c)
            region_1_pred = (X1_pred_centered + self.X1_mean).reshape(region_1_data.shape)
            
            return [region_1_pred, region_2_pred]
            
        else:
            # Single input - assume region_1, predict region_2
            X1 = data.reshape(-1, data.shape[-1])
            X1_centered = X1 - self.X1_mean
            X1_c = X1_centered @ self.cca.x_weights_
            X2_pred_centered = self.ridge_1to2.predict(X1_c)
            X2_pred = X2_pred_centered + self.X2_mean
            return X2_pred.reshape(data.shape[:-1] + (-1,))

    
    def get_canonical_correlations(self):
        """Get canonical correlations"""
        return self.train_corr if self.fitted else None

def create_cca_model(config):
    """Create CCA model from config"""
    return CCABaseline(config['model_params'])
