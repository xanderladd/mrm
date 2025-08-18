"""Bidirectional Reduced Rank Regression for multi-region neural dynamics"""
import os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import torch

class RRRBaseline:
    """Bidirectional RRR: trains two models for region_1↔region_2"""
    
    def __init__(self, params):
        self.rank = params.get('rank', 5)
        self.alpha = params.get('alpha', 1.0)
        self.delay = params.get('delay', 0)
        self.fitted = False
        
    def _fit_single_direction(self, X, Y):
        """Fit single RRR model from X to Y"""
        # Apply delay
        if self.delay > 0:
            X_delayed = X[:, :-self.delay, :]
            Y_delayed = Y[:, self.delay:, :]
        else:
            X_delayed, Y_delayed = X, Y
            
        # Flatten and center
        X_flat = X_delayed.reshape(-1, X_delayed.shape[-1])
        Y_flat = Y_delayed.reshape(-1, Y_delayed.shape[-1])
        
        X_mean = np.mean(X_flat, axis=0)
        Y_mean = np.mean(Y_flat, axis=0)
        X_centered = X_flat - X_mean
        Y_centered = Y_flat - Y_mean
        
        # Ridge regression
        ridge = Ridge(alpha=self.alpha, fit_intercept=False)
        ridge.fit(X_centered, Y_centered)
        W_ridge = ridge.coef_.T
        
        # SVD for rank reduction
        XW = X_centered @ W_ridge
        U, s, Vt = np.linalg.svd(XW, full_matrices=False)
        k = min(self.rank, len(s))
        V_k = Vt[:k, :].T
        W_rrr = W_ridge @ V_k @ V_k.T
        
        return {
            'W': W_rrr,
            'X_mean': X_mean,
            'Y_mean': Y_mean,
            'singular_values': s[:k],
            'components': V_k
        }
    
    def fit(self, train_data, region_info=None, **training_params):
        """Train bidirectional RRR models"""
        region_1 = train_data['region_1']
        region_2 = train_data['region_2']
        
        print(f"  Training region_1 → region_2...")
        self.model_12 = self._fit_single_direction(region_1, region_2)
        
        print(f"  Training region_2 → region_1...")
        self.model_21 = self._fit_single_direction(region_2, region_1)
        
        self.fitted = True
        
        # Evaluate both directions
        pred_2 = self._predict_single(region_1, self.model_12, region_2.shape)
        pred_1 = self._predict_single(region_2, self.model_21, region_1.shape)
        
        # Handle delay for evaluation
        if self.delay > 0:
            region_1_eval = region_1[:, self.delay:, :]
            region_2_eval = region_2[:, self.delay:, :]
            pred_1_eval = pred_1[:, self.delay:, :]
            pred_2_eval = pred_2[:, self.delay:, :]
        else:
            region_1_eval, region_2_eval = region_1, region_2
            pred_1_eval, pred_2_eval = pred_1, pred_2
        
        # Calculate metrics
        mse_12 = np.mean((region_2_eval.flatten() - pred_2_eval.flatten())**2)
        mse_21 = np.mean((region_1_eval.flatten() - pred_1_eval.flatten())**2)
        r2_12 = r2_score(region_2_eval.flatten(), pred_2_eval.flatten())
        r2_21 = r2_score(region_1_eval.flatten(), pred_1_eval.flatten())
        
        avg_mse = (mse_12 + mse_21) / 2
        avg_r2 = (r2_12 + r2_21) / 2
        
        print(f"  Region 1→2: MSE={mse_12:.4f}, R²={r2_12:.3f}")
        print(f"  Region 2→1: MSE={mse_21:.4f}, R²={r2_21:.3f}")
        print(f"  Average: MSE={avg_mse:.4f}, R²={avg_r2:.3f}")
        
        return {
            'mse': float(avg_mse),
            'r2': float(avg_r2),
            'mse_12': float(mse_12),
            'mse_21': float(mse_21),
            'r2_12': float(r2_12),
            'r2_21': float(r2_21),
            'model_type': 'bidirectional_rrr',
            'effective_rank': self.rank
        }
    
    def _predict_single(self, X, model, target_shape):
        """Predict using single direction model"""
        # Apply delay
        if self.delay > 0:
            X_delayed = X[:, :-self.delay, :]
            pred_shape = target_shape
        else:
            X_delayed = X
            pred_shape = target_shape
            
        X_flat = X_delayed.reshape(-1, X_delayed.shape[-1])
        X_centered = X_flat - model['X_mean']
        Y_pred = X_centered @ model['W'] + model['Y_mean']
        
        if self.delay > 0:
            Y_reshaped = Y_pred.reshape(X_delayed.shape[:-1] + (-1,))
            Y_full = np.zeros(pred_shape)
            Y_full[:, self.delay:, :] = Y_reshaped
            return Y_full
        else:
            return Y_pred.reshape(pred_shape)
    
    def predict(self, data, region_ids=None):
        """Bidirectional prediction"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        if isinstance(data, list) and len(data) == 2:
            # Input: [region_1_data, region_2_data]
            region_1_data, region_2_data = data
            
            # Predict both directions
            pred_2 = self._predict_single(region_1_data, self.model_12, region_2_data.shape)
            pred_1 = self._predict_single(region_2_data, self.model_21, region_1_data.shape)
            
            return [pred_1, pred_2]  # Return in same order as input
        else:
            raise ValueError("Expected list of 2 regions for bidirectional prediction")

def create_rrr_model(config):
    """Create bidirectional RRR model"""
    return BidirectionalRRR(config['model_params'])

def fit_rrr(config):
    """Fit bidirectional RRR model"""
    from utils import extract_trajectories
    from models.motor_rnn import MultiRegionRNN
    from utils import load_config
    import torch
    
    print("    Loading Motor RNN to extract trajectories...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load RNN model
    rnn_config = load_config('configs/motor_rnn.json')
    rnn_model = MultiRegionRNN(**rnn_config['model_params']).to(device)
    
    rnn_path = os.path.join(rnn_config['cache_path'], 'model.pt')
    if os.path.exists(rnn_path):
        state_dict = torch.load(rnn_path, map_location=device)
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        rnn_model.load_state_dict(state_dict)
    else:
        raise ValueError("Motor RNN must be trained first!")
    
    # Extract trajectories
    print("    Extracting trajectories...")
    trajectories = extract_trajectories(rnn_model, 
                                      n_trials=config['data_params']['n_trials'],
                                      noise_scale=config['data_params'].get('noise_scale', 0.01))
    
    # Prepare data
    train_data = {
        'region_1': np.array(trajectories['evidence_states']),
        'region_2': np.array(trajectories['motor_states'])
    }
    
    print(f"    Data shapes: Region 1 {train_data['region_1'].shape}, Region 2 {train_data['region_2'].shape}")
    
    # Create and fit model
    model = create_rrr_model(config)
    metadata = model.fit(train_data)
    
    return model, metadata