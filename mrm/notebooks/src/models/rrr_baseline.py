"""Reduced Rank Regression baseline model for multi-region neural dynamics"""
import os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

class RRRBaseline:
    """Reduced Rank Regression baseline for multi-region modeling"""
    
    def __init__(self, params):
        self.rank = params.get('rank', 5)
        self.alpha = params.get('alpha', 1.0)  # Ridge regularization
        self.delay = params.get('delay', 0)   # Time delay
        self.fitted = False
        
    def fit(self, source_data, target_data):
        """Fit RRR model
        Args:
            source_data: [trials, time, neurons_source] 
            target_data: [trials, time, neurons_target]
        """
        # Apply time delay if specified
        if self.delay > 0:
            X = source_data[:, :-self.delay, :]
            Y = target_data[:, self.delay:, :]
        else:
            X = source_data
            Y = target_data
            
        # Flatten across trials and time
        X_flat = X.reshape(-1, X.shape[-1])
        Y_flat = Y.reshape(-1, Y.shape[-1])
        
        # Center data
        self.X_mean = np.mean(X_flat, axis=0)
        self.Y_mean = np.mean(Y_flat, axis=0)
        X_centered = X_flat - self.X_mean
        Y_centered = Y_flat - self.Y_mean
        
        # Step 1: Ridge regression
        ridge = Ridge(alpha=self.alpha, fit_intercept=False)
        ridge.fit(X_centered, Y_centered)
        W_ridge = ridge.coef_.T  # [n_source, n_target]
        
        # Step 2: SVD on X @ W_ridge for rank reduction
        XW = X_centered @ W_ridge
        U, s, Vt = np.linalg.svd(XW, full_matrices=False)
        
        # Keep top-k components
        k = min(self.rank, len(s))
        V_k = Vt[:k, :].T  # [n_target, k]
        
        # Step 3: Reduced rank weight matrix
        self.W_rrr = W_ridge @ V_k @ V_k.T
        
        self.fitted = True
        
        # Store components for analysis
        self.ridge_weights = W_ridge
        self.components = V_k
        self.singular_values = s[:k]
        
    def predict(self, source_data):
        """Predict target from source using RRR mapping"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
            
        # Apply same delay structure as training
        if self.delay > 0:
            X = source_data[:, :-self.delay, :]
            pred_shape = source_data.shape[:-1]  # Keep original time dim
            pred_shape = (pred_shape[0], pred_shape[1], self.W_rrr.shape[1])
        else:
            X = source_data
            pred_shape = source_data.shape[:-1] + (self.W_rrr.shape[1],)
            
        X_flat = X.reshape(-1, X.shape[-1])
        X_centered = X_flat - self.X_mean
        
        # Predict using reduced rank weights
        Y_pred = X_centered @ self.W_rrr + self.Y_mean
        
        if self.delay > 0:
            # Pad with zeros for delayed timesteps
            Y_reshaped = Y_pred.reshape(X.shape[:-1] + (-1,))
            Y_full = np.zeros(pred_shape)
            Y_full[:, self.delay:, :] = Y_reshaped
            return Y_full
        else:
            return Y_pred.reshape(pred_shape)
    
    def get_explained_variance_ratio(self):
        """Get explained variance ratio of components"""
        if not self.fitted:
            return None
        total_var = np.sum(self.singular_values**2)
        return (self.singular_values**2) / total_var

def create_rrr_model(config):
    """Create RRR model from config"""
    return RRRBaseline(config['model_params'])

def fit_rrr(config):
    """Fit RRR model following MRM conventions using RNN trajectories"""
    # Load RNN model and extract trajectories (same as other models)
    from utils import extract_trajectories
    from models.motor_rnn import MultiRegionRNN
    from utils import load_config
    import torch
    
    print("    Loading Motor RNN to extract trajectories...")
    
    # Use same device logic as other models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the motor_rnn model
    rnn_config = load_config('configs/motor_rnn.json')
    rnn_model = MultiRegionRNN(**rnn_config['model_params']).to(device)
    
    # Load RNN weights
    rnn_path = os.path.join(rnn_config['cache_path'], 'model.pt')
    if os.path.exists(rnn_path):
        state_dict = torch.load(rnn_path, map_location=device)
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        rnn_model.load_state_dict(state_dict)
    else:
        raise ValueError("Motor RNN must be trained first! Run train_motor_rnn.py")
    
    # Extract trajectories
    print("    Extracting trajectories from Motor RNN...")
    trajectories = extract_trajectories(rnn_model, 
                                      n_trials=config['data_params']['n_trials'],
                                      noise_scale=config['data_params'].get('noise_scale', 0.01))
    
    # Convert to numpy arrays
    evidence_data = np.array(trajectories['evidence_states'])  # [trials, time, features]
    motor_data = np.array(trajectories['motor_states'])
    
    print(f"    Data shapes: Evidence {evidence_data.shape}, Motor {motor_data.shape}")
    
    # Create model  
    model = create_rrr_model(config)
    
    # Fit model
    model.fit(evidence_data, motor_data)
    
    # Evaluate
    motor_pred = model.predict(evidence_data)
    
    # Handle different shapes due to potential delay
    if model.delay > 0:
        motor_eval = motor_data[:, model.delay:, :]
        pred_eval = motor_pred[:, model.delay:, :]
    else:
        motor_eval = motor_data
        pred_eval = motor_pred
        
    motor_flat = motor_eval.reshape(-1, motor_eval.shape[-1])
    pred_flat = pred_eval.reshape(-1, pred_eval.shape[-1])
    
    mse = np.mean((motor_flat - pred_flat)**2)
    r2 = r2_score(motor_flat, pred_flat)
    
    # Get explained variance
    exp_var = model.get_explained_variance_ratio()
    
    metadata = {
        'mse': float(mse),
        'r2': float(r2),
        'explained_variance_ratio': exp_var.tolist() if exp_var is not None else [],
        'total_explained_variance': float(np.sum(exp_var)) if exp_var is not None else 0.0,
        'effective_rank': model.rank,
        'model_type': 'rrr',
        'data_shapes': {
            'evidence': evidence_data.shape,
            'motor': motor_data.shape
        }
    }
    
    return model, metadata