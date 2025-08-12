"""CCA baseline model for multi-region neural dynamics"""
import os
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

class CCABaseline:
    """Canonical Correlation Analysis baseline for multi-region modeling"""
    
    def __init__(self, params):
        self.n_components = params.get('n_components', 5)
        self.alpha = params.get('alpha', 0.0)  
        self.cca = CCA(n_components=self.n_components, scale=True)
        self.ridge = Ridge(alpha=1.0)  # Linear mapping from X canonical to Y space
        self.fitted = False
        
    def fit(self, source_data, target_data):
        """Fit CCA model"""
        # Flatten across trials and time
        X = source_data.reshape(-1, source_data.shape[-1])
        Y = target_data.reshape(-1, target_data.shape[-1])
        
        # Remove mean
        self.X_mean = np.mean(X, axis=0)
        self.Y_mean = np.mean(Y, axis=0)
        X_centered = X - self.X_mean
        Y_centered = Y - self.Y_mean
        
        # Fit CCA to find canonical components
        self.cca.fit(X_centered, Y_centered)
        
        # Transform to canonical space
        X_c, Y_c = self.cca.transform(X_centered, Y_centered)
        
        # Fit direct mapping from X canonical components to original Y space
        self.ridge.fit(X_c, Y_centered)
        
        self.fitted = True
        
        # Store canonical correlations for analysis
        self.train_corr = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(self.n_components)]
        
    def predict(self, source_data):
        """Predict target from source"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
            
        X = source_data.reshape(-1, source_data.shape[-1])
        X_centered = X - self.X_mean
        
        # Transform to canonical space
        X_c = self.cca.transform(X_centered)
        
        # Predict Y using direct mapping
        Y_pred_centered = self.ridge.predict(X_c)
        Y_pred = Y_pred_centered + self.Y_mean
        
        return Y_pred.reshape(source_data.shape[:-1] + (-1,))
    
    def get_canonical_correlations(self):
        """Get canonical correlations"""
        return self.train_corr if self.fitted else None

def create_cca_model(config):
    """Create CCA model from config"""
    return CCABaseline(config['model_params'])

def fit_cca(config):
    """Fit CCA model following MRM conventions using RNN trajectories"""
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
    model = create_cca_model(config)
    
    # Fit model
    model.fit(evidence_data, motor_data)
    
    # Evaluate
    motor_pred = model.predict(evidence_data)
    motor_flat = motor_data.reshape(-1, motor_data.shape[-1])
    pred_flat = motor_pred.reshape(-1, motor_pred.shape[-1])
    
    mse = np.mean((motor_flat - pred_flat)**2)
    r2 = r2_score(motor_flat, pred_flat)
    
    # Get canonical correlations
    corrs = model.get_canonical_correlations()
    
    metadata = {
        'mse': float(mse),
        'r2': float(r2),
        'canonical_correlations': corrs,
        'mean_correlation': float(np.mean(corrs)) if corrs else 0.0,
        'model_type': 'cca',
        'data_shapes': {
            'evidence': evidence_data.shape,
            'motor': motor_data.shape
        }
    }
    
    return model, metadata