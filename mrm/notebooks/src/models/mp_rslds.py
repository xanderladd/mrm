"""MP-rSLDS model wrapper for multi-region dynamics"""
import numpy as np
import ssm
from ssm.extensions.mp_srslds.emissions_ext import GaussianOrthogonalCompoundEmissions  
from ssm.extensions.mp_srslds.transitions_ext import StickyRecurrentOnlyTransitions
import torch


def create_mp_rslds(params):
    """Create MP-rSLDS model with given parameters"""
    
    # Extract parameters
    K = params['K']  # Number of discrete states
    D_evidence = params['D_evidence']  # Latent dims for evidence
    D_motor = params['D_motor']  # Latent dims for motor
    N_evidence = params['N_evidence']  # Number of evidence neurons
    N_motor = params['N_motor']  # Number of motor neurons
    
    # Setup vectors
    D_vec = [D_evidence, D_motor]
    N_vec = [N_evidence, N_motor]
    D_total = sum(D_vec)
    N_total = sum(N_vec)
    
    # Create emissions
    emissions = GaussianOrthogonalCompoundEmissions(
        N=N_total, K=1, D=D_total,
        D_vec=D_vec, N_vec=N_vec
    )
    
    # Create transitions  
    transitions = StickyRecurrentOnlyTransitions(
        K=K, D=D_total,
        l2_penalty_similarity=10,
        l1_penalty=10
    )
    
    # Create model
    rslds = ssm.SLDS(
        N=N_total, K=K, D=D_total,
        dynamics="gaussian",
        emissions=emissions,
        transitions=transitions,
        dynamics_kwargs=dict(l2_penalty_A=100)
    )
    
    return rslds

class MPrSLDSWrapper:
    """Wrapper class for MP-rSLDS to match interface"""
    
    def __init__(self, params):
        self.model = create_mp_rslds(params)
        self.params = params
        self.fitted = False
        self.data_mean = None
        self.data_std = None
        
    def fit(self, train_data, region_info=None, **training_params):
        """Fit MP-rSLDS model to trajectory data"""
        import ssm
        from ssm.extensions.mp_srslds.emissions_ext import GaussianOrthogonalCompoundEmissions
        from ssm.extensions.mp_srslds.transitions_ext import StickyRecurrentOnlyTransitions
        from utils import compute_r2
        
        # Prepare data - concatenate region_1 and region_2
        region_1_data = train_data['region_1']  # evidence
        region_2_data = train_data['region_2']  # motor
        data = np.concatenate([region_1_data, region_2_data], axis=2)
        
        # Normalize
        self.data_mean = np.mean(data.reshape(-1, data.shape[-1]), axis=0)
        self.data_std = np.std(data.reshape(-1, data.shape[-1]), axis=0) + 1e-8
        data = (data - self.data_mean) / self.data_std
        
        # Setup model parameters
        D_vec = [self.params['D_evidence'], self.params['D_motor']]
        N_vec = [self.params['N_evidence'], self.params['N_motor']]
        D_total = sum(D_vec)
        N_total = sum(N_vec)
        
        emissions = GaussianOrthogonalCompoundEmissions(
            N=N_total, K=1, D=D_total, D_vec=D_vec, N_vec=N_vec)
        
        transitions = StickyRecurrentOnlyTransitions(
            K=self.params['K'], D=D_total, l2_penalty_similarity=10, l1_penalty=10)
        
        self.model = ssm.SLDS(N=N_total, K=self.params['K'], D=D_total,
                            dynamics="gaussian", emissions=emissions, transitions=transitions,
                            dynamics_kwargs=dict(l2_penalty_A=100))
        
        # Fit model
        data_flat = data.reshape(-1, N_total)
        elbos, self.posterior = self.model.fit(
            data_flat,
            method="laplace_em",
            variational_posterior="structured_meanfield",
            continuous_optimizer='newton',
            initialize=True,
            num_init_restarts=training_params.get('num_init_restarts', 3),
            num_iters=training_params.get('num_iters', 20),
            alpha=training_params.get('alpha', 0.0)
        )
        
        self.fitted = True
        
        # Compute metrics
        preds = self.model.smooth(self.posterior.mean_continuous_states[0], data_flat)
        mse = np.mean((data_flat - preds)**2)
        r2 = 1 - np.var(data_flat - preds) / np.var(data_flat)
        
        # Compute per-region metrics
        evidence_preds = preds[:, :self.params['N_evidence']]
        motor_preds = preds[:, self.params['N_evidence']:]
        evidence_data = data_flat[:, :self.params['N_evidence']]
        motor_data_flat = data_flat[:, self.params['N_evidence']:]
        
        evidence_mse = np.mean((evidence_data - evidence_preds)**2)
        motor_mse = np.mean((motor_data_flat - motor_preds)**2)
        evidence_r2 = compute_r2(evidence_preds.flatten(), evidence_data.flatten())
        motor_r2 = compute_r2(motor_preds.flatten(), motor_data_flat.flatten())
        
        print(f"\n  Final MSE: {mse:.4f}")
        print(f"  Final R²: {r2:.3f}")
        print(f"  Evidence MSE: {evidence_mse:.4f}, R²: {evidence_r2:.3f}")
        print(f"  Motor MSE: {motor_mse:.4f}, R²: {motor_r2:.3f}")
        
        return {
            'elbos': elbos.tolist(),
            'mse': float(mse),
            'r2': float(r2),
            'evidence_mse': float(evidence_mse),
            'motor_mse': float(motor_mse),
            'evidence_r2': float(evidence_r2),
            'motor_r2': float(motor_r2),
            'model_type': 'mp_rslds'
        }
    
    def predict(self, data, region_ids=None):
        """Get predictions for data using MP-rSLDS reconstruction
        
        Args:
            data: Either single tensor [trials, time, features] or list of tensors
            region_ids: List specifying region ordering if data is a list
            
        Returns:
            Predictions in same format as input
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Handle different input formats
        if isinstance(data, list):
            if region_ids is None:
                raise ValueError("region_ids must be provided when data is a list")
            # Concatenate all regions for full state reconstruction
            all_data = torch.cat(data, dim=-1) if isinstance(data[0], torch.Tensor) else np.concatenate(data, axis=-1)
            input_data = all_data
            
        else:
            # Single tensor - assume this is evidence, need to pad with motor states
            # For MP-rSLDS we need full state (evidence + motor) for reconstruction
            # If only evidence is provided, we'll pad with zeros for motor dimensions
            if hasattr(self, 'motor_dim'):
                motor_dim = self.motor_dim
            else:
                # Infer from training data dimensions
                motor_dim = data.shape[-1]  # Assume same size as evidence for now
            
            # Pad with zeros for motor dimensions
            motor_padding = np.zeros(data.shape[:-1] + (motor_dim,))
            input_data = np.concatenate([data, motor_padding], axis=-1)
        
        # Normalize data
        data_norm = (input_data - self.data_mean) / self.data_std
        n_trials, n_time, n_dims = data_norm.shape
        data_flat = data_norm.reshape(-1, n_dims).astype(np.float64)
        
        # Handle both wrapper and direct SLDS objects
        if hasattr(self.model, 'model'):
            rslds_model = self.model.model
        else:
            rslds_model = self.model
        
        # Run inference to get reconstructions
        try:
            _, posterior = rslds_model.fit(data_flat, method="laplace_em", 
                                        initialize=False, num_iters=1)
            preds_flat = rslds_model.smooth(posterior.mean_continuous_states[0], data_flat)
        except:
            # Fallback approach
            preds_flat = rslds_model.smooth(data_flat, data_flat)
        
        # Reshape and denormalize
        preds = preds_flat.reshape(input_data.shape)
        preds = preds * self.data_std + self.data_mean
        
        # Return in same format as input
        if isinstance(data, list):
            # Split back into list format matching input
            start_idx = 0
            result = []
            for tensor in data:
                end_idx = start_idx + tensor.shape[-1]
                result.append(preds[..., start_idx:end_idx])
                start_idx = end_idx
            return result
        else:
            # Return only motor reconstruction for single tensor input
            evidence_dim = data.shape[-1]
            return preds[..., evidence_dim:]
            
    def get_discrete_states(self, data):
        """Get most likely discrete states"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        data_norm = (data - self.data_mean) / self.data_std
        data_flat = data_norm.reshape(-1, data_norm.shape[-1])
        
        z_inferred = self.model.most_likely_states(
            self.posterior.mean_continuous_states[0],
            data_flat
        )
        
        return z_inferred.reshape(data.shape[:-1])
    
    def compute_metrics(self, data):
        """Compute reconstruction metrics"""
        preds = self.predict(data)
        mse = np.mean((data - preds)**2)
        r2 = 1 - np.var(data - preds) / np.var(data)
        
        return {'mse': mse, 'r2': r2}