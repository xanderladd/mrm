"""MP-rSLDS model wrapper for multi-region dynamics"""
import numpy as np
import ssm
from ssm.extensions.mp_srslds.emissions_ext import GaussianOrthogonalCompoundEmissions  
from ssm.extensions.mp_srslds.transitions_ext import StickyRecurrentOnlyTransitions

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
        
    def fit(self, data, **kwargs):
        """Fit model to data"""
                
        # Reshape for SSM
        data_flat = data.reshape(-1, data_norm.shape[-1])
        
        # Fit model
        elbos, posterior = self.model.fit(
            data_flat,
            method="laplace_em",
            variational_posterior="structured_meanfield",
            continuous_optimizer='newton',
            initialize=True,
            **kwargs
        )
        
        self.posterior = posterior
        self.elbos = elbos
        self.fitted = True
        
        return elbos, posterior
    
    def predict(self, data):
        """Get predictions for data"""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get predictions
        preds_flat = self.model.smooth(
            self.posterior.mean_continuous_states[0], 
            data_flat
        )
        
        # Reshape and denormalize
        preds = preds_flat.reshape(data.shape)
        preds = preds * self.data_std + self.data_mean
        
        return preds
    
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