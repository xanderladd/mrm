"""Shared utilities for multi-region neural dynamics models"""
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_task_data(batch_size, sequence_length, coherence=0.8, noise_level=0.5, seed=0):
    """Generate evidence integration + reaching task data with confidence-modulated velocities"""
    EVIDENCE_PHASE = 15
    
    # Random target direction for each trial (0=LEFT, 1=RIGHT)
    targets = torch.randint(0, 2, (batch_size,), device=device)
    
    # Evidence streams (first 15 timesteps)
    evidence = torch.zeros(batch_size, sequence_length, 2, device=device)
    evidence[:, :EVIDENCE_PHASE, 0] = (1 - targets.float()).unsqueeze(1) * coherence + \
                                      torch.randn(batch_size, EVIDENCE_PHASE, device=device) * noise_level
    evidence[:, :EVIDENCE_PHASE, 1] = targets.float().unsqueeze(1) * coherence + \
                                      torch.randn(batch_size, EVIDENCE_PHASE, device=device) * noise_level
    
    # True evidence integration
    raw_int =  torch.cumsum(evidence[:, :, 1] - evidence[:, :, 0], dim=1)
    true_integration = torch.tanh(raw_int)
    # Compute confidence from evidence strength at decision time
    confidence = torch.abs(raw_int[:, EVIDENCE_PHASE-1]) / 20 #torch.mean(torch.abs(raw_int[:, EVIDENCE_PHASE-1]))  # Confidence at end of evidence
    confidence = torch.clamp(confidence, 0, 1.0)  # Keep speeds reasonable (0.2x to 1.0x)
    # Target positions (same as before)
    target_positions = torch.zeros(batch_size, sequence_length, 2, device=device)
    target_velocities = torch.zeros(batch_size, sequence_length, 2, device=device)
    
    for i in range(batch_size):
        if targets[i] == 0:  # LEFT reach
            target_positions[i, :, 0] = -1.0
        else:  # RIGHT reach  
            target_positions[i, :, 0] = 1.0
    
    # Generate smooth noise for each trial (low-frequency variations)
    smooth_noise = torch.zeros(batch_size, sequence_length, device=device)
    for i in range(batch_size):
        # Create smooth noise with random phase and frequency
        t_vals = torch.linspace(0, 4*np.pi, sequence_length, device=device)
        phase = torch.rand(1, device=device) * 2 * np.pi
        freq_scale = 0.5 + torch.rand(1, device=device) * 1.0  # Random frequency
        smooth_noise[i] = 0.4 * torch.sin(freq_scale * t_vals + phase)
    
    # Confidence-modulated velocity trajectories with smooth noise
    for t in range(EVIDENCE_PHASE, sequence_length):
        progress = (t - EVIDENCE_PHASE) / (sequence_length - EVIDENCE_PHASE)
        base_vel = 0.5 * np.sin(np.pi * progress) if progress < 1.0 else 0.0
        
        for i in range(batch_size):
            # Scale velocity by confidence and add smooth noise
            modulated_vel = base_vel * confidence[i].item()
            noise_vel = modulated_vel + smooth_noise[i, t].item()
            
            if targets[i] == 0:  # LEFT
                target_velocities[i, t, 0] = -noise_vel
            else:  # RIGHT
                target_velocities[i, t, 0] = noise_vel
    
    return {
        'evidence': evidence,
        'targets': targets,
        'true_integration': true_integration,
        'target_positions': target_positions,
        'target_velocities': target_velocities
    }

def extract_trajectories(model, n_trials: int = 20, noise_scale: float = 0, seed=0):
    """Extract neural trajectories from trained model"""
    model.eval()
    trajectories = {'evidence_states': [], 'motor_states': [], 'trial_info': []}
    
    with torch.no_grad():
        for trial in range(n_trials):
            if seed is not None: torch.manual_seed(seed + trial)
            data = generate_task_data(1, 30, seed=seed)
            outputs = model(data['evidence'])
            
            # Add noise for variability
            evidence_states = outputs['evidence_states'] + torch.randn_like(outputs['evidence_states']) * noise_scale
            motor_states = outputs['motor_states'] + torch.randn_like(outputs['motor_states']) * noise_scale
            
            trajectories['evidence_states'].append(evidence_states.cpu().numpy())
            trajectories['motor_states'].append(motor_states.cpu().numpy())
            trajectories['trial_info'].append({
                'evidence': data['evidence'].cpu().numpy(),
                'target': int(data['true_integration'][0, -1] > 0)
            })
    
    trajectories['evidence_states'] = np.array(trajectories['evidence_states']).squeeze()
    trajectories['motor_states'] = np.array(trajectories['motor_states']).squeeze()
    
    return trajectories

# Model I/O
def load_cached_model(cache_path: str, model_class=None, config: Dict = None):
    """Fixed load_cached_model that handles both torch and pickle models"""
    import pickle
    
    # Check for metadata
    meta_path = os.path.join(cache_path, "metadata.json")
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    
    model_type = metadata.get('model_type', '')
    
    # Try pickle format first (baseline models)
    pickle_path = os.path.join(cache_path, "model.pkl")
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                model = pickle.load(f)
            return model, metadata, True
        except Exception as e:
            print(f"  Failed to load pickle model: {e}")
    
    # Try torch format (neural models)  
    torch_path = os.path.join(cache_path, "model.pt")
    if os.path.exists(torch_path):
        try:
            if model_class and config:
                # Create model instance and load state dict
                model = model_class(**config['model_params'])
                state_dict = torch.load(torch_path, map_location=device)
                
                # Handle _orig_mod. prefix from torch compilation
                if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
                    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                
                model.load_state_dict(state_dict)
                model.to(device)
                return model, metadata, True
            else:
                # Load full model (fallback)
                model = torch.load(torch_path, map_location=device)
                return model, metadata, True
        except Exception as e:
            print(f"  Failed to load torch model: {e}")
    
    return None, None, False
    
def save_model(model, cache_path, metadata):
    """Save model - handles both torch models and baseline models"""
    import pickle
    import torch
    import json
    
    os.makedirs(cache_path, exist_ok=True)
    
    # Save metadata
    with open(os.path.join(cache_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save model based on type
    model_type = metadata.get('model_type', '')
    
    if model_type in ['cca', 'rrr', 'mp_rslds']:
        # Baseline models use pickle
        with open(os.path.join(cache_path, 'model.pkl'), 'wb') as f:
            pickle.dump(model, f)
    else:
        # Torch models use torch.save
        torch.save(model.state_dict(), os.path.join(cache_path, 'model.pt'))
              
# Metrics
def compute_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute mean squared error"""
    return np.mean((pred - target) ** 2)

def compute_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute R-squared score"""
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

# Visualization
def plot_training_curves(losses: Dict[str, list], save_path: str = None):
    """Plot training loss curves"""
    fig, axes = plt.subplots(1, len(losses), figsize=(5*len(losses), 4))
    
    if len(losses) == 1:
        axes = [axes]
    
    for ax, (name, loss) in zip(axes, losses.items()):
        ax.plot(loss)
        ax.set_title(name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_comparison(results: Dict[str, Dict[str, float]]):
    """Plot model comparison results"""
    models = list(results.keys())
    mse_values = [results[m]['mse'] for m in models]
    r2_values = [results[m]['r2'] for m in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # MSE comparison
    ax1.bar(models, mse_values)
    ax1.set_ylabel('MSE')
    ax1.set_title('Mean Squared Error')
    ax1.grid(True, alpha=0.3)
    
    # R² comparison
    ax2.bar(models, r2_values)
    ax2.set_ylabel('R²')
    ax2.set_title('R-Squared Score')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Config handling
def load_config(config_path: str) -> Dict:
    """Load configuration from JSON"""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_model(config: Dict):
    """Enhanced model factory function for all model types"""
    model_type = config['model_type']
    model_params = config['model_params']
    
    if model_type == 'motor_rnn':
        from models.motor_rnn import MultiRegionRNN
        return MultiRegionRNN(**model_params).to(device)
    
    elif model_type == 'mr_gnode':
        from models.mr_gnode import MRgnODE_DynamicComm
        return MRgnODE_DynamicComm(**model_params).to(device)
    
    elif model_type == 'mp_rslds':
        from models.mp_rslds import MPrSLDSWrapper
        return MPrSLDSWrapper(model_params)
    
    elif model_type == 'cca':
        from models.cca_baseline import CCABaseline
        return CCABaseline(model_params)
    
    elif model_type == 'rrr':
        from models.rrr_baseline import RRRBaseline
        return RRRBaseline(model_params)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")