"""Shared utilities for multi-region neural dynamics models"""
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data generation
def generate_task_data(batch_size, sequence_length, coherence=0.8, noise_level=0.5):
    """Generate evidence integration + optimal reaching trajectories"""
    EVIDENCE_PHASE = 15
    
    targets = torch.randint(0, 2, (batch_size,), device=device)
    
    # Evidence streams (same as before)
    evidence = torch.zeros(batch_size, sequence_length, 2, device=device)
    evidence[:, :EVIDENCE_PHASE, 0] = (1 - targets.float()).unsqueeze(1) * coherence + \
                                      torch.randn(batch_size, EVIDENCE_PHASE, device=device) * noise_level
    evidence[:, :EVIDENCE_PHASE, 1] = targets.float().unsqueeze(1) * coherence + \
                                      torch.randn(batch_size, EVIDENCE_PHASE, device=device) * noise_level
    
    # True integration (same as before)
    true_integration = torch.cumsum(evidence[:, :, 1] - evidence[:, :, 0], dim=1)
    true_integration = torch.tanh(true_integration)
    
    # OPTIMAL REACHING TRAJECTORIES
    target_positions_sequence = torch.zeros(batch_size, sequence_length, 2, device=device)
    target_velocities_sequence = torch.zeros(batch_size, sequence_length, 2, device=device)
    
    for i in range(batch_size):
        # Final target location
        final_target = torch.tensor([-1.0, 0.0] if targets[i] == 0 else [1.0, 0.0], device=device)
        
        # EVIDENCE PHASE (0-14): Stay at origin
        target_positions_sequence[i, :EVIDENCE_PHASE, :] = 0.0  # No movement
        target_velocities_sequence[i, :EVIDENCE_PHASE, :] = 0.0  # No velocity
        
        # REACH PHASE (15-29): Linear interpolation to target
        reach_steps = sequence_length - EVIDENCE_PHASE  # 15 steps
        for t in range(EVIDENCE_PHASE, sequence_length):
            # Linear progress: 0.0 at step 15 → 1.0 at step 29
            progress = (t - EVIDENCE_PHASE) / (reach_steps - 1)
            
            # Linear interpolation from origin to target
            current_position = progress * final_target  # [2]
            target_positions_sequence[i, t, :] = current_position
            
            # Velocity = change in position
            if t > EVIDENCE_PHASE:
                velocity = target_positions_sequence[i, t, :] - target_positions_sequence[i, t-1, :]
                target_velocities_sequence[i, t, :] = velocity
    
    return {
        'evidence': evidence,
        'targets': targets,
        'true_integration': true_integration,
        'target_positions_sequence': target_positions_sequence,  # [batch, seq_len, 2] - position at each timestep
        'target_velocities_sequence': target_velocities_sequence  # [batch, seq_len, 2] - velocity at each timestep
    }


def extract_trajectories(model, n_trials: int = 20, noise_scale: float = 0):
    """Extract neural trajectories from trained model"""
    model.eval()
    trajectories = {'evidence_states': [], 'motor_states': [], 'trial_info': []}
    
    with torch.no_grad():
        for _ in range(n_trials):
            data = generate_task_data(1, 30)
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
    """Load model from cache"""
    model_path = os.path.join(cache_path, "model.pt")
    meta_path = os.path.join(cache_path, "metadata.json")
    
    if not os.path.exists(model_path):
        return None, None, False
    
    # Load metadata
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    # Create model instance if class provided
    if model_class and config:
        model = model_class(**config['model_params'])
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
    else:
        model = torch.load(model_path, map_location=device)
    
    return model, metadata, True

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
    
    if model_type in ['cca', 'rrr']:
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
    """Create model instance from config"""
    model_type = config['model_type']
    
    if model_type == 'motor_rnn':
        from models.motor_rnn import MultiRegionRNN
        return MultiRegionRNN(**config['model_params']).to(device)
    elif model_type == 'mr_gnode':
        from models.mr_gnode import MRgnODE_DynamicComm
        return MRgnODE_DynamicComm(**config['model_params']).to(device)
    elif model_type == 'mp_rslds':
        from models.mp_rslds import create_mp_rslds
        return create_mp_rslds(config['model_params'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")