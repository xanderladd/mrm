"""Shared utilities for multi-region neural dynamics models"""
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data generation
def generate_task_data(batch_size: int, seq_len: int, task_type: str = 'decision_motor'):
    """Generate task data for training"""
    if task_type == 'decision_motor':
        evidence = torch.randn(batch_size, seq_len, 2, device=device) * 0.3
        evidence[:, 15:, :] += torch.randn(batch_size, 1, 2, device=device) * 0.5
        
        targets = torch.cumsum(evidence, dim=1) / 10
        motor_targets = torch.tanh(targets)
        
        return {
            'evidence': evidence,
            'true_integration': targets.mean(dim=-1),
            'target_velocities': motor_targets
        }
    else:
        raise ValueError(f"Unknown task type: {task_type}")

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