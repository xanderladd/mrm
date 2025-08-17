"""Standalone training script for Motor-Decision RNN"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

from models.motor_rnn import MultiRegionRNN
from utils import generate_task_data, save_model, extract_trajectories, load_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training parameters
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
N_TRIALS = 500
SEQUENCE_LENGTH = 30
EVIDENCE_PHASE = 15
CACHE_DIR = "cache/decision_motor_rnn"
def train(config):
    """Train Multi-Region RNN - ORIGINAL simple approach"""
    model_params = config['model_params']
    training_params = config['training_params']
    data_params = config['data_params']
    
    # Initialize model
    model = MultiRegionRNN(**model_params).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=training_params['lr'], 
        weight_decay=training_params.get('weight_decay', 1e-4)
    )
    
    if training_params.get('use_scheduler', True):
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=training_params.get('max_lr', 5e-3), 
            epochs=training_params['epochs'],
            steps_per_epoch=data_params['n_trials'] // training_params['batch_size']
        )
    else:
        scheduler = None
    
    losses = []
    evidence_losses = []
    motor_losses = []
    
    print("Training Multi-Region RNN - SIMPLE original approach...")
    
    for epoch in tqdm(range(training_params['epochs']), desc="Epochs"):
        epoch_loss = 0
        epoch_evidence_loss = 0
        epoch_motor_loss = 0
        
        n_batches = data_params['n_trials'] // training_params['batch_size']
        for batch_idx in range(n_batches):
            # Generate batch
            batch_data = generate_task_data(
                training_params['batch_size'], 
                data_params['sequence_length'],
                coherence=data_params.get('coherence', 0.8),
                noise_level=data_params.get('noise_level', 0.5)
            )
            
            # Forward pass
            outputs = model(batch_data['evidence'])
            
            # SIMPLE LOSSES (from your document)
            # Region 1 loss: Evidence integration accuracy
            evidence_loss = nn.MSELoss()(
                outputs['evidence_integration'], 
                batch_data['true_integration'].unsqueeze(-1)
            )
            
            # Region 2 loss: Motor control accuracy  
            motor_loss = nn.MSELoss()(
                outputs['motor_velocity'],
                batch_data['target_velocities']
            )
            
            # Combined loss (simple!)
            total_loss = evidence_loss + motor_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            epoch_loss += total_loss.item()
            epoch_evidence_loss += evidence_loss.item()
            epoch_motor_loss += motor_loss.item()
        
        # Record losses
        losses.append(epoch_loss / n_batches)
        evidence_losses.append(epoch_evidence_loss / n_batches)
        motor_losses.append(epoch_motor_loss / n_batches)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Total={losses[-1]:.4f}, Evidence={evidence_losses[-1]:.4f}, Motor={motor_losses[-1]:.4f}")
    
    # Save model
    metadata = {
        'losses': losses,
        'evidence_losses': evidence_losses,
        'motor_losses': motor_losses,
        'config': config
    }
    save_model(model, config['cache_path'], metadata)
    
    return model, losses, evidence_losses, motor_losses

# ===== Keep our existing analysis function =====
def analyze_model(model, config):
    """Analyze trained model"""
    from sklearn.decomposition import PCA
    
    print("\nAnalyzing model behavior...")
    trajectories = extract_trajectories(model, n_trials=100)
    
    # PCA analysis
    evidence_states = trajectories['evidence_states'].reshape(-1, config['model_params']['evidence_hidden'])
    pca_evidence = PCA(n_components=3)
    evidence_pca = pca_evidence.fit_transform(evidence_states)
    
    motor_states = trajectories['motor_states'].reshape(-1, config['model_params']['motor_hidden'])
    pca_motor = PCA(n_components=3)
    motor_pca = pca_motor.fit_transform(motor_states)
    
    # Plot trajectories
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(121, projection='3d')
    n_trials = trajectories['evidence_states'].shape[0]
    for trial in range(n_trials):
        start_idx = trial * config['data_params']['sequence_length']
        end_idx = (trial + 1) * config['data_params']['sequence_length']
        traj = evidence_pca[start_idx:end_idx]
        color = 'red' if trajectories['trial_info'][trial]['target'] == 0 else 'blue'
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, alpha=0.7)
    ax1.set_title('Evidence Region Trajectories')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    
    ax2 = fig.add_subplot(122, projection='3d')
    for trial in range(n_trials):
        start_idx = trial * config['data_params']['sequence_length']
        end_idx = (trial + 1) * config['data_params']['sequence_length']
        traj = motor_pca[start_idx:end_idx]
        color = 'red' if trajectories['trial_info'][trial]['target'] == 0 else 'blue'
        ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, alpha=0.7)
    ax2.set_title('Motor Region Trajectories')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_zlabel('PC3')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['cache_path'], 'trajectories.png'))
    plt.show()
    
    print(f"Evidence variance explained: {pca_evidence.explained_variance_ratio_[:3].sum():.2%}")
    print(f"Motor variance explained: {pca_motor.explained_variance_ratio_[:3].sum():.2%}")

# ===== Keep existing main function =====
def main():
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = 'configs/motor_rnn.json'
    
    config = load_config(config_path)
    
    # Check if model exists
    model_path = os.path.join(config['cache_path'], "model.pt")
    if os.path.exists(model_path):
        print("Loading cached model...")
        model = MultiRegionRNN(**config['model_params']).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")
    else:
        # Train new model
        model, losses, evidence_losses, motor_losses = train(config)
        print(f"\nTraining complete! Final loss: {losses[-1]:.4f}")
    
    # Analyze model
    analyze_model(model, config)

if __name__ == "__main__":
    main()