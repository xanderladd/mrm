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


# ===== UPDATED training loop with constrained movement =====
def train(config):
    """Train Multi-Region RNN with SIMPLE position supervision"""
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
    
    print("Training Multi-Region RNN with SIMPLE position supervision...")
    
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
            
            # 1. Evidence loss: Learn evidence integration
            evidence_loss = nn.MSELoss()(
                outputs['evidence_integration'], 
                batch_data['true_integration'].unsqueeze(-1))
            
            # 2. Motor loss: Learn position trajectory (SIMPLE!)
            position_loss = nn.MSELoss()(
                outputs['motor_position'],
                batch_data['target_positions_sequence'])
            
            # THAT'S IT! Just two losses.
            total_loss = evidence_loss + position_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            epoch_loss += total_loss.item()
            epoch_evidence_loss += evidence_loss.item()
            epoch_motor_loss += position_loss.item()
        
        # Record losses
        losses.append(epoch_loss / n_batches)
        evidence_losses.append(epoch_evidence_loss / n_batches)
        motor_losses.append(epoch_motor_loss / n_batches)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Loss={losses[-1]:.4f}, Ev={evidence_losses[-1]:.4f}, Pos={motor_losses[-1]:.4f}")
    
    # Save model
    metadata = {
        'losses': losses,
        'evidence_losses': evidence_losses,
        'motor_losses': motor_losses,
        'config': config
    }
    save_model(model, config['cache_path'], metadata)
    
    return model, losses, evidence_losses, motor_losses

# ===== SIMPLE analysis =====
def analyze_trajectory_learning(model, config):
    """Analyze how well the model learned the target trajectories"""
    model.eval()
    EVIDENCE_PHASE = config['data_params'].get('evidence_phase', 15)
    
    with torch.no_grad():
        test_data = generate_task_data(5, config['data_params']['sequence_length'])
        outputs = model(test_data['evidence'])
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot trajectories for first 3 trials
        for trial in range(3):
            # Predicted vs target positions
            pred_positions = outputs['motor_position'][trial].cpu().numpy()  # [30, 2]
            target_positions = test_data['target_positions_sequence'][trial].cpu().numpy()  # [30, 2]
            
            # Position trajectory
            axes[0, trial].plot(pred_positions[:, 0], pred_positions[:, 1], 'b-', linewidth=2, label='Predicted')
            axes[0, trial].plot(target_positions[:, 0], target_positions[:, 1], 'r--', linewidth=2, label='Target')
            axes[0, trial].scatter(0, 0, color='black', s=50, marker='o')  # Origin
            axes[0, trial].axvline(x=0, color='gray', alpha=0.3)
            axes[0, trial].axhline(y=0, color='gray', alpha=0.3)
            axes[0, trial].set_xlabel('X Position')
            axes[0, trial].set_ylabel('Y Position')
            axes[0, trial].set_title(f'Trial {trial+1}: Position Trajectory')
            axes[0, trial].legend()
            axes[0, trial].grid(True, alpha=0.3)
            axes[0, trial].set_xlim(-1.2, 1.2)
            axes[0, trial].set_ylim(-0.5, 0.5)
            
            # Position over time
            time_steps = range(len(pred_positions))
            axes[1, trial].plot(time_steps, pred_positions[:, 0], 'b-', linewidth=2, label='Pred X')
            axes[1, trial].plot(time_steps, target_positions[:, 0], 'r--', linewidth=2, label='Target X')
            axes[1, trial].axvline(x=EVIDENCE_PHASE, color='gray', linestyle=':', alpha=0.7, label='Reach Start')
            axes[1, trial].set_xlabel('Time Step')
            axes[1, trial].set_ylabel('X Position')
            axes[1, trial].set_title(f'Trial {trial+1}: Position Over Time')
            axes[1, trial].legend()
            axes[1, trial].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Calculate trajectory accuracy
        position_errors = torch.norm(
            outputs['motor_position'] - test_data['target_positions_sequence'], 
            dim=-1
        )  # [5, 30]
        
        evidence_phase_error = position_errors[:, :EVIDENCE_PHASE].mean().item()
        reach_phase_error = position_errors[:, EVIDENCE_PHASE:].mean().item()
        final_error = position_errors[:, -1].mean().item()
        
        print(f"Position tracking accuracy:")
        print(f"  Evidence phase error: {evidence_phase_error:.4f} (should be ~0)")
        print(f"  Reach phase error: {reach_phase_error:.4f}")
        print(f"  Final position error: {final_error:.4f}")

def analyze_model(model):
    """Analyze trained model behavior"""
    from sklearn.decomposition import PCA
    import numpy as np
    
    print("\nAnalyzing model behavior...")
    
    # Extract trajectories
    trajectories = extract_trajectories(model, n_trials=50)
    # PCA on evidence states
    evidence_states = trajectories['evidence_states'].reshape(-1, 64)
    pca_evidence = PCA(n_components=3)
    evidence_pca = pca_evidence.fit_transform(evidence_states)
    
    # PCA on motor states  
    motor_states = trajectories['motor_states'].reshape(-1, 64)
    pca_motor = PCA(n_components=3)
    motor_pca = pca_motor.fit_transform(motor_states)
    
    # Plot trajectories
    fig = plt.figure(figsize=(12, 5))
    
    # Evidence trajectories
    ax1 = fig.add_subplot(121, projection='3d')
    n_trials = trajectories['evidence_states'].shape[0]
    for trial in range(n_trials):
        start_idx = trial * 30
        end_idx = (trial + 1) * 30
        traj = evidence_pca[start_idx:end_idx]
        color = 'red' if trajectories['trial_info'][trial]['target'] == 0 else 'blue'
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, alpha=0.7)
    ax1.set_title('Evidence Region Trajectories')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    
    # Motor trajectories
    ax2 = fig.add_subplot(122, projection='3d')
    for trial in range(min(5, n_trials)):
        start_idx = trial * 30
        end_idx = (trial + 1) * 30
        traj = motor_pca[start_idx:end_idx]
        color = 'red' if trajectories['trial_info'][trial]['target'] == 0 else 'blue'
        ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, alpha=0.7)
    ax2.set_title('Motor Region Trajectories')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_zlabel('PC3')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CACHE_DIR, 'trajectories.png'))
    plt.show()
    
    print(f"Evidence variance explained: {pca_evidence.explained_variance_ratio_[:3].sum():.2%}")
    print(f"Motor variance explained: {pca_motor.explained_variance_ratio_[:3].sum():.2%}")


# Update main function
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
    analyze_model(model)
    analyze_trajectory_learning(model, config)

if __name__ == "__main__":
    main()