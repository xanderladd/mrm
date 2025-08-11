"""Standalone training script for Motor-Decision RNN"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

from models.motor_rnn import MultiRegionRNN
from utils import generate_task_data, save_model, extract_trajectories

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training parameters
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
N_TRIALS = 500
SEQUENCE_LENGTH = 30
CACHE_DIR = "cache/decision_motor_rnn"

def train():
    """Train Multi-Region RNN for decision-motor task"""
    
    # Initialize model
    model = MultiRegionRNN(evidence_hidden=64, motor_hidden=64).to(device)
    
    # Optimizer with OneCycleLR scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-3, epochs=EPOCHS, steps_per_epoch=N_TRIALS//BATCH_SIZE
    )
    
    # Training history
    losses = []
    evidence_losses = []
    motor_losses = []
    train_mse_history = []
    train_r2_history = []
    
    print("Training Multi-Region RNN...")
    print("-" * 50)
    
    for epoch in tqdm(range(EPOCHS), desc="Epochs"):
        epoch_loss = 0
        epoch_evidence_loss = 0
        epoch_motor_loss = 0
        all_preds = []
        all_targets = []
        
        for batch_idx in range(N_TRIALS // BATCH_SIZE):
            # Generate batch
            batch_data = generate_task_data(BATCH_SIZE, SEQUENCE_LENGTH)
            
            # Forward pass
            outputs = model(batch_data['evidence'])
            
            # Compute losses
            evidence_loss = nn.MSELoss()(
                outputs['evidence_integration'], 
                batch_data['true_integration'].unsqueeze(-1)
            )
            
            motor_loss = nn.MSELoss()(
                outputs['motor_velocity'],
                batch_data['target_velocities']
            )
            
            total_loss = evidence_loss + motor_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Track losses
            epoch_loss += total_loss.item()
            epoch_evidence_loss += evidence_loss.item()
            epoch_motor_loss += motor_loss.item()
            
            # Store predictions for metrics
            combined_pred = torch.cat([outputs['evidence_integration'], 
                                      outputs['motor_velocity']], dim=-1)
            combined_target = torch.cat([batch_data['true_integration'].unsqueeze(-1),
                                        batch_data['target_velocities']], dim=-1)
            all_preds.append(combined_pred.detach().cpu())
            all_targets.append(combined_target.detach().cpu())
        
        # Average losses
        n_batches = N_TRIALS // BATCH_SIZE
        losses.append(epoch_loss / n_batches)
        evidence_losses.append(epoch_evidence_loss / n_batches)
        motor_losses.append(epoch_motor_loss / n_batches)
        
        # Compute MSE and R²
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        train_mse = float(np.mean((all_preds - all_targets) ** 2))
        train_r2 = float(1 - np.var(all_preds - all_targets) / np.var(all_targets))
        train_mse_history.append(train_mse)
        train_r2_history.append(train_r2)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Loss={losses[-1]:.4f}, MSE={train_mse:.4f}, R²={train_r2:.3f} | Ev={evidence_losses[-1]:.4f}, Mot={motor_losses[-1]:.4f}")
    
    print("-" * 50)
    print(f"Training complete!")
    print(f"Final MSE: {train_mse:.4f}")
    print(f"Final R²: {train_r2:.3f}")
    
    # Save model
    metadata = {
        'losses': losses,
        'evidence_losses': evidence_losses,
        'motor_losses': motor_losses,
        'train_mse': train_mse_history,
        'train_r2': train_r2_history,
        'final_mse': train_mse,
        'final_r2': train_r2,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'sequence_length': SEQUENCE_LENGTH
    }
    save_model(model, CACHE_DIR, metadata)
    
    # Plot training curves
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    axes[0, 0].plot(losses)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(evidence_losses)
    axes[0, 1].set_title('Evidence Integration Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(motor_losses)
    axes[0, 2].set_title('Motor Control Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].plot(train_mse_history)
    axes[1, 0].set_title('Training MSE')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(train_r2_history)
    axes[1, 1].set_title('Training R²')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot both MSE and R² on same plot with two y-axes
    ax1 = axes[1, 2]
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE', color=color)
    ax1.plot(train_mse_history, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('R²', color=color)
    ax2.plot(train_r2_history, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_title('MSE & R² Progress')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CACHE_DIR, 'training_curves.png'))
    plt.show()
    
    return model, losses, evidence_losses, motor_losses

def analyze_model(model):
    """Analyze trained model behavior"""
    from sklearn.decomposition import PCA
    import numpy as np
    
    print("\nAnalyzing model behavior...")
    
    # Extract trajectories
    trajectories = extract_trajectories(model, n_trials=20)
    
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
    for trial in range(min(5, n_trials)):
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

def main():
    # Check if model exists
    if os.path.exists(os.path.join(CACHE_DIR, "model.pt")):
        print("Loading cached model...")
        model = MultiRegionRNN(evidence_hidden=64, motor_hidden=64).to(device)
        model.load_state_dict(torch.load(os.path.join(CACHE_DIR, "model.pt"), map_location=device))
        print("Model loaded successfully!")
    else:
        # Train new model
        model, losses, evidence_losses, motor_losses = train()
        print(f"\nTraining complete! Final loss: {losses[-1]:.4f}")
    
    # Analyze model
    analyze_model(model)

if __name__ == "__main__":
    main()