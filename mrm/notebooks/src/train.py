"""Universal training script for all models"""
import argparse
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from utils import (load_config, create_model, load_cached_model, save_model, 
                   generate_task_data, extract_trajectories, plot_training_curves,
                   compute_r2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def communication_loss(model, outputs, targets, comm_norm, comm_penalty=0.01):
    """Simplified loss function with communication penalty"""
    mse_loss = nn.MSELoss()(outputs, targets)
    comm_cost = comm_penalty * comm_norm
    total_loss = mse_loss + comm_cost
    return total_loss, {'mse': mse_loss.item(), 'comm_cost': comm_cost.item()}

def create_batched_data_tensors(train_data, batch_size):
    """Convert training data to batched tensors"""
    inputs_batched = []
    targets_batched = []
    
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i+batch_size]
        batch_inputs = torch.stack([torch.tensor(x[0], dtype=torch.float32) for x in batch]).to(device)
        batch_targets = torch.stack([torch.tensor(x[1], dtype=torch.float32) for x in batch]).to(device)
        inputs_batched.append(batch_inputs)
        targets_batched.append(batch_targets)
    
    return inputs_batched, targets_batched

def train_motor_rnn(model, config):
    """Train MultiRegionRNN model"""
    # Store original model for saving
    original_model = model
    
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config['training_params']['lr'],
                           weight_decay=config['training_params']['weight_decay'])
    
    losses = []
    evidence_losses = []
    motor_losses = []
    train_mse_history = []
    train_r2_history = []
    
    n_epochs = config['training_params']['epochs']
    batch_size = config['training_params']['batch_size']
    seq_len = config['data_params']['sequence_length']
    n_trials = config['data_params']['n_trials']
    
    for epoch in tqdm(range(n_epochs), desc="Training Motor RNN"):
        epoch_loss = 0
        epoch_ev_loss = 0
        epoch_mot_loss = 0
        all_preds = []
        all_targets = []
        
        for _ in range(n_trials // batch_size):
            batch_data = generate_task_data(batch_size, seq_len)
            outputs = model(batch_data['evidence'])
            
            # Evidence loss
            ev_loss = nn.MSELoss()(outputs['evidence_integration'], 
                                   batch_data['true_integration'].unsqueeze(-1))
            # Motor loss
            mot_loss = nn.MSELoss()(outputs['motor_velocity'],
                                    batch_data['target_velocities'])
            
            total_loss = ev_loss + mot_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_ev_loss += ev_loss.item()
            epoch_mot_loss += mot_loss.item()
            
            # Store for metrics
            combined_pred = torch.cat([outputs['evidence_integration'], 
                                      outputs['motor_velocity']], dim=-1)
            combined_target = torch.cat([batch_data['true_integration'].unsqueeze(-1),
                                        batch_data['target_velocities']], dim=-1)
            all_preds.append(combined_pred.detach().cpu())
            all_targets.append(combined_target.detach().cpu())
        
        losses.append(epoch_loss / (n_trials // batch_size))
        evidence_losses.append(epoch_ev_loss / (n_trials // batch_size))
        motor_losses.append(epoch_mot_loss / (n_trials // batch_size))
        
        # Compute MSE and R2
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        train_mse = float(np.mean((all_preds - all_targets) ** 2))
        train_r2 = float(compute_r2(all_preds.flatten(), all_targets.flatten()))
        train_mse_history.append(train_mse)
        train_r2_history.append(train_r2)
        
        # Print metrics every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: Loss={losses[-1]:.4f}, MSE={train_mse:.4f}, R²={train_r2:.3f}")
    
    metadata = {
        'losses': losses,
        'evidence_losses': evidence_losses,
        'motor_losses': motor_losses,
        'train_mse': train_mse_history,
        'train_r2': train_r2_history,
        'final_mse': train_mse_history[-1],
        'final_r2': train_r2_history[-1],
        'config': config
    }
    
    return original_model, metadata

def fit_mp_rslds(config):
    """Fit MP-rSLDS model to RNN trajectories"""
    import ssm
    from ssm.extensions.mp_srslds.emissions_ext import GaussianOrthogonalCompoundEmissions
    from ssm.extensions.mp_srslds.transitions_ext import StickyRecurrentOnlyTransitions
    
    # First load the trained RNN to extract trajectories
    rnn_config = load_config('configs/motor_rnn.json')
    from models.motor_rnn import MultiRegionRNN
    rnn_model, _, _ = load_cached_model(rnn_config['cache_path'], MultiRegionRNN, rnn_config)
    
    if rnn_model is None:
        raise ValueError("Must train Motor RNN first before fitting MP-rSLDS")
    
    # Extract trajectories
    trajectories = extract_trajectories(rnn_model, 
                                       n_trials=config['data_params']['n_trials'],
                                       noise_scale=config['data_params']['noise_scale'])
    
    # Prepare data
    evidence_data = trajectories['evidence_states']
    motor_data = trajectories['motor_states']
    data = np.concatenate([evidence_data, motor_data], axis=2)
    
    # Normalize
    data_mean = np.mean(data.reshape(-1, data.shape[-1]), axis=0)
    data_std = np.std(data.reshape(-1, data.shape[-1]), axis=0) + 1e-8
    data = (data - data_mean) / data_std
    
    # Setup model
    p = config['model_params']
    D_vec = [p['D_evidence'], p['D_motor']]
    N_vec = [p['N_evidence'], p['N_motor']]
    D_total = sum(D_vec)
    N_total = sum(N_vec)
    
    emissions = GaussianOrthogonalCompoundEmissions(
        N=N_total, K=1, D=D_total, D_vec=D_vec, N_vec=N_vec)
    
    transitions = StickyRecurrentOnlyTransitions(
        K=p['K'], D=D_total, l2_penalty_similarity=10, l1_penalty=10)
    
    rslds = ssm.SLDS(N=N_total, K=p['K'], D=D_total,
                     dynamics="gaussian", emissions=emissions, transitions=transitions,
                     dynamics_kwargs=dict(l2_penalty_A=100))
    
    # Fit model
    data_flat = data.reshape(-1, N_total)
    elbos, posterior = rslds.fit(
        data_flat,
        method="laplace_em",
        variational_posterior="structured_meanfield",
        continuous_optimizer='newton',
        initialize=True,
        num_init_restarts=config['training_params']['num_init_restarts'],
        num_iters=config['training_params']['num_iters'],
        alpha=config['training_params']['alpha']
    )
    
    # Compute metrics
    preds = rslds.smooth(posterior.mean_continuous_states[0], data_flat)
    mse = np.mean((data_flat - preds)**2)
    r2 = 1 - np.var(data_flat - preds) / np.var(data_flat)
    
    # Compute per-region metrics
    evidence_preds = preds[:, :p['N_evidence']]
    motor_preds = preds[:, p['N_evidence']:]
    evidence_data = data_flat[:, :p['N_evidence']]
    motor_data = data_flat[:, p['N_evidence']:]
    
    evidence_mse = np.mean((evidence_data - evidence_preds)**2)
    motor_mse = np.mean((motor_data - motor_preds)**2)
    evidence_r2 = compute_r2(evidence_preds.flatten(), evidence_data.flatten())
    motor_r2 = compute_r2(motor_preds.flatten(), motor_data.flatten())
    
    print(f"\n  Final MSE: {mse:.4f}")
    print(f"  Final R²: {r2:.3f}")
    print(f"  Evidence MSE: {evidence_mse:.4f}, R²: {evidence_r2:.3f}")
    print(f"  Motor MSE: {motor_mse:.4f}, R²: {motor_r2:.3f}")
    
    metadata = {
        'elbos': elbos.tolist(),
        'mse': float(mse),
        'r2': float(r2),
        'evidence_mse': float(evidence_mse),
        'motor_mse': float(motor_mse),
        'evidence_r2': float(evidence_r2),
        'motor_r2': float(motor_r2),
        'data_mean': data_mean.tolist(),
        'data_std': data_std.tolist(),
        'config': config
    }
    
    return (rslds, posterior), metadata

def train_mr_gnode_dynamic_comm(model, config):
    """Train MR-GNODE with dynamic communication"""
    optimizer = optim.AdamW(model.parameters(), lr=config['training_params']['lr'], weight_decay=1e-2)
    
    n_epochs = config['training_params']['epochs']
    batch_size = config['training_params']['batch_size']
    dt = config['training_params'].get('dt', 0.01)
    comm_penalty = config['training_params'].get('comm_penalty', 0.01)
    
    # Check if using motor_rnn as source
    if 'source_model' in config['data_params'] and config['data_params']['source_model'] == 'motor_rnn':
        # Train on motor_rnn style data
        from utils import extract_trajectories
        from models.motor_rnn import MultiRegionRNN
        
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
            raise ValueError("Motor RNN must be trained first!")
        
        # Extract trajectories as training data
        print("    Extracting trajectories from Motor RNN...")
        trajectories = extract_trajectories(rnn_model, 
                                          n_trials=config['data_params']['n_trials'],
                                          noise_scale=config['data_params'].get('noise_scale', 0.03))
        
        # Prepare data for MR-GNODE
        evidence_states = torch.tensor(trajectories['evidence_states'], dtype=torch.float32).to(device)
        motor_states = torch.tensor(trajectories['motor_states'], dtype=torch.float32).to(device)
        
        # Setup scheduler for motor_rnn data
        n_trials = evidence_states.shape[0]
        steps_per_epoch = n_trials // batch_size + 1
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['training_params']['lr']*5, 
                                                epochs=n_epochs, steps_per_epoch=steps_per_epoch)
        
        losses, train_mse_history, train_r2_history = [], [], []
        
        for epoch in tqdm(range(n_epochs), desc="Training Dynamic Communication MR-GNODE"):
            epoch_loss = 0
            all_preds, all_targets = [], []
            
            # Shuffle indices
            indices = torch.randperm(n_trials)
            
            for i in range(0, n_trials, batch_size):
                batch_indices = indices[i:i+batch_size]
                
                # Get batch
                batch_evidence = evidence_states[batch_indices]  # [batch, time, 64]
                batch_motor = motor_states[batch_indices]  # [batch, time, 64]
                
                # Format input for MR-GNODE: [batch, time, regions, channels]
                batch_input = batch_evidence[:, :, :2].unsqueeze(2)  # [batch, time, 1, 2]
                
                # If model has 2 regions, duplicate input
                if model.num_regions == 2:
                    batch_input = batch_input.repeat(1, 1, 2, 1)  # [batch, time, 2, 2]
                
                # Target is to reconstruct the motor states (first 2 dims)
                batch_target = batch_motor[:, :, :2]
                
                optimizer.zero_grad()
                
                batch_size_actual, seq_len, n_regions, input_dim = batch_input.shape
                
                # Initialize hidden state
                h = torch.empty(batch_size_actual, model.N_total, device=device)
                nn.init.xavier_uniform_(h)
                outputs, total_comm = [], 0
                
                # Forward pass through time
                for t in range(seq_len):
                    h_dot, comm_norm = model(h, batch_input[:, t], timestep=t)
                    h = h + dt * h_dot
                    total_comm += comm_norm
                    step_output = model.get_outputs(h)
                    outputs.append(step_output)
                
                outputs = torch.stack(outputs, dim=1)  # [batch, time, regions, output_dim]
                
                # Average across regions if needed
                if outputs.shape[2] > 1:
                    outputs = outputs.mean(dim=2)  # [batch, time, 2]
                else:
                    outputs = outputs.squeeze(2)
                
                # Compute loss
                loss, loss_components = communication_loss(model, outputs, batch_target, total_comm, comm_penalty)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                all_preds.append(outputs.detach().cpu().numpy())
                all_targets.append(batch_target.detach().cpu().numpy())
            
            losses.append(epoch_loss / (n_trials // batch_size))
            
            # Compute metrics
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            train_mse = float(np.mean((all_preds - all_targets) ** 2))
            train_r2 = float(compute_r2(all_preds.flatten(), all_targets.flatten()))
            train_mse_history.append(train_mse)
            train_r2_history.append(train_r2)
            
            if epoch % (n_epochs // 10) == 0:
                print(f"Epoch {epoch}, Loss: {losses[-1]:.6f}, MSE: {train_mse:.4f}, R²: {train_r2:.3f}")
    
    else:
        # Original flip-flop training with dynamic communication
        from models.mr_gnode import generate_mr_flip_flop_data
        train_data = generate_mr_flip_flop_data(
            config['data_params']['n_trials'],
            task_type=config['data_params'].get('task_type', 'square'),
            combine_method=config['data_params'].get('combine_method', 'sum')
        )
        
        # Setup scheduler
        steps_per_epoch = len(train_data) // batch_size + 1
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['training_params']['lr']*5, 
                                                epochs=n_epochs, steps_per_epoch=steps_per_epoch)
        
        inputs_batched, targets_batched = create_batched_data_tensors(train_data, batch_size)
        
        losses, train_mse_history, train_r2_history = [], [], []
        
        for epoch in tqdm(range(n_epochs), desc="Training Dynamic Communication MR-GNODE"):
            total_loss = 0
            all_preds, all_targets = [], []
            
            for batch_inputs, batch_targets in zip(inputs_batched, targets_batched):
                optimizer.zero_grad()
                
                batch_size_actual, seq_len, n_regions, input_dim = batch_inputs.shape
                
                # Initialize hidden state
                h = torch.empty(batch_size_actual, model.N_total, device=device)
                nn.init.xavier_uniform_(h)
                outputs, total_comm = [], 0
                
                # Forward pass through time
                for t in range(seq_len):
                    h_dot, comm_norm = model(h, batch_inputs[:, t], timestep=t)
                    h = h + dt * h_dot
                    total_comm += comm_norm
                    outputs.append(model.get_outputs(h))
                
                outputs = torch.stack(outputs, dim=1)  # [batch, time, regions, output_dim]
                
                # Compute loss
                loss, loss_components = communication_loss(model, outputs, batch_targets, total_comm, comm_penalty)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                all_preds.append(outputs.detach().cpu().numpy())
                all_targets.append(batch_targets.detach().cpu().numpy())
            
            losses.append(total_loss / len(inputs_batched))
            
            # Compute metrics
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            train_mse = float(np.mean((all_preds - all_targets) ** 2))
            train_r2 = float(compute_r2(all_preds.flatten(), all_targets.flatten()))
            train_mse_history.append(train_mse)
            train_r2_history.append(train_r2)
            
            if epoch % (n_epochs // 10) == 0:
                print(f"Epoch {epoch}, Loss: {losses[-1]:.6f}, MSE: {train_mse:.4f}, R²: {train_r2:.3f}")
    
    model.eval()
    
    metadata = {
        'losses': losses,
        'train_mse': train_mse_history,
        'train_r2': train_r2_history,
        'final_mse': train_mse_history[-1],
        'final_r2': train_r2_history[-1],
        'config': config
    }
    
    return model, metadata

def main():
    parser = argparse.ArgumentParser(description='Train neural dynamics models')
    parser.add_argument('config', type=str, help='Path to config JSON file')
    parser.add_argument('--force', action='store_true', help='Force retrain even if cached')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    print(f"\nTraining {config['model_type']} model...")
    print(f"Cache path: {config['cache_path']}")
    print("-" * 50)
    
    # Check cache
    if not args.force:
        if config['model_type'] == 'mp_rslds':
            # MP-rSLDS doesn't use torch models
            import os
            if os.path.exists(os.path.join(config['cache_path'], 'metadata.json')):
                print(f"Found cached {config['model_type']} model")
                with open(os.path.join(config['cache_path'], 'metadata.json'), 'r') as f:
                    metadata = json.load(f)
                print(f"Cached performance: MSE={metadata.get('mse', 'N/A'):.4f}, R²={metadata.get('r2', 'N/A'):.3f}")
                return
        else:
            model, metadata, loaded = load_cached_model(config['cache_path'])
            if loaded:
                print(f"Found cached {config['model_type']} model")
                if metadata and 'final_mse' in metadata:
                    print(f"Cached performance: MSE={metadata['final_mse']:.4f}, R²={metadata['final_r2']:.3f}")
                return
    
    # Train model
    if config['model_type'] == 'motor_rnn':
        model = create_model(config)
        model, metadata = train_motor_rnn(model, config)
        save_model(model, config['cache_path'], metadata)
        
        print("\n" + "="*50)
        print(f"Training complete!")
        print(f"Final MSE: {metadata['final_mse']:.4f}")
        print(f"Final R²: {metadata['final_r2']:.3f}")
        print("="*50)
        
    elif config['model_type'] == 'mp_rslds':
        result, metadata = fit_mp_rslds(config)
        # Save MP-rSLDS differently (pickle)
        import pickle
        import os
        os.makedirs(config['cache_path'], exist_ok=True)
        
        # Unpack the model and posterior
        model, posterior = result
        
        # Save both model and posterior
        with open(os.path.join(config['cache_path'], 'model.pkl'), 'wb') as f:
            pickle.dump(model, f)
        with open(os.path.join(config['cache_path'], 'posterior.pkl'), 'wb') as f:
            pickle.dump(posterior, f)
        
        import json
        with open(os.path.join(config['cache_path'], 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\n" + "="*50)
        print(f"Training complete!")
        print(f"Final MSE: {metadata['mse']:.4f}")
        print(f"Final R²: {metadata['r2']:.3f}")
        print("="*50)
            
    elif config['model_type'] == 'mr_gnode':
        model = create_model(config)
        model, metadata = train_mr_gnode_dynamic_comm(model, config)
        save_model(model, config['cache_path'], metadata)
        
        print("\n" + "="*50)
        print(f"Training complete!")
        print(f"Final MSE: {metadata['final_mse']:.4f}")
        print(f"Final R²: {metadata['final_r2']:.3f}")
        print("="*50)
    
    print(f"\nModel saved to {config['cache_path']}")
    
    # Plot training curves if available
    if 'losses' in metadata:
        plot_training_curves({'Total Loss': metadata['losses']})
    
    # Plot MSE and R2 history if available
    if 'train_mse' in metadata:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        ax1.plot(metadata['train_mse'])
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE')
        ax1.set_title('Training MSE')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(metadata['train_r2'])
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('R²')
        ax2.set_title('Training R²')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{config['cache_path']}/metrics.png")
        plt.show()

if __name__ == '__main__':
    main()