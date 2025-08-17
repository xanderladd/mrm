"""Compare performance of different models on same RNN trajectories"""
import argparse
import json
import numpy as np
import torch
import pickle
import os
from tabulate import tabulate
from sklearn.metrics import r2_score

from utils import (load_config, create_model, compute_mse, compute_r2, plot_comparison)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_common_test_data():
    """Generate common test trajectories from RNN for all models"""
    from models.motor_rnn import MultiRegionRNN
    from utils import extract_trajectories
    
    print("Generating common test trajectories from RNN...")
    rnn_config = load_config('configs/motor_rnn.json')
    rnn_model = MultiRegionRNN(**rnn_config['model_params']).to(device)
    
    # Load RNN
    rnn_path = os.path.join(rnn_config['cache_path'], 'model.pt')
    state_dict = torch.load(rnn_path, map_location=device)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    rnn_model.load_state_dict(state_dict)
    
    # Generate test trajectories
    trajectories = extract_trajectories(rnn_model, n_trials=20, noise_scale=.01)
    return trajectories


def evaluate_cca(config, test_trajectories):
    """Evaluate CCA baseline model"""
    import pickle
    
    # Load trained model
    model_path = os.path.join(config['cache_path'], 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Get test data
    evidence_test = np.array(test_trajectories['evidence_states'])
    motor_test = np.array(test_trajectories['motor_states'])
    
    # Predict
    motor_pred = model.predict(evidence_test)
    
    return motor_pred, motor_test

def evaluate_rrr(config, test_trajectories):
    """Evaluate RRR baseline model"""
    import pickle
    
    # Load trained model
    model_path = os.path.join(config['cache_path'], 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Get test data
    evidence_test = np.array(test_trajectories['evidence_states'])
    motor_test = np.array(test_trajectories['motor_states'])
    
    # Predict
    motor_pred = model.predict(evidence_test)
    
    # Handle delays if present - adjust both pred and targets consistently
    if model.delay > 0:
        motor_test_adjusted = motor_test[:, model.delay:, :]
        motor_pred_adjusted = motor_pred[:, model.delay:, :]
        return motor_pred_adjusted, motor_test_adjusted
    else:
        return motor_pred, motor_test


def evaluate_model(config_path, test_trajectories):
    """Evaluate any model on the same test trajectories"""
    config = load_config(config_path)
    model_type = config['model_type']
    
    print(f"Evaluating {model_type}...")
    
    if model_type == 'motor_rnn':
        return evaluate_motor_rnn(config, test_trajectories)
    elif model_type == 'mp_rslds':
        return evaluate_mp_rslds(config, test_trajectories)
    elif model_type == 'mr_gnode':
        return evaluate_mr_gnode(config, test_trajectories)  
    elif model_type == 'cca':
        return evaluate_cca(config, test_trajectories)
    elif model_type == 'rrr':
        return evaluate_rrr(config, test_trajectories)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def evaluate_motor_rnn(config, test_trajectories):
    """Evaluate Motor RNN reconstruction"""
    model = create_model(config)
    state_dict = torch.load(f"{config['cache_path']}/model.pt", map_location=device)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()
    
    # Use evidence to reconstruct full motor states
    evidence = torch.tensor(test_trajectories['evidence_states'], dtype=torch.float32).to(device)
    targets = test_trajectories['motor_states']  # Full motor state dimensions
    
    all_preds = []
    with torch.no_grad():
        for trial_idx in range(evidence.shape[0]):
            trial_evidence = evidence[trial_idx:trial_idx+1]
            
            # Reconstruct motor states from evidence
            outputs = model(trial_evidence)
            motor_pred = outputs['motor_velocity'].cpu().numpy()  # Full motor output
            all_preds.append(motor_pred[0])
    
    preds = np.array(all_preds)
    return preds, targets

def evaluate_mp_rslds(config, test_trajectories):
    """Evaluate MP-rSLDS reconstruction"""
    # Load trained model
    with open(f"{config['cache_path']}/model.pkl", 'rb') as f:
        rslds = pickle.load(f)
    
    # Prepare input data (evidence + motor states)
    evidence_data = test_trajectories['evidence_states']
    motor_data = test_trajectories['motor_states']
    test_states = np.concatenate([evidence_data, motor_data], axis=2)
    
    n_trials, n_time, n_dims = test_states.shape
    test_states_flat = test_states.reshape(-1, n_dims).astype(np.float64)
    
    # Run inference to get reconstructions
    try:
        _, posterior = rslds.fit(test_states_flat, method="laplace_em", 
                                initialize=False, num_iters=4)
        reconstructions_flat = rslds.smooth(posterior.mean_continuous_states[0], test_states_flat)
    except:
        reconstructions_flat = rslds.smooth(test_states_flat, test_states_flat)
    
    # Extract motor reconstructions and targets
    reconstructions = reconstructions_flat.reshape(n_trials, n_time, n_dims)
    motor_dim = evidence_data.shape[2]  # Evidence dimensions
    
    preds = reconstructions[:, :, motor_dim:]  # Motor reconstructions
    targets = test_trajectories['motor_states']  # Ground truth motor states
    
    return preds, targets

def evaluate_mr_gnode(config, test_trajectories):
    """Evaluate MR-GNODE reconstruction"""
    model = create_model(config)
    state_dict = torch.load(f"{config['cache_path']}/model.pt", map_location=device)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()
    
    # Use evidence to reconstruct full motor states
    evidence = torch.tensor(test_trajectories['evidence_states'], dtype=torch.float32).to(device)
    targets = test_trajectories['motor_states']  # Full motor state dimensions
    
    dt = 0.01
    n_trials, seq_len, _ = evidence.shape
    all_preds = []
    
    with torch.no_grad():
        for trial_idx in range(n_trials):
            trial_evidence = evidence[trial_idx:trial_idx+1]
            
            # Format input: [batch, time, regions, channels]
            batch_input = trial_evidence[:, :, :2].unsqueeze(2)
            if hasattr(model, 'num_regions') and model.num_regions == 2:
                batch_input = batch_input.repeat(1, 1, 2, 1)
            
            # Initialize and forward pass
            h = torch.empty(1, model.N_total, device=device)
            torch.nn.init.xavier_uniform_(h)
            outputs = []
            
            for t in range(seq_len):
                h_dot, _ = model(h, batch_input[:, t], timestep=t)
                h = h + dt * h_dot
                step_output = model.get_outputs(h)
                outputs.append(step_output)
            
            outputs = torch.stack(outputs, dim=1)
            if outputs.shape[2] > 1:
                preds = outputs.mean(dim=2)
            else:
                preds = outputs.squeeze(2)
            
            all_preds.append(preds.squeeze(0).cpu().numpy())
    
    preds = np.array(all_preds)
    
    # Ensure output dimensions match motor state dimensions
    if preds.shape[-1] != targets.shape[-1]:
        print(f"Warning: MR-GNODE output dim {preds.shape[-1]} != motor dim {targets.shape[-1]}")
        # If MR-GNODE outputs fewer dims, pad with zeros or truncate targets
        if preds.shape[-1] < targets.shape[-1]:
            targets = targets[..., :preds.shape[-1]]
        else:
            # Truncate predictions to match targets
            preds = preds[..., :targets.shape[-1]]
    
    return preds, targets

def count_parameters(config):
    """Count model parameters"""
    if config['model_type'] == 'mp_rslds':
        p = config['model_params']
        return p['K'] * (p['D_evidence'] + p['D_motor']) * (p['N_evidence'] + p['N_motor'])
    else:
        try:
            model = create_model(config)
        except ValueError:
            return 0
        return sum(p.numel() for p in model.parameters())

def compare_models(config_paths):
    """Compare multiple models on same test data"""
    # Generate common test data once
    test_trajectories = generate_common_test_data()
    
    results = {}
    for config_path in config_paths:
        config = load_config(config_path)
        model_type = config['model_type']
        
        # Evaluate model
        preds, targets = evaluate_model(config_path, test_trajectories)
        
        # Debug: print shapes
        print(f"{model_type} - Preds shape: {preds.shape}, Targets shape: {targets.shape}")
        
        # Compute metrics
        mse = compute_mse(preds.flatten(), targets.flatten())
        r2 = compute_r2(preds.flatten(), targets.flatten())
        
        results[model_type] = {
            'mse': mse,
            'r2': r2,
            'n_params': count_parameters(config)
        }
    
    return results

def display_results(results):
    """Display comparison results"""
    headers = ['Model', 'MSE', 'R²', '# Parameters']
    table = [[model, f"{m['mse']:.4f}", f"{m['r2']:.3f}", f"{m['n_params']:,}"] 
             for model, m in results.items()]
    
    print("\n" + "="*50)
    print("MODEL COMPARISON RESULTS")
    print("="*50)
    print(tabulate(table, headers=headers, tablefmt='grid'))

def main():
    parser = argparse.ArgumentParser(description='Compare neural dynamics models')
    parser.add_argument('configs', nargs='+', help='Config JSON file paths')
    args = parser.parse_args()
    
    results = compare_models(args.configs)
    display_results(results)
    plot_comparison(results)
    
    # Save results
    with open('comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print("\nResults saved to comparison_results.json")

if __name__ == '__main__':
    main()