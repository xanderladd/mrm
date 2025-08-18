"""Cross-validation script for neural dynamics models"""
import argparse
import json
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm

from utils import (load_config, create_model, load_cached_model, save_model, 
                   extract_trajectories, compute_mse, compute_r2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_cv_splits(n_folds=5, base_seed=42):
    """Create deterministic CV splits using different seeds"""
    train_seeds = []
    test_seeds = []
    
    for fold in range(n_folds):
        # Use different seeds for train and test to ensure independence
        train_seed = base_seed + fold * 1000
        test_seed = base_seed + fold * 1000 + 500
        train_seeds.append(train_seed)
        test_seeds.append(test_seed)
    
    return train_seeds, test_seeds

def run_single_fold(model_config, motor_rnn, fold_idx, train_seed, test_seed, 
                   train_trials, test_trials, noise_scale):
    """Run training and evaluation for a single CV fold"""
    
    print(f"\n--- Fold {fold_idx} ---")
    print(f"Train seed: {train_seed}, Test seed: {test_seed}")
    
    # Create fresh model instance
    model = create_model(model_config)
    print(f"Created {model_config['model_type']} model")
    
    # Extract training data with train_seed
    print("  Extracting training trajectories...")
    train_trajectories = extract_trajectories(motor_rnn, 
                                            n_trials=train_trials,
                                            noise_scale=noise_scale,
                                            seed=train_seed)
    
    # Fix data shapes to ensure consistency
    evidence_states = train_trajectories['evidence_states']
    motor_states = train_trajectories['motor_states']
    
    print(f"  Raw shapes: evidence {evidence_states.shape}, motor {motor_states.shape}")
    
    train_data = {
        'region_1': evidence_states,  # evidence -> region_1
        'region_2': motor_states      # motor -> region_2
    }
    
    print(f"  Fixed shapes: Region 1 {train_data['region_1'].shape}, Region 2 {train_data['region_2'].shape}")
    print(f"  Extracted {len(train_trajectories['trial_info'])} training trajectories")
    
    # Set region info for unidirectional models
    region_info = None
    if model_config['model_type'] in ['cca', 'rrr']:
        region_info = {'source': 'region_1', 'target': 'region_2'}
    
    # Train model
    print(f"  Training {model_config['model_type']} model...")
    if region_info is not None:
        train_metadata = model.fit(train_data, region_info=region_info, **model_config['training_params'])
    else:
        train_metadata = model.fit(train_data, **model_config['training_params'])
    
    # Extract test data with test_seed
    print("  Extracting test trajectories...")
    test_trajectories = extract_trajectories(motor_rnn,
                                           n_trials=test_trials,
                                           noise_scale=noise_scale,
                                           seed=test_seed)
    
    # Fix test data shapes the same way
    evidence_states_test = test_trajectories['evidence_states']
    motor_states_test = test_trajectories['motor_states']
    
    print(f"  Raw test shapes: evidence {evidence_states_test.shape}, motor {motor_states_test.shape}")
    
    # Ensure evidence_states_test is 3D: [trials, time, features]
    if evidence_states_test.ndim == 2:
        evidence_states_test = evidence_states_test[:, :, np.newaxis]
    elif evidence_states_test.ndim == 1:
        n_trials = test_trials
        seq_len = len(evidence_states_test) // n_trials
        evidence_states_test = evidence_states_test.reshape(n_trials, seq_len, 1)
    
    # Ensure motor_states_test is 3D: [trials, time, features]
    if motor_states_test.ndim == 2:
        motor_states_test = motor_states_test[:, np.newaxis, :]
    elif motor_states_test.ndim == 1:
        n_trials = test_trials
        seq_len = len(motor_states_test) // n_trials
        motor_states_test = motor_states_test.reshape(n_trials, seq_len, 1)
    
    test_data = {
        'region_1': evidence_states_test,
        'region_2': motor_states_test
    }
    
    print(f"  Fixed test shapes: Region 1 {test_data['region_1'].shape}, Region 2 {test_data['region_2'].shape}")
    print(f"  Extracted {len(test_trajectories['trial_info'])} test trajectories")
    
    # Make predictions on test data
    print("  Making predictions...")
    # Use standardized predict interface
    
    # All models now use bidirectional prediction
    all_regions = [test_data['region_1'], test_data['region_2']]
    region_ids = ['region_1', 'region_2']
    predictions_list = model.predict(all_regions, region_ids=region_ids)

    # Extract predictions for both regions
    region_1_pred, region_2_pred = predictions_list
        
    # Convert to numpy if needed
    if hasattr(predictions, 'numpy'):
        predictions = predictions.numpy()
    
    ground_truth = test_data['region_2']
    import pdb; pdb.set_trace()
    # Compute metrics
    predictions_flat = predictions.flatten()
    ground_truth_flat = ground_truth.flatten()
    
    test_mse = compute_mse(predictions_flat, ground_truth_flat)
    test_r2 = compute_r2(predictions_flat, ground_truth_flat)
    
    print(f"  Test MSE: {test_mse:.4f}, Test R²: {test_r2:.3f}")
    
    # Create fold metadata
    fold_metadata = {
        'fold_idx': fold_idx,
        'train_seed': train_seed,
        'test_seed': test_seed,
        'test_mse': float(test_mse),
        'test_r2': float(test_r2),
        'train_metadata': train_metadata,
        'model_type': model_config['model_type'],
        'data_shapes': {
            'train_region_1': train_data['region_1'].shape,
            'train_region_2': train_data['region_2'].shape,
            'test_region_1': test_data['region_1'].shape,
            'test_region_2': test_data['region_2'].shape,
            'predictions': predictions.shape
        }
    }
    
    return {
        'model': model,
        'predictions': predictions,
        'ground_truth': ground_truth,
        'metadata': fold_metadata
    }

def main():
    parser = argparse.ArgumentParser(description='Cross-validate neural dynamics models')
    parser.add_argument('config', type=str, help='Path to config JSON file')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--base-seed', type=int, default=42, help='Base seed for reproducible splits')
    parser.add_argument('--train-trials', type=int, default=80, help='Number of training trials per fold')
    parser.add_argument('--test-trials', type=int, default=20, help='Number of test trials per fold')
    parser.add_argument('--force', action='store_true', help='Force re-run even if results exist')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create CV cache path
    cv_cache_path = config['cache_path'] + '_cv'
    cv_results_path = os.path.join(cv_cache_path, 'cv_results.json')
    
    print(f"\nCross-validating {config['model_type']} model...")
    print(f"CV cache path: {cv_cache_path}")
    print(f"Folds: {args.n_folds}, Train trials: {args.train_trials}, Test trials: {args.test_trials}")
    print("-" * 70)
    
    # Check if CV results already exist
    if not args.force and os.path.exists(cv_results_path):
        print("Cross-validation results already exist. Use --force to re-run.")
        with open(cv_results_path, 'r') as f:
            results = json.load(f)
        print(f"Existing results: MSE = {results['mean_test_mse']:.4f} ± {results['std_test_mse']:.4f}")
        print(f"                 R² = {results['mean_test_r2']:.3f} ± {results['std_test_r2']:.3f}")
        return
    
    # Load motor_rnn model
    print("Loading Motor RNN for trajectory extraction...")
    motor_rnn_config = load_config('configs/motor_rnn.json')
    motor_rnn, _, loaded = load_cached_model(motor_rnn_config['cache_path'], 
                                            create_model(motor_rnn_config).__class__, 
                                            motor_rnn_config)
    
    if not loaded:
        raise ValueError("Motor RNN must be trained first! Run: python train.py configs/motor_rnn.json")
    
    # Create CV splits
    train_seeds, test_seeds = create_cv_splits(args.n_folds, args.base_seed)
    
    # Run cross-validation
    os.makedirs(cv_cache_path, exist_ok=True)
    fold_results = []
    
    noise_scale = config['data_params'].get('noise_scale', 0.01)
    
    for fold_idx in range(args.n_folds):
        fold_result = run_single_fold(
            config, motor_rnn, fold_idx, 
            train_seeds[fold_idx], test_seeds[fold_idx],
            args.train_trials, args.test_trials, noise_scale
        )
        
        # Save fold results
        fold_cache_path = os.path.join(cv_cache_path, f'cv_fold_{fold_idx}')
        os.makedirs(fold_cache_path, exist_ok=True)
        
        # Save model
        save_model(fold_result['model'], fold_cache_path, fold_result['metadata'])
        
        # Save predictions and ground truth
        with open(os.path.join(fold_cache_path, 'predictions.pkl'), 'wb') as f:
            pickle.dump(fold_result['predictions'], f)
        
        with open(os.path.join(fold_cache_path, 'ground_truth.pkl'), 'wb') as f:
            pickle.dump(fold_result['ground_truth'], f)
        
        fold_results.append(fold_result['metadata'])
    
    # Aggregate results
    test_mses = [r['test_mse'] for r in fold_results]
    test_r2s = [r['test_r2'] for r in fold_results]
    
    cv_results = {
        'model_type': config['model_type'],
        'n_folds': args.n_folds,
        'base_seed': args.base_seed,
        'train_trials_per_fold': args.train_trials,
        'test_trials_per_fold': args.test_trials,
        'noise_scale': noise_scale,
        'mean_test_mse': float(np.mean(test_mses)),
        'std_test_mse': float(np.std(test_mses)),
        'mean_test_r2': float(np.mean(test_r2s)),
        'std_test_r2': float(np.std(test_r2s)),
        'fold_results': fold_results,
        'config': config
    }
    
    # Save aggregated results
    with open(cv_results_path, 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("CROSS-VALIDATION COMPLETE")
    print("="*70)
    print(f"Model: {config['model_type']}")
    print(f"Test MSE: {cv_results['mean_test_mse']:.4f} ± {cv_results['std_test_mse']:.4f}")
    print(f"Test R²:  {cv_results['mean_test_r2']:.3f} ± {cv_results['std_test_r2']:.3f}")
    print(f"\nIndividual fold results:")
    for i, result in enumerate(fold_results):
        print(f"  Fold {i}: MSE={result['test_mse']:.4f}, R²={result['test_r2']:.3f}")
    print(f"\nResults saved to: {cv_cache_path}")
    print("="*70)

if __name__ == '__main__':
    main()