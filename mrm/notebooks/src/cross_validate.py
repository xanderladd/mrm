"""Cross-validation script for bidirectional neural dynamics models"""
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
    """Create deterministic CV splits"""
    train_seeds = []
    test_seeds = []
    
    for fold in range(n_folds):
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
    
    # Create model
    model = create_model(model_config)
    print(f"Created {model_config['model_type']} model")
    
    # Extract training data
    print("  Extracting training trajectories...")
    train_trajectories = extract_trajectories(motor_rnn, 
                                            n_trials=train_trials,
                                            noise_scale=noise_scale,
                                            seed=train_seed)
    
    train_data = {
        'region_1': train_trajectories['evidence_states'],
        'region_2': train_trajectories['motor_states']
    }
    
    print(f"  Train shapes: Region 1 {train_data['region_1'].shape}, Region 2 {train_data['region_2'].shape}")
    
    # Train model
    print(f"  Training {model_config['model_type']} model...")
    train_metadata = model.fit(train_data, **model_config['training_params'])
    
    # Extract test data
    print("  Extracting test trajectories...")
    test_trajectories = extract_trajectories(motor_rnn,
                                           n_trials=test_trials,
                                           noise_scale=noise_scale,
                                           seed=test_seed)
    
    test_data = {
        'region_1': test_trajectories['evidence_states'],
        'region_2': test_trajectories['motor_states']
    }
    
    print(f"  Test shapes: Region 1 {test_data['region_1'].shape}, Region 2 {test_data['region_2'].shape}")
    
    # Make bidirectional predictions
    print("  Making bidirectional predictions...")
    test_regions = [test_data['region_1'], test_data['region_2']]
    predictions = model.predict(test_regions, region_ids=['region_1', 'region_2'])
    
    # predictions = [pred_region_1, pred_region_2]
    pred_region_1, pred_region_2 = predictions
    
    # Compute metrics for both directions
    # Handle delay if present
    if hasattr(model, 'delay') and model.delay > 0:
        true_1 = test_data['region_1'][:, model.delay:, :]
        true_2 = test_data['region_2'][:, model.delay:, :]
        pred_1_eval = pred_region_1[:, model.delay:, :]
        pred_2_eval = pred_region_2[:, model.delay:, :]
    else:
        true_1 = test_data['region_1']
        true_2 = test_data['region_2']
        pred_1_eval = pred_region_1
        pred_2_eval = pred_region_2
    
    # Calculate metrics for both directions
    mse_21 = compute_mse(true_1.flatten(), pred_1_eval.flatten())  # region_2 → region_1
    mse_12 = compute_mse(true_2.flatten(), pred_2_eval.flatten())  # region_1 → region_2
    r2_21 = compute_r2(true_1.flatten(), pred_1_eval.flatten())
    r2_12 = compute_r2(true_2.flatten(), pred_2_eval.flatten())
    
    # Average metrics
    test_mse = (mse_12 + mse_21) / 2
    test_r2 = (r2_12 + r2_21) / 2
    
    print(f"  Test Region 1→2: MSE={mse_12:.4f}, R²={r2_12:.3f}")
    print(f"  Test Region 2→1: MSE={mse_21:.4f}, R²={r2_21:.3f}")
    print(f"  Test Average: MSE={test_mse:.4f}, R²={test_r2:.3f}")
    
    fold_metadata = {
        'fold_idx': fold_idx,
        'train_seed': train_seed,
        'test_seed': test_seed,
        'test_mse': float(test_mse),
        'test_r2': float(test_r2),
        'test_mse_12': float(mse_12),
        'test_mse_21': float(mse_21),
        'test_r2_12': float(r2_12),
        'test_r2_21': float(r2_21),
        'train_metadata': train_metadata,
        'model_type': model_config['model_type']
    }
    
    return {
        'model': model,
        'predictions': predictions,
        'ground_truth': [test_data['region_1'], test_data['region_2']],
        'metadata': fold_metadata
    }

def main():
    parser = argparse.ArgumentParser(description='Cross-validate bidirectional neural dynamics models')
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