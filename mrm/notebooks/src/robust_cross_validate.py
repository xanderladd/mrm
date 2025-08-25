"""Denoising/masking cross-validation for neural dynamics models"""
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

def corrupt_data(data, corruption_type, level, seed=None):
    """Apply corruption to data"""
    if seed: np.random.seed(seed)
    
    if corruption_type == 'noise':
        return data + np.random.randn(*data.shape) * level, None
    elif corruption_type == 'mask':
        mask = np.random.random((1, 1, data.shape[-1])) > level
        return data * mask, mask
    return data, None

def run_fold(config, motor_rnn, fold, train_seed, test_seed, n_train, n_test, corrupt_type, corrupt_level):
    """Run single CV fold with corruption"""
    
    # Extract clean data
    train_traj = extract_trajectories(motor_rnn, n_train, 0.0, train_seed)  # No noise in extraction
    test_traj = extract_trajectories(motor_rnn, n_test, 0.0, test_seed)
    
    train_clean = {'region_1': train_traj['evidence_states'], 
                   'region_2': train_traj['motor_states']}
    test_clean = {'region_1': test_traj['evidence_states'],
                  'region_2': test_traj['motor_states']}
    
    # Apply corruption
    if corrupt_type == 'none':
        train_input = train_clean
        test_input = test_clean
        train_masks = None
        test_masks = None
    else:
        r1_corrupt, m1 = corrupt_data(train_clean['region_1'], corrupt_type, corrupt_level, train_seed+1000)
        r2_corrupt, m2 = corrupt_data(train_clean['region_2'], corrupt_type, corrupt_level, train_seed+1001)
        train_input = {'region_1': r1_corrupt, 'region_2': r2_corrupt}
        train_masks = {'region_1': m1, 'region_2': m2} if corrupt_type == 'mask' else None
        
        r1_test, m1_test = corrupt_data(test_clean['region_1'], corrupt_type, corrupt_level, test_seed+1000)
        r2_test, m2_test = corrupt_data(test_clean['region_2'], corrupt_type, corrupt_level, test_seed+1001)
        test_input = {'region_1': r1_test, 'region_2': r2_test}
        test_masks = {'region_1': m1_test, 'region_2': m2_test} if corrupt_type == 'mask' else None
    
    # Train model
    model = create_model(config)
    targets = train_clean if corrupt_type != 'none' else {}
    train_metadata = model.fit(train_input, targets=targets, **config['training_params'])
    
    # Predict
    preds = model.predict([test_input['region_1'], test_input['region_2']], 
                         ['region_1', 'region_2'])
    
    # Compute metrics
    metrics = {}
    for i, region in enumerate(['region_1', 'region_2']):
        true = test_clean[region]
        pred = preds[i]
        
        # Handle delay if present
        if hasattr(model, 'delay') and model.delay > 0:
            true = true[:, model.delay:, :]
            pred = pred[:, model.delay:, :]
        
        mse = compute_mse(true.flatten(), pred.flatten())
        r2 = compute_r2(true.flatten(), pred.flatten())
        metrics[f'{region}_mse'] = float(mse)
        metrics[f'{region}_r2'] = float(r2)
        
        # Masked metrics
        if test_masks and test_masks[region] is not None:
            mask_flat = np.broadcast_to(test_masks[region], true.shape).flatten()
            true_flat = true.flatten()
            pred_flat = pred.flatten()
            
            masked_idx = mask_flat == 0
            if masked_idx.sum() > 0:
                metrics[f'{region}_masked_mse'] = float(compute_mse(
                    true_flat[masked_idx], pred_flat[masked_idx]))
                metrics[f'{region}_masked_r2'] = float(compute_r2(
                    true_flat[masked_idx], pred_flat[masked_idx]))
    
    metrics['avg_mse'] = (metrics['region_1_mse'] + metrics['region_2_mse']) / 2
    metrics['avg_r2'] = (metrics['region_1_r2'] + metrics['region_2_r2']) / 2
    
    if 'region_1_masked_mse' in metrics:
        metrics['avg_masked_mse'] = (metrics['region_1_masked_mse'] + 
                                     metrics['region_2_masked_mse']) / 2
        metrics['avg_masked_r2'] = (metrics['region_1_masked_r2'] + 
                                    metrics['region_2_masked_r2']) / 2
    
    # Build metadata with all necessary fields
    fold_metadata = {
        'fold_idx': fold,
        'train_seed': train_seed,
        'test_seed': test_seed,
        'corruption_type': corrupt_type,
        'corruption_level': float(corrupt_level),
        'test_mse': metrics['avg_mse'],
        'test_r2': metrics['avg_r2'],
        'test_mse_12': metrics.get('region_2_mse', 0),  # region_1 → region_2
        'test_mse_21': metrics.get('region_1_mse', 0),  # region_2 → region_1
        'test_r2_12': metrics.get('region_2_r2', 0),
        'test_r2_21': metrics.get('region_1_r2', 0),
        'train_metadata': train_metadata,
        'model_type': config['model_type']
    }
    
    # Add masked metrics if present
    if 'avg_masked_mse' in metrics:
        fold_metadata['masked_mse'] = metrics['avg_masked_mse']
        fold_metadata['masked_r2'] = metrics['avg_masked_r2']
        fold_metadata['masked_mse_r1'] = metrics.get('region_1_masked_mse', 0)
        fold_metadata['masked_mse_r2'] = metrics.get('region_2_masked_mse', 0)
        fold_metadata['masked_r2_r1'] = metrics.get('region_1_masked_r2', 0)
        fold_metadata['masked_r2_r2'] = metrics.get('region_2_masked_r2', 0)
    
    # Return in same format as original CV
    return {
        'model': model,
        'predictions': preds,
        'ground_truth': [test_clean['region_1'], test_clean['region_2']],
        'corrupted_input': [test_input['region_1'], test_input['region_2']],
        'masks': test_masks,
        'metadata': fold_metadata
    }

def run_experiment(config, motor_rnn, args, corrupt_type, corrupt_level):
    """Run CV experiment with corruption and save results"""
    
    # Create cache path with corruption info
    corruption_str = f"{corrupt_type}_{corrupt_level:.2f}".replace('.', 'p')
    cv_cache_path = config['cache_path'] + f'_cv_denoising_{corruption_str}'
    cv_results_path = os.path.join(cv_cache_path, 'cv_results.json')
    
    print(f"\n{corrupt_type.upper()} (level={corrupt_level})")
    print(f"Cache path: {cv_cache_path}")
    
    # Check if results exist
    if not args.force and os.path.exists(cv_results_path):
        print("  Results already exist, loading...")
        with open(cv_results_path, 'r') as f:
            return json.load(f)
    
    # Create output directory
    os.makedirs(cv_cache_path, exist_ok=True)
    
    fold_results = []
    for fold in range(args.n_folds):
        train_seed = args.base_seed + fold * 1000
        test_seed = train_seed + 500
        
        # Run fold
        fold_result = run_fold(config, motor_rnn, fold, train_seed, test_seed,
                              args.train_trials, args.test_trials, 
                              corrupt_type, corrupt_level)
        
        # Save fold results
        fold_cache_path = os.path.join(cv_cache_path, f'cv_fold_{fold}')
        os.makedirs(fold_cache_path, exist_ok=True)
        
        # Save model
        save_model(fold_result['model'], fold_cache_path, fold_result['metadata'])
        
        # Save predictions and ground truth
        with open(os.path.join(fold_cache_path, 'predictions.pkl'), 'wb') as f:
            pickle.dump(fold_result['predictions'], f)
        
        with open(os.path.join(fold_cache_path, 'ground_truth.pkl'), 'wb') as f:
            pickle.dump(fold_result['ground_truth'], f)
        
        # Save corrupted input
        with open(os.path.join(fold_cache_path, 'corrupted_input.pkl'), 'wb') as f:
            pickle.dump(fold_result['corrupted_input'], f)
        
        # Save masks if present
        if fold_result['masks'] is not None:
            with open(os.path.join(fold_cache_path, 'masks.pkl'), 'wb') as f:
                pickle.dump(fold_result['masks'], f)
        
        fold_results.append(fold_result['metadata'])
        
        # Print fold results
        metrics = fold_result['metadata']
        print(f"  Fold {fold}: MSE={metrics['test_mse']:.4f}, R²={metrics['test_r2']:.3f}", end='')
        if 'masked_mse' in metrics:
            print(f", Masked MSE={metrics['masked_mse']:.4f}", end='')
        print()
    
    # Aggregate results
    test_mses = [r['test_mse'] for r in fold_results]
    test_r2s = [r['test_r2'] for r in fold_results]
    
    cv_results = {
        'model_type': config['model_type'],
        'corruption_type': corrupt_type,
        'corruption_level': float(corrupt_level),
        'n_folds': args.n_folds,
        'base_seed': args.base_seed,
        'train_trials_per_fold': args.train_trials,
        'test_trials_per_fold': args.test_trials,
        'mean_test_mse': float(np.mean(test_mses)),
        'std_test_mse': float(np.std(test_mses)),
        'mean_test_r2': float(np.mean(test_r2s)),
        'std_test_r2': float(np.std(test_r2s)),
        'fold_results': fold_results,
        'config': config
    }
    
    # Add masked metrics aggregation if present
    if any('masked_mse' in r for r in fold_results):
        masked_mses = [r['masked_mse'] for r in fold_results if 'masked_mse' in r]
        masked_r2s = [r['masked_r2'] for r in fold_results if 'masked_r2' in r]
        cv_results['mean_masked_mse'] = float(np.mean(masked_mses))
        cv_results['std_masked_mse'] = float(np.std(masked_mses))
        cv_results['mean_masked_r2'] = float(np.mean(masked_r2s))
        cv_results['std_masked_r2'] = float(np.std(masked_r2s))
    
    # Save aggregated results
    with open(cv_results_path, 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    print(f"  Results saved to: {cv_cache_path}")
    
    return cv_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config JSON')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--base-seed', type=int, default=42)
    parser.add_argument('--train-trials', type=int, default=80)
    parser.add_argument('--test-trials', type=int, default=20)
    parser.add_argument('--force', action='store_true', help='Force re-run even if results exist')
    args = parser.parse_args()
    
    config = load_config(args.config)
    model_type = config['model_type']
    
    print(f"DENOISING CV: {model_type}")
    print("="*50)
    
    # Load Motor RNN
    from utils import load_cached_model
    motor_config = load_config('configs/motor_rnn.json')
    motor_rnn, _, _ = load_cached_model(motor_config['cache_path'],
                                        create_model(motor_config).__class__,
                                        motor_config)
    
    # Get corruption params from config
    noise_level = config['data_params'].get('noise_scale', 0.2)
    mask_ratio = config['data_params'].get('mask_ratio', 0.2)
    
    results = []
    
    # Baseline
    results.append(run_experiment(config, motor_rnn, args, 'none', 0.0))
    
    # Noise
    results.append(run_experiment(config, motor_rnn, args, 'noise', noise_level))
    
    # Masking
    results.append(run_experiment(config, motor_rnn, args, 'mask', mask_ratio))
    
    # Summary
    print("\nSUMMARY:")
    for exp in results:
        print(f"{exp['corruption_type']:6s} {exp['corruption_level']:.1f}: "
              f"MSE={exp['mean_test_mse']:.4f}±{exp['std_test_mse']:.4f}, "
              f"R²={exp['mean_test_r2']:.3f}±{exp['std_test_r2']:.3f}", end='')
        if 'mean_masked_mse' in exp:
            print(f", Masked={exp['mean_masked_mse']:.4f}±{exp['std_masked_mse']:.4f}", end='')
        print()

if __name__ == '__main__':
    main()