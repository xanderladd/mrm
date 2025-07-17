# evaluate_reconstruction.py
"""
Evaluate reconstruction accuracy for trained neural models
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Optional, Tuple
import warnings

# Import framework components
from mrm.trainer import load_trained_model
from mrm.dataset import IBLDataset


def compute_reconstruction_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive reconstruction metrics
    
    Args:
        original: Original neural data (n_trials, n_timepoints, n_neurons)
        reconstructed: Reconstructed neural data (same shape)
    
    Returns:
        Dictionary of reconstruction metrics
    """
    # Flatten for overall metrics
    orig_flat = original.flatten()
    recon_flat = reconstructed.flatten()
    
    metrics = {}
    
    # Overall metrics
    metrics['mse_overall'] = mean_squared_error(orig_flat, recon_flat)
    metrics['rmse_overall'] = np.sqrt(metrics['mse_overall'])
    
    # Correlation
    try:
        correlation, p_value = pearsonr(orig_flat, recon_flat)
        metrics['correlation_overall'] = correlation
        metrics['correlation_p_value'] = p_value
    except:
        metrics['correlation_overall'] = np.nan
        metrics['correlation_p_value'] = np.nan
    
    # R² score
    try:
        metrics['r2_overall'] = r2_score(orig_flat, recon_flat)
    except:
        metrics['r2_overall'] = np.nan
    
    # Explained variance
    total_var = np.var(orig_flat)
    error_var = np.var(orig_flat - recon_flat)
    metrics['explained_variance'] = 1 - (error_var / total_var) if total_var > 0 else 0
    
    # Mean absolute error
    metrics['mae_overall'] = np.mean(np.abs(orig_flat - recon_flat))
    
    # Signal-to-noise ratio (higher is better)
    signal_power = np.mean(orig_flat**2)
    noise_power = np.mean((orig_flat - recon_flat)**2)
    metrics['snr_db'] = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf
    
    return metrics


def compute_per_neuron_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute per-neuron reconstruction metrics"""
    n_trials, n_timepoints, n_neurons = original.shape
    
    per_neuron_metrics = {
        'mse_per_neuron': np.zeros(n_neurons),
        'correlation_per_neuron': np.zeros(n_neurons),
        'r2_per_neuron': np.zeros(n_neurons)
    }
    
    for neuron in range(n_neurons):
        orig_neuron = original[:, :, neuron].flatten()
        recon_neuron = reconstructed[:, :, neuron].flatten()
        
        # MSE
        per_neuron_metrics['mse_per_neuron'][neuron] = mean_squared_error(orig_neuron, recon_neuron)
        
        # Correlation
        try:
            corr, _ = pearsonr(orig_neuron, recon_neuron)
            per_neuron_metrics['correlation_per_neuron'][neuron] = corr
        except:
            per_neuron_metrics['correlation_per_neuron'][neuron] = np.nan
            
        # R²
        try:
            per_neuron_metrics['r2_per_neuron'][neuron] = r2_score(orig_neuron, recon_neuron)
        except:
            per_neuron_metrics['r2_per_neuron'][neuron] = np.nan
    
    return per_neuron_metrics


def compute_trial_averaged_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """Compute metrics on trial-averaged responses"""
    # Average across trials for each timepoint and neuron
    orig_avg = np.mean(original, axis=0)  # (n_timepoints, n_neurons)
    recon_avg = np.mean(reconstructed, axis=0)
    
    # Flatten for metrics
    orig_flat = orig_avg.flatten()
    recon_flat = recon_avg.flatten()
    
    metrics = {}
    metrics['mse_trial_avg'] = mean_squared_error(orig_flat, recon_flat)
    metrics['rmse_trial_avg'] = np.sqrt(metrics['mse_trial_avg'])
    
    try:
        correlation, _ = pearsonr(orig_flat, recon_flat)
        metrics['correlation_trial_avg'] = correlation
    except:
        metrics['correlation_trial_avg'] = np.nan
        
    try:
        metrics['r2_trial_avg'] = r2_score(orig_flat, recon_flat)
    except:
        metrics['r2_trial_avg'] = np.nan
    
    return metrics


def create_reconstruction_plots(original: np.ndarray, reconstructed: np.ndarray, 
                              per_neuron_metrics: Dict, output_dir: Path) -> Dict[str, plt.Figure]:
    """Create visualization plots for reconstruction evaluation"""
    figures = {}
    
    # 1. Scatter plot: Original vs Reconstructed (subsampled)
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    
    # Subsample for plotting (to avoid memory issues)
    max_points = 10000
    orig_flat = original.flatten()
    recon_flat = reconstructed.flatten()
    
    if len(orig_flat) > max_points:
        indices = np.random.choice(len(orig_flat), max_points, replace=False)
        orig_plot = orig_flat[indices]
        recon_plot = recon_flat[indices]
    else:
        orig_plot = orig_flat
        recon_plot = recon_flat
    
    ax1.scatter(orig_plot, recon_plot, alpha=0.5, s=1)
    
    # Add identity line
    min_val = min(orig_plot.min(), recon_plot.min())
    max_val = max(orig_plot.max(), recon_plot.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    
    ax1.set_xlabel('Original Activity')
    ax1.set_ylabel('Reconstructed Activity')
    ax1.set_title('Original vs Reconstructed Activity')
    ax1.grid(True, alpha=0.3)
    
    figures['scatter'] = fig1
    
    # 2. Per-neuron correlation distribution
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    
    # Correlation histogram
    valid_corrs = per_neuron_metrics['correlation_per_neuron'][
        ~np.isnan(per_neuron_metrics['correlation_per_neuron'])
    ]
    axes2[0].hist(valid_corrs, bins=30, alpha=0.7, edgecolor='black')
    axes2[0].set_xlabel('Correlation')
    axes2[0].set_ylabel('Number of Neurons')
    axes2[0].set_title('Per-Neuron Correlation Distribution')
    axes2[0].axvline(np.median(valid_corrs), color='red', linestyle='--', 
                     label=f'Median: {np.median(valid_corrs):.3f}')
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)
    
    # MSE histogram
    axes2[1].hist(per_neuron_metrics['mse_per_neuron'], bins=30, alpha=0.7, edgecolor='black')
    axes2[1].set_xlabel('MSE')
    axes2[1].set_ylabel('Number of Neurons')
    axes2[1].set_title('Per-Neuron MSE Distribution')
    axes2[1].axvline(np.median(per_neuron_metrics['mse_per_neuron']), color='red', linestyle='--',
                     label=f'Median: {np.median(per_neuron_metrics["mse_per_neuron"]):.6f}')
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)
    
    # R² histogram
    valid_r2 = per_neuron_metrics['r2_per_neuron'][
        ~np.isnan(per_neuron_metrics['r2_per_neuron'])
    ]
    axes2[2].hist(valid_r2, bins=30, alpha=0.7, edgecolor='black')
    axes2[2].set_xlabel('R²')
    axes2[2].set_ylabel('Number of Neurons')
    axes2[2].set_title('Per-Neuron R² Distribution')
    axes2[2].axvline(np.median(valid_r2), color='red', linestyle='--',
                     label=f'Median: {np.median(valid_r2):.3f}')
    axes2[2].legend()
    axes2[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    figures['per_neuron_distributions'] = fig2
    
    # 3. Example neuron traces
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))
    axes3 = axes3.flatten()
    
    # Show 4 example neurons
    n_neurons = original.shape[2]
    neuron_indices = np.linspace(0, n_neurons-1, 4, dtype=int)
    
    for i, neuron_idx in enumerate(neuron_indices):
        if i >= 4:
            break
            
        # Show first trial for this neuron
        trial_idx = 0
        orig_trace = original[trial_idx, :, neuron_idx]
        recon_trace = reconstructed[trial_idx, :, neuron_idx]
        
        time_points = np.arange(len(orig_trace))
        
        axes3[i].plot(time_points, orig_trace, 'b-', alpha=0.7, linewidth=2, label='Original')
        axes3[i].plot(time_points, recon_trace, 'r--', alpha=0.7, linewidth=2, label='Reconstructed')
        
        corr = per_neuron_metrics['correlation_per_neuron'][neuron_idx]
        axes3[i].set_title(f'Neuron {neuron_idx} (r={corr:.3f})')
        axes3[i].set_xlabel('Time Bins')
        axes3[i].set_ylabel('Activity')
        axes3[i].legend()
        axes3[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    figures['example_traces'] = fig3
    
    # Save plots
    for name, fig in figures.items():
        save_path = output_dir / f"reconstruction_{name}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    
    return figures


def evaluate_reconstruction(config_path: str, splits: List[str] = ['test'], 
                          create_plots: bool = True) -> Dict:
    """
    Main function to evaluate reconstruction accuracy
    
    Args:
        config_path: Path to JSON config file
        splits: Which data splits to evaluate on
        create_plots: Whether to create visualization plots
    
    Returns:
        Dictionary containing all evaluation results
    """
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract paths and settings
    experiment_name = config['experiment']['name']
    session_id = config['experiment']['session_id']
    model_type = config['model']['type']
    save_dir = config.get('save_dir', 'outputs')
    
    # Construct paths
    model_dir = Path(save_dir) / experiment_name / session_id / model_type
    output_dir = model_dir / "reconstruction_evaluation"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Evaluating reconstruction for model: {model_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load trained model
    print("Loading trained model...")
    model = load_trained_model(str(model_dir))
    
    # Load dataset
    print("Loading dataset...")
    dataset_config = config['dataset']
    if dataset_config['type'] == 'ibl':
        dataset = IBLDataset(**dataset_config['params'])
        dataset.prepare()
    else:
        raise ValueError(f"Unknown dataset type: {dataset_config['type']}")
    
    # Evaluate reconstruction for each split
    results = {}
    all_figures = {}
    
    for split in splits:
        print(f"\nEvaluating reconstruction on {split} split...")
        
        # Get neural data
        neural_data = dataset.get_neural_data(split)
        print(f"Neural data shape: {neural_data.shape}")
        
        # Get reconstructions
        print("Computing reconstructions...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Encode to latents
            latents = model.encode(neural_data)
            print(f"Latents shape: {latents.shape}")
            
            # Decode back to neural space
            reconstructed = model.decode(latents)
            print(f"Reconstructed shape: {reconstructed.shape}")
        
        # Ensure shapes match
        if reconstructed.shape != neural_data.shape:
            print(f"Warning: Shape mismatch. Original: {neural_data.shape}, Reconstructed: {reconstructed.shape}")
            # Try to handle common shape mismatches
            if len(reconstructed.shape) == 2 and len(neural_data.shape) == 3:
                # Reshape 2D back to 3D
                n_trials, n_timepoints, n_neurons = neural_data.shape
                reconstructed = reconstructed.reshape(n_trials, n_timepoints, n_neurons)
                print(f"Reshaped reconstructed to: {reconstructed.shape}")
        
        # Compute metrics
        print("Computing reconstruction metrics...")
        
        # Overall metrics
        overall_metrics = compute_reconstruction_metrics(neural_data, reconstructed)
        
        # Per-neuron metrics
        per_neuron_metrics = compute_per_neuron_metrics(neural_data, reconstructed)
        
        # Trial-averaged metrics
        trial_avg_metrics = compute_trial_averaged_metrics(neural_data, reconstructed)
        
        # Store results
        split_results = {
            'overall_metrics': overall_metrics,
            'per_neuron_metrics': per_neuron_metrics,
            'trial_averaged_metrics': trial_avg_metrics,
            'data_shapes': {
                'original': neural_data.shape,
                'reconstructed': reconstructed.shape,
                'latents': latents.shape
            }
        }
        
        results[split] = split_results
        
        # Create plots
        if create_plots:
            print("Creating visualization plots...")
            split_output_dir = output_dir / split
            split_output_dir.mkdir(exist_ok=True)
            
            figures = create_reconstruction_plots(
                neural_data, reconstructed, per_neuron_metrics, split_output_dir
            )
            all_figures[split] = figures
        
        # Print summary
        print(f"\n=== {split.upper()} SPLIT RECONSTRUCTION SUMMARY ===")
        print(f"Overall MSE: {overall_metrics['mse_overall']:.6f}")
        print(f"Overall RMSE: {overall_metrics['rmse_overall']:.6f}")
        print(f"Overall Correlation: {overall_metrics['correlation_overall']:.4f}")
        print(f"Overall R²: {overall_metrics['r2_overall']:.4f}")
        print(f"Explained Variance: {overall_metrics['explained_variance']:.4f}")
        print(f"SNR (dB): {overall_metrics['snr_db']:.2f}")
        print(f"")
        print(f"Trial-averaged MSE: {trial_avg_metrics['mse_trial_avg']:.6f}")
        print(f"Trial-averaged Correlation: {trial_avg_metrics['correlation_trial_avg']:.4f}")
        print(f"")
        print(f"Median per-neuron correlation: {np.nanmedian(per_neuron_metrics['correlation_per_neuron']):.4f}")
        print(f"Neurons with correlation > 0.5: {np.sum(per_neuron_metrics['correlation_per_neuron'] > 0.5)}/{len(per_neuron_metrics['correlation_per_neuron'])}")
        print(f"Neurons with correlation > 0.7: {np.sum(per_neuron_metrics['correlation_per_neuron'] > 0.7)}/{len(per_neuron_metrics['correlation_per_neuron'])}")
    
    # Save results to JSON
    results_path = output_dir / "reconstruction_results.json"
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for split, split_results in results.items():
        json_split_results = {
            'overall_metrics': split_results['overall_metrics'],
            'trial_averaged_metrics': split_results['trial_averaged_metrics'],
            'data_shapes': split_results['data_shapes']
        }
        
        # Add per-neuron summary statistics
        per_neuron = split_results['per_neuron_metrics']
        json_split_results['per_neuron_summary'] = {
            'correlation_median': float(np.nanmedian(per_neuron['correlation_per_neuron'])),
            'correlation_mean': float(np.nanmean(per_neuron['correlation_per_neuron'])),
            'correlation_std': float(np.nanstd(per_neuron['correlation_per_neuron'])),
            'mse_median': float(np.median(per_neuron['mse_per_neuron'])),
            'mse_mean': float(np.mean(per_neuron['mse_per_neuron'])),
            'r2_median': float(np.nanmedian(per_neuron['r2_per_neuron'])),
            'r2_mean': float(np.nanmean(per_neuron['r2_per_neuron'])),
            'neurons_corr_gt_05': int(np.sum(per_neuron['correlation_per_neuron'] > 0.5)),
            'neurons_corr_gt_07': int(np.sum(per_neuron['correlation_per_neuron'] > 0.7)),
            'total_neurons': len(per_neuron['correlation_per_neuron'])
        }
        
        json_results[split] = json_split_results
    
    # Add model and dataset info
    json_results['model_info'] = {
        'model_type': model_type,
        'model_config': model.get_config() if hasattr(model, 'get_config') else 'Not available',
        'experiment_name': experiment_name,
        'session_id': session_id
    }
    
    json_results['dataset_info'] = dataset.get_config()
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Close figures to save memory
    if create_plots:
        for split_figures in all_figures.values():
            for fig in split_figures.values():
                plt.close(fig)
    
    return results


def main():
    """Main function for command line usage"""
    if len(sys.argv) < 2:
        print("Usage: python evaluate_reconstruction.py <config_file.json> [splits]")
        print("Example: python evaluate_reconstruction.py configs/lfads_config.json test,val")
        print("Default splits: test")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    # Parse splits argument
    if len(sys.argv) > 2:
        splits = sys.argv[2].split(',')
    else:
        splits = ['test']
    
    if not Path(config_path).exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    print(f"Evaluating reconstruction with config: {config_path}")
    print(f"Splits: {splits}")
    
    try:
        results = evaluate_reconstruction(config_path, splits=splits, create_plots=True)
        print(f"\n✅ Reconstruction evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Reconstruction evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()