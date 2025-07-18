# visualize_embeddings.py
"""
Visualize neural embeddings with mean trajectory plotting, scatter plots, and smoothing options
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Optional, Tuple, Union

# Import your existing framework components
from mrm.trainer import load_trained_model
from mrm.dataset import IBLDataset


def create_scatter_plots(
    embeddings: np.ndarray,
    behavior_data: Dict[str, np.ndarray],
    behavioral_signals: List[str],
    splits: List[str],
    output_dir: Path,
    max_points_per_condition: int = 5000
) -> Dict[str, plt.Figure]:
    """
    Create scatter plots showing all timepoints colored by behavioral condition
    """
    
    figures = {}
    
    for signal in behavioral_signals:
        print(f"Creating scatter plot for {signal}...")
        
        # Create figure
        n_splits = len(splits)
        fig, axes = plt.subplots(1, n_splits, figsize=(8*n_splits, 6), squeeze=False)
        axes = axes.flatten()
        
        for split_idx, split in enumerate(splits):
            ax = axes[split_idx]
            
            # Get behavioral signal data
            behavior_signal = behavior_data.get(signal, np.array([]))
            
            if len(behavior_signal) == 0:
                ax.text(0.5, 0.5, f'No {signal} data', ha='center', va='center', 
                       transform=ax.transAxes)
                continue
            
            # Group trials by behavioral condition
            conditions, colors = get_behavioral_conditions(signal, behavior_signal)
            
            # Plot scatter points for each condition
            for condition_name, condition_mask in conditions.items():
                if not np.any(condition_mask):
                    continue
                
                # Get all timepoints for this condition
                condition_trials = np.where(condition_mask)[0]
                
                
                # Collect all timepoints from all trials in this condition
                all_points = []
                for trial_idx in condition_trials:
                    trial_points = embeddings[trial_idx, :, :2]  # First 2 components
                    all_points.append(trial_points)
                
                if len(all_points) == 0:
                    continue
                    
                # Stack all points
                condition_points = np.vstack(all_points)
                
                # Subsample if too many points for performance
                if len(condition_points) > max_points_per_condition:
                    subsample_idx = np.random.choice(
                        len(condition_points), max_points_per_condition, replace=False
                    )
                    condition_points = condition_points[subsample_idx]
                
                color = colors.get(condition_name, '#1f77b4')
                
                # Plot scatter points
                ax.scatter(condition_points[:, 0], condition_points[:, 1], 
                          c=color, alpha=0.6, s=5, 
                          label=f'{condition_name} (n={len(condition_trials)} trials)')
            
            # Formatting
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            ax.set_title(f'All Timepoints - {signal.replace("_", " ").title()}\n{split.title()} Set')
            ax.grid(True, alpha=0.3)
            ax.legend(markerscale=3)  # Make legend markers bigger
        
        plt.tight_layout()
        figures[f"scatter_{signal}"] = fig
        
        # Save figure
        save_path = output_dir / f"scatter_plot_{signal}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    return figures


def create_mean_trajectory_plots(
    embeddings: np.ndarray,
    behavior_data: Dict[str, np.ndarray],
    time_info: Dict,
    behavioral_signals: List[str],
    splits: List[str],
    output_dir: Path,
    smoothing_sigma: Optional[float] = 1.0,
    temporal_smoothing: Optional[float] = None,
    confidence_intervals: bool = True,
    max_trials_per_condition: int = 100
) -> Dict[str, plt.Figure]:
    """
    Create mean trajectory plots integrated with existing framework
    """
    
    figures = {}
    
    for signal in behavioral_signals:
        print(f"Creating mean trajectory plot for {signal}...")
        
        # Create figure
        n_splits = len(splits)
        fig, axes = plt.subplots(1, n_splits, figsize=(8*n_splits, 6), squeeze=False)
        axes = axes.flatten()
        
        for split_idx, split in enumerate(splits):
            ax = axes[split_idx]
            
            # Get behavioral signal data
            behavior_signal = behavior_data.get(signal, np.array([]))
            
            if len(behavior_signal) == 0:
                ax.text(0.5, 0.5, f'No {signal} data', ha='center', va='center', 
                       transform=ax.transAxes)
                continue
            
            # Group trials by behavioral condition
            conditions, colors = get_behavioral_conditions(signal, behavior_signal)
            
            # Plot mean trajectories for each condition
            for condition_name, condition_mask in conditions.items():
                if not np.any(condition_mask):
                    continue
                
                # Subsample trials if too many
                trial_indices = np.where(condition_mask)[0]
                if len(trial_indices) > max_trials_per_condition:
                    trial_indices = np.random.choice(
                        trial_indices, max_trials_per_condition, replace=False
                    )
                    condition_mask = np.zeros_like(condition_mask)
                    condition_mask[trial_indices] = True
                
                # Get trajectories for this condition (first 2 components only)
                condition_trajectories = embeddings[condition_mask, :, :2]
                
                if len(condition_trajectories) == 0:
                    continue
                
                # Apply temporal smoothing if requested
                if temporal_smoothing is not None:
                    smoothed_trajectories = np.zeros_like(condition_trajectories)
                    for trial in range(len(condition_trajectories)):
                        for dim in range(2):
                            smoothed_trajectories[trial, :, dim] = gaussian_filter1d(
                                condition_trajectories[trial, :, dim], 
                                sigma=temporal_smoothing
                            )
                    condition_trajectories = smoothed_trajectories
                
                # Calculate mean trajectory
                mean_trajectory = np.mean(condition_trajectories, axis=0)
                
                # Apply spatial smoothing if requested
                if smoothing_sigma is not None:
                    for dim in range(2):
                        mean_trajectory[:, dim] = gaussian_filter1d(
                            mean_trajectory[:, dim], 
                            sigma=smoothing_sigma
                        )
                
                # Plot mean trajectory
                color = colors.get(condition_name, '#1f77b4')
                ax.plot(mean_trajectory[:, 0], mean_trajectory[:, 1], 
                       color=color, linewidth=3, 
                       label=f'{condition_name} (n={np.sum(condition_mask)})',
                       alpha=0.9)
                
                # Add confidence intervals if requested
                if confidence_intervals and len(condition_trajectories) > 1:
                    std_trajectory = np.std(condition_trajectories, axis=0)
                    
                    # Apply same spatial smoothing to std
                    if smoothing_sigma is not None:
                        for dim in range(2):
                            std_trajectory[:, dim] = gaussian_filter1d(
                                std_trajectory[:, dim], 
                                sigma=smoothing_sigma
                            )
                    
                    # Create confidence band
                    ax.fill_between(
                        mean_trajectory[:, 0],
                        mean_trajectory[:, 1] - std_trajectory[:, 1],
                        mean_trajectory[:, 1] + std_trajectory[:, 1],
                        color=color, alpha=0.2
                    )
                
                # Mark start and end points
                ax.scatter(mean_trajectory[0, 0], mean_trajectory[0, 1], 
                          color=color, s=150, marker='o', edgecolor='black', 
                          linewidth=2, zorder=10)
                ax.scatter(mean_trajectory[-1, 0], mean_trajectory[-1, 1], 
                          color=color, s=150, marker='s', edgecolor='black', 
                          linewidth=2, zorder=10)
            
            
            # Formatting
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            ax.set_title(f'Mean Trajectories - {signal.replace("_", " ").title()}\n{split.title()} Set')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        figures[signal] = fig
        
        # Save figure
        save_path = output_dir / f"mean_trajectories_{signal}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    return figures


def create_individual_trajectory_plots(
    embeddings: np.ndarray,
    behavior_data: Dict[str, np.ndarray],
    behavioral_signals: List[str],
    splits: List[str],
    output_dir: Path,
    max_trials_traj: int = 15
) -> Dict[str, plt.Figure]:
    """
    Create individual trajectory plots
    """
    
    figures = {}
    
    for signal in behavioral_signals:
        print(f"Creating individual trajectory plot for {signal}...")
        
        # Create figure
        n_splits = len(splits)
        fig, axes = plt.subplots(1, n_splits, figsize=(8*n_splits, 6), squeeze=False)
        axes = axes.flatten()
        
        for split_idx, split in enumerate(splits):
            ax = axes[split_idx]
            
            # Get behavioral signal data
            behavior_signal = behavior_data.get(signal, np.array([]))
            
            if len(behavior_signal) == 0:
                ax.text(0.5, 0.5, f'No {signal} data', ha='center', va='center', 
                       transform=ax.transAxes)
                continue
            
            # Group trials by behavioral condition
            conditions, colors = get_behavioral_conditions(signal, behavior_signal)
            
            # Plot individual trajectories for each condition
            for condition_name, condition_mask in conditions.items():
                if not np.any(condition_mask):
                    continue
                
                # Get trial indices for this condition
                trial_indices = np.where(condition_mask)[0]
                n_trials_to_plot = min(len(trial_indices), max_trials_traj)
                
                if n_trials_to_plot > len(trial_indices):
                    selected_trials = trial_indices
                else:
                    selected_trials = np.random.choice(trial_indices, n_trials_to_plot, replace=False)
                
                color = colors.get(condition_name, '#1f77b4')
                
                # Plot individual trajectories
                for i, trial_idx in enumerate(selected_trials):
                    traj = embeddings[trial_idx, :, :2]  # First 2 components
                    
                    alpha = 0.7 if i == 0 else 0.4  # First trajectory more visible
                    label = condition_name if i == 0 else None
                    
                    ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=alpha, 
                           linewidth=1.5, label=label)
                    
                    # Mark start and end
                    ax.scatter(traj[0, 0], traj[0, 1], color=color, s=30, 
                              marker='o', edgecolor='black', linewidth=1, alpha=0.8)
                    ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=30, 
                              marker='s', edgecolor='black', linewidth=1, alpha=0.8)
            
            # Formatting
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            ax.set_title(f'Individual Trajectories - {signal.replace("_", " ").title()}\n{split.title()} Set')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        figures[f"individual_{signal}"] = fig
        
        # Save figure
        save_path = output_dir / f"individual_trajectories_{signal}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    return figures


def get_behavioral_conditions(signal: str, behavior_signal: np.ndarray) -> Tuple[Dict, Dict]:
    """Get conditions and colors for a behavioral signal"""
    
    if signal == 'choice':
        conditions = {
            'Left Choice': behavior_signal == -1,
            'Right Choice': behavior_signal == 1
        }
        colors = {'Left Choice': '#1f77b4', 'Right Choice': '#ff7f0e'}
        
    elif signal == 'feedback_type':
        conditions = {
            'Correct': behavior_signal == 1,
            'Error': behavior_signal == -1
        }
        colors = {'Correct': '#2ca02c', 'Error': '#d62728'}
        
    elif 'contrast' in signal:
        # Handle contrast signals
        valid_mask = ~np.isnan(behavior_signal)
        if np.any(valid_mask):
            contrast_vals = behavior_signal[valid_mask]
            unique_vals = np.unique(contrast_vals)
            
            if len(unique_vals) > 4:
                # Bin continuous values
                percentiles = np.percentile(contrast_vals[contrast_vals > 0], [50]) if np.any(contrast_vals > 0) else [0]
                conditions = {
                    'Low Contrast': (behavior_signal > 0) & (behavior_signal <= percentiles[0]),
                    'High Contrast': behavior_signal > percentiles[0],
                    'No Stimulus': behavior_signal == 0
                }
            else:
                # Use discrete values
                conditions = {f'Contrast {val:.2f}': behavior_signal == val for val in unique_vals}
                
            colors = {cond: plt.cm.viridis(i/len(conditions)) for i, cond in enumerate(conditions.keys())}
        else:
            conditions = {'All': np.ones(len(behavior_signal), dtype=bool)}
            colors = {'All': '#1f77b4'}
            
    elif signal == 'reaction_time':
        # Bin reaction times
        valid_mask = ~np.isnan(behavior_signal)
        if np.any(valid_mask):
            rt_vals = behavior_signal[valid_mask]
            percentiles = np.percentile(rt_vals, [33, 67])
            conditions = {
                'Fast RT': behavior_signal <= percentiles[0],
                'Medium RT': (behavior_signal > percentiles[0]) & (behavior_signal <= percentiles[1]),
                'Slow RT': behavior_signal > percentiles[1]
            }
            colors = {'Fast RT': '#440154', 'Medium RT': '#21908c', 'Slow RT': '#fde725'}
        else:
            conditions = {'All': np.ones(len(behavior_signal), dtype=bool)}
            colors = {'All': '#1f77b4'}
    else:
        # Default case
        conditions = {'All': np.ones(len(behavior_signal), dtype=bool)}
        colors = {'All': '#1f77b4'}
        
    return conditions, colors



def visualize_embeddings(config_path: str):
    """
    Create visualization plots from trained model
    """
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract paths and settings
    experiment_name = config['experiment']['name']
    session_id = config['experiment']['session_id']
    model_type = config['model']['type']
    save_dir = config.get('save_dir', 'outputs')
    
    # Load visualization config
    viz_config = config.get('visualization', {})
    splits = viz_config.get('splits', ['train', 'test', 'val'])
    behavioral_signals = viz_config.get('behavioral_signals', [
        'choice', 'feedback_type', 'stimulus_contrast_left', 
        'stimulus_contrast_right', 'reaction_time'
    ])
    
    # Visualization parameters
    smoothing_sigma = viz_config.get('smoothing_sigma', 1.0)
    temporal_smoothing = viz_config.get('temporal_smoothing', None)
    confidence_intervals = viz_config.get('confidence_intervals', True)
    
    # Construct paths
    model_dir = Path(save_dir) / experiment_name / session_id / model_type
    output_dir = model_dir / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading model from: {model_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load trained model
    model = load_trained_model(str(model_dir))
    
    # Load dataset
    dataset_config = config['dataset']
    if dataset_config['type'] == 'ibl':
        dataset = IBLDataset(**dataset_config['params'])
        dataset.prepare()
    else:
        raise ValueError(f"Unknown dataset type: {dataset_config['type']}")
    
    print("Creating visualizations...")
    
    all_figures = {}
    
    # Process each split
    for split in splits:
        print(f"\nProcessing {split} split...")
        
        # Get data
        neural_data = dataset.get_neural_data(split)
        behavior_data = dataset.get_behavior_data(split)
       
        # Check the behavioral correlation
        choices = behavior_data['choice']  # -1=left, 1=right
        feedback = behavior_data['feedback_type']  # -1=error, 1=correct

        # Calculate correlation
        correlation = np.corrcoef(choices, feedback)[0,1]
        print(f"Choice-Feedback correlation: {correlation}")
    

        # Crosstab analysis
        left_correct = np.sum((choices == -1) & (feedback == 1))
        left_error = np.sum((choices == -1) & (feedback == -1))
        right_correct = np.sum((choices == 1) & (feedback == 1))
        right_error = np.sum((choices == 1) & (feedback == -1))

        print(f"Left-Correct: {left_correct}, Left-Error: {left_error}")
        print(f"Right-Correct: {right_correct}, Right-Error: {right_error}")
        
        time_info = dataset.get_time_info(split)
        
        # Get embeddings
        embeddings = model.encode(neural_data, behavior_data)
        print(f"Embeddings shape: {embeddings.shape}")
        
        
        # Ensure embeddings are in trial format (n_trials, n_timepoints, n_dims)
        if len(embeddings.shape) == 2:
            n_trials = neural_data.shape[0]
            n_timepoints = neural_data.shape[1]
            n_dims = embeddings.shape[1]
            embeddings = embeddings.reshape(n_trials, n_timepoints, n_dims)
        
        # Create split-specific output directory
        split_output_dir = output_dir / split
        split_output_dir.mkdir(exist_ok=True)
        
        # Create scatter plots
        print(f"Creating scatter plots for {split}...")
        scatter_figures = create_scatter_plots(
            embeddings=embeddings,
            behavior_data=behavior_data,
            behavioral_signals=behavioral_signals,
            splits=[split],
            output_dir=split_output_dir,
            max_points_per_condition=10000
        )
        all_figures.update({f"{split}_{k}": v for k, v in scatter_figures.items()})
        
        # Create mean trajectory plots
        print(f"Creating mean trajectory plots for {split}...")
        mean_figures = create_mean_trajectory_plots(
            embeddings=embeddings,
            behavior_data=behavior_data,
            time_info=time_info,
            behavioral_signals=behavioral_signals,
            splits=[split],
            output_dir=split_output_dir,
            smoothing_sigma=smoothing_sigma,
            temporal_smoothing=temporal_smoothing,
            confidence_intervals=confidence_intervals
        )
        all_figures.update({f"{split}_{k}": v for k, v in mean_figures.items()})
        
        # Create individual trajectory plots (optional)
        if viz_config.get('save_trajectories', False):
            print(f"Creating individual trajectory plots for {split}...")
            max_trials_traj = viz_config.get('max_trials_traj', 15)
            individual_figures = create_individual_trajectory_plots(
                embeddings=embeddings,
                behavior_data=behavior_data,
                behavioral_signals=behavioral_signals,
                splits=[split],
                output_dir=split_output_dir,
                max_trials_traj=max_trials_traj
            )
            all_figures.update({f"{split}_{k}": v for k, v in individual_figures.items()})
        
        # Close figures to save memory
        for fig_dict in [scatter_figures, mean_figures]:
            for fig in fig_dict.values():
                plt.close(fig)
        
        if viz_config.get('save_trajectories', False):
            for fig in individual_figures.values():
                plt.close(fig)
    
    # Save summary
    summary_path = output_dir / "visualization_summary.json"
    summary_data = {
        "model_type": model_type,
        "dataset_config": dataset_config,
        "visualization_config": viz_config,
        "figures_created": list(all_figures.keys()),
        "splits_processed": splits,
        "behavioral_signals": behavioral_signals
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nVisualization complete! Created {len(all_figures)} figures.")
    print(f"Summary saved to {summary_path}")
    
    return all_figures


def main():
    """Main function to run visualization"""
    
    if len(sys.argv) != 2:
        print("Usage: python visualize_embeddings.py <config_file.json>")
        print("Example: python visualize_embeddings.py ../configs/pca_config.json")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not Path(config_path).exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    print(f"Running visualization with config: {config_path}")
    
    try:
        visualize_embeddings(config_path)
        print(f"\n✅ Visualization completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()