"""
embeddings_step2.py

Simple visualization of trajectory distances over time.
Shows how different behavioral conditions separate in latent space as a function of time.
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from scipy.spatial.distance import euclidean
from scipy.ndimage import gaussian_filter1d
plt.rcParams['svg.fonttype'] = 'none'

# Import your existing framework components
from mrm.trainer import load_trained_model
from mrm.dataset import IBLDataset

def smooth_trajectories(trajectories: np.ndarray, sigma: float, acausal: bool = False) -> np.ndarray:
    """
    Apply Gaussian smoothing to trajectories
    
    Parameters:
    -----------
    trajectories : np.ndarray
        Shape (n_trials, n_timepoints, n_components) or (n_timepoints, n_components)
    sigma : float
        Gaussian smoothing parameter
    acausal : bool
        If True, use acausal smoothing (only past/present data)
        If False, use standard bidirectional Gaussian smoothing
        
    Returns:
    --------
    smoothed : np.ndarray
        Smoothed trajectories with same shape as input
    """
    if sigma <= 0:
        return trajectories
        
    smoothed = np.zeros_like(trajectories)
    
    if acausal:
        # Acausal exponential smoothing (only past data)
        alpha = 1.0 / (1.0 + sigma)  # Convert sigma to decay parameter
        
        if len(trajectories.shape) == 3:
            # (n_trials, n_timepoints, n_components)
            for trial in range(trajectories.shape[0]):
                for comp in range(trajectories.shape[2]):
                    data = trajectories[trial, :, comp]
                    smoothed[trial, 0, comp] = data[0]  # Initialize with first value
                    for t in range(1, len(data)):
                        smoothed[trial, t, comp] = alpha * data[t] + (1 - alpha) * smoothed[trial, t-1, comp]
        elif len(trajectories.shape) == 2:
            # (n_timepoints, n_components)
            for comp in range(trajectories.shape[1]):
                data = trajectories[:, comp]
                smoothed[0, comp] = data[0]  # Initialize with first value
                for t in range(1, len(data)):
                    smoothed[t, comp] = alpha * data[t] + (1 - alpha) * smoothed[t-1, comp]
        else:
            # 1D case
            data = trajectories
            smoothed[0] = data[0]
            for t in range(1, len(data)):
                smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t-1]
    else:
        # Standard bidirectional Gaussian smoothing
        if len(trajectories.shape) == 3:
            # (n_trials, n_timepoints, n_components)
            for trial in range(trajectories.shape[0]):
                for comp in range(trajectories.shape[2]):
                    smoothed[trial, :, comp] = gaussian_filter1d(
                        trajectories[trial, :, comp], sigma=sigma
                    )
        elif len(trajectories.shape) == 2:
            # (n_timepoints, n_components)
            for comp in range(trajectories.shape[1]):
                smoothed[:, comp] = gaussian_filter1d(
                    trajectories[:, comp], sigma=sigma
                )
        else:
            # 1D case
            smoothed = gaussian_filter1d(trajectories, sigma=sigma)
    
    return smoothed

def compute_trajectory_distance(latents_condition1: np.ndarray, 
                               latents_condition2: np.ndarray,
                               metric: str = 'euclidean') -> np.ndarray:
    """
    Compute distance between two sets of trajectories over time.
    
    Parameters:
    -----------
    latents_condition1 : np.ndarray
        Shape (n_trials, n_timepoints, n_components)
    latents_condition2 : np.ndarray  
        Shape (n_trials, n_timepoints, n_components)
    metric : str
        Distance metric ('euclidean', 'manhattan', 'cosine')
    
    Returns:
    --------
    distances : np.ndarray
        Shape (n_timepoints,) - distance at each timepoint
    """
    # Average across trials for each condition
    mean_traj1 = np.mean(latents_condition1, axis=0)  # (n_timepoints, n_components)
    mean_traj2 = np.mean(latents_condition2, axis=0)  # (n_timepoints, n_components)
    
    # Compute distance at each timepoint
    distances = np.zeros(mean_traj1.shape[0])
    
    for t in range(mean_traj1.shape[0]):
        if metric == 'euclidean':
            distances[t] = euclidean(mean_traj1[t], mean_traj2[t])
        elif metric == 'manhattan':
            distances[t] = np.sum(np.abs(mean_traj1[t] - mean_traj2[t]))
        elif metric == 'cosine':
            dot_product = np.dot(mean_traj1[t], mean_traj2[t])
            norms = np.linalg.norm(mean_traj1[t]) * np.linalg.norm(mean_traj2[t])
            distances[t] = 1 - (dot_product / norms) if norms > 0 else 0
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    return distances

def create_condition_masks(behavior_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Create boolean masks for different behavioral conditions."""
    
    masks = {}
    
    # Choice conditions
    if 'choice' in behavior_data:
        choices = behavior_data['choice']
        masks['left_choice'] = choices == -1
        masks['right_choice'] = choices == 1
    
    # Stimulus position conditions  
    if 'stimulus_contrast_left' in behavior_data and 'stimulus_contrast_right' in behavior_data:
        contrast_left = behavior_data['stimulus_contrast_left']
        contrast_right = behavior_data['stimulus_contrast_right']
        
        # Left stimulus present
        masks['left_stimulus'] = (~np.isnan(contrast_left)) & (contrast_left > 0)
        # Right stimulus present  
        masks['right_stimulus'] = (~np.isnan(contrast_right)) & (contrast_right > 0)
    
    # Feedback conditions
    if 'feedback_type' in behavior_data:
        feedback = behavior_data['feedback_type']
        masks['correct'] = feedback == 1
        masks['error'] = feedback == -1
    
    return masks

def plot_trajectory_distances(config: Dict[str, Any]) -> plt.Figure:
    """
    Plot distances between different behavioral conditions over time.
    
    Parameters:
    -----------
    config : Dict[str, Any]
        Configuration containing latents, behavior_data, time_vector, region_name, 
        and visualization parameters including temporal_smoothing
    """
    
    # Extract data from config
    latents = config['latents']
    behavior_data = config['behavior_data']
    time_vector = config['time_vector']
    region_name = config.get('region_name', "Region")
    smoothing_sigma = config.get('visualization', {}).get('temporal_smoothing', 2.0)
    acausal_smoothing = config.get('visualization', {}).get('acausal_smoothing', False)
    
    # Get condition masks
    masks = create_condition_masks(behavior_data)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Choice Distance (Evidence Accumulation Signal)
    if 'left_choice' in masks and 'right_choice' in masks:
        left_trials = np.sum(masks['left_choice'])
        right_trials = np.sum(masks['right_choice'])
        
        if left_trials >= 3 and right_trials >= 3:
            left_latents = latents[masks['left_choice']]
            right_latents = latents[masks['right_choice']]
            
            choice_distances = compute_trajectory_distance(left_latents, right_latents)
            
            if smoothing_sigma > 0:
                choice_distances = smooth_trajectories(choice_distances.reshape(1, -1, 1), 
                                                     smoothing_sigma, acausal_smoothing)[0, :, 0]
            
            axes[0].plot(time_vector, choice_distances, 'b-', linewidth=3, 
                        label=f'Left vs Right Choice\n(n={left_trials} vs {right_trials})')
            axes[0].fill_between(time_vector, choice_distances, alpha=0.3, color='blue')
    
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.7, label='Stimulus Onset')
    axes[0].set_title(f'{region_name}\nChoice Distance Over Time\n(Evidence Accumulation)', 
                     fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Euclidean Distance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Stimulus Position Distance (Sensory Processing Signal)
    if 'left_stimulus' in masks and 'right_stimulus' in masks:
        left_stim_trials = np.sum(masks['left_stimulus'])
        right_stim_trials = np.sum(masks['right_stimulus'])
        
        if left_stim_trials >= 3 and right_stim_trials >= 3:
            left_stim_latents = latents[masks['left_stimulus']]
            right_stim_latents = latents[masks['right_stimulus']]
            
            stim_distances = compute_trajectory_distance(left_stim_latents, right_stim_latents)
            
            if smoothing_sigma > 0:
                stim_distances = smooth_trajectories(stim_distances.reshape(1, -1, 1), 
                                                   smoothing_sigma, acausal_smoothing)[0, :, 0]
            
            axes[1].plot(time_vector, stim_distances, 'g-', linewidth=3,
                        label=f'Left vs Right Stimulus\n(n={left_stim_trials} vs {right_stim_trials})')
            axes[1].fill_between(time_vector, stim_distances, alpha=0.3, color='green')
    
    axes[1].axvline(0, color='red', linestyle='--', alpha=0.7, label='Stimulus Onset')
    axes[1].set_title(f'{region_name}\nStimulus Distance Over Time\n(Sensory Processing)', 
                     fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Euclidean Distance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Feedback Distance (Performance Monitoring Signal)
    if 'correct' in masks and 'error' in masks:
        correct_trials = np.sum(masks['correct'])
        error_trials = np.sum(masks['error'])
        
        if correct_trials >= 3 and error_trials >= 3:
            correct_latents = latents[masks['correct']]
            error_latents = latents[masks['error']]
            
            feedback_distances = compute_trajectory_distance(correct_latents, error_latents)
            
            if smoothing_sigma > 0:
                feedback_distances = smooth_trajectories(feedback_distances.reshape(1, -1, 1), 
                                                       smoothing_sigma, acausal_smoothing)[0, :, 0]
            
            axes[2].plot(time_vector, feedback_distances, 'purple', linewidth=3,
                        label=f'Correct vs Error\n(n={correct_trials} vs {error_trials})')
            axes[2].fill_between(time_vector, feedback_distances, alpha=0.3, color='purple')
    
    axes[2].axvline(0, color='red', linestyle='--', alpha=0.7, label='Stimulus Onset')
    axes[2].set_title(f'{region_name}\nFeedback Distance Over Time\n(Performance Monitoring)', 
                     fontweight='bold', fontsize=12)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Euclidean Distance')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_stimulus_position_variable(behavior_data: Dict[str, np.ndarray]) -> np.ndarray:
    """Create stimulus position variable from left/right contrasts"""
    
    contrast_left = behavior_data.get('stimulus_contrast_left', np.full(len(behavior_data['choice']), np.nan))
    contrast_right = behavior_data.get('stimulus_contrast_right', np.full(len(behavior_data['choice']), np.nan))
    
    # Create stimulus position: -1=left, 0=no_stim, 1=right
    stimulus_position = np.zeros_like(contrast_left)
    
    for i in range(len(stimulus_position)):
        if not np.isnan(contrast_left[i]) and contrast_left[i] > 0:
            stimulus_position[i] = -1  # Left stimulus
        elif not np.isnan(contrast_right[i]) and contrast_right[i] > 0:
            stimulus_position[i] = 1   # Right stimulus
        else:
            stimulus_position[i] = 0   # No stimulus or unclear
    
    return stimulus_position

def trajectory_distance_analysis(config_path: str):
    """
    Main function to run trajectory distance analysis from config file
    
    Parameters:
    -----------
    config_path : str
        Path to JSON configuration file
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
    splits = viz_config.get('splits', ['test'])
    temporal_smoothing = viz_config.get('temporal_smoothing', 2.0)
    causal_smoothing = viz_config.get('causal_smoothing', False)
    
    # Construct paths
    model_dir = Path(save_dir) / experiment_name / session_id / model_type
    output_dir = model_dir / "trajectory_distances"
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
    
    print("Creating trajectory distance visualizations...")
    
    # Process each split
    for split in splits:
        print(f"\nProcessing {split} split...")
        
        # Get data
        neural_data = dataset.get_neural_data(split)
        behavior_data = dataset.get_behavior_data(split)
        
        # Add stimulus position variable
        stimulus_position = create_stimulus_position_variable(behavior_data)
        behavior_data['stimulus_position'] = stimulus_position
        
        time_info = dataset.get_time_info(split)
        time_vector = np.linspace(-time_info['pre_time'], time_info['post_time'], 
                                 neural_data.shape[1])
        
        # Get embeddings
        embeddings = model.encode(neural_data, behavior_data)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Ensure embeddings are in trial format (n_trials, n_timepoints, n_dims)
        if len(embeddings.shape) == 2:
            n_trials = neural_data.shape[0]
            n_timepoints = neural_data.shape[1]
            n_dims = embeddings.shape[1]
            embeddings = embeddings.reshape(n_trials, n_timepoints, n_dims)
        
        # Create config for plotting functions
        plot_config = {
            'latents': embeddings,
            'behavior_data': behavior_data,
            'time_vector': time_vector,
            'region_name': f"{config['experiment']['description']} - {split.title()}",
            'visualization': {
                'temporal_smoothing': temporal_smoothing,
                'causal_smoothing': causal_smoothing
            }
        }
        
        # Create trajectory distance plots
        print(f"Creating trajectory distance plots for {split}...")
        
        distance_fig = plot_trajectory_distances(plot_config)
        
        # Save figure
        save_path = output_dir / f"trajectory_distances_{split}.png"
        distance_fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved trajectory distances to {save_path}")
        
        plt.close(distance_fig)
    
    print(f"\nTrajectory distance analysis complete!")
    print(f"Results saved to {output_dir}")

def compare_two_regions_from_configs(cp_config_path: str, vis_config_path: str, 
                                   split: str = 'test', output_dir: Optional[str] = None):
    """
    Compare trajectory distances between two regions using separate config files
    
    Parameters:
    -----------
    cp_config_path : str
        Path to CP region config file
    vis_config_path : str  
        Path to VIS region config file
    split : str
        Data split to use
    output_dir : str, optional
        Where to save comparison plots
    """
    
    # Helper function to load region data
    def load_region_data(config_path, region_name):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load model and dataset
        experiment_name = config['experiment']['name']
        session_id = config['experiment']['session_id']
        model_type = config['model']['type']
        save_dir = config.get('save_dir', 'outputs')
        
        model_dir = Path(save_dir) / experiment_name / session_id / model_type
        model = load_trained_model(str(model_dir))
        
        dataset_config = config['dataset']
        if dataset_config['type'] == 'ibl':
            dataset = IBLDataset(**dataset_config['params'])
            dataset.prepare()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_config['type']}")
        
        # Get data for specified split
        neural_data = dataset.get_neural_data(split)
        behavior_data = dataset.get_behavior_data(split)
        
        # Add stimulus position
        stimulus_position = create_stimulus_position_variable(behavior_data)
        behavior_data['stimulus_position'] = stimulus_position
        
        time_info = dataset.get_time_info(split)
        time_vector = np.linspace(-time_info['pre_time'], time_info['post_time'], 
                                 neural_data.shape[1])
        
        # Get embeddings
        embeddings = model.encode(neural_data, behavior_data)
        
        # Ensure proper format
        if len(embeddings.shape) == 2:
            n_trials = neural_data.shape[0]
            n_timepoints = neural_data.shape[1]
            n_dims = embeddings.shape[1]
            embeddings = embeddings.reshape(n_trials, n_timepoints, n_dims)
        
        # Get temporal smoothing from config
        temporal_smoothing = config.get('visualization', {}).get('temporal_smoothing', 2.0)
        causal_smoothing = config.get('visualization', {}).get('causal_smoothing', False)
        
        return embeddings, behavior_data, time_vector, temporal_smoothing, causal_smoothing
    
    print("Loading CP region data...")
    cp_latents, cp_behavior, time_vector, temporal_smoothing, causal_smoothing = load_region_data(cp_config_path, "CP")
    
    print("Loading VIS region data...")
    vis_latents, vis_behavior, _, _, _ = load_region_data(vis_config_path, "VIS")
    
    # Create comparison config
    comparison_config = {
        'cp_latents': cp_latents,
        'cp_behavior': cp_behavior,
        'vis_latents': vis_latents,
        'vis_behavior': vis_behavior,
        'time_vector': time_vector,
        'visualization': {
            'temporal_smoothing': temporal_smoothing,
            'causal_smoothing': causal_smoothing
        }
    }
    
    print("Creating region comparison plot...")
    comparison_fig = compare_regions_distance_dynamics(comparison_config)
    
    # Save comparison plot
    if output_dir is None:
        output_dir = Path("trajectory_distance_comparison")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / f"region_comparison_{split}.png"
    comparison_fig.savefig(save_path, dpi=300, bbox_inches='tight')

    save_path = output_dir / f"region_comparison_{split}.svg"
    comparison_fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved region comparison to {save_path}")
    
    plt.close(comparison_fig)
    
    return comparison_config

def main():
    """Main function for CLI usage"""
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single region analysis:")
        print("    python embeddings_step2.py <config_file.json>")
        print("  Two region comparison:")  
        print("    python embeddings_step2.py <cp_config.json> <vis_config.json>")
        print()
        print("Examples:")
        print("  python embeddings_step2.py configs/pca_cp_config.json")
        print("  python embeddings_step2.py configs/pca_cp_config.json configs/pca_vis_config.json")
        sys.exit(1)
    
    try:
        if len(sys.argv) == 2:
            # Single region analysis
            config_path = sys.argv[1]
            
            if not Path(config_path).exists():
                print(f"Config file not found: {config_path}")
                sys.exit(1)
            
            print(f"Running trajectory distance analysis with: {config_path}")
            trajectory_distance_analysis(config_path)
            
        elif len(sys.argv) == 3:
            # Two region comparison
            cp_config_path = sys.argv[1]
            vis_config_path = sys.argv[2]
            
            if not Path(cp_config_path).exists():
                print(f"CP config file not found: {cp_config_path}")
                sys.exit(1)
                
            if not Path(vis_config_path).exists():
                print(f"VIS config file not found: {vis_config_path}")
                sys.exit(1)
            
            print(f"Running region comparison with:")
            print(f"  CP config: {cp_config_path}")
            print(f"  VIS config: {vis_config_path}")
            
            compare_two_regions_from_configs(cp_config_path, vis_config_path)
            
        else:
            print("Error: Too many arguments")
            sys.exit(1)
            
        print("\n✅ Trajectory distance analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def compare_regions_distance_dynamics(config: Dict[str, Any]) -> plt.Figure:
    """
    Compare distance dynamics between two regions (e.g., CP vs VIS).
    This is the key plot for evidence accumulation vs sensory processing.
    
    Parameters:
    -----------
    config : Dict[str, Any]
        Configuration containing cp_latents, cp_behavior, vis_latents, vis_behavior,
        time_vector, and visualization parameters
    """
    
    # Extract data from config
    cp_latents = config['cp_latents']
    cp_behavior = config['cp_behavior'] 
    vis_latents = config['vis_latents']
    vis_behavior = config['vis_behavior']
    time_vector = config['time_vector']
    smoothing_sigma = config.get('visualization', {}).get('temporal_smoothing', 2.0)
    causal_smoothing = config.get('visualization', {}).get('causal_smoothing', False)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Compute distances for both regions
    regions_data = [
        (cp_latents, cp_behavior, 'CP (Striatum)', 'blue'),
        (vis_latents, vis_behavior, 'VIS (Visual Cortex)', 'red')
    ]
    
    choice_distances = {}
    stim_distances = {}
    
    for latents, behavior, region_name, color in regions_data:
        masks = create_condition_masks(behavior)
        
        # Choice distances
        if 'left_choice' in masks and 'right_choice' in masks:
            left_trials = np.sum(masks['left_choice'])
            right_trials = np.sum(masks['right_choice'])
            
            if left_trials >= 3 and right_trials >= 3:
                left_latents = latents[masks['left_choice']]
                right_latents = latents[masks['right_choice']]
                distances = compute_trajectory_distance(left_latents, right_latents)
                
                if smoothing_sigma > 0:
                    distances = smooth_trajectories(distances.reshape(1, -1, 1), 
                                                  smoothing_sigma, causal_smoothing)[0, :, 0]
                
                choice_distances[region_name] = distances
                axes[0].plot(time_vector, distances, color=color, linewidth=3, 
                           label=f'{region_name} (n={left_trials}/{right_trials})')
        
        # Stimulus distances
        if 'left_stimulus' in masks and 'right_stimulus' in masks:
            left_stim_trials = np.sum(masks['left_stimulus'])
            right_stim_trials = np.sum(masks['right_stimulus'])
            
            if left_stim_trials >= 3 and right_stim_trials >= 3:
                left_stim_latents = latents[masks['left_stimulus']]
                right_stim_latents = latents[masks['right_stimulus']]
                distances = compute_trajectory_distance(left_stim_latents, right_stim_latents)
                
                if smoothing_sigma > 0:
                    distances = smooth_trajectories(distances.reshape(1, -1, 1), 
                                                  smoothing_sigma, causal_smoothing)[0, :, 0]
                
                stim_distances[region_name] = distances
                axes[1].plot(time_vector, distances, color=color, linewidth=3,
                           label=f'{region_name} (n={left_stim_trials}/{right_stim_trials})')
    
    # Format choice distance plot
    axes[0].axvline(-0.01, color='black', linestyle='--', alpha=0.7, label='Stimulus Onset')
    axes[0].set_title('Choice Distance Over Time\n(Evidence Accumulation Signature)', 
                     fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Distance (Left vs Right Choice)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Format stimulus distance plot  
    axes[1].axvline(-0.01, color='black', linestyle='--', alpha=0.7, label='Stimulus Onset')
    axes[1].set_title('Stimulus Distance Over Time\n(Sensory Processing Signature)', 
                     fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Distance (Left vs Right Stimulus)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# CLI Usage Examples
if __name__ == "__main__":
    main()