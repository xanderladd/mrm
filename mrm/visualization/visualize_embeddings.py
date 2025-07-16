# visualization/visualize_embeddings.py
"""
Embedding visualization for neural models with IBL data
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from mrm.dataset import IBLDataset
from mrm.models.base import BaseModel
from mrm.trainer import load_trained_model


class EmbeddingVisualizer:
    """Visualizer for neural embeddings with multiple dimensionality reduction techniques"""
    
    def __init__(self, model_dir: str, output_dir: Optional[str] = None):
        """
        Initialize visualizer
        
        Args:
            model_dir: Directory containing trained model
            output_dir: Directory to save visualizations (default: model_dir/viz)
        """
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir) if output_dir else self.model_dir / "viz"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and configs
        self.model = load_trained_model(str(self.model_dir))
        
        with open(self.model_dir / "training_config.json", 'r') as f:
            self.training_config = json.load(f)
            
        with open(self.model_dir / "dataset_config.json", 'r') as f:
            self.dataset_config = json.load(f)
            
        with open(self.model_dir / "data_splits.json", 'r') as f:
            self.splits = json.load(f)
            
        # Initialize dataset
        self.dataset = IBLDataset(**self.dataset_config)
        self.dataset.prepare()
        
        print(f"Loaded model from {self.model_dir}")
        print(f"Visualizations will be saved to {self.output_dir}")
        
    def extract_embeddings(self, split: str = 'test') -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Extract embeddings and behavioral data for visualization
        
        Args:
            split: Data split to use ('train', 'val', 'test')
            
        Returns:
            embeddings: (n_trials, n_timepoints, embedding_dim)
            behavior_data: Dict of behavioral signals
        """
        # Get neural data and encode to embeddings
        neural_data = self.dataset.get_neural_data(split)
        embeddings = self.model.encode(neural_data)
        
        # Get behavioral data for coloring
        behavior_data = self.dataset.get_behavior_data(split)
        
        print(f"Extracted embeddings: {embeddings.shape}")
        return embeddings, behavior_data
        
    def reduce_dimensionality(self, embeddings: np.ndarray, 
                            method: str = 'pca',
                            n_components: int = 2,
                            **kwargs) -> np.ndarray:
        """
        Apply dimensionality reduction to embeddings
        
        Args:
            embeddings: Input embeddings (n_samples, embedding_dim)
            method: Reduction method ('pca')
            n_components: Number of output dimensions
            **kwargs: Additional arguments for the reduction method
            
        Returns:
            reduced_embeddings: (n_samples, n_components)
        """
        if method == 'pca':
            reducer = PCA(n_components=n_components, **kwargs)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
            
        return reducer.fit_transform(embeddings)
        
    def create_behavioral_colormap(self, behavior_data: Dict[str, np.ndarray], 
                                 signal_name: str) -> Tuple[np.ndarray, str, str]:
        """
        Create colormap based on behavioral signal
        
        Args:
            behavior_data: Behavioral signals
            signal_name: Which signal to use for coloring
            
        Returns:
            colors: Color values for each trial
            label: Label for colorbar
            cmap: Colormap name
        """
        if signal_name not in behavior_data:
            print(f"Warning: {signal_name} not found, using choice instead")
            signal_name = 'choice'
            
        signal = behavior_data[signal_name]
        
        # Handle different signal types
        if signal_name == 'choice':
            # Categorical: left (-1), right (+1), no-go (0)
            valid_mask = ~np.isnan(signal)
            colors = np.full(len(signal), 0.5)  # Default gray
            colors[valid_mask] = signal[valid_mask]
            label = "Choice (L=-1, R=+1)"
            cmap = 'RdBu_r'
            
        elif signal_name == 'feedback_type':
            # Binary: correct (1), incorrect (-1)
            valid_mask = ~np.isnan(signal)
            colors = np.full(len(signal), 0.5)
            colors[valid_mask] = signal[valid_mask]
            label = "Feedback (Correct=1, Error=-1)"
            cmap = 'RdYlGn'
            
        elif 'contrast' in signal_name:
            # Continuous stimulus contrast
            colors = np.abs(signal)
            colors[np.isnan(colors)] = 0
            label = f"{signal_name.replace('_', ' ').title()}"
            cmap = 'viridis'
            
        elif signal_name == 'reaction_time':
            # Continuous reaction time
            colors = signal.copy()
            # Handle outliers
            valid_mask = ~np.isnan(colors)
            if np.any(valid_mask):
                p95 = np.percentile(colors[valid_mask], 95)
                colors[colors > p95] = p95
                colors[np.isnan(colors)] = np.median(colors[valid_mask])
            label = "Reaction Time (s)"
            cmap = 'plasma'
            
        else:
            # Generic continuous signal
            colors = signal.copy()
            colors[np.isnan(colors)] = np.nanmedian(colors)
            label = signal_name.replace('_', ' ').title()
            cmap = 'viridis'
            
        return colors, label, cmap
        
    def plot_embedding_space(self, embeddings_2d: np.ndarray, 
                           colors: np.ndarray,
                           title: str,
                           color_label: str,
                           cmap: str,
                           method: str = 'pca',
                           trial_indices: Optional[np.ndarray] = None) -> plt.Figure:
        """
        Create embedding space visualization
        
        Args:
            embeddings_2d: 2D embeddings to plot
            colors: Color values for each point
            title: Plot title
            color_label: Label for colorbar
            cmap: Colormap name
            method: Dimensionality reduction method used
            trial_indices: Optional trial indices for annotation
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                           c=colors, cmap=cmap, alpha=0.7, s=30)
        
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        ax.set_title(title)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(color_label)
        
        # Add trial annotations if requested (for small datasets)
        if trial_indices is not None and len(trial_indices) < 50:
            for i, trial_idx in enumerate(trial_indices):
                ax.annotate(f'T{trial_idx}', (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                          xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        return fig
        
    def plot_trajectory_embeddings(self, embeddings: np.ndarray,
                                 behavior_data: Dict[str, np.ndarray],
                                 signal_name: str = 'choice',
                                 trial_indices: Optional[List[int]] = None,
                                 max_trials: int = 20) -> plt.Figure:
        """
        Plot individual trial trajectories in embedding space
        
        Args:
            embeddings: Full embeddings (n_trials, n_timepoints, embedding_dim)
            behavior_data: Behavioral data for coloring
            signal_name: Behavioral signal for coloring
            trial_indices: Specific trials to plot (default: first max_trials)
            max_trials: Maximum number of trials to plot
            
        Returns:
            Figure object
        """
        if trial_indices is None:
            trial_indices = list(range(min(max_trials, embeddings.shape[0])))
        
        colors, color_label, cmap = self.create_behavioral_colormap(behavior_data, signal_name)
        
        # Apply PCA to reduce to 2D for visualization
        n_trials, n_timepoints, embed_dim = embeddings.shape
        embeddings_flat = embeddings.reshape(-1, embed_dim)
        embeddings_2d = self.reduce_dimensionality(embeddings_flat, 'pca', 2)
        embeddings_2d = embeddings_2d.reshape(n_trials, n_timepoints, 2)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Plot each trial trajectory
        for i, trial_idx in enumerate(trial_indices):
            trial_traj = embeddings_2d[trial_idx]
            trial_color = colors[trial_idx] if not np.isnan(colors[trial_idx]) else 0.5
            
            # Plot trajectory
            ax.plot(trial_traj[:, 0], trial_traj[:, 1], alpha=0.6, linewidth=2,
                   color=plt.cm.get_cmap(cmap)(0.5 + trial_color * 0.4))
            
            # Mark start and end
            ax.scatter(trial_traj[0, 0], trial_traj[0, 1], 
                      marker='o', s=100, alpha=0.8, 
                      color=plt.cm.get_cmap(cmap)(0.5 + trial_color * 0.4),
                      edgecolors='black', linewidth=1)
            ax.scatter(trial_traj[-1, 0], trial_traj[-1, 1], 
                      marker='s', s=100, alpha=0.8,
                      color=plt.cm.get_cmap(cmap)(0.5 + trial_color * 0.4),
                      edgecolors='black', linewidth=1)
            
            # Add trial number
            ax.annotate(f'{trial_idx}', 
                       (trial_traj[0, 0], trial_traj[0, 1]),
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, alpha=0.7)
        
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_title(f'Trial Trajectories in Embedding Space\n'
                    f'Colored by {color_label} (○=start, □=end)')
        
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='gray', linestyle='-', 
                   markersize=8, alpha=0.7, label='Trial start'),
            Line2D([0], [0], marker='s', color='gray', linestyle='-', 
                   markersize=8, alpha=0.7, label='Trial end')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        return fig
        
    def visualize_embeddings(self, 
                           split: str = 'test',
                           methods: List[str] = ['pca', 'tsne', 'umap'],
                           behavioral_signals: List[str] = ['choice', 'feedback_type', 'stimulus_contrast_left'],
                           save_trajectories: bool = True,
                           max_trials_traj: int = 20) -> Dict[str, plt.Figure]:
        """
        Create comprehensive embedding visualizations
        
        Args:
            split: Data split to visualize
            methods: Dimensionality reduction methods to use
            behavioral_signals: Behavioral signals for coloring
            save_trajectories: Whether to create trajectory plots
            max_trials_traj: Max trials for trajectory visualization
            
        Returns:
            Dictionary of figure objects
        """
        print(f"Creating embedding visualizations for {split} split...")
        
        # Extract embeddings and behavior
        embeddings, behavior_data = self.extract_embeddings(split)
        
        # Flatten embeddings for dimensionality reduction
        n_trials, n_timepoints, embed_dim = embeddings.shape
        embeddings_flat = embeddings.reshape(-1, embed_dim)
        
        figures = {}
        
        # Create visualizations for each method and behavioral signal
        for method in methods:
            print(f"  Applying {method.upper()}...")
            
            try:
                # Apply dimensionality reduction
                if method == 'tsne' and embeddings_flat.shape[0] > 10000:
                    # Subsample for t-SNE if too many points
                    indices = np.random.choice(embeddings_flat.shape[0], 10000, replace=False)
                    embeddings_2d = self.reduce_dimensionality(embeddings_flat[indices], method, 2)
                    # Map back to trial structure (approximately)
                    trial_indices = indices // n_timepoints
                else:
                    embeddings_2d = self.reduce_dimensionality(embeddings_flat, method, 2)
                    trial_indices = np.arange(n_trials)
                
                for signal_name in behavioral_signals:
                    if signal_name not in behavior_data:
                        continue
                        
                    print(f"    Plotting {signal_name}...")
                    
                    # Create colors for this signal
                    colors, color_label, cmap = self.create_behavioral_colormap(
                        behavior_data, signal_name
                    )
                    
                    # If we subsampled, adjust colors
                    if method == 'tsne' and embeddings_flat.shape[0] > 10000:
                        # Map colors to subsampled points
                        colors_sub = np.array([colors[idx] for idx in trial_indices])
                    else:
                        # Repeat colors for each timepoint
                        colors_sub = np.repeat(colors, n_timepoints)
                    
                    # Create plot
                    title = f'{method.upper()} Visualization - {color_label}\n{split.title()} Set'
                    fig = self.plot_embedding_space(
                        embeddings_2d, colors_sub, title, color_label, cmap, method
                    )
                    
                    figures[f'{method}_{signal_name}_{split}'] = fig
                    
            except Exception as e:
                print(f"    Error with {method}: {e}")
                continue
        
        # Create trajectory visualizations
        if save_trajectories and n_timepoints > 1:
            print("  Creating trajectory visualizations...")
            
            for signal_name in behavioral_signals[:2]:  # Limit to avoid too many plots
                if signal_name not in behavior_data:
                    continue
                    
                try:
                    fig = self.plot_trajectory_embeddings(
                        embeddings, behavior_data, signal_name, 
                        max_trials=max_trials_traj
                    )
                    figures[f'trajectories_{signal_name}_{split}'] = fig
                except Exception as e:
                    print(f"    Error creating trajectories for {signal_name}: {e}")
        
        print(f"Created {len(figures)} visualizations")
        return figures
        
    def save_figures(self, figures: Dict[str, plt.Figure], dpi: int = 300):
        """Save all figures to output directory"""
        print(f"Saving {len(figures)} figures to {self.output_dir}")
        
        for name, fig in figures.items():
            filepath = self.output_dir / f"{name}.png"
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"  Saved {filepath}")
            
        # Save summary info
        summary = {
            'dataset_config': self.dataset_config,
            'model_type': self.training_config['model']['type'],
            'experiment_name': self.training_config['experiment']['name'],
            'n_figures': len(figures),
            'figure_names': list(figures.keys())
        }
        
        summary_path = self.output_dir / "visualization_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Saved summary to {summary_path}")
        
    def run_full_visualization(self, 
                             splits: List[str] = ['test'],
                             methods: List[str] = ['pca', 'tsne', 'umap'],
                             behavioral_signals: List[str] = ['choice', 'feedback_type', 'stimulus_contrast_left'],
                             save_trajectories: bool = True, **kwargs) -> Dict[str, plt.Figure]:
        """
        Run complete visualization pipeline
        
        Args:
            splits: Data splits to visualize
            methods: Dimensionality reduction methods
            behavioral_signals: Behavioral signals for coloring
            save_trajectories: Whether to include trajectory plots
            
        Returns:
            All generated figures
        """
        all_figures = {}
        
        for split in splits:
            print(f"\nProcessing {split} split...")
            figures = self.visualize_embeddings(
                split=split,
                methods=methods,
                behavioral_signals=behavioral_signals,
                save_trajectories=save_trajectories
            )
            all_figures.update(figures)
        
        # Save all figures
        self.save_figures(all_figures)
        
        return all_figures


def visualize_from_config(config_path: str, 
                         model_dir: Optional[str] = None,
                         output_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
    """
    Create visualizations from a configuration file
    
    Args:
        config_path: Path to config JSON (with visualization block)
        model_dir: Override model directory from config
        output_dir: Override output directory
        
    Returns:
        Generated figures
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Determine model directory from config
    if model_dir is None:
        # Build model directory from config structure
        save_dir = config.get('save_dir', 'outputs')
        experiment_name = config['experiment']['name']
        session_id = config['experiment']['session_id']
        model_type = config['model']['type']
        
        model_dir = os.path.join(save_dir, experiment_name, session_id, model_type)
    
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")
    
    print(f"Using model directory: {model_dir}")
    
    # Create visualizer
    visualizer = EmbeddingVisualizer(model_dir, output_dir)
    
    # Get visualization parameters from config
    viz_config = config.get('visualization', {})
    
    # Extract parameters with defaults
    viz_params = {
        'splits': viz_config.get('splits', ['test']),
        'methods': viz_config.get('methods', ['pca', 'tsne']),
        'behavioral_signals': viz_config.get('behavioral_signals', ['choice', 'feedback_type']),
        'save_trajectories': viz_config.get('save_trajectories', True),
        'max_trials_traj': viz_config.get('max_trials_traj', 20)
    }
    
    print(f"Visualization parameters: {viz_params}")
    
    # Run visualization
    return visualizer.run_full_visualization(**viz_params)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_embeddings.py <config.json>")
        print("Example: python visualization/visualize_embeddings.py ../configs/pca_config.json")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    print(f"Creating visualizations from config: {config_path}")
    
    try:
        figures = visualize_from_config(config_path)
        print(f"\n✅ Visualization complete! Generated {len(figures)} plots.")
    except Exception as e:
        print(f"\n❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)