#!/usr/bin/env python3
"""
Plot PSTHs (Peri-Stimulus Time Histograms) for neural units to check responsiveness

Usage:
    python plot_psths.py configs/lfads_config.json
    python plot_psths.py configs/pca_config.json --max-units 50 --splits train test
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.ndimage import gaussian_filter1d

# Import your existing framework components
from mrm.dataset import IBLDataset


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Plot PSTHs for neural units")
    parser.add_argument("config", help="Path to config JSON file")
    parser.add_argument("--max-units", type=int, default=50, 
                       help="Maximum number of units to plot (default: 50)")
    parser.add_argument("--splits", nargs="+", default=["test"], 
                       choices=["train", "val", "test"],
                       help="Data splits to analyze (default: test)")
    parser.add_argument("--smooth-sigma", type=float, default=2.0,
                       help="Gaussian smoothing sigma for PSTHs (default: 2.0)")
    parser.add_argument("--min-trials", type=int, default=5,
                       help="Minimum trials per condition (default: 5)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: same as model)")
    return parser.parse_args()


def load_dataset_from_config(config_path: str):
    """Load dataset from config file"""
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load dataset
    dataset_config = config['dataset']
    if dataset_config['type'] == 'ibl':
        dataset = IBLDataset(**dataset_config['params'])
        dataset.prepare()
    else:
        raise ValueError(f"Unknown dataset type: {dataset_config['type']}")
    
    return config, dataset


def compute_firing_rates(neural_data: np.ndarray, time_info: Dict) -> np.ndarray:
    """
    Convert spike counts to firing rates
    
    Args:
        neural_data: Shape (n_trials, n_timepoints, n_neurons) - spike counts
        time_info: Dictionary with bin_size information
        
    Returns:
        firing_rates: Shape (n_trials, n_timepoints, n_neurons) - in Hz
    """
    bin_size = time_info.get('bin_size', 0.01)  # seconds
    return neural_data / bin_size


def create_behavioral_conditions(behavior_data: Dict[str, np.ndarray], 
                                min_trials: int = 5) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Create behavioral condition masks
    
    Returns:
        Dict with condition categories and their masks
    """
    conditions = {}
    
    # Choice conditions
    if 'choice' in behavior_data:
        choice = behavior_data['choice']
        choice_conditions = {}
        
        left_mask = choice == -1
        right_mask = choice == 1
        
        if np.sum(left_mask) >= min_trials:
            choice_conditions['Left Choice'] = left_mask
        if np.sum(right_mask) >= min_trials:
            choice_conditions['Right Choice'] = right_mask
            
        if choice_conditions:
            conditions['choice'] = choice_conditions
    
    # Feedback conditions
    if 'feedback_type' in behavior_data:
        feedback = behavior_data['feedback_type']
        feedback_conditions = {}
        
        correct_mask = feedback == 1
        error_mask = feedback == -1
        
        if np.sum(correct_mask) >= min_trials:
            feedback_conditions['Correct'] = correct_mask
        if np.sum(error_mask) >= min_trials:
            feedback_conditions['Error'] = error_mask
            
        if feedback_conditions:
            conditions['feedback'] = feedback_conditions
    
    # Contrast conditions
    if 'stimulus_contrast_left' in behavior_data:
        contrast_left = behavior_data['stimulus_contrast_left']
        contrast_right = behavior_data['stimulus_contrast_right']
        
        # Create signed contrast
        contrast_combined = np.zeros_like(contrast_left)
        left_stim = ~np.isnan(contrast_left) & (contrast_left > 0)
        right_stim = ~np.isnan(contrast_right) & (contrast_right > 0)
        
        contrast_combined[left_stim] = -contrast_left[left_stim]
        contrast_combined[right_stim] = contrast_right[right_stim]
        
        contrast_conditions = {}
        
        # Bin contrasts
        unique_contrasts = np.unique(contrast_combined)
        unique_contrasts = unique_contrasts[~np.isnan(unique_contrasts)]
        
        for contrast in unique_contrasts:
            mask = contrast_combined == contrast
            if np.sum(mask) >= min_trials:
                if contrast == 0:
                    contrast_conditions['No Stimulus'] = mask
                elif contrast < 0:
                    contrast_conditions[f'Left {abs(contrast):.2f}'] = mask
                else:
                    contrast_conditions[f'Right {contrast:.2f}'] = mask
        
        if contrast_conditions:
            conditions['contrast'] = contrast_conditions
    
    # Reaction time conditions (tertile split)
    if 'reaction_time' in behavior_data:
        rt = behavior_data['reaction_time']
        valid_rt = rt[~np.isnan(rt)]
        
        if len(valid_rt) >= min_trials * 3:
            rt_conditions = {}
            
            # Create tertiles
            rt_33 = np.percentile(valid_rt, 33.3)
            rt_67 = np.percentile(valid_rt, 66.7)
            
            fast_mask = (rt <= rt_33) & ~np.isnan(rt)
            medium_mask = (rt > rt_33) & (rt <= rt_67) & ~np.isnan(rt)
            slow_mask = (rt > rt_67) & ~np.isnan(rt)
            
            if np.sum(fast_mask) >= min_trials:
                rt_conditions['Fast RT'] = fast_mask
            if np.sum(medium_mask) >= min_trials:
                rt_conditions['Medium RT'] = medium_mask
            if np.sum(slow_mask) >= min_trials:
                rt_conditions['Slow RT'] = slow_mask
                
            if rt_conditions:
                conditions['reaction_time'] = rt_conditions
    
    return conditions


def compute_unit_responsiveness(firing_rates: np.ndarray, 
                               conditions: Dict[str, Dict[str, np.ndarray]],
                               time_info: Dict) -> pd.DataFrame:
    """
    Compute responsiveness metrics for each unit
    
    Returns:
        DataFrame with responsiveness metrics per unit
    """
    n_trials, n_timepoints, n_units = firing_rates.shape
    
    results = []
    
    for unit_idx in range(n_units):
        unit_rates = firing_rates[:, :, unit_idx]
        
        # Basic statistics
        mean_rate = np.mean(unit_rates)
        max_rate = np.max(unit_rates)
        cv_rate = np.std(unit_rates) / (mean_rate + 1e-8)
        
        # Test for significant differences between conditions
        max_f_stat = 0
        max_condition_type = None
        significant_conditions = 0
        
        for condition_type, condition_dict in conditions.items():
            if len(condition_dict) < 2:
                continue
                
            # Collect data for each condition
            condition_data = []
            for condition_name, condition_mask in condition_dict.items():
                if np.sum(condition_mask) >= 3:  # Minimum for stats
                    condition_rates = unit_rates[condition_mask].flatten()
                    condition_data.append(condition_rates)
            
            if len(condition_data) >= 2:
                # One-way ANOVA
                try:
                    f_stat, p_val = stats.f_oneway(*condition_data)
                    
                    if p_val < 0.05:
                        significant_conditions += 1
                        
                    if f_stat > max_f_stat:
                        max_f_stat = f_stat
                        max_condition_type = condition_type
                        
                except:
                    continue
        
        # Temporal variance (how much does firing change over time)
        time_var = np.var(np.mean(unit_rates, axis=0))
        
        results.append({
            'unit': unit_idx,
            'mean_rate': mean_rate,
            'max_rate': max_rate,
            'cv_rate': cv_rate,
            'max_f_stat': max_f_stat,
            'max_condition_type': max_condition_type,
            'significant_conditions': significant_conditions,
            'temporal_variance': time_var,
            'responsiveness_score': max_f_stat * np.log(max_rate + 1) * np.log(time_var + 1)
        })
    
    return pd.DataFrame(results)


def plot_unit_psth(firing_rates: np.ndarray, 
                   unit_idx: int,
                   conditions: Dict[str, Dict[str, np.ndarray]],
                   time_info: Dict,
                   condition_type: str,
                   smooth_sigma: float = 2.0,
                   ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot PSTH for a single unit
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create time vector
    n_timepoints = firing_rates.shape[1]
    bin_size = time_info.get('bin_size', 0.01)
    pre_time = time_info.get('pre_time', 0.5)
    time_vector = np.arange(n_timepoints) * bin_size - pre_time
    
    unit_rates = firing_rates[:, :, unit_idx]
    
    if condition_type not in conditions:
        ax.text(0.5, 0.5, f'No {condition_type} conditions', 
               ha='center', va='center', transform=ax.transAxes)
        return ax
    
    condition_dict = conditions[condition_type]
    colors = plt.cm.tab10(np.linspace(0, 1, len(condition_dict)))
    
    max_rate = 0
    
    for i, (condition_name, condition_mask) in enumerate(condition_dict.items()):
        if np.sum(condition_mask) < 3:
            continue
            
        # Get rates for this condition
        condition_rates = unit_rates[condition_mask]  # (n_trials, n_timepoints)
        
        # Compute mean and SEM
        mean_rate = np.mean(condition_rates, axis=0)
        sem_rate = np.std(condition_rates, axis=0) / np.sqrt(len(condition_rates))
        
        # Apply smoothing
        if smooth_sigma > 0:
            mean_rate = gaussian_filter1d(mean_rate, sigma=smooth_sigma)
            sem_rate = gaussian_filter1d(sem_rate, sigma=smooth_sigma)
        
        max_rate = max(max_rate, np.max(mean_rate + sem_rate))
        
        # Plot
        color = colors[i]
        ax.plot(time_vector, mean_rate, color=color, linewidth=2, 
               label=f'{condition_name} (n={np.sum(condition_mask)})')
        ax.fill_between(time_vector, mean_rate - sem_rate, mean_rate + sem_rate,
                       color=color, alpha=0.3)
    
    # Formatting
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Stimulus Onset')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title(f'Unit {unit_idx} - {condition_type.replace("_", " ").title()}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max_rate * 1.1)
    
    return ax


def create_summary_plots(responsiveness_df: pd.DataFrame, 
                        output_dir: Path) -> None:
    """Create summary plots of unit responsiveness"""
    
    # 1. Responsiveness distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Responsiveness score distribution
    axes[0, 0].hist(responsiveness_df['responsiveness_score'], bins=30, alpha=0.7)
    axes[0, 0].set_xlabel('Responsiveness Score')
    axes[0, 0].set_ylabel('Number of Units')
    axes[0, 0].set_title('Distribution of Responsiveness Scores')
    
    # Mean firing rate vs responsiveness
    axes[0, 1].scatter(responsiveness_df['mean_rate'], 
                      responsiveness_df['responsiveness_score'], alpha=0.6)
    axes[0, 1].set_xlabel('Mean Firing Rate (Hz)')
    axes[0, 1].set_ylabel('Responsiveness Score')
    axes[0, 1].set_title('Firing Rate vs Responsiveness')
    
    # F-statistic distribution
    axes[1, 0].hist(responsiveness_df['max_f_stat'], bins=30, alpha=0.7)
    axes[1, 0].set_xlabel('Max F-statistic')
    axes[1, 0].set_ylabel('Number of Units')
    axes[1, 0].set_title('Distribution of F-statistics')
    
    # Significant conditions
    axes[1, 1].hist(responsiveness_df['significant_conditions'], 
                   bins=range(6), alpha=0.7, align='left')
    axes[1, 1].set_xlabel('Number of Significant Conditions')
    axes[1, 1].set_ylabel('Number of Units')
    axes[1, 1].set_title('Units with Significant Condition Effects')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'responsiveness_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Top responsive units by condition type
    condition_types = responsiveness_df['max_condition_type'].value_counts()
    
    if len(condition_types) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        condition_types.plot(kind='bar', ax=ax)
        ax.set_xlabel('Condition Type')
        ax.set_ylabel('Number of Most Responsive Units')
        ax.set_title('Primary Condition Type for Most Responsive Units')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'condition_preferences.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function"""
    
    args = parse_args()
    
    if not Path(args.config).exists():
        print(f"Config file not found: {args.config}")
        sys.exit(1)
    
    print(f"🧠 Analyzing neural unit PSTHs with config: {args.config}")
    print("=" * 80)
    
    try:
        # Load dataset
        config, dataset = load_dataset_from_config(args.config)
        
        # Set up output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            save_dir = config.get('save_dir', 'outputs')
            experiment_name = config['experiment']['name']
            session_id = config['experiment']['session_id']
            output_dir = Path(save_dir) / experiment_name / session_id / "psths"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split in args.splits:
            print(f"\n📊 Analyzing {split} split...")
            
            # Get data
            neural_data = dataset.get_neural_data(split)
            behavior_data = dataset.get_behavior_data(split)
            time_info = dataset.get_time_info(split)
            
            print(f"Neural data shape: {neural_data.shape}")
            print(f"Behavioral variables: {list(behavior_data.keys())}")
            
            # Convert to firing rates
            firing_rates = compute_firing_rates(neural_data, time_info)
            print(f"Mean firing rate: {np.mean(firing_rates):.2f} Hz")
            print(f"Max firing rate: {np.max(firing_rates):.2f} Hz")
            
            # Create behavioral conditions
            conditions = create_behavioral_conditions(behavior_data, args.min_trials)
            
            print(f"\nBehavioral conditions found:")
            for condition_type, condition_dict in conditions.items():
                print(f"  {condition_type}: {list(condition_dict.keys())}")
            
            # Compute responsiveness
            print("\nComputing unit responsiveness...")
            responsiveness_df = compute_unit_responsiveness(firing_rates, conditions, time_info)
            
            # Sort by responsiveness
            responsiveness_df = responsiveness_df.sort_values('responsiveness_score', ascending=False)
            
            # Print summary
            print(f"\n📈 RESPONSIVENESS SUMMARY ({split}):")
            print("-" * 50)
            print(f"Total units: {len(responsiveness_df)}")
            print(f"Units with any significant condition: {len(responsiveness_df[responsiveness_df['significant_conditions'] > 0])}")
            print(f"Mean responsiveness score: {responsiveness_df['responsiveness_score'].mean():.2f}")
            print(f"Mean firing rate: {responsiveness_df['mean_rate'].mean():.2f} Hz")
            
            # Top responsive units
            print(f"\n🏆 TOP 10 MOST RESPONSIVE UNITS ({split}):")
            print("-" * 70)
            top_units = responsiveness_df.head(10)
            
            for _, row in top_units.iterrows():
                print(f"Unit {row['unit']:3d} | Score: {row['responsiveness_score']:6.1f} | "
                      f"Rate: {row['mean_rate']:5.1f} Hz | F-stat: {row['max_f_stat']:5.1f} | "
                      f"Best: {row['max_condition_type']}")
            
            # Create summary plots
            split_output_dir = output_dir / split
            split_output_dir.mkdir(exist_ok=True)
            
            create_summary_plots(responsiveness_df, split_output_dir)
            
            # Plot PSTHs for top units
            n_units_to_plot = min(args.max_units, len(responsiveness_df))
            top_units_to_plot = responsiveness_df.head(n_units_to_plot)
            
            print(f"\n📈 Creating PSTH plots for top {n_units_to_plot} units...")
            
            # Create individual PSTH plots for each unit and condition type
            individual_psth_dir = split_output_dir / "individual_units"
            individual_psth_dir.mkdir(exist_ok=True)
            
            # Group by condition type for plotting
            condition_types = ['choice', 'feedback', 'contrast', 'reaction_time']
            
            # Plot individual PSTHs for each unit
            for _, unit_row in top_units_to_plot.iterrows():
                unit_idx = int(unit_row['unit'])
                
                for condition_type in condition_types:
                    if condition_type not in conditions:
                        continue
                    
                    # Create individual plot for this unit and condition
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plot_unit_psth(firing_rates, unit_idx, conditions, time_info,
                                 condition_type, args.smooth_sigma, ax)
                    
                    plt.tight_layout()
                    plt.savefig(individual_psth_dir / f'unit_{unit_idx:03d}_{condition_type}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
            
            # Also create summary grids as before
            for condition_type in condition_types:
                if condition_type not in conditions:
                    continue
                
                print(f"  Creating {condition_type} summary grid...")
                
                # Create subplot grid for summary
                units_for_condition = top_units_to_plot[
                    top_units_to_plot['max_condition_type'] == condition_type
                ].head(min(12, len(top_units_to_plot)))
                
                if len(units_for_condition) == 0:
                    units_for_condition = top_units_to_plot.head(min(12, len(top_units_to_plot)))
                
                n_plots = len(units_for_condition)
                if n_plots == 0:
                    continue
                
                n_cols = min(4, n_plots)
                n_rows = (n_plots + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                if n_plots == 1:
                    axes = [axes]
                elif n_rows == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()
                
                for i, (_, unit_row) in enumerate(units_for_condition.iterrows()):
                    if i >= len(axes):
                        break
                        
                    unit_idx = int(unit_row['unit'])
                    plot_unit_psth(firing_rates, unit_idx, conditions, time_info,
                                 condition_type, args.smooth_sigma, axes[i])
                
                # Hide empty subplots
                for i in range(len(units_for_condition), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                plt.savefig(split_output_dir / f'psth_{condition_type}_summary_grid.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            # Save responsiveness data
            responsiveness_df.to_csv(split_output_dir / 'unit_responsiveness.csv', index=False)
            
            print(f"📁 Results saved to: {split_output_dir}")
        
        print(f"\n✅ PSTH analysis complete!")
        print(f"📁 All results saved in: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()