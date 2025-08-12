"""
Usage example for CCA and RRR baselines in MRM
Run this script to train and compare all baseline models
"""

import json
import numpy as np
from pathlib import Path

# Example usage and configuration setup
def create_configs():
    """Create configuration files for all baselines"""
    
    configs = {
        'cca': {
            "model_type": "cca",
            "cache_path": "cache/cca_baseline",
            "model_params": {
                "n_components": 8,  # Number of canonical components
                "alpha": 0.0        # Ridge regularization (0 = no regularization)
            },
            "data_params": {
                "n_trials": 100,
                "noise_scale": 0.01,
                "source_model": "motor_rnn"
            }
        },
        
        'rrr': {
            "model_type": "rrr", 
            "cache_path": "cache/rrr_baseline",
            "model_params": {
                "rank": 10,         # Reduced rank constraint
                "alpha": 1.0,       # Ridge regularization strength
                "delay": 1          # Time delay (0 = no delay)
            },
            "data_params": {
                "n_trials": 100,
                "noise_scale": 0.01,
                "source_model": "motor_rnn"
            }
        },
        
        'rrr_nodelay': {
            "model_type": "rrr",
            "cache_path": "cache/rrr_nodelay",
            "model_params": {
                "rank": 10,
                "alpha": 0.5,
                "delay": 0          # Compare with no delay
            },
            "data_params": {
                "n_trials": 100,
                "noise_scale": 0.01,
                "source_model": "motor_rnn"
            }
        }
    }
    
    # Save configs
    Path("configs").mkdir(exist_ok=True)
    for name, config in configs.items():
        with open(f"configs/{name}.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    print("Created configuration files:")
    for name in configs.keys():
        print(f"  configs/{name}.json")

def quick_demo():
    """Quick demonstration of baseline usage"""
    
    print("="*60)
    print("MRM BASELINE MODELS DEMO")
    print("="*60)
    
    # Create sample data
    n_trials, n_time, n_source, n_target = 50, 100, 32, 24
    source_data = np.random.randn(n_trials, n_time, n_source)
    target_data = np.random.randn(n_trials, n_time, n_target)
    
    print(f"Demo data shape: {source_data.shape} -> {target_data.shape}")
    
    # Test CCA
    print("\n1. Testing CCA baseline...")
    from models.cca_baseline import CCABaseline
    
    cca = CCABaseline({'n_components': 5, 'alpha': 0.0})
    cca.fit(source_data, target_data)
    pred_cca = cca.predict(source_data)
    corrs = cca.get_canonical_correlations()
    
    print(f"   CCA canonical correlations: {[f'{c:.3f}' for c in corrs[:3]]}")
    print(f"   Prediction shape: {pred_cca.shape}")
    
    # Test RRR
    print("\n2. Testing RRR baseline...")
    from models.rrr_baseline import RRRBaseline
    
    rrr = RRRBaseline({'rank': 8, 'alpha': 1.0, 'delay': 0})
    rrr.fit(source_data, target_data)
    pred_rrr = rrr.predict(source_data)
    exp_var = rrr.get_explained_variance_ratio()
    
    print(f"   RRR explained variance: {[f'{v:.3f}' for v in exp_var[:3]]}")
    print(f"   Total explained variance: {np.sum(exp_var):.3f}")
    print(f"   Prediction shape: {pred_rrr.shape}")
    
    # Compare performance
    mse_cca = np.mean((target_data - pred_cca)**2)
    mse_rrr = np.mean((target_data - pred_rrr)**2)
    
    print(f"\n3. Performance comparison on random data:")
    print(f"   CCA MSE: {mse_cca:.4f}")
    print(f"   RRR MSE: {mse_rrr:.4f}")
    print(f"   Better model: {'CCA' if mse_cca < mse_rrr else 'RRR'}")

def run_full_pipeline():
    """Run the complete baseline comparison pipeline"""
    
    print("\n" + "="*60)
    print("RUNNING FULL BASELINE PIPELINE")
    print("="*60)
    
    # Step 1: Create configs
    print("\nStep 1: Creating configuration files...")
    create_configs()
    
    # Step 2: Train models (would need data loading functions)
    print("\nStep 2: To train models, run:")
    print("  python train_baselines.py")
    
    # Step 3: Compare models
    print("\nStep 3: To compare models, run:")
    print("  python model_comparison.py")
    
    print("\nWorkflow summary:")
    print("1. Create configs (✓ Done)")
    print("2. Train all baselines")
    print("3. Compare performance")
    print("4. Generate reports and plots")

# Hyperparameter suggestions
def get_hyperparameter_suggestions():
    """Get hyperparameter tuning suggestions"""
    
    suggestions = {
        'CCA': {
            'n_components': [3, 5, 8, 10, 15],
            'alpha': [0.0, 0.01, 0.1, 1.0],
            'notes': 'Start with n_components = min(n_source, n_target) // 2'
        },
        
        'RRR': {
            'rank': [5, 10, 15, 20],
            'alpha': [0.1, 1.0, 10.0, 100.0],
            'delay': [0, 1, 2, 3],
            'notes': 'Delay should reflect expected neural timing'
        }
    }
    
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING SUGGESTIONS")
    print("="*50)
    
    for model, params in suggestions.items():
        print(f"\n{model}:")
        for param, values in params.items():
            if param != 'notes':
                print(f"  {param}: {values}")
        if 'notes' in params:
            print(f"  Notes: {params['notes']}")

if __name__ == '__main__':
    # Run quick demo with synthetic data
    quick_demo()
    
    # Show full pipeline
    run_full_pipeline()
    
    # Show hyperparameter suggestions
    get_hyperparameter_suggestions()
    
    print(f"\n{'='*60}")
    print("Demo complete! Check the generated config files and run the training scripts.")
    print(f"{'='*60}")