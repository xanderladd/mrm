"""Updated training script with CCA and RRR baselines"""
import json
import os
import pickle
import numpy as np
from models.cca_baseline import fit_cca
from models.rrr_baseline import fit_rrr

def save_baseline_model(model, cache_path, metadata):
    """Save baseline model (CCA/RRR) using pickle"""
    os.makedirs(cache_path, exist_ok=True)
    
    # Save model
    with open(os.path.join(cache_path, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    # Save metadata
    with open(os.path.join(cache_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

def load_baseline_model(cache_path):
    """Load baseline model from cache"""
    model_path = os.path.join(cache_path, 'model.pkl')
    metadata_path = os.path.join(cache_path, 'metadata.json')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return model, metadata

def train_baseline_models(config_dir="configs/"):
    """Train all baseline models"""
    
    configs = {
        'cca': f"{config_dir}/cca.json",
        'rrr': f"{config_dir}/rrr.json", 
        'mp_rslds': f"{config_dir}/mp_rslds.json"
    }
    
    results = {}
    
    for model_name, config_path in configs.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()} baseline...")
        print(f"{'='*60}")
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Train model
        if config['model_type'] == 'cca':
            model, metadata = fit_cca(config)
            save_baseline_model(model, config['cache_path'], metadata)
            
            print(f"CCA Results:")
            print(f"  MSE: {metadata['mse']:.4f}")
            print(f"  R²: {metadata['r2']:.3f}")
            print(f"  Mean Canonical Correlation: {metadata['mean_correlation']:.3f}")
            
        elif config['model_type'] == 'rrr':
            model, metadata = fit_rrr(config)
            save_baseline_model(model, config['cache_path'], metadata)
            
            print(f"RRR Results:")
            print(f"  MSE: {metadata['mse']:.4f}")
            print(f"  R²: {metadata['r2']:.3f}")
            print(f"  Total Explained Variance: {metadata['total_explained_variance']:.3f}")
            print(f"  Effective Rank: {metadata['effective_rank']}")
            
        elif config['model_type'] == 'mp_rslds':
           raise NotImplementedError
        
        results[model_name] = metadata
        print(f"Model saved to: {config['cache_path']}")
    
    return results

def compare_baselines(results):
    """Compare baseline model performance"""
    print(f"\n{'='*80}")
    print("BASELINE MODEL COMPARISON")
    print(f"{'='*80}")
    
    print(f"{'Model':<12} {'MSE':<10} {'R²':<8} {'Special Metric':<20}")
    print(f"{'-'*50}")
    
    for model_name, metadata in results.items():
        mse = metadata['mse']
        r2 = metadata['r2']
        
        if model_name == 'cca':
            special = f"Corr: {metadata['mean_correlation']:.3f}"
        elif model_name == 'rrr':
            special = f"ExpVar: {metadata['total_explained_variance']:.3f}"
        else:
            special = "N/A"
            
        print(f"{model_name:<12} {mse:<10.4f} {r2:<8.3f} {special:<20}")
    
    # Find best model by R²
    best_model = max(results.keys(), key=lambda k: results[k]['r2'])
    print(f"\nBest performing model: {best_model} (R² = {results[best_model]['r2']:.3f})")

def main():
    """Main training function"""
    # Train all baselines
    results = train_baseline_models()
    
    # Compare results
    compare_baselines(results)
    
    print(f"\n{'='*80}")
    print("Training complete! All models saved to their respective cache directories.")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()