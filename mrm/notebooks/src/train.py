"""Universal training script for all models with consistent interface"""
import argparse
import json
import os
import torch
import numpy as np

from utils import (load_config, create_model, load_cached_model, save_model, 
                   extract_trajectories, plot_training_curves)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    parser = argparse.ArgumentParser(description='Train neural dynamics models')
    parser.add_argument('config', type=str, help='Path to config JSON file')
    parser.add_argument('--force', action='store_true', help='Force retrain even if cached')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    print(f"\nTraining {config['model_type']} model...")
    print(f"Cache path: {config['cache_path']}")
    print("-" * 50)
    
    # Check cache
    if not args.force:
        if config['model_type'] == 'mp_rslds':
            # MP-rSLDS uses pickle format
            import os
            if os.path.exists(os.path.join(config['cache_path'], 'metadata.json')):
                print(f"Found cached {config['model_type']} model")
                with open(os.path.join(config['cache_path'], 'metadata.json'), 'r') as f:
                    metadata = json.load(f)
                print(f"Cached performance: MSE={metadata.get('mse', 'N/A'):.4f}, R²={metadata.get('r2', 'N/A'):.3f}")
                return
        else:
            model, metadata, loaded = load_cached_model(config['cache_path'])
            if loaded:
                print(f"Found cached {config['model_type']} model")
                if metadata and 'final_mse' in metadata:
                    print(f"Cached performance: MSE={metadata['final_mse']:.4f}, R²={metadata['final_r2']:.3f}")
                elif metadata and 'mse' in metadata:
                    print(f"Cached performance: MSE={metadata['mse']:.4f}, R²={metadata['r2']:.3f}")
                return
    
    # Create model
    model = create_model(config)
    print(f"Created {config['model_type']} model")
    
    # Prepare training data
    train_data = None
    region_info = None
    
    if config['model_type'] != 'motor_rnn':
        # Load motor_rnn to extract trajectories
        print("Loading Motor RNN to extract training trajectories...")
        motor_rnn_config = load_config('configs/motor_rnn.json')
        motor_rnn, _, loaded = load_cached_model(motor_rnn_config['cache_path'], 
                                                 create_model(motor_rnn_config).__class__, 
                                                 motor_rnn_config)
        
        if not loaded:
            raise ValueError("Motor RNN must be trained first! Run: python train.py configs/motor_rnn.json")
        
        # Extract trajectories
        trajectories = extract_trajectories(motor_rnn, 
                                           n_trials=config['data_params']['n_trials'],
                                           noise_scale=config['data_params'].get('noise_scale', 0.01))
        
        # Format training data
        train_data = {
            'region_1': trajectories['evidence_states'],  # evidence -> region_1
            'region_2': trajectories['motor_states']      # motor -> region_2
        }
        
        print(f"Extracted {len(trajectories['trial_info'])} training trajectories")
        print(f"Region 1 shape: {train_data['region_1'].shape}")
        print(f"Region 2 shape: {train_data['region_2'].shape}")
        
        # Set region info for unidirectional models
        if config['model_type'] in ['cca', 'rrr']:
            region_info = {'source': 'region_1', 'target': 'region_2'}
    
    # Train model
    print(f"Training {config['model_type']} model...")

    if region_info is not None:
        metadata = model.fit(train_data, region_info=region_info, **config['training_params'])
    else:
        metadata = model.fit(train_data, **config['training_params'])
    
    # Add config to metadata
    metadata['config'] = config
    
    # Save model
    save_model(model, config['cache_path'], metadata)
    
    # Print results
    print("\n" + "="*50)
    print(f"Training complete!")
    if 'final_mse' in metadata:
        print(f"Final MSE: {metadata['final_mse']:.4f}")
        print(f"Final R²: {metadata['final_r2']:.3f}")
    elif 'mse' in metadata:
        print(f"Final MSE: {metadata['mse']:.4f}")
        print(f"Final R²: {metadata['r2']:.3f}")
    
    # Model-specific metrics
    if config['model_type'] == 'cca':
        print(f"Mean Canonical Correlation: {metadata['mean_correlation']:.3f}")
    elif config['model_type'] == 'rrr':
        print(f"Total Explained Variance: {metadata['total_explained_variance']:.3f}")
        print(f"Effective Rank: {metadata['effective_rank']}")
    
    print("="*50)
    print(f"\nModel saved to {config['cache_path']}")
    
    # Plot training curves if available
    if 'losses' in metadata:
        plot_training_curves({'Total Loss': metadata['losses']})
    
    # Plot MSE and R2 history if available
    if 'train_mse' in metadata:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        ax1.plot(metadata['train_mse'])
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE')
        ax1.set_title('Training MSE')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(metadata['train_r2'])
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('R²')
        ax2.set_title('Training R²')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{config['cache_path']}/metrics.png")
        plt.show()

if __name__ == '__main__':
    main()