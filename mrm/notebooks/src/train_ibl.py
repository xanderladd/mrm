#!/usr/bin/env python
"""
Train MR-gNODE on IBL neural data with multi-region support
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Import your modules
from ibl_dataset import IBLDataset
from models.mr_gnode import MRgnODE_DynamicComm
from utils import compute_r2, save_model, load_cached_model


def assign_neurons_to_regions(dataset: IBLDataset, region_mapping: Dict[str, List[str]]) -> Dict:
    """
    Assign neurons to regions based on brain area mappings (optimized version)
    
    Args:
        dataset: Prepared IBLDataset with cluster_brain_regions attribute
        region_mapping: Dict mapping region names to brain area lists
            e.g., {'region_1': ['MOs'], 'region_2': ['MOp']}
    
    Returns:
        Dict with neuron assignments and metadata
    """
    import numpy as np
    
    if not dataset.is_prepared:
        raise ValueError("Dataset must be prepared before assigning neurons to regions")
    
    # Check if dataset has brain region info stored
    if not hasattr(dataset, 'cluster_brain_regions'):
        # Fall back to reloading (use the full version from previous artifact)
        print("Warning: Dataset doesn't have cluster_brain_regions stored. Consider updating IBLDataset class.")
        # Would call the full version here
        raise NotImplementedError("Dataset needs cluster_brain_regions attribute. Update IBLDataset._load_raw_data()")
    
    neuron_brain_regions = dataset.cluster_brain_regions
    
    # Initialize neuron assignments
    neuron_assignments = {}
    assigned_neurons = set()
    
    # Assign neurons to regions based on mapping
    for region_name, target_areas in region_mapping.items():
        neuron_indices = []
        actual_areas = set()
        
        for i, brain_region in enumerate(neuron_brain_regions):
            if i in assigned_neurons:
                continue  # Skip already assigned neurons
                
            # Check if neuron's brain region matches any target area
            for target_area in target_areas:
                # Hierarchical matching - "MOs" matches "MOs", "MOs1", "MOs2", etc.
                if brain_region.startswith(target_area):
                    neuron_indices.append(i)
                    assigned_neurons.add(i)
                    actual_areas.add(brain_region)
                    break
        
        neuron_assignments[region_name] = {
            'neuron_indices': neuron_indices,
            'brain_areas': target_areas,
            'actual_areas': list(actual_areas),
            'n_neurons': len(neuron_indices)
        }
    
    # Handle unassigned neurons
    unassigned = [i for i in range(len(neuron_brain_regions)) if i not in assigned_neurons]
    if unassigned:
        print(f"Warning: {len(unassigned)} neurons not assigned to any region")
        unassigned_areas = set([neuron_brain_regions[i] for i in unassigned])
        print(f"  Unassigned brain areas: {unassigned_areas}")
    
    # Validate and report
    total_assigned = sum(info['n_neurons'] for info in neuron_assignments.values())
    total_neurons = dataset.aligned_neural_data.shape[-1]
    
    print(f"\nNeuron assignments:")
    for region_name, info in neuron_assignments.items():
        print(f"  {region_name}: {info['n_neurons']} neurons")
        print(f"    Target areas: {info['brain_areas']}")
        print(f"    Actual areas found: {info['actual_areas']}")
    
    if total_assigned == 0:
        raise ValueError("No neurons assigned to any region! Check region_mapping and available brain areas.")
    
    # Add metadata
    neuron_assignments['_metadata'] = {
        'total_neurons': total_neurons,
        'assigned_neurons': total_assigned,
        'unassigned_neurons': len(unassigned),
        'all_brain_regions': list(set(neuron_brain_regions))
    }
    
    return neuron_assignments


def prepare_region_data(dataset: IBLDataset, 
                        neuron_assignments: Dict,
                        split: str = 'train') -> Dict[str, np.ndarray]:
    """
    Extract region-specific neural data
    
    Args:
        dataset: Prepared IBLDataset
        neuron_assignments: Neuron to region mapping
        split: Data split to use
    
    Returns:
        Dict with region data
    """
    # Get full neural data
    neural_data = dataset.get_neural_data(split)  # [trials, time, neurons]
    
    # Split by regions
    region_data = {}
    for region_name, info in neuron_assignments.items():
        if region_name == "_metadata": continue
        indices = info['neuron_indices']
        region_data[region_name] = neural_data[..., indices]

    return region_data


def cache_ibl_data(cache_path: str, 
                   dataset: IBLDataset,
                   neuron_assignments: Dict,
                   config: Dict) -> Dict:
    """
    Cache processed IBL data for faster loading
    
    Args:
        cache_path: Path to cache directory
        dataset: Prepared IBLDataset
        neuron_assignments: Neuron to region mapping
        config: Configuration dict
    
    Returns:
        Dict with all cached data
    """
    cache_file = Path(cache_path) / 'ibl_processed_data.pkl'
    
    # Check if cache exists and is valid
    if cache_file.exists():
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            
        # Validate cache
        if (cached_data.get('config_hash') == hash(json.dumps(config, sort_keys=True)) and
            cached_data.get('eid') == dataset.eid):
            print("✓ Cache is valid")
            return cached_data
        else:
            print("✗ Cache is outdated, reprocessing...")
    
    print("Processing IBL data for caching...")
    
    # Prepare data for all splits
    processed_data = {
        'train': {},
        'val': {},
        'test': {}
    }
    
    for split in ['train', 'val', 'test']:
        # Get region data
        region_data = prepare_region_data(dataset, neuron_assignments, split)
        
        # Store with consistent naming for MR-gNODE
        for i, (region_name, data) in enumerate(region_data.items()):
            processed_data[split][f'region_{i+1}'] = data
        
        # Also store behavioral data if needed
        processed_data[split]['behavior'] = dataset.get_behavior_data(split)
        processed_data[split]['trial_info'] = dataset.get_trial_info(split)
    
    # Create cache data structure
    cached_data = {
        'processed_data': processed_data,
        'neuron_assignments': neuron_assignments,
        'time_info': dataset.get_time_info('train'),
        'dataset_config': dataset.get_config(),
        'config_hash': hash(json.dumps(config, sort_keys=True)),
        'eid': dataset.eid,
        'cache_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'data_shapes': {
            split: {k: v.shape for k, v in data.items() if isinstance(v, np.ndarray)}
            for split, data in processed_data.items()
        }
    }
    
    # Save cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)
    
    print(f"✓ Cached processed data to {cache_file}")
    
    # Print summary
    print("\nData shapes:")
    for split in ['train', 'val', 'test']:
        print(f"  {split}:")
        for region in [k for k in processed_data[split].keys() if k.startswith('region_')]:
            shape = processed_data[split][region].shape
            print(f"    {region}: {shape}")
    
    return cached_data


def create_model_from_config(config: Dict, cached_data: Dict) -> MRgnODE_DynamicComm:
    """
    Create MR-gNODE model with appropriate dimensions
    
    Args:
        config: Model configuration
        cached_data: Cached data with dimension info
    
    Returns:
        MRgnODE_DynamicComm model
    """
    # Extract dimensions from cached data
    train_data = cached_data['processed_data']['train']
    region_dims_actual = []
    
    for i in range(1, 10):  # Check up to 10 regions
        key = f'region_{i}'
        if key in train_data:
            region_dims_actual.append(train_data[key].shape[-1])
        else:
            break
    
    num_regions = len(region_dims_actual)
    
    # Get model params from config
    model_params = config.get('model_params', {})
    
    # Set variable dimensions
    model_params['region_dims'] = model_params.get('region_dims', region_dims_actual)
    model_params['input_dims'] = model_params.get('input_dims', region_dims_actual)
    model_params['output_dims'] = model_params.get('output_dims', region_dims_actual)
    model_params['num_regions'] = num_regions
    
    # Use defaults for other params if not specified
    model_params.setdefault('hidden_dim', 64)
    model_params.setdefault('comm_dim', 32)
    model_params.setdefault('tau_regions', 0.01)
    model_params.setdefault('tau_comm', 0.01)
    
    print(f"\nCreating MR-gNODE model:")
    print(f"  Regions: {num_regions}")
    print(f"  Region dimensions: {model_params['region_dims']}")
    print(f"  Input dimensions: {model_params['input_dims']}")
    print(f"  Communication dim: {model_params['comm_dim']}")
    print(f"  Hidden dim: {model_params['hidden_dim']}")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MRgnODE_DynamicComm(**model_params).to(device)
    
    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def train_model(model: MRgnODE_DynamicComm, 
                cached_data: Dict,
                training_params: Dict) -> Dict:
    """
    Train MR-gNODE model on IBL data
    
    Args:
        model: MR-gNODE model
        cached_data: Cached processed data
        training_params: Training hyperparameters
    
    Returns:
        Training metadata
    """
    # Get training data
    train_data = cached_data['processed_data']['train']
    val_data = cached_data['processed_data']['val']
    
    # Filter to just region data
    train_regions = {k: v for k, v in train_data.items() if k.startswith('region_')}
    val_regions = {k: v for k, v in val_data.items() if k.startswith('region_')}
    
    print(f"\nTraining MR-gNODE model...")
    print(f"  Training samples: {list(train_regions.values())[0].shape[0]}")
    print(f"  Validation samples: {list(val_regions.values())[0].shape[0]}")
    print(f"  Sequence length: {list(train_regions.values())[0].shape[1]}")
    # Train model
    metadata = model.fit(train_regions, **training_params)
    
    # Add validation metrics
    print("\nComputing validation metrics...")
    with torch.no_grad():
        model.eval()
        val_preds = model.predict(val_regions)
        
        # Compute validation MSE and R2
        # Add this case before the existing if/else
        if isinstance(val_preds, dict):
            # Dict format - concatenate values
            val_preds_concat = np.concatenate(list(val_preds.values()), axis=-1)
            val_targets_concat = np.concatenate(list(val_regions.values()), axis=-1)
        elif isinstance(val_preds, list):
            # Variable dims - concatenate for metrics
            val_preds_concat = np.concatenate(val_preds, axis=-1)
            val_targets_concat = np.concatenate([val_regions[f'region_{i+1}'] 
                                                for i in range(len(val_preds))], axis=-1)
        else:
            val_preds_concat = val_preds
            val_targets_concat = np.concatenate(list(val_regions.values()), axis=-1)
            
        val_mse = np.mean((val_preds_concat - val_targets_concat) ** 2)
        val_r2 = compute_r2(val_preds_concat.flatten(), val_targets_concat.flatten())
        
        metadata['val_mse'] = float(val_mse)
        metadata['val_r2'] = float(val_r2)
        
        print(f"  Validation MSE: {val_mse:.4f}")
        print(f"  Validation R²: {val_r2:.3f}")
    
    # Add data info to metadata
    metadata['data_info'] = {
        'eid': cached_data['eid'],
        'n_regions': len([k for k in train_regions.keys() if k.startswith('region_')]),
        'region_dims': [v.shape[-1] for k, v in train_regions.items() if k.startswith('region_')],
        'n_train_trials': list(train_regions.values())[0].shape[0],
        'n_val_trials': list(val_regions.values())[0].shape[0],
        'sequence_length': list(train_regions.values())[0].shape[1],
        'neuron_assignments': cached_data['neuron_assignments']
    }
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Train MR-gNODE on IBL data')
    parser.add_argument('config', type=str, help='Path to config JSON file')
    parser.add_argument('--force-data', action='store_true', help='Force reload data (ignore cache)')
    parser.add_argument('--force-train', action='store_true', help='Force retrain model')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    print(f"IBL Multi-Region Training")
    print(f"=" * 50)
    print(f"Config: {args.config}")
    print(f"Session: {config['ibl_params']['eid']}")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Create cache directory
    cache_path = config.get('cache_path', f"cache/ibl_{config['ibl_params']['eid']}")
    Path(cache_path).mkdir(parents=True, exist_ok=True)
    
    # Check for existing model
    model_path = Path(cache_path) / 'model.pt'
    metadata_path = Path(cache_path) / 'metadata.json'
    
    if model_path.exists() and metadata_path.exists() and not args.force_train:
        print(f"\n✓ Found trained model at {model_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"  Final MSE: {metadata.get('final_mse', 'N/A'):.4f}")
        print(f"  Final R²: {metadata.get('final_r2', 'N/A'):.3f}")
        print(f"  Val MSE: {metadata.get('val_mse', 'N/A'):.4f}")
        print(f"  Val R²: {metadata.get('val_r2', 'N/A'):.3f}")
        
        if not args.force_train:
            print("\nUse --force-train to retrain")
            return
    
    # Load or prepare IBL dataset
    print(f"\n{'='*50}")
    print("Loading IBL Dataset...")
    
    dataset = IBLDataset(**config['ibl_params'])
    
    # Check if we should use cached data
    cache_file = Path(cache_path) / 'ibl_processed_data.pkl'
    if cache_file.exists() and not args.force_data:
        # Try loading cached data first
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Validate it matches our config
            if (cached_data.get('eid') == config['ibl_params']['eid'] and
                cached_data.get('config_hash') == hash(json.dumps(config, sort_keys=True))):
                print("✓ Using cached processed data")
                
                # Still need to prepare dataset for metadata
                if not dataset.is_prepared:
                    dataset.prepare()
            else:
                print("Cache outdated, reprocessing...")
                dataset.prepare()
                neuron_assignments = assign_neurons_to_regions(dataset, config['region_mapping'])
                cached_data = cache_ibl_data(cache_path, dataset, neuron_assignments, config)
        except:
            print("Cache loading failed, reprocessing...")
            dataset.prepare()
            neuron_assignments = assign_neurons_to_regions(dataset, config['region_mapping'])
            cached_data = cache_ibl_data(cache_path, dataset, neuron_assignments, config)
    else:
        # Prepare dataset and cache
        dataset.prepare()
        neuron_assignments = assign_neurons_to_regions(dataset, config['region_mapping'])
        cached_data = cache_ibl_data(cache_path, dataset, neuron_assignments, config)
    
    # Create model
    print(f"\n{'='*50}")
    model = create_model_from_config(config, cached_data)
    model = model.to(device)
    
    # Train model
    print(f"\n{'='*50}")
    training_params = config.get('training_params', {})
    training_params.setdefault('epochs', 150)
    training_params.setdefault('batch_size', 32)
    training_params.setdefault('lr', 0.001)
    training_params.setdefault('comm_penalty', 0.01)
    training_params.setdefault('clip_grad_norm', 10.0)
    
    metadata = train_model(model, cached_data, training_params)
    
    # Save model and metadata
    print(f"\n{'='*50}")
    print("Saving model...")
    
    torch.save(model.state_dict(), model_path)
    
    # Add config to metadata
    metadata['config'] = config
    metadata['training_completed'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Model saved to {model_path}")
    print(f"✓ Metadata saved to {metadata_path}")
    
    # Print final summary
    print(f"\n{'='*50}")
    print("Training Complete!")
    print(f"  Final Training MSE: {metadata['final_mse']:.4f}")
    print(f"  Final Training R²: {metadata['final_r2']:.3f}")
    print(f"  Validation MSE: {metadata['val_mse']:.4f}")
    print(f"  Validation R²: {metadata['val_r2']:.3f}")
    
    # Save training curves if available
    if 'losses' in metadata:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Loss curve
        axes[0].plot(metadata['losses'])
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].grid(True, alpha=0.3)
        
        # MSE curve
        axes[1].plot(metadata['train_mse'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE')
        axes[1].set_title('Training MSE')
        axes[1].grid(True, alpha=0.3)
        
        # R2 curve
        axes[2].plot(metadata['train_r2'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('R²')
        axes[2].set_title('Training R²')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(Path(cache_path) / 'training_curves.png')
        print(f"✓ Training curves saved to {Path(cache_path) / 'training_curves.png'}")


if __name__ == '__main__':
    main()