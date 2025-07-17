#!/usr/bin/env python3
"""
Simple test script to debug LFADS encode/decode using config file
"""

import json
import numpy as np
from pathlib import Path
from mrm.trainer import load_trained_model

def test_model_from_config(config_path: str):
    """Test model encode/decode using config file to find model"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Find model directory from config
    experiment_name = config['experiment']['name']
    session_id = config['experiment']['session_id']
    model_type = config['model']['type']
    save_dir = config.get('save_dir', 'outputs')
    
    model_dir = Path(save_dir) / experiment_name / session_id / model_type
    
    print(f"=== TESTING MODEL FROM CONFIG ===")
    print(f"Config: {config_path}")
    print(f"Model dir: {model_dir}")
    print(f"Model type: {model_type}")
    
    # Check if model exists
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return
    
    model_weights = model_dir / "model_weights.pkl"
    if not model_weights.exists():
        print(f"❌ Model weights not found: {model_weights}")
        return
    
    # Load model
    print(f"\n--- LOADING MODEL ---")
    try:
        model = load_trained_model(str(model_dir))
        print(f"✓ Model loaded: {type(model)}")
        print(f"✓ Model fitted: {getattr(model, 'is_fitted', 'unknown')}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Create test data
    print(f"\n--- CREATING TEST DATA ---")
    test_data = np.random.randn(3, 20, 10)  # 3 trials, 20 timepoints, 10 neurons
    print(f"Test data shape: {test_data.shape}")
    print(f"Test data stats: min={test_data.min():.3f}, max={test_data.max():.3f}, mean={test_data.mean():.3f}")
    print(f"Test data sample (trial 0, first 3 timepoints, first 3 neurons):")
    print(test_data[0, :3, :3])
    
    # Test encode
    print(f"\n--- TESTING ENCODE ---")
    try:
        latents1 = model.encode(test_data)
        print(f"✓ Encode successful")
        print(f"Latents shape: {latents1.shape}")
        print(f"Latents stats: min={latents1.min():.3f}, max={latents1.max():.3f}, mean={latents1.mean():.3f}, std={latents1.std():.3f}")
        print(f"Latents sample:")
        if latents1.ndim == 3:
            print(latents1[0, :3, :min(3, latents1.shape[2])])
        else:
            print(latents1[:3, :min(3, latents1.shape[1])])
    except Exception as e:
        print(f"❌ Encode failed: {e}")
        return
    
    # Test encode determinism
    print(f"\n--- TESTING ENCODE DETERMINISM ---")
    try:
        latents2 = model.encode(test_data)
        is_deterministic = np.allclose(latents1, latents2)
        print(f"Encode deterministic: {is_deterministic}")
        if not is_deterministic:
            print(f"Max difference: {np.max(np.abs(latents1 - latents2))}")
    except Exception as e:
        print(f"❌ Second encode failed: {e}")
        return
    
    # Test decode
    print(f"\n--- TESTING DECODE ---")
    try:
        reconstructed1 = model.decode(latents1)
        print(f"✓ Decode successful")
        print(f"Reconstructed shape: {reconstructed1.shape}")
        print(f"Reconstructed stats: min={reconstructed1.min():.3f}, max={reconstructed1.max():.3f}, mean={reconstructed1.mean():.3f}")
        print(f"Reconstructed sample:")
        if reconstructed1.ndim == 3:
            print(reconstructed1[0, :3, :3])
        else:
            print(reconstructed1[:3, :3])
    except Exception as e:
        print(f"❌ Decode failed: {e}")
        return
    
    # Test decode determinism
    print(f"\n--- TESTING DECODE DETERMINISM ---")
    try:
        reconstructed2 = model.decode(latents1)
        is_deterministic = np.allclose(reconstructed1, reconstructed2)
        print(f"Decode deterministic: {is_deterministic}")
        if not is_deterministic:
            print(f"Max difference: {np.max(np.abs(reconstructed1 - reconstructed2))}")
    except Exception as e:
        print(f"❌ Second decode failed: {e}")
        return
    
    # Test with different input
    print(f"\n--- TESTING WITH DIFFERENT INPUT ---")
    test_data2 = np.random.randn(3, 20, 10)
    try:
        latents3 = model.encode(test_data2)
        different_inputs_different_latents = not np.allclose(latents1, latents3, rtol=1e-3)
        print(f"Different inputs → different latents: {different_inputs_different_latents}")
        if not different_inputs_different_latents:
            print(f"⚠️  WARNING: Same latents for different inputs!")
    except Exception as e:
        print(f"❌ Encode with different input failed: {e}")
        return
    
    # Sanity checks
    print(f"\n--- SANITY CHECKS ---")
    print(f"Reconstruction == input: {np.allclose(reconstructed1, test_data, rtol=1e-3)}")
    print(f"Reconstruction is constant: {np.allclose(reconstructed1, reconstructed1.flat[0])}")
    print(f"Reconstruction variance: {np.var(reconstructed1):.6f}")
    
    # Simple correlation test
    orig_flat = test_data.flatten()
    recon_flat = reconstructed1.flatten()
    if len(orig_flat) > 1:
        correlation = np.corrcoef(orig_flat, recon_flat)[0, 1]
        print(f"Test correlation: {correlation:.6f}")
    
    print(f"\n=== TEST COMPLETE ===")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python test_model_encode_decode.py <config_file.json>")
        print("Example: python test_model_encode_decode.py configs/lfads_cp_config.json")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not Path(config_path).exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    test_model_from_config(config_path)