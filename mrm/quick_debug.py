#!/usr/bin/env python3
"""
Quick LFADS Debugging Script

Minimal script to identify why evaluate_reconstruction gives identical results.
This script focuses on the most common causes of identical results.
"""

import numpy as np
import torch
import json
from pathlib import Path

def quick_lfads_debug(config_path):
    """
    Quick debugging function to identify identical reconstruction issues
    
    Args:
        config_path: Path to your LFADS config JSON
    """
    
    print("🔍 Quick LFADS Debug - Identifying Identical Results Issue")
    print("=" * 60)
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Config loaded: {config['experiment']['name']}")
    
    # Import your modules (adjust imports as needed)
    try:
        from mrm.models.lfads import LFADSModel
        # Import the actual dataset implementation your MRM repo uses
        # Look for the concrete implementation in your codebase
        print("✓ MRM LFADS model imported")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Skip dataset loading for now and create synthetic data for testing
    print("\n📚 Creating synthetic test data (skipping dataset loading)...")
    eid = config['dataset']['params']['eid']
    print(f"Config EID: {eid}")
    
    # Create minimal synthetic data that matches your expected format
    # This lets us test just the LFADS model without dataset issues
    n_trials = 20
    n_timepoints = int((config['dataset']['params']['post_time'] - config['dataset']['params']['pre_time']) / config['dataset']['params']['bin_size'])
    n_neurons = 50  # Small number for testing
    
    print(f"Creating synthetic data: {n_trials} trials, {n_timepoints} timepoints, {n_neurons} neurons")
    
    # Create realistic-looking spike data (Poisson-like)
    np.random.seed(42)
    train_data = np.random.poisson(2.0, size=(n_trials, n_timepoints, n_neurons)).astype(np.float32)
    
    # Create different test data (should give different reconstructions)
    np.random.seed(123)  # Different seed
    test_data = np.random.poisson(1.5, size=(10, n_timepoints, n_neurons)).astype(np.float32)
    
    print(f"Synthetic train data: {train_data.shape}")
    print(f"Synthetic test data: {test_data.shape}")
    
    print(f"Train shape: {train_data.shape}")
    print(f"Test shape: {test_data.shape}")
    
    # Create a simple dataset-like object for LFADS
    class SimpleDataset:
        def __init__(self, train_data, test_data):
            self.train_data = train_data
            self.test_data = test_data
            
        def get_neural_data(self, split):
            if split == 'train':
                return self.train_data
            elif split in ['val', 'test']:
                return self.test_data
            else:
                raise ValueError(f"Unknown split: {split}")
                
        def get_behavior_data(self, split):
            # Return empty dict for behavior data
            return {}
    
    # Create simple dataset object
    simple_dataset = SimpleDataset(train_data, test_data)
    
    # Create model with minimal parameters for quick testing
    print("\n🧠 Creating LFADS model...")
    model = None
    try:
        model = LFADSModel(
            latent_dim=8,  # Small for quick testing
            generator_dim=32,
            controller_dim=16,
            ic_dim=8,
            co_dim=4,
            learning_rate=0.01,
            max_epochs=3,  # Just enough to see if training works
            batch_size=8,
            external_inputs=[],
            reconstruction_type='poisson'
        )
        print("✓ LFADS model created successfully")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        print("This could be due to:")
        print("- Missing lfads-torch dependency")
        print("- Incompatible model parameters")
        print("- Issues with your LFADSModel implementation")
        return {
            'model_trained': False,
            'identical_reconstructions': None,
            'reconstruction_mse': None,
            'reconstruction_correlation': None,
            'error': f"Model creation failed: {e}"
        }
    
    if model is None:
        print("❌ Could not create model")
        return {
            'model_trained': False,
            'identical_reconstructions': None,
            'reconstruction_mse': None,
            'reconstruction_correlation': None,
            'error': "Model is None"
        }
    
    # Test 1: Check initial reconstructions (should be poor/random)
    print("\n🧪 Test 1: Initial reconstructions (before training)")
    
    # Get small sample - make sure we have data first
    if train_data is None:
        print("❌ No training data available for testing")
        return None
        
    sample_data = train_data[:2]  # Just 2 trials
    print(f"Testing with sample shape: {sample_data.shape}")
    
    try:
        # This should fail or give random results since model isn't trained
        initial_recon = model.encode(sample_data)
        print(f"⚠️  Model gave output before training - this might be a problem!")
        print(f"Initial encoding shape: {initial_recon.shape}")
    except Exception as e:
        print(f"✓ Model correctly failed before training: {e}")
    
    # Test 2: Train model briefly (or skip if issues)
    print("\n🏋️ Test 2: Brief training...")
    
    model_trained = False
    if 'simple_dataset' in locals():
        try:
            model.fit(simple_dataset)
            print("✓ Training completed")
            model_trained = True
        except Exception as e:
            print(f"❌ Training failed: {e}")
            print("This might be because your LFADS model expects a different dataset format.")
            print("Let's try to test the model without full training...")
            model_trained = False
    else:
        print("⚠️  No dataset available - skipping training")
        print("   (Will test model initialization only)")
        model_trained = False
    
    # Test 3: Check if we can at least do encode/decode (even without training)
    print("\n🔄 Test 3: Model encode/decode test")
    
    # Get small sample for testing
    sample_data = train_data[:2]  # Just 2 trials
    
    try:
        if model_trained:
            # Test with trained model
            recon1 = model.decode(model.encode(sample_data))
            recon2 = model.decode(model.encode(sample_data))
            
            if np.allclose(recon1, recon2, atol=1e-10):
                print("✓ Reconstructions are identical between calls (deterministic - good)")
            else:
                print("⚠️  Reconstructions differ between calls (stochastic - check if intended)")
                print(f"Max difference: {np.max(np.abs(recon1 - recon2))}")
        else:
            # Test model architecture without training
            print("Testing model architecture without training...")
            
            # Try to get the underlying model architecture
            if hasattr(model, 'lfads_model'):
                print(f"✓ Model has lfads_model attribute")
                if model.lfads_model is not None:
                    print(f"✓ LFADS model is initialized")
                else:
                    print(f"⚠️  LFADS model is None")
            else:
                print(f"⚠️  Model missing lfads_model attribute")
                
            # Try basic encoding/decoding (might fail without training)
            try:
                encoded = model.encode(sample_data)
                print(f"✓ Encoding works, shape: {encoded.shape}")
                
                decoded = model.decode(encoded)
                print(f"✓ Decoding works, shape: {decoded.shape}")
                
                # This is the key test - different inputs should give different outputs
                different_sample = test_data[:2]
                encoded2 = model.encode(different_sample)
                decoded2 = model.decode(encoded2)
                
                if np.allclose(decoded, decoded2, atol=1e-6):
                    print("🚨 IDENTICAL OUTPUTS for different inputs (even without training)")
                    print("   This suggests a fundamental issue with the model")
                else:
                    print("✓ Different inputs give different outputs (good)")
                    
            except Exception as e:
                print(f"⚠️  Encode/decode failed: {e}")
                print("   This is expected if model needs training first")
                
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 5: Check model parameters
    print("\n⚙️  Test 5: Model parameter check")
    
    if hasattr(model, 'lfads_model') and model.lfads_model:
        param_count = sum(p.numel() for p in model.lfads_model.parameters())
        trainable_count = sum(p.numel() for p in model.lfads_model.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {param_count}")
        print(f"   Trainable parameters: {trainable_count}")
        
        # Check if parameters are reasonable
        param_std = []
        for name, param in model.lfads_model.named_parameters():
            if param.requires_grad:
                std = torch.std(param).item()
                param_std.append(std)
        
        avg_param_std = np.mean(param_std)
        print(f"   Average parameter std: {avg_param_std:.6f}")
        
        if avg_param_std < 1e-6:
            print("   🚨 Parameters have very low variance - model may not be training!")
        else:
            print("   ✓ Parameters have reasonable variance")
    
    # Test 5: Check model parameters (if accessible)
    print("\n⚙️  Test 5: Model parameter check")
    
    if model is None:
        print("   ⚠️  No model available for parameter check")
    else:
        try:
            if hasattr(model, 'lfads_model') and model.lfads_model:
                param_count = sum(p.numel() for p in model.lfads_model.parameters())
                trainable_count = sum(p.numel() for p in model.lfads_model.parameters() if p.requires_grad)
                
                print(f"   Total parameters: {param_count}")
                print(f"   Trainable parameters: {trainable_count}")
                
                # Check if parameters are reasonable
                param_std = []
                for name, param in model.lfads_model.named_parameters():
                    if param.requires_grad:
                        std = torch.std(param).item()
                        param_std.append(std)
                
                if param_std:  # Only if we have parameters
                    avg_param_std = np.mean(param_std)
                    print(f"   Average parameter std: {avg_param_std:.6f}")
                    
                    if avg_param_std < 1e-6:
                        print("   🚨 Parameters have very low variance - model may not be training!")
                    else:
                        print("   ✓ Parameters have reasonable variance")
                else:
                    print("   ⚠️  No trainable parameters found")
                    
            else:
                print("   ⚠️  Cannot access model parameters")
                
        except Exception as e:
            print(f"   ⚠️  Parameter check failed: {e}")
    
    # Test 6: Quick reconstruction quality (if model trained)
    if model_trained and model is not None:
        print("\n📈 Test 6: Reconstruction quality check")
        
        try:
            full_train_recon = model.decode(model.encode(train_data[:5]))  # First 5 trials
            
            # Simple metrics
            mse = np.mean((train_data[:5] - full_train_recon) ** 2)
            correlation = np.corrcoef(train_data[:5].flatten(), full_train_recon.flatten())[0, 1]
            
            print(f"   MSE: {mse:.6f}")
            print(f"   Correlation: {correlation:.4f}")
            
            if correlation < 0.1:
                print("   🚨 Very low correlation - model likely not learning!")
            elif correlation < 0.5:
                print("   ⚠️  Low correlation - may need more training")
            else:
                print("   ✓ Reasonable correlation")
                
            recon_mse = mse
            recon_corr = correlation
            
        except Exception as e:
            print(f"   ❌ Reconstruction quality test failed: {e}")
            recon_mse = None
            recon_corr = None
    else:
        print("\n📈 Test 6: Skipping reconstruction quality (model not trained or model is None)")
        recon_mse = None
        recon_corr = None
    
    # Summary and recommendations
    print("\n📋 SUMMARY AND RECOMMENDATIONS")
    print("=" * 40)
    
    if identical_results is True:
        print("🚨 IDENTICAL RESULTS CONFIRMED")
        print("\nMost likely causes:")
        print("1. Model not training properly (check training loss)")
        print("2. Caching issue - old model being loaded")
        print("3. Random seed fixed incorrectly")
        print("4. Model outputting constants/zeros")
        print("\nRecommended fixes:")
        print("- Set force_retrain=True in config")
        print("- Increase max_epochs to 50+")
        print("- Check training loss decreases")
        print("- Clear any cached model files")
        print("- Verify model architecture parameters")
        
    elif identical_results is False:
        print("✓ Model appears to be working correctly")
        print("The 'identical results' issue may be elsewhere in your pipeline")
        
        if model_trained and recon_corr is not None:
            if recon_corr < 0.3:
                print("⚠️  However, reconstruction quality is low - consider more training")
            else:
                print("✓ Reconstruction quality looks reasonable")
        
    else:
        print("⚠️  Could not complete full test due to model/training issues")
        print("\nThis suggests:")
        print("1. Dataset loading problems")
        print("2. Model initialization issues") 
        print("3. LFADS dependencies missing")
        print("\nNext steps:")
        print("- Check your dataset implementation")
        print("- Verify lfads-torch installation")
        print("- Try with a working LFADS config")
    
    return {
        'model_trained': model_trained,
        'identical_reconstructions': identical_results,
        'reconstruction_mse': recon_mse,
        'reconstruction_correlation': recon_corr,
    }

def main():
    """Run quick debug"""
    print("🔍 LFADS Quick Debug Tool")
    print("This script will help identify why evaluate_reconstruction gives identical results")
    print("It uses synthetic data to test your LFADS model implementation")
    print()
    
    config_path = input("Enter path to your LFADS config JSON (or press Enter for test config): ").strip()
    
    if not config_path:
        print("Creating minimal test config...")
        test_config = {
            "experiment": {"name": "debug_test"},
            "dataset": {
                "params": {
                    "eid": "ebce500b-c530-47de-8cb1-963c552703ea",
                    "alignment_event": "stimOn_times",
                    "pre_time": 0.1,
                    "post_time": 0.2,
                    "bin_size": 0.01
                }
            },
            "model": {
                "params": {
                    "latent_dim": 8,
                    "max_epochs": 3,
                    "learning_rate": 0.01
                }
            }
        }
        
        test_config_path = "test_debug_config.json"
        with open(test_config_path, 'w') as f:
            json.dump(test_config, f, indent=2)
        
        config_path = test_config_path
        print(f"Test config created: {test_config_path}")
    
    if not Path(config_path).exists():
        print(f"❌ Config file not found: {config_path}")
        return
    
    try:
        results = quick_lfads_debug(config_path)
        
        if results:
            print(f"\n✅ Debug completed!")
            
            if results['identical_reconstructions'] is True:
                print("🚨 CONFIRMED: Your model has the 'identical results' problem")
                print("   Check the recommendations above to fix it")
            elif results['identical_reconstructions'] is False:
                print("✅ Your model appears to be working correctly")
                print("   The issue might be elsewhere in your evaluation pipeline")
            else:
                print("⚠️  Could not fully test due to setup issues")
                print("   Check the error messages above for guidance")
        else:
            print("❌ Debug could not complete - check error messages above")
        
    except Exception as e:
        print(f"\n❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\n🔧 Common issues:")
        print(f"1. Make sure lfads-torch is installed: pip install git+https://github.com/arsedler9/lfads-torch.git")
        print(f"2. Ensure MRM repo is in your Python path")
        print(f"3. Check that your config JSON is valid")
        print(f"4. Try with a simpler LFADS configuration")

if __name__ == "__main__":
    main()