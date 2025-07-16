# trainer.py
"""
Trainer class for neural models with JSON-based configuration
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

from mrm.dataset import NeuralDataset
from mrm.models.dummy import DummyModel
from mrm.models.pca import PCAModel
from mrm.models.base import BaseModel
from mrm.models.lfads import LFADSModel


class NeuralTrainer:
    """Single trainer for all model types"""
    
    def __init__(self):
        self.model_registry = {}
        self._register_default_models()
        
    def _register_default_models(self):
        """Register default available model types"""
        self.register_model('dummy', DummyModel)
        self.register_model('pca', PCAModel)
        # TODO: Register actual models when implemented
        self.register_model('lfads', LFADSModel)
        # self.register_model('ssm', SSMModel)
        # self.register_model('gnode', GNODEModel)
        
    def register_model(self, model_name: str, model_class: type):
        """Register a new model type"""
        if not issubclass(model_class, BaseModel):
            raise ValueError(f"Model class must inherit from BaseNeuralModel")
        self.model_registry[model_name] = model_class
        print(f"Registered model: {model_name}")
        
    def train_from_config(self, config_path: str, dataset: Optional[NeuralDataset] = None) -> BaseModel:
        """Train a model using JSON configuration file"""
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        return self.train_from_dict(config, dataset)
        
    def train_from_dict(self, config: Dict[str, Any], dataset: Optional[NeuralDataset] = None) -> BaseModel:
        """Train a model using configuration dictionary"""
        
        # Validate configuration
        self._validate_config(config)
        
        # Create or use provided dataset
        if dataset is None:
            dataset = self._create_dataset_from_config(config['dataset'])
        
        if not dataset.is_prepared:
            dataset.prepare()
            
        # Create output directory structure
        output_dir = self._create_output_directory(config)
        
        # Instantiate model
        model_class = self.model_registry[config['model']['type']]
        model = model_class(**config['model']['params'])
        
        print(f"Training {config['model']['type']} model...")
        print(f"Output directory: {output_dir}")
        
        # Record training start time
        start_time = time.time()
        
        # Train model
        trained_model = model.fit(dataset)
        
        # Record training end time
        end_time = time.time()
        training_time = end_time - start_time
        
        # Save everything
        self._save_results(trained_model, dataset, config, output_dir, training_time)
        
        print(f"Training complete! Results saved to {output_dir}")
        print(f"Training time: {training_time:.2f} seconds")
        return trained_model
        
    def _validate_config(self, config: Dict[str, Any]):
        """Validate training configuration"""
        required_keys = ['experiment', 'dataset', 'model']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
                
        # Validate model type
        model_type = config['model']['type']
        if model_type not in self.model_registry:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available: {list(self.model_registry.keys())}")
                           
        # Validate experiment config
        exp_required = ['name', 'session_id']
        for key in exp_required:
            if key not in config['experiment']:
                raise ValueError(f"Missing required experiment config key: {key}")
                
    def _create_dataset_from_config(self, dataset_config: Dict[str, Any]) -> NeuralDataset:
        """Create dataset from configuration"""
        dataset_type = dataset_config.get('type', 'ibl')
        
        if dataset_type == 'ibl':
            from dataset import IBLDataset
            return IBLDataset(**dataset_config.get('params', {}))
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
            
    def _create_output_directory(self, config: Dict[str, Any]) -> Path:
        """Create output directory structure"""
        save_dir = config.get('save_dir', 'outputs')
        experiment_name = config['experiment']['name']
        session_id = config['experiment']['session_id']
        model_type = config['model']['type']
        
        output_dir = Path(save_dir) / experiment_name / session_id / model_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir
        
    def _save_results(self, model: BaseModel, dataset: NeuralDataset, 
                     config: Dict[str, Any], output_dir: Path, training_time: float):
        """Save training results"""
        
        # Save model
        model.save(str(output_dir / "model_weights.pkl"))
        
        # Save model configuration
        with open(output_dir / "model_config.json", 'w') as f:
            json.dump(model.get_config(), f, indent=2)
            
        # Save full training configuration
        with open(output_dir / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)
            
        # Save dataset configuration
        with open(output_dir / "dataset_config.json", 'w') as f:
            json.dump(dataset.get_config(), f, indent=2)
            
        # Save splits info
        with open(output_dir / "data_splits.json", 'w') as f:
            json.dump(dataset.get_splits(), f, indent=2)
            
        # Save training metadata
        metadata = {
            'training_time_seconds': training_time,
            'data_shapes': {
                'train': dataset.get_neural_data('train').shape,
                'val': dataset.get_neural_data('val').shape,
                'test': dataset.get_neural_data('test').shape
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(output_dir / "training_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Saved files:")
        for file_path in output_dir.iterdir():
            print(f"  - {file_path.name}")


def load_trained_model(model_dir: str) -> BaseModel:
    """Load a trained model from directory"""
    model_dir = Path(model_dir)
    
    # Load model config to determine type
    with open(model_dir / "model_config.json", 'r') as f:
        model_config = json.load(f)
        
    model_type = model_config['model_type']
    
    # Create trainer to access model registry
    trainer = NeuralTrainer()
    
    if model_type not in trainer.model_registry:
        raise ValueError(f"Unknown model type: {model_type}")
        
    # Load the model
    model_class = trainer.model_registry[model_type]
    model = model_class.load(str(model_dir / "model_weights.pkl"))
    
    print(f"Loaded {model_type} model from {model_dir}")
    return model


def main():
    """Main function for command line usage"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python trainer.py <config_file.json>")
        print("Example: python trainer.py configs/pca_config.json")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    print(f"Training model with config: {config_path}")
    
    # Create trainer and train model
    trainer = NeuralTrainer()
    
    try:
        model = trainer.train_from_config(config_path)
        print(f"\n🎉 Training completed successfully!")
        print(f"Model type: {type(model).__name__}")
        print(f"Model config: {model.get_config()}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()