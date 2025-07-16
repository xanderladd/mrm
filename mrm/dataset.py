# dataset.py
"""
Neural dataset classes with standardized interface for different data sources
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import json
import pickle
import os
from pathlib import Path


class NeuralDataset(ABC):
    """Base class for neural datasets with standardized interface"""
    
    def __init__(self):
        self.splits = {'train': [], 'val': [], 'test': []}
        self.is_prepared = False
        
    @abstractmethod
    def get_neural_data(self, split: str = 'train') -> np.ndarray:
        """
        Returns neural activity data
        
        Returns:
            np.ndarray: Shape (n_trials, n_timepoints, n_neurons)
        """
        pass
        
    @abstractmethod  
    def get_behavior_data(self, split: str = 'train') -> Dict[str, np.ndarray]:
        """
        Returns behavioral signals
        
        Returns:
            Dict with keys like 'choice', 'stimulus_contrast', 'reaction_time', etc.
            Each value is np.ndarray of length n_trials
        """
        pass
        
    @abstractmethod
    def get_trial_info(self, split: str = 'train') -> pd.DataFrame:
        """
        Returns trial-level metadata
        
        Returns:
            pd.DataFrame: Columns like 'trial_id', 'session_id', 'block_id', etc.
        """
        pass
        
    @abstractmethod
    def get_time_info(self, split: str = 'train') -> Dict[str, Any]:
        """
        Returns timing information
        
        Returns:
            Dict with keys:
            - 'bin_size': float (seconds)
            - 'n_timepoints': int
            - 'alignment_event': str
            - 'pre_time': float (seconds before alignment)
            - 'post_time': float (seconds after alignment)
        """
        pass
        
    def get_splits(self) -> Dict[str, List[int]]:
        """Returns trial indices for each split"""
        return self.splits
        
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Returns dataset configuration for reproducibility"""
        pass


class IBLDataset(NeuralDataset):
    """IBL-specific dataset implementation with event alignment"""
    
    def __init__(self, config_path: str = None, **kwargs):
        """
        Initialize IBL Dataset
        
        Args:
            config_path: Path to JSON config file
            use_cache: Whether to use cached data (default True)
            cache_dir: Directory for cached data (default "D://multi_region//ibl/")
            **kwargs: Direct parameters (override config file)
                Required: eid (IBL session ID)
        """
        super().__init__()
        
        # Load config from file if provided, otherwise use kwargs
        if config_path:
            with open(config_path, 'r') as f:
                config = json.load(f)
            # Override with any kwargs
            config.update(kwargs)
        else:
            config = kwargs
            
        # Set parameters with defaults
        self.eid = config.get('eid', None)
        if self.eid is None:
            raise ValueError("IBL session EID is required")
            
        self.alignment_event = config.get('alignment_event', 'stimOn_times')
        self.pre_time = config.get('pre_time', .2)
        self.post_time = config.get('post_time', .5)
        self.bin_size = config.get('bin_size', 0.01)
        self.min_trial_length = config.get('min_trial_length', None)
        if self.min_trial_length is None:
            self.min_trial_length = self.pre_time + self.post_time
        self.brain_regions = config.get('brain_regions', None)
        self.val_fraction = config.get('val_fraction', 0.15)
        self.test_fraction = config.get('test_fraction', 0.15)
        self.random_seed = config.get('random_seed', 42)
        
        # Cache settings
        self.use_cache = config.get('use_cache', True)
        self.cache_dir = config.get("cache_dir",  "D://multi_region//ibl/")
        
        # Data containers - populated during prepare()
        self.neural_data = {}  # split -> np.ndarray
        self.behavior_data = {}  # split -> Dict[str, np.ndarray]
        self.trial_info = {}  # split -> pd.DataFrame
        self.time_info = {}  # split -> Dict
        self.valid_trials = {}  # split -> List[int] (original trial indices)
        
    def _get_cache_path(self):
        """Get path for cached data"""
        cache_path = os.path.join(self.cache_dir, self.eid)
        os.makedirs(cache_path, exist_ok=True)
        return os.path.join(cache_path, 'raw_data.pkl')
        
    def _save_to_cache(self):
        """Save raw data to cache"""
        cache_data = {
            'raw_behavior': self.raw_behavior,
            'raw_events': self.raw_events,
            'raw_trial_info': self.raw_trial_info,
            'raw_neural_data': self.raw_neural_data,
            'time_bins': self.time_bins,
            'session_start': self.session_start,
            'good_cluster_ids': self.good_cluster_ids,
            'config_subset': {
                'eid': self.eid,
                'bin_size': self.bin_size,
                'brain_regions': self.brain_regions
            },
            'cache_version': '1.0'
        }
        
        cache_path = self._get_cache_path()
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"  Cached raw data to {cache_path}")
        
    def _load_from_cache(self):
        """Load raw data from cache if available and valid"""
        cache_path = self._get_cache_path()
        
        if not os.path.exists(cache_path):
            return False
            
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Validate cache matches current config
            cached_config = cache_data.get('config_subset', {})
            if (cached_config.get('eid') != self.eid or
                cached_config.get('bin_size') != self.bin_size or
                cached_config.get('brain_regions') != self.brain_regions):
                print("  Cache config mismatch, reloading...")
                return False
                
            # Load cached data
            self.raw_behavior = cache_data['raw_behavior']
            self.raw_events = cache_data['raw_events']
            self.raw_trial_info = cache_data['raw_trial_info']
            self.raw_neural_data = cache_data['raw_neural_data']
            self.time_bins = cache_data['time_bins']
            self.session_start = cache_data['session_start']
            self.good_cluster_ids = cache_data['good_cluster_ids']
            
            return True
            
        except Exception as e:
            print(f"  Error loading cache: {e}")
            return False
        
    def prepare(self):
        """Load and process the data"""
        if self.is_prepared:
            return
            
        print("Loading IBL session data...")
        if self.use_cache and self._load_from_cache():
            print("✓ Loaded from cache")
        else:
            self._load_raw_data()
            if self.use_cache:
                self._save_to_cache()
        
        print("Aligning trials to events...")
        self._align_trials_to_events()
        
        print("Creating train/val/test splits...")
        self._create_splits()
        
        self.is_prepared = True
        print(f"Dataset prepared: {len(self.splits['train'])} train, "
              f"{len(self.splits['val'])} val, {len(self.splits['test'])} test trials")
        
    def _load_raw_data(self):
        """Load raw IBL data using ONE API"""
        try:
            from one.api import ONE
            import numpy as np
        except ImportError:
            raise ImportError("IBL ONE API not installed. Please install with: pip install ONE-api")
        
        # Initialize ONE
        one = ONE(base_url='https://openalyx.internationalbrainlab.org', password='international', silent=True)
        
        print(f"Loading IBL session: {self.eid}")
        
        # Load behavioral data
        print("  Loading behavioral data...")
        trials = one.load_object(self.eid, 'trials')
        
        if trials is None:
            raise ValueError(f"Could not load trials data for session {self.eid}")
        
        # Extract behavioral signals
        self.raw_behavior = {
            'choice': trials.get('choice', np.full(len(trials.intervals), np.nan)),
            'stimulus_contrast_left': trials.get('contrastLeft', np.full(len(trials.intervals), np.nan)),
            'stimulus_contrast_right': trials.get('contrastRight', np.full(len(trials.intervals), np.nan)),
            'reaction_time': trials.get('firstMovement_times', np.full(len(trials.intervals), np.nan)) - 
                            trials.get('stimOn_times', np.full(len(trials.intervals), np.nan)),
            'feedback_type': trials.get('feedbackType', np.full(len(trials.intervals), np.nan)),
        }
        
        # Extract event times (absolute times)
        self.raw_events = {
            'stimOn_times': trials.get('stimOn_times', np.full(len(trials.intervals), np.nan)),
            'movement_times': trials.get('firstMovement_times', np.full(len(trials.intervals), np.nan)),
            'feedback_times': trials.get('feedback_times', np.full(len(trials.intervals), np.nan)),
        }
        
        n_trials_raw = len(trials.intervals)
        print(f"    Loaded {n_trials_raw} trials")
        
        # Load neural data 
        print("  Loading neural data...")
        spikes = one.load_object(self.eid, 'spikes')
        clusters = one.load_object(self.eid, 'clusters')
        
        if spikes is None or clusters is None:
            raise ValueError(f"Could not load spikes or clusters data for session {self.eid}")
        
        # Get unique cluster IDs from spikes data
        cluster_ids = np.unique(spikes.clusters)
        
        # Get good clusters using quality metrics
        if hasattr(clusters, 'metrics') and hasattr(clusters.metrics, 'label'):
            good_cluster_mask = clusters.metrics.label == 1
            # Map cluster IDs to indices in the metrics array
            good_cluster_indices = np.where(good_cluster_mask)[0]
            good_cluster_ids = good_cluster_indices  # Use indices as cluster IDs
        else:
            # No quality metrics available, use all clusters
            good_cluster_ids = cluster_ids
            print(f"    No quality metrics found, using all {len(good_cluster_ids)} clusters")
        
        # Filter for brain regions if specified
        if self.brain_regions:
            print(f"    Brain region filtering requested but not fully implemented yet")
        
        n_neurons = len(good_cluster_ids)
        print(f"    Using {n_neurons} good clusters out of {len(cluster_ids)} total")
        
        # Create mapping from cluster ID to neuron index
        cluster_to_neuron = {cluster_id: idx for idx, cluster_id in enumerate(good_cluster_ids)}
        
        # Get session bounds with buffer
        all_event_times = np.concatenate([
            self.raw_events['stimOn_times'][~np.isnan(self.raw_events['stimOn_times'])],
            self.raw_events['feedback_times'][~np.isnan(self.raw_events['feedback_times'])]
        ])
        
        if len(all_event_times) == 0:
            raise ValueError("No valid event times found")
            
        session_start = np.min(all_event_times) - 5.0  # 5s buffer
        session_end = np.max(all_event_times) + 5.0     # 5s buffer
        session_duration = session_end - session_start
        
        # Create time bins for the entire session
        time_bins = np.arange(session_start, session_end + self.bin_size, self.bin_size)
        n_time_bins = len(time_bins) - 1
        
        print(f"    Session duration: {session_duration:.1f}s, {n_time_bins} time bins")
        print(f"    {n_neurons} good clusters, {len(spikes.times)} total spikes")
        
        # Initialize spike count matrix (time_bins x neurons)
        spike_counts = np.zeros((n_time_bins, n_neurons))
        
        # Bin spikes for each neuron
        print(f"    Binning spikes...")
        for cluster_id in good_cluster_ids:
            if cluster_id not in cluster_to_neuron:
                continue
                
            neuron_idx = cluster_to_neuron[cluster_id]
            
            # Get spike times for this cluster
            cluster_spike_mask = spikes.clusters == cluster_id
            cluster_spike_times = spikes.times[cluster_spike_mask]
            
            # Only include spikes within session bounds
            valid_spikes = cluster_spike_times[
                (cluster_spike_times >= session_start) & 
                (cluster_spike_times <= session_end)
            ]
            
            if len(valid_spikes) > 0:
                # Bin the spikes
                counts, _ = np.histogram(valid_spikes, bins=time_bins)
                spike_counts[:, neuron_idx] = counts
        
        # Store session-wide binned data for later trial extraction
        self.raw_neural_data = spike_counts
        self.time_bins = time_bins
        self.session_start = session_start
        self.good_cluster_ids = good_cluster_ids
        
        # Create trial info
        self.raw_trial_info = pd.DataFrame({
            'trial_id': range(n_trials_raw),
            'session_id': [self.eid] * n_trials_raw,
            'eid': [self.eid] * n_trials_raw,
        })
        
        print(f"  Data loading complete: {n_trials_raw} trials, {n_neurons} neurons")
        print(f"  Total spikes binned: {spike_counts.sum():.0f}")
        
    def _align_trials_to_events(self):
        """Extract fixed windows around behavioral events using real spike data"""
        n_timepoints = int((self.pre_time + self.post_time) / self.bin_size)
        n_neurons = self.raw_neural_data.shape[1]
        
        aligned_neural = []
        aligned_behavior = {key: [] for key in self.raw_behavior.keys()}
        aligned_trial_info = []
        valid_trial_indices = []
        
        for trial_idx in range(len(self.raw_events[self.alignment_event])):
            # Get alignment time for this trial (absolute time)
            alignment_time = self.raw_events[self.alignment_event][trial_idx]
            
            # Skip trials with NaN alignment times
            if np.isnan(alignment_time):
                continue
            
            # Check trial length requirement
            stim_time = self.raw_events['stimOn_times'][trial_idx]
            feedback_time = self.raw_events['feedback_times'][trial_idx]
            
            if (not np.isnan(stim_time) and not np.isnan(feedback_time) and 
                feedback_time - stim_time < self.min_trial_length):
                continue
            
            # Calculate time window around alignment
            window_start = alignment_time - self.pre_time
            window_end = alignment_time + self.post_time
            
            # Find corresponding bins in session data
            start_bin_idx = np.searchsorted(self.time_bins, window_start)
            end_bin_idx = np.searchsorted(self.time_bins, window_end)
            
            # Check if we have enough data
            if (start_bin_idx >= 0 and 
                end_bin_idx <= len(self.time_bins) - 1 and
                end_bin_idx - start_bin_idx == n_timepoints):
                
                # Extract neural data for this trial
                trial_neural = self.raw_neural_data[start_bin_idx:end_bin_idx, :]
                
                if trial_neural.shape[0] == n_timepoints:
                    aligned_neural.append(trial_neural)
                    
                    # Store behavioral data for this trial
                    for key in self.raw_behavior.keys():
                        if trial_idx < len(self.raw_behavior[key]):
                            aligned_behavior[key].append(self.raw_behavior[key][trial_idx])
                        else:
                            aligned_behavior[key].append(np.nan)
                    
                    # Store trial info
                    aligned_trial_info.append(self.raw_trial_info.iloc[trial_idx])
                    valid_trial_indices.append(trial_idx)
        
        if len(aligned_neural) == 0:
            raise ValueError("No valid trials found after alignment. Check data and parameters.")
        
        # Convert to arrays
        self.aligned_neural_data = np.array(aligned_neural)
        self.aligned_behavior_data = {key: np.array(vals) for key, vals in aligned_behavior.items()}
        self.aligned_trial_info = pd.DataFrame(aligned_trial_info).reset_index(drop=True)
        self.all_valid_trials = valid_trial_indices
        
        retention_rate = len(self.all_valid_trials)/len(self.raw_events[self.alignment_event])
        print(f"Aligned {len(self.all_valid_trials)} trials (kept {retention_rate:.1%})")
        print(f"Neural data shape: {self.aligned_neural_data.shape}")
        
        # Clean up session data to save memory
        del self.raw_neural_data
        del self.time_bins
        
    def _create_splits(self):
        """Create train/validation/test splits"""
        np.random.seed(self.random_seed)
        n_trials = len(self.all_valid_trials)
        indices = np.arange(n_trials)
        np.random.shuffle(indices)
        
        n_test = int(n_trials * self.test_fraction)
        n_val = int(n_trials * self.val_fraction)
        n_train = n_trials - n_test - n_val
        
        self.splits = {
            'train': indices[:n_train].tolist(),
            'val': indices[n_train:n_train+n_val].tolist(), 
            'test': indices[n_train+n_val:].tolist()
        }
        
        # Populate split-specific data
        for split in ['train', 'val', 'test']:
            split_indices = self.splits[split]
            
            self.neural_data[split] = self.aligned_neural_data[split_indices]
            self.behavior_data[split] = {key: vals[split_indices] 
                                       for key, vals in self.aligned_behavior_data.items()}
            self.trial_info[split] = self.aligned_trial_info.iloc[split_indices].reset_index(drop=True)
            self.valid_trials[split] = [self.all_valid_trials[i] for i in split_indices]
            
            self.time_info[split] = {
                'bin_size': self.bin_size,
                'n_timepoints': self.aligned_neural_data.shape[1],
                'alignment_event': self.alignment_event,
                'pre_time': self.pre_time,
                'post_time': self.post_time
            }
    
    def get_neural_data(self, split: str = 'train') -> np.ndarray:
        """Returns neural activity data"""
        if not self.is_prepared:
            self.prepare()
        return self.neural_data[split]
        
    def get_behavior_data(self, split: str = 'train') -> Dict[str, np.ndarray]:
        """Returns behavioral signals"""
        if not self.is_prepared:
            self.prepare()
        return self.behavior_data[split]
        
    def get_trial_info(self, split: str = 'train') -> pd.DataFrame:
        """Returns trial-level metadata"""
        if not self.is_prepared:
            self.prepare()
        return self.trial_info[split]
        
    def get_time_info(self, split: str = 'train') -> Dict[str, Any]:
        """Returns timing information"""
        if not self.is_prepared:
            self.prepare()
        return self.time_info[split]
        
    def get_config(self) -> Dict[str, Any]:
        """Returns dataset configuration"""
        return {
            'eid': self.eid,
            'alignment_event': self.alignment_event,
            'pre_time': self.pre_time,
            'post_time': self.post_time,
            'bin_size': self.bin_size,
            'min_trial_length': self.min_trial_length,
            'brain_regions': self.brain_regions,
            'val_fraction': self.val_fraction,
            'test_fraction': self.test_fraction,
            'random_seed': self.random_seed,
        }
        
    @classmethod
    def from_config(cls, config_path: str, **kwargs):
        """Create dataset from JSON config file"""
        return cls(config_path=config_path, **kwargs)


if __name__ == "__main__":
    # Test with specific IBL session EID
    eid = "ebce500b-c530-47de-8cb1-963c552703ea"
    
    print(f"Testing IBL dataset loading with EID: {eid}")
    
    # Test with a real IBL session
    dataset = IBLDataset(
        eid=eid,
        alignment_event="stimOn_times",
        pre_time=0.1,
        post_time=0.3,
        bin_size=0.01
    )
    
    dataset.prepare()
    
    # Test the interface
    neural_data = dataset.get_neural_data('train')
    behavior_data = dataset.get_behavior_data('train')
    trial_info = dataset.get_trial_info('train')
    time_info = dataset.get_time_info('train')
    
    print(f"\n=== Dataset Successfully Loaded ===")
    print(f"EID: {dataset.eid}")
    print(f"Neural data shape: {neural_data.shape}")
    print(f"Behavior keys: {list(behavior_data.keys())}")
    print(f"Trial info columns: {list(trial_info.columns)}")
    print(f"Time info: {time_info}")
    print(f"Data splits: {[(k, len(v)) for k, v in dataset.get_splits().items()]}")
    
    # Show some basic statistics
    print(f"\n=== Data Statistics ===")
    print(f"Total spikes in training data: {neural_data.sum():.0f}")
    print(f"Mean firing rate: {neural_data.mean():.3f} Hz")
    print(f"Alignment event: {dataset.alignment_event}")
    print(f"Time window: -{dataset.pre_time}s to +{dataset.post_time}s")
    
    # Show behavioral summary
    print(f"\n=== Behavioral Summary ===")
    choices = behavior_data['choice']
    valid_choices = choices[~np.isnan(choices)]
    if len(valid_choices) > 0:
        print(f"Choice distribution: {np.bincount(valid_choices.astype(int) + 1)}")
    
    feedback = behavior_data['feedback_type']
    valid_feedback = feedback[~np.isnan(feedback)]
    if len(valid_feedback) > 0:
        print(f"Performance: {np.mean(valid_feedback == 1):.2%} correct")
    
    print(f"\n=== Success! Real IBL data loaded ===")