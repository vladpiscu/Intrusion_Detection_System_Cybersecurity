"""
Fault Injection Layer for IoT Cybersecurity Dataset
Simulates real-world data corruption scenarios:
1. Random feature corruption: Adds Gaussian noise to 10-30% of numeric features
2. Partial payload loss: Zero-out payload_len or drop selected features for random samples
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Tuple, Dict
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class FaultInjector:
    """
    Fault injection layer for simulating data corruption in IoT cybersecurity datasets
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize fault injector
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
    def inject_feature_corruption(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        corruption_rate: float = 0.2,
        noise_std: float = 0.1,
        exclude_features: Optional[List[str]] = None
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], dict]:
        """
        Inject random feature corruption by adding Gaussian noise to a percentage of numeric features
        
        Args:
            data: Input data (DataFrame or numpy array)
            feature_names: List of feature names (required if data is numpy array)
            corruption_rate: Percentage of features to corrupt (0.0 to 1.0, default 0.2 = 20%)
                           Recommended range: 0.1-0.3 (10-30%), but any value 0-100% is allowed
            noise_std: Standard deviation of Gaussian noise (default 0.1)
            exclude_features: List of feature names to exclude from corruption (e.g., 'Target')
            
        Returns:
            Tuple of (corrupted_data, corruption_info)
            corruption_info contains:
                - corrupted_features: List of feature indices/names that were corrupted
                - corruption_rate: Actual corruption rate applied
                - noise_std: Standard deviation used
        """
        if exclude_features is None:
            exclude_features = ['Target']
        
        # Convert to DataFrame if numpy array
        is_dataframe = isinstance(data, pd.DataFrame)
        if is_dataframe:
            df = data.copy()
            feature_names = df.columns.tolist()
        else:
            if feature_names is None:
                raise ValueError("feature_names must be provided when data is a numpy array")
            df = pd.DataFrame(data, columns=feature_names)
        
        # Identify numeric features (exclude target and specified features)
        numeric_features = []
        for col in df.columns:
            if col not in exclude_features and pd.api.types.is_numeric_dtype(df[col]):
                numeric_features.append(col)
        
        if len(numeric_features) == 0:
            warnings.warn("No numeric features found for corruption")
            return (data, {'corrupted_features': [], 'corruption_rate': 0.0, 'noise_std': noise_std})
        
        # Clamp corruption_rate to valid range (0-100%)
        # Note: Original requirement was 10-30%, but now allows any percentage
        corruption_rate = np.clip(corruption_rate, 0.0, 1.0)
        
        # Warn if outside recommended range but allow it
        if corruption_rate < 0.1 or corruption_rate > 0.3:
            warnings.warn(f"corruption_rate={corruption_rate:.1%} is outside recommended range (10-30%). "
                         f"Proceeding with {corruption_rate:.1%}.")
        
        # Select random features to corrupt
        n_features_to_corrupt = max(1, int(len(numeric_features) * corruption_rate))
        corrupted_features = self.rng.choice(
            numeric_features,
            size=n_features_to_corrupt,
            replace=False
        ).tolist()
        
        # Add Gaussian noise to selected features
        for feature in corrupted_features:
            # Calculate noise based on feature's standard deviation
            feature_std = df[feature].std()
            if feature_std > 0:
                # Scale noise by feature std and noise_std parameter
                noise = self.rng.normal(0, feature_std * noise_std, size=len(df))
                df[feature] = df[feature] + noise
            else:
                # If std is 0, use absolute noise
                noise = self.rng.normal(0, noise_std, size=len(df))
                df[feature] = df[feature] + noise
        
        corruption_info = {
            'corrupted_features': corrupted_features,
            'corruption_rate': len(corrupted_features) / len(numeric_features),
            'noise_std': noise_std,
            'n_features_corrupted': len(corrupted_features),
            'n_total_numeric_features': len(numeric_features)
        }
        
        # Convert back to original format
        if not is_dataframe:
            result = df.values
        else:
            result = df
        
        return result, corruption_info
    
    def inject_payload_loss(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        sample_rate: float = 0.15,
        payload_feature: str = 'mqtt_payload_len',
        drop_features: Optional[List[str]] = None,
        zero_payload: bool = True,
        drop_features_mode: bool = False
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], dict]:
        """
        Inject partial payload loss by zeroing-out payload_len or dropping features for random samples
        
        Args:
            data: Input data (DataFrame or numpy array)
            feature_names: List of feature names (required if data is numpy array)
            sample_rate: Percentage of samples to affect (default 0.15 = 15%)
            payload_feature: Name of payload length feature to zero-out (default 'mqtt_payload_len')
            drop_features: List of feature names to drop/zero for affected samples
            zero_payload: If True, zero-out payload_feature (default True)
            drop_features_mode: If True, drop selected features entirely (set to 0 or NaN)
            
        Returns:
            Tuple of (corrupted_data, loss_info)
            loss_info contains:
                - affected_samples: Number of samples affected
                - affected_indices: Indices of affected samples
                - payload_feature: Name of payload feature
                - dropped_features: Features that were dropped/zeroed
        """
        # Convert to DataFrame if numpy array
        is_dataframe = isinstance(data, pd.DataFrame)
        if is_dataframe:
            df = data.copy()
            feature_names = df.columns.tolist()
        else:
            if feature_names is None:
                raise ValueError("feature_names must be provided when data is a numpy array")
            df = pd.DataFrame(data, columns=feature_names)
        
        n_samples = len(df)
        n_affected = max(1, int(n_samples * sample_rate))
        
        # Select random samples to affect
        affected_indices = self.rng.choice(
            n_samples,
            size=n_affected,
            replace=False
        )
        
        loss_info = {
            'affected_samples': n_affected,
            'affected_indices': affected_indices.tolist(),
            'sample_rate': n_affected / n_samples,
            'payload_feature': payload_feature,
            'dropped_features': []
        }
        
        # Zero-out payload feature if requested and it exists
        if zero_payload and payload_feature in df.columns:
            df.loc[affected_indices, payload_feature] = 0.0
            loss_info['payload_zeroed'] = True
        else:
            loss_info['payload_zeroed'] = False
        
        # Drop/zero selected features if specified
        if drop_features is not None:
            available_drop_features = [f for f in drop_features if f in df.columns]
            if available_drop_features:
                if drop_features_mode:
                    # Set to NaN (simulating feature loss)
                    df.loc[affected_indices, available_drop_features] = np.nan
                else:
                    # Zero-out features
                    df.loc[affected_indices, available_drop_features] = 0.0
                loss_info['dropped_features'] = available_drop_features
                loss_info['drop_mode'] = 'nan' if drop_features_mode else 'zero'
        
        # Convert back to original format
        if not is_dataframe:
            result = df.values
        else:
            result = df
        
        return result, loss_info
    
    def inject_combined_faults(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        corruption_rate: float = 0.2,
        noise_std: float = 0.1,
        payload_loss_rate: float = 0.15,
        payload_feature: str = 'mqtt_payload_len',
        drop_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], dict]:
        """
        Apply both feature corruption and payload loss
        
        Args:
            data: Input data (DataFrame or numpy array)
            feature_names: List of feature names (required if data is numpy array)
            corruption_rate: Percentage of features to corrupt (0.0-1.0, default 0.2 = 20%)
                           Recommended: 0.1-0.3, but any value 0-100% is allowed
            noise_std: Standard deviation of Gaussian noise (default 0.1)
            payload_loss_rate: Percentage of samples to affect with payload loss (0.0-1.0, default 0.15 = 15%)
            payload_feature: Name of payload length feature (default 'mqtt_payload_len')
            drop_features: Features to drop/zero for payload loss
            exclude_features: Features to exclude from corruption
            
        Returns:
            Tuple of (corrupted_data, combined_info)
        """
        # Apply feature corruption first
        corrupted_data, corruption_info = self.inject_feature_corruption(
            data=data,
            feature_names=feature_names,
            corruption_rate=corruption_rate,
            noise_std=noise_std,
            exclude_features=exclude_features
        )
        
        # Then apply payload loss
        final_data, loss_info = self.inject_payload_loss(
            data=corrupted_data,
            feature_names=feature_names,
            sample_rate=payload_loss_rate,
            payload_feature=payload_feature,
            drop_features=drop_features,
            zero_payload=True,
            drop_features_mode=False
        )
        
        combined_info = {
            'corruption': corruption_info,
            'payload_loss': loss_info
        }
        
        return final_data, combined_info
    
    def reset_random_state(self, random_state: int = None):
        """Reset random number generator with new seed"""
        if random_state is None:
            random_state = self.random_state
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)


def apply_fault_injection_to_training_data(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    corruption_rate: float = 0.2,
    noise_std: float = 0.1,
    payload_loss_rate: float = 0.15,
    payload_feature: str = 'mqtt_payload_len',
    apply_to: str = 'test',
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Convenience function to apply fault injection to training datasets
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        feature_names: List of feature names
        corruption_rate: Percentage of features to corrupt
        noise_std: Standard deviation of Gaussian noise
        payload_loss_rate: Percentage of samples to affect with payload loss
        payload_feature: Name of payload length feature
        apply_to: Which dataset to apply faults to ('train', 'val', 'test', 'all')
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_val, X_test, fault_info)
    """
    injector = FaultInjector(random_state=random_state)
    
    fault_info = {}
    
    if apply_to in ['train', 'all']:
        X_train, info_train = injector.inject_combined_faults(
            data=X_train,
            feature_names=feature_names,
            corruption_rate=corruption_rate,
            noise_std=noise_std,
            payload_loss_rate=payload_loss_rate,
            payload_feature=payload_feature,
            exclude_features=['Target']
        )
        fault_info['train'] = info_train
    
    if apply_to in ['val', 'all']:
        X_val, info_val = injector.inject_combined_faults(
            data=X_val,
            feature_names=feature_names,
            corruption_rate=corruption_rate,
            noise_std=noise_std,
            payload_loss_rate=payload_loss_rate,
            payload_feature=payload_feature,
            exclude_features=['Target']
        )
        fault_info['val'] = info_val
    
    if apply_to in ['test', 'all']:
        X_test, info_test = injector.inject_combined_faults(
            data=X_test,
            feature_names=feature_names,
            corruption_rate=corruption_rate,
            noise_std=noise_std,
            payload_loss_rate=payload_loss_rate,
            payload_feature=payload_feature,
            exclude_features=['Target']
        )
        fault_info['test'] = info_test
    
    return X_train, X_val, X_test, fault_info


def load_fault_injection_config(config_path: Union[str, Path]) -> Dict:
    """
    Load fault injection configuration from JSON file
    
    Args:
        config_path: Path to configuration JSON file
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
        ValueError: If required configuration keys are missing
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate required keys
    required_keys = ['enabled', 'corruption', 'payload_loss', 'target']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate corruption section
    if 'rate' not in config['corruption']:
        raise ValueError("Missing 'rate' in corruption configuration")
    if 'noise_std' not in config['corruption']:
        raise ValueError("Missing 'noise_std' in corruption configuration")
    
    # Validate payload_loss section
    if 'rate' not in config['payload_loss']:
        raise ValueError("Missing 'rate' in payload_loss configuration")
    if 'payload_feature' not in config['payload_loss']:
        raise ValueError("Missing 'payload_feature' in payload_loss configuration")
    
    # Set defaults for optional fields
    if 'random_state' not in config:
        config['random_state'] = 42
    
    if 'drop_features' not in config['payload_loss']:
        config['payload_loss']['drop_features'] = []
    
    if 'zero_payload' not in config['payload_loss']:
        config['payload_loss']['zero_payload'] = True
    
    if 'drop_features_mode' not in config['payload_loss']:
        config['payload_loss']['drop_features_mode'] = False
    
    return config


def get_fault_injection_params_from_config(config_path: Union[str, Path]) -> Dict:
    """
    Load and extract fault injection parameters from config file
    
    Args:
        config_path: Path to configuration JSON file
        
    Returns:
        Dictionary with parameters ready to use in training scripts:
        - enabled: bool
        - corruption_rate: float
        - noise_std: float
        - payload_loss_rate: float
        - payload_feature: str
        - fault_injection_target: str
        - random_state: int
        - drop_features: List[str]
        - zero_payload: bool
        - drop_features_mode: bool
    """
    config = load_fault_injection_config(config_path)
    
    params = {
        'enabled': config['enabled'],
        'corruption_rate': config['corruption']['rate'],
        'noise_std': config['corruption']['noise_std'],
        'payload_loss_rate': config['payload_loss']['rate'],
        'payload_feature': config['payload_loss']['payload_feature'],
        'fault_injection_target': config['target'],
        'random_state': config.get('random_state', 42),
        'drop_features': config['payload_loss'].get('drop_features', []),
        'zero_payload': config['payload_loss'].get('zero_payload', True),
        'drop_features_mode': config['payload_loss'].get('drop_features_mode', False)
    }
    
    return params


def generate_output_folder_name(
    base_dir: str = 'models',
    enable_fault_injection: bool = False,
    corruption_rate: float = None,
    noise_std: float = None,
    payload_loss_rate: float = None,
    fault_injection_target: str = None
) -> str:
    """
    Generate output folder name based on fault injection parameters
    
    Args:
        base_dir: Base directory name (default: 'models')
        enable_fault_injection: Whether fault injection is enabled
        corruption_rate: Corruption rate (if enabled)
        noise_std: Noise standard deviation (if enabled)
        payload_loss_rate: Payload loss rate (if enabled)
        fault_injection_target: Target dataset ('train', 'val', 'test', 'all')
        
    Returns:
        Folder name string, e.g., 'models_corruption0.2_noise0.1_payload0.15_test'
        or 'models' if fault injection is disabled
        
    Examples:
        >>> generate_output_folder_name(enable_fault_injection=True, corruption_rate=0.2, noise_std=0.1, payload_loss_rate=0.15, fault_injection_target='test')
        'models_corruption0.2_noise0.1_payload0.15_test'
        
        >>> generate_output_folder_name(enable_fault_injection=False)
        'models'
        
        >>> generate_output_folder_name(enable_fault_injection=True, corruption_rate=0.5, fault_injection_target='all')
        'models_corruption0.5_all'
    """
    if not enable_fault_injection:
        return base_dir
    
    # Build folder name components
    parts = [base_dir]
    
    if corruption_rate is not None:
        # Format as corruption0.2 (remove trailing zeros, but keep at least one decimal if needed)
        rate_str = f"{corruption_rate:.2f}".rstrip('0').rstrip('.')
        if '.' not in rate_str:
            rate_str = f"{corruption_rate:.2f}"  # Keep decimals for clarity
        parts.append(f"corruption{rate_str}")
    
    if noise_std is not None:
        # Format as noise0.1 (remove trailing zeros)
        noise_str = f"{noise_std:.2f}".rstrip('0').rstrip('.')
        if '.' not in noise_str:
            noise_str = f"{noise_std:.2f}"
        parts.append(f"noise{noise_str}")
    
    if payload_loss_rate is not None:
        # Format as payload0.15 (remove trailing zeros)
        payload_str = f"{payload_loss_rate:.2f}".rstrip('0').rstrip('.')
        if '.' not in payload_str:
            payload_str = f"{payload_loss_rate:.2f}"
        parts.append(f"payload{payload_str}")
    
    if fault_injection_target:
        parts.append(fault_injection_target)
    
    return "_".join(parts)


def create_default_config(output_path: Union[str, Path] = 'fault_injection_config.json'):
    """
    Create a default fault injection configuration file
    
    Args:
        output_path: Path where to save the default configuration
    """
    default_config = {
        "enabled": False,
        "corruption": {
            "rate": 0.2,
            "noise_std": 0.1,
            "description": "Random feature corruption: adds Gaussian noise to a percentage of numeric features"
        },
        "payload_loss": {
            "rate": 0.15,
            "payload_feature": "mqtt_payload_len",
            "drop_features": [],
            "zero_payload": True,
            "drop_features_mode": False,
            "description": "Partial payload loss: zero-out payload_len or drop selected features for random samples"
        },
        "target": "test",
        "description": "Target dataset to apply faults to: 'train', 'val', 'test', or 'all'",
        "random_state": 42,
        "notes": {
            "corruption_rate_range": "0.0 to 1.0 (0% to 100%), recommended: 0.1-0.3 (10-30%)",
            "payload_loss_rate_range": "0.0 to 1.0 (0% to 100%), recommended: 0.1-0.2 (10-20%)",
            "noise_std": "Standard deviation of Gaussian noise. Recommended: 0.05-0.2"
        }
    }
    
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    print(f"Default configuration file created: {output_path}")
    return output_path

