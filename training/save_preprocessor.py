"""
Utility script to save preprocessing pipeline (scaler and feature names)
from training data for use in streaming predictions
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def save_preprocessor(train_data_path, output_path='models/preprocessor.joblib'):
    """
    Save preprocessor (scaler and feature names) from training data
    
    Args:
        train_data_path: Path to training CSV file
        output_path: Path to save preprocessor
    """
    print(f"Loading training data from {train_data_path}...")
    train_df = pd.read_csv(train_data_path)
    
    # Get feature columns (exclude Target)
    feature_cols = [col for col in train_df.columns if col != 'Target']
    print(f"Found {len(feature_cols)} features")
    
    # Columns to normalize (same as in preprocess_dataset.py)
    columns_to_normalize = [
        'frame_length', 'mqtt_payload_len', 'iat',
        'msg_rate_1s', 'msg_rate_10s', 'msg_rate_60s',
        'rolling_mean_payload_1s', 'rolling_mean_payload_10s', 'rolling_mean_payload_60s',
        'rolling_std_payload_1s', 'rolling_std_payload_10s', 'rolling_std_payload_60s'
    ]
    
    # Filter to columns that exist
    normalize_cols = [col for col in columns_to_normalize if col in train_df.columns]
    print(f"Normalizing {len(normalize_cols)} columns: {normalize_cols}")
    
    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(train_df[normalize_cols])
    
    # Calculate outlier caps
    outlier_caps = {}
    for col in ['tcp_len', 'mqtt_msg', 'iat']:
        if col in train_df.columns:
            Q1 = train_df[col].quantile(0.25)
            Q3 = train_df[col].quantile(0.75)
            IQR = Q3 - Q1
            factor = 3.0
            outlier_caps[col] = {
                'lower': float(Q1 - factor * IQR),
                'upper': float(Q3 + factor * IQR)
            }
    
    # Save preprocessor
    preprocessor_data = {
        'scaler': scaler,
        'feature_names': feature_cols,
        'normalize_columns': normalize_cols,
        'outlier_caps': outlier_caps
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor_data, output_path)
    
    print(f"\nPreprocessor saved to {output_path}")
    print(f"  - Scaler fitted on {len(normalize_cols)} columns")
    print(f"  - Feature names: {len(feature_cols)} features")
    print(f"  - Outlier caps: {list(outlier_caps.keys())}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Save preprocessor from training data')
    parser.add_argument('--train-data', type=str, default='processed_output/train.csv',
                       help='Path to training data CSV')
    parser.add_argument('--output', type=str, default='models/preprocessor.joblib',
                       help='Path to save preprocessor')
    
    args = parser.parse_args()
    
    save_preprocessor(args.train_data, args.output)
