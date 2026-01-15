"""
Real-time Streaming Prediction Script for IoT Cybersecurity Dataset
Loads trained model + preprocessing pipeline, streams records in timestamp order,
extracts features in real-time, and produces continuous predictions with confidence scores.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from collections import deque
from datetime import datetime
import warnings
import sys
import csv
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import preprocessing components
from preprocessing import IoTDatasetPreprocessor


class StreamingPredictor:
    """
    Real-time streaming predictor that processes records in timestamp order
    and produces predictions with confidence scores
    """
    
    def __init__(self, model_path, preprocessor_path=None, train_data_path=None):
        """
        Initialize streaming predictor
        
        Args:
            model_path: Path to saved model (.joblib file)
            preprocessor_path: Path to saved preprocessor/scaler (optional)
            train_data_path: Path to training data CSV (used to fit scaler if preprocessor not saved)
        """
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path) if preprocessor_path else None
        self.train_data_path = Path(train_data_path) if train_data_path else None
        
        # Load model
        print(f"Loading model from {self.model_path}...")
        self.model = joblib.load(self.model_path)
        print(f"Model loaded: {type(self.model).__name__}")
        
        # Initialize preprocessor for feature extraction
        self.preprocessor = IoTDatasetPreprocessor(
            data_path='dummy',  # Will be overridden
            metadata_dir='MQTTEEB-D_Final_Dataset/Preprocessed_Data'
        )
        self.preprocessor.load_metadata()
        
        # Feature columns expected by model (excluding Target)
        self.expected_features = [
            'frame_length', 'protocol_flags', 'mqtt_msgtype', 'mqtt_qos',
            'mqtt_payload_len', 'mqtt_retain', 'mqtt_client_id_hash', 'iat',
            'msg_rate_1s', 'msg_rate_10s', 'msg_rate_60s',
            'rolling_mean_payload_1s', 'rolling_std_payload_1s',
            'rolling_mean_payload_10s', 'rolling_std_payload_10s',
            'rolling_mean_payload_60s', 'rolling_std_payload_60s'
        ]
        
        # Columns to normalize (same as in preprocess_dataset.py)
        # Must be defined before _load_or_fit_scaler() is called
        self.columns_to_normalize = [
            'frame_length', 'mqtt_payload_len', 'iat',
            'msg_rate_1s', 'msg_rate_10s', 'msg_rate_60s',
            'rolling_mean_payload_1s', 'rolling_mean_payload_10s', 'rolling_mean_payload_60s',
            'rolling_std_payload_1s', 'rolling_std_payload_10s', 'rolling_std_payload_60s'
        ]
        
        # Outlier capping parameters (from training)
        self.outlier_caps = {}
        
        # Load or fit scaler (must be called after columns_to_normalize is defined)
        self.scaler = None
        self.feature_names = None
        self._load_or_fit_scaler()
        
        # Load outlier caps if not already loaded by preprocessor
        self._load_outlier_caps()
        
        # Buffer for temporal features (sliding window)
        self.record_buffer = deque(maxlen=10000)  # Keep last 10k records for rolling stats
        self.timestamp_buffer = deque(maxlen=10000)
        self.payload_buffer = deque(maxlen=10000)
        
        # Track previous timestamp for IAT calculation
        self.prev_timestamp = None
        
    def _load_or_fit_scaler(self):
        """Load scaler from file or fit from training data"""
        if self.preprocessor_path and self.preprocessor_path.exists():
            print(f"Loading preprocessor from {self.preprocessor_path}...")
            preprocessor_data = joblib.load(self.preprocessor_path)
            self.scaler = preprocessor_data.get('scaler')
            self.feature_names = preprocessor_data.get('feature_names')
            self.outlier_caps = preprocessor_data.get('outlier_caps', {})
            normalize_cols = preprocessor_data.get('normalize_columns', [])
            if normalize_cols:
                self.columns_to_normalize = normalize_cols
            print("Preprocessor loaded successfully")
            print(f"  - Scaler: {type(self.scaler).__name__}")
            print(f"  - Features: {len(self.feature_names) if self.feature_names else 0}")
            print(f"  - Outlier caps: {list(self.outlier_caps.keys())}")
        elif self.train_data_path and self.train_data_path.exists():
            print(f"Fitting scaler from training data: {self.train_data_path}...")
            train_df = pd.read_csv(self.train_data_path)
            feature_cols = [col for col in train_df.columns if col != 'Target']
            self.feature_names = feature_cols
            
            # Fit scaler on normalized columns
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            normalize_cols = [col for col in self.columns_to_normalize if col in train_df.columns]
            if normalize_cols:
                self.scaler.fit(train_df[normalize_cols])
                print(f"Scaler fitted on {len(normalize_cols)} columns")
        else:
            print("Warning: No scaler found. Using identity transformation.")
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
    
    def _load_outlier_caps(self):
        """Load outlier capping parameters from training data or preprocessor"""
        # Already loaded in _load_or_fit_scaler if preprocessor exists
        if self.outlier_caps:
            return
            
        if self.train_data_path and self.train_data_path.exists():
            train_df = pd.read_csv(self.train_data_path)
            for col in ['tcp_len', 'mqtt_msg', 'iat']:
                if col in train_df.columns:
                    Q1 = train_df[col].quantile(0.25)
                    Q3 = train_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    factor = 3.0
                    self.outlier_caps[col] = {
                        'lower': float(Q1 - factor * IQR),
                        'upper': float(Q3 + factor * IQR)
                    }
    
    def _convert_to_int(self, value, default=0):
        """
        Convert value to integer, handling hex strings, numeric strings, and numeric types
        
        Args:
            value: Value to convert (can be hex string like '0x0018', numeric string, or number)
            default: Default value if conversion fails
            
        Returns:
            int: Converted integer value
        """
        if pd.isna(value) or value is None:
            return default
        
        # If already an integer or float, convert directly
        if isinstance(value, (int, float, np.integer, np.floating)):
            return int(value)
        
        # Convert to string for processing
        value_str = str(value).strip()
        
        # Handle hex strings (e.g., '0x0018', '0X0018')
        if value_str.startswith(('0x', '0X')):
            try:
                return int(value_str, 16)
            except (ValueError, TypeError):
                return default
        
        # Try numeric conversion
        try:
            # First try pd.to_numeric which handles various formats
            numeric_val = pd.to_numeric(value_str, errors='coerce')
            if pd.isna(numeric_val):
                return default
            return int(numeric_val)
        except (ValueError, TypeError):
            return default
    
    def _convert_to_float(self, value, default=0.0):
        """
        Convert value to float, handling hex strings, numeric strings, and numeric types
        
        Args:
            value: Value to convert (can be hex string like '0x0018', numeric string, or number)
            default: Default value if conversion fails
            
        Returns:
            float: Converted float value
        """
        if pd.isna(value) or value is None:
            return default
        
        # If already numeric, convert directly
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        
        # Convert to string for processing
        value_str = str(value).strip()
        
        # Handle hex strings (e.g., '0x0018', '0X0018') - convert to int first, then float
        if value_str.startswith(('0x', '0X')):
            try:
                return float(int(value_str, 16))
            except (ValueError, TypeError):
                return default
        
        # Try numeric conversion
        try:
            numeric_val = pd.to_numeric(value_str, errors='coerce')
            if pd.isna(numeric_val):
                return default
            return float(numeric_val)
        except (ValueError, TypeError):
            return default
    
    def _preprocess_raw_row(self, raw_row):
        """
        Preprocess a single raw row using the preprocessing module's logic.
        This applies the same preprocessing steps as the batch preprocessing pipeline.
        
        Args:
            raw_row: Dictionary with raw data fields
            
        Returns:
            dict: Preprocessed features dictionary
        """
        # Convert raw row to DataFrame for preprocessing (single row)
        df_row = pd.DataFrame([raw_row])
        
        # Step 1: Map features (using preprocessor's logic)
        df_row = self._apply_feature_mapping(df_row)
        
        # Step 2: Apply encoding
        df_row = self._apply_encoding_single_row(df_row)
        
        # Convert back to dictionary
        features = df_row.iloc[0].to_dict()
        
        return features
    
    def _apply_feature_mapping(self, df):
        """
        Apply feature mapping (same as preprocess_dataset.py map_features)
        Adapted for single-row processing
        """
        df_mapped = df.copy()
        
        # Packet-level features - handle missing columns
        if 'tcp_len' in df_mapped.columns:
            df_mapped['frame_length'] = pd.to_numeric(df_mapped['tcp_len'], errors='coerce').fillna(0)
        else:
            df_mapped['frame_length'] = 0
            
        if 'tcp_flags' in df_mapped.columns:
            df_mapped['protocol_flags'] = pd.to_numeric(df_mapped['tcp_flags'], errors='coerce').fillna(0)
        else:
            df_mapped['protocol_flags'] = 0
        
        # MQTT-level features
        if 'mqtt_qos' in df_mapped.columns:
            df_mapped['mqtt_qos'] = pd.to_numeric(df_mapped['mqtt_qos'], errors='coerce').fillna(0)
        else:
            df_mapped['mqtt_qos'] = 0
            
        if 'mqtt_msg' in df_mapped.columns:
            df_mapped['mqtt_payload_len'] = pd.to_numeric(df_mapped['mqtt_msg'], errors='coerce').fillna(0)
        else:
            df_mapped['mqtt_payload_len'] = 0
        
        # Derive mqtt_msgtype using preprocessor's logic
        if 'mqtt_hdrflags' in df_mapped.columns:
            hdrflags = pd.to_numeric(df_mapped['mqtt_hdrflags'], errors='coerce').fillna(0).astype(int).values
        else:
            hdrflags = np.array([0])
            
        if 'mqtt_conflags' in df_mapped.columns:
            conflags = pd.to_numeric(df_mapped['mqtt_conflags'], errors='coerce').fillna(0).astype(int).values
        else:
            conflags = np.array([0])
        
        # Extract message type from hdrflags (bits 4-7, shifted right by 4)
        msgtype = (hdrflags >> 4) & 0x0F
        
        # Special handling: CONNECT messages
        connect_mask = (conflags > 0) & (msgtype == 1)
        msgtype = np.where(connect_mask, 1, msgtype)
        
        # If msgtype is 0 but we have MQTT data, infer from context
        zero_mask = (msgtype == 0) & (hdrflags > 0)
        msgtype = np.where(zero_mask & (hdrflags > 16), 3, msgtype)
        
        df_mapped['mqtt_msgtype'] = msgtype[0] if len(msgtype) > 0 else 0
        
        # Extract retain flag from mqtt_hdrflags (bit 0)
        if 'mqtt_hdrflags' in df_mapped.columns:
            df_mapped['mqtt_retain'] = (pd.to_numeric(df_mapped['mqtt_hdrflags'], errors='coerce').fillna(0).astype(int) & 0x01).astype(int)
        else:
            df_mapped['mqtt_retain'] = 0
        
        # Client ID hash
        if 'mqtt_conack_flags' in df_mapped.columns:
            df_mapped['mqtt_client_id_hash'] = pd.to_numeric(df_mapped['mqtt_conack_flags'], errors='coerce').fillna(0).astype(int)
        else:
            df_mapped['mqtt_client_id_hash'] = 0
        
        return df_mapped
    
    def _apply_encoding_single_row(self, df):
        """
        Apply categorical encoding (same as preprocess_dataset.py apply_encoding)
        Adapted for single-row processing
        """
        df_encoded = df.copy()
        
        # Encode mqtt_qos if it's categorical
        if 'mqtt_qos' in df_encoded.columns:
            df_encoded['mqtt_qos'] = pd.to_numeric(df_encoded['mqtt_qos'], errors='coerce').fillna(0).astype(int)
        
        return df_encoded
    
    def _map_features(self, record):
        """
        Map raw record to feature space using preprocessing module logic.
        This ensures consistency with the batch preprocessing pipeline.
        """
        # Use the preprocessing method
        features = self._preprocess_raw_row(record)
        
        # Extract only the mapped features we need
        mapped_features = {
            'frame_length': features.get('frame_length', 0),
            'protocol_flags': features.get('protocol_flags', 0),
            'mqtt_qos': features.get('mqtt_qos', 0),
            'mqtt_payload_len': features.get('mqtt_payload_len', 0),
            'mqtt_msgtype': features.get('mqtt_msgtype', 0),
            'mqtt_retain': features.get('mqtt_retain', 0),
            'mqtt_client_id_hash': features.get('mqtt_client_id_hash', 0)
        }
        
        return mapped_features
    
    def _calculate_temporal_features(self, record, timestamp_epoch):
        """Calculate temporal features using sliding window buffer"""
        features = {}
        
        # Calculate IAT (Inter-Arrival Time)
        if self.prev_timestamp is not None:
            iat = timestamp_epoch - self.prev_timestamp
        else:
            iat = 0.0
        features['iat'] = iat
        self.prev_timestamp = timestamp_epoch
        
        # Cap IAT outlier
        if 'iat' in self.outlier_caps:
            caps = self.outlier_caps['iat']
            features['iat'] = np.clip(features['iat'], caps['lower'], caps['upper'])
        
        # Calculate message rates (packets per second in windows)
        for window_sec in [1, 10, 60]:
            window_start = timestamp_epoch - window_sec
            count = sum(1 for ts in self.timestamp_buffer if window_start <= ts <= timestamp_epoch)
            features[f'msg_rate_{window_sec}s'] = count / window_sec if window_sec > 0 else 0
        
        # Calculate rolling statistics for payload - convert hex strings to float
        payload = self._convert_to_float(record.get('mqtt_msg', 0))
        
        # Cap payload outlier before adding to buffer
        if 'mqtt_msg' in self.outlier_caps:
            caps = self.outlier_caps['mqtt_msg']
            payload = np.clip(payload, caps['lower'], caps['upper'])
        
        for window_sec in [1, 10, 60]:
            window_start = timestamp_epoch - window_sec
            window_payloads = [
                p for ts, p in zip(self.timestamp_buffer, self.payload_buffer)
                if window_start <= ts <= timestamp_epoch
            ]
            
            if len(window_payloads) > 0:
                features[f'rolling_mean_payload_{window_sec}s'] = np.mean(window_payloads)
                features[f'rolling_std_payload_{window_sec}s'] = np.std(window_payloads) if len(window_payloads) > 1 else 0
            else:
                features[f'rolling_mean_payload_{window_sec}s'] = 0
                features[f'rolling_std_payload_{window_sec}s'] = 0
        
        # Update buffers
        self.timestamp_buffer.append(timestamp_epoch)
        self.payload_buffer.append(payload)
        self.record_buffer.append(record)
        
        return features
    
    def _normalize_features(self, features_dict):
        """Apply Z-score normalization to specified features"""
        # Create DataFrame for normalization
        normalize_cols = [col for col in self.columns_to_normalize if col in features_dict]
        
        if not normalize_cols or self.scaler is None:
            return features_dict
        
        # Prepare data for scaler - ensure all values are numeric (convert hex strings if needed)
        values_list = []
        for col in normalize_cols:
            val = features_dict[col]
            # Convert to float if it's not already a numeric type
            # Check for numpy string types (np.str_) and other non-numeric types
            if isinstance(val, (str, np.str_)) or not isinstance(val, (int, float, np.integer, np.floating, np.number)):
                val = self._convert_to_float(val, 0.0)
            values_list.append(float(val))
        
        values = np.array([values_list])
        
        # Transform
        try:
            normalized = self.scaler.transform(values)[0]
            for i, col in enumerate(normalize_cols):
                features_dict[col] = normalized[i]
        except Exception as e:
            print(f"Warning: Normalization failed: {e}")
        
        return features_dict
    
    def _extract_features(self, record):
        """Extract all features from a single record"""
        # Convert timestamp to epoch - handle both string and numeric Unix timestamps
        timestamp_raw = record.get('timestamp')
        
        # Check if timestamp is already a numeric Unix epoch (large number)
        if isinstance(timestamp_raw, (int, float, np.integer, np.floating)):
            # If it's a large number (> 1e9), it's likely a Unix epoch timestamp
            if timestamp_raw > 1e9:
                # It's already in seconds (Unix epoch)
                timestamp_epoch = float(timestamp_raw)
                timestamp = pd.Timestamp.fromtimestamp(timestamp_epoch)
            else:
                # Small number, treat as days since epoch (unlikely but handle it)
                timestamp = pd.to_datetime(timestamp_raw, unit='D', origin='unix')
                timestamp_epoch = timestamp.timestamp()
        else:
            # String timestamp - try to parse it
            try:
                timestamp = pd.to_datetime(timestamp_raw)
                timestamp_epoch = timestamp.timestamp() if hasattr(timestamp, 'timestamp') else pd.Timestamp(timestamp).timestamp()
            except (ValueError, TypeError) as e:
                # If parsing fails, try converting to numeric first
                try:
                    numeric_ts = float(timestamp_raw)
                    if numeric_ts > 1e9:
                        timestamp_epoch = numeric_ts
                        timestamp = pd.Timestamp.fromtimestamp(timestamp_epoch)
                    else:
                        raise ValueError(f"Cannot parse timestamp: {timestamp_raw}")
                except (ValueError, TypeError):
                    raise ValueError(f"Cannot parse timestamp: {timestamp_raw}. Error: {e}")
        
        # Map basic features
        features = self._map_features(record)
        
        # Cap outliers for frame_length and payload_len
        if 'tcp_len' in self.outlier_caps:
            caps = self.outlier_caps['tcp_len']
            features['frame_length'] = np.clip(features['frame_length'], caps['lower'], caps['upper'])
        
        if 'mqtt_msg' in self.outlier_caps:
            caps = self.outlier_caps['mqtt_msg']
            features['mqtt_payload_len'] = np.clip(features['mqtt_payload_len'], caps['lower'], caps['upper'])
        
        # Calculate temporal features
        temporal_features = self._calculate_temporal_features(record, timestamp_epoch)
        features.update(temporal_features)
        
        # Normalize features
        features = self._normalize_features(features)
        
        return features, timestamp
    
    def predict(self, record):
        """
        Predict label and confidence for a single record
        
        Args:
            record: Dictionary or pandas Series with record data
            
        Returns:
            tuple: (timestamp, features_dict, predicted_label, confidence)
        """
        # Extract features
        features, timestamp = self._extract_features(record)
        
        # Prepare feature vector in correct order - ensure all values are numeric
        feature_values = []
        for col in self.expected_features:
            val = features.get(col, 0)
            # Ensure value is numeric (convert hex strings if needed)
            if not isinstance(val, (int, float, np.integer, np.floating)):
                val = self._convert_to_float(val, 0.0)
            feature_values.append(float(val))
        feature_vector = np.array([feature_values])
        
        # Make prediction
        if hasattr(self.model, 'predict_proba'):
            # Classification model (Random Forest)
            pred_proba = self.model.predict_proba(feature_vector)[0]
            predicted_label = self.model.predict(feature_vector)[0]
            confidence = float(np.max(pred_proba))
        elif hasattr(self.model, 'score_samples'):
            # Isolation Forest (anomaly detection)
            score = self.model.score_samples(feature_vector)[0]
            predicted_label = self.model.predict(feature_vector)[0]
            # Convert score to confidence-like metric (lower score = more anomalous)
            # Normalize to [0, 1] where 1 = most confident anomaly
            confidence = float(1.0 / (1.0 + np.exp(score)))  # Sigmoid transformation
        else:
            # Fallback
            predicted_label = self.model.predict(feature_vector)[0]
            confidence = 1.0
        
        return timestamp, features, predicted_label, confidence
    
    def process_single_row(self, raw_row):
        """
        Process a single raw row of data without requiring the full dataset.
        This method can be called repeatedly for streaming one row at a time.
        
        Args:
            raw_row: Dictionary with raw data fields (e.g., from CSV row)
                    Must contain at minimum: 'timestamp' and other required fields
                    
        Returns:
            dict: Dictionary containing:
                - timestamp: Processed timestamp
                - features: Extracted features dictionary
                - predicted_label: Model prediction
                - confidence: Prediction confidence score
                
        Example:
            >>> predictor = StreamingPredictor(model_path='model.joblib', ...)
            >>> row = {'timestamp': '2023-01-01 10:00:00', 'tcp_len': 100, ...}
            >>> result = predictor.process_single_row(row)
        """
        try:
            timestamp, features, predicted_label, confidence = self.predict(raw_row)
            return {
                'timestamp': timestamp,
                'features': features,
                'predicted_label': predicted_label,
                'confidence': confidence
            }
        except Exception as e:
            raise ValueError(f"Error processing row: {e}")
    
    def stream_predictions(self, input_csv_path, output_path=None, max_records=None, verbose=True):
        """
        Stream predictions from CSV file, processing one row at a time without loading entire dataset.
        Each raw row is preprocessed using the preprocessing module before prediction.
        
        Args:
            input_csv_path: Path to input CSV file with records
            output_path: Path to output CSV file (optional, prints to stdout if None)
            max_records: Maximum number of records to process (None for all)
            verbose: If True, print progress updates
        """
        print(f"\n{'='*60}")
        print("Streaming Predictions (Row-by-Row Processing)")
        print(f"{'='*60}")
        print(f"Input: {input_csv_path}")
        if output_path:
            print(f"Output: {output_path}")
        print(f"Mode: Continuous row-by-row streaming with preprocessing")
        
        # Prepare output
        results = []
        output_file = None
        if output_path:
            output_file = open(output_path, 'w')
            # Write header: timestamp | features | predicted_label | confidence
            header = "timestamp|features|predicted_label|confidence\n"
            output_file.write(header)
        
        # Process row-by-row without loading entire dataset
        print(f"\nProcessing records row-by-row (streaming mode)...")
        record_count = 0
        
        try:
            with open(input_csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for idx, row_dict in enumerate(reader):
                    # Check max_records limit
                    if max_records and record_count >= max_records:
                        break
                    
                    if verbose and (record_count + 1) % 1000 == 0:
                        print(f"  Processed {record_count + 1} records...")
                    
                    try:
                        # Process single row (includes preprocessing)
                        result = self.process_single_row(row_dict)
                        timestamp = result['timestamp']
                        features = result['features']
                        predicted_label = result['predicted_label']
                        confidence = result['confidence']
                        
                        # Format features as string (pipe-separated key:value pairs)
                        features_str = '|'.join([f"{k}:{v:.6f}" if isinstance(v, (float, np.floating)) else f"{k}:{v}" 
                                                for k, v in sorted(features.items())])
                        
                        # Format output line: timestamp | features | predicted_label | confidence
                        output_line = f"{timestamp}|{features_str}|{predicted_label}|{confidence:.6f}\n"
                        
                        if output_file:
                            output_file.write(output_line)
                        else:
                            # Print formatted output to stdout
                            print(f"{timestamp} | {len(features)} features | label={predicted_label} | confidence={confidence:.4f}")
                        
                        results.append(result)
                        record_count += 1
                        
                    except Exception as e:
                        if verbose:
                            print(f"Error processing record {idx}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            if output_file:
                output_file.close()
            raise
        
        if output_file:
            output_file.close()
            print(f"\nResults saved to {output_path}")
        
        print(f"\n{'='*60}")
        print("Streaming Complete!")
        print(f"{'='*60}")
        print(f"Processed {len(results)} records")
        
        return results


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stream predictions from CSV file')
    parser.add_argument('--model', type=str, default='models/random_forest_model.joblib',
                       help='Path to trained model (.joblib file)')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input CSV file with records')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to output CSV file (prints to stdout if not specified)')
    parser.add_argument('--train-data', type=str, default='processed_output/train.csv',
                       help='Path to training data CSV (for scaler fitting)')
    parser.add_argument('--preprocessor', type=str, default=None,
                       help='Path to saved preprocessor/scaler (.joblib file)')
    parser.add_argument('--max-records', type=int, default=None,
                       help='Maximum number of records to process')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = StreamingPredictor(
        model_path=args.model,
        preprocessor_path=args.preprocessor,
        train_data_path=args.train_data
    )
    
    # Stream predictions (row-by-row processing with preprocessing)
    results = predictor.stream_predictions(
        input_csv_path=args.input,
        output_path=args.output,
        max_records=args.max_records
    )
    
    # Print summary
    if results:
        labels = [r['predicted_label'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        print(f"\nSummary:")
        print(f"  Total predictions: {len(results)}")
        print(f"  Average confidence: {np.mean(confidences):.4f}")
        print(f"  Unique predicted labels: {sorted(set(labels))}")


if __name__ == '__main__':
    main()
