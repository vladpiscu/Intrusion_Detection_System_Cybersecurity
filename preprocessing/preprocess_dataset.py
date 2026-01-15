"""
IoT Cybersecurity Dataset Preprocessing Script
Implements feature mapping, temporal feature generation, preprocessing, and time-aware splitting
"""

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class IoTDatasetPreprocessor:
    """
    Preprocessing pipeline for MQTTEEB-D IoT cybersecurity dataset
    """
    
    def __init__(self, data_path, metadata_dir=None):
        """
        Initialize preprocessor
        
        Args:
            data_path: Path to the input CSV file
            metadata_dir: Directory containing metadata JSON files
        """
        self.data_path = Path(data_path)
        self.metadata_dir = Path(metadata_dir) if metadata_dir else self.data_path.parent
        self.scaler = StandardScaler()
        self.label_encoding = None
        self.categorical_encoding = None
        
    def load_data(self):
        """Load the dataset"""
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp to ensure chronological order
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Loaded {len(df)} records")
        return df
    
    def load_metadata(self):
        """Load encoding metadata from JSON files"""
        label_metadata_path = self.metadata_dir / 'label_encoding_metadata.json'
        categorical_metadata_path = self.metadata_dir / 'categorical_processing_metadata.json'
        
        if label_metadata_path.exists():
            with open(label_metadata_path, 'r') as f:
                self.label_encoding = json.load(f)
            print(f"Loaded label encoding metadata: {self.label_encoding}")
        
        if categorical_metadata_path.exists():
            with open(categorical_metadata_path, 'r') as f:
                self.categorical_encoding = json.load(f)
            print("Loaded categorical processing metadata")
    
    def map_features(self, df):
        """
        Map CSV columns to required features
        
        Packet-level Features:
        - frame_length: from tcp_len
        - protocol_flags: from tcp_flags
        - ports: Note - not available in current dataset
        
        MQTT-level Features:
        - mqtt.msgtype: derive from mqtt_conflags and mqtt_hdrflags
        - mqtt.qos: from mqtt_qos
        - mqtt.payload_len: from mqtt_msg
        - retain/client_id: extract from metadata
        """
        print("\n=== Mapping Features ===")
        df_mapped = df.copy()
        
        # Packet-level features
        df_mapped['frame_length'] = df_mapped['tcp_len'].copy()
        df_mapped['protocol_flags'] = df_mapped['tcp_flags'].copy()
        
        # Note: Ports are not available in the dataset
        # If ports exist, they would be extracted here: df_mapped['ports'] = ...
        
        # MQTT-level features
        df_mapped['mqtt_qos'] = df_mapped['mqtt_qos'].fillna(0)
        df_mapped['mqtt_payload_len'] = df_mapped['mqtt_msg'].copy()
        
        # Derive mqtt.msgtype from mqtt_conflags and mqtt_hdrflags
        # MQTT message types are encoded in the first 4 bits of mqtt_hdrflags
        # Common types: CONNECT (1), CONNACK (2), PUBLISH (3), PUBACK (4), etc.
        df_mapped['mqtt_msgtype'] = self._derive_mqtt_msgtype(
            df_mapped['mqtt_conflags'], 
            df_mapped['mqtt_hdrflags']
        )
        
        # Extract retain flag from mqtt_hdrflags (bit 0 of the flags byte)
        # Retain flag is typically in bit 0 of the MQTT fixed header
        df_mapped['mqtt_retain'] = (df_mapped['mqtt_hdrflags'].astype(int) & 0x01).astype(int)
        
        # Client ID extraction would typically come from mqtt_conack_flags or payload
        # For now, we'll use mqtt_conack_flags as a placeholder
        df_mapped['mqtt_client_id_hash'] = df_mapped['mqtt_conack_flags'].fillna(0).astype(int)
        
        print(f"Feature mapping complete. New columns: frame_length, protocol_flags, mqtt_msgtype, mqtt_retain, mqtt_client_id_hash")
        
        return df_mapped
    
    def _derive_mqtt_msgtype(self, conflags, hdrflags):
        """
        Derive MQTT message type from mqtt_conflags and mqtt_hdrflags
        
        MQTT message type is in bits 4-7 of the first byte (fixed header)
        Types: CONNECT(1), CONNACK(2), PUBLISH(3), PUBACK(4), PUBREC(5), 
               PUBREL(6), PUBCOMP(7), SUBSCRIBE(8), SUBACK(9), UNSUBSCRIBE(10), 
               UNSUBACK(11), PINGREQ(12), PINGRESP(13), DISCONNECT(14)
        """
        # Convert to numeric if hex strings, then to numpy array
        hdrflags_int = pd.to_numeric(hdrflags, errors='coerce').fillna(0).astype(int).values
        conflags_int = pd.to_numeric(conflags, errors='coerce').fillna(0).astype(int).values
        
        # Extract message type from hdrflags (bits 4-7, shifted right by 4)
        msgtype = (hdrflags_int >> 4) & 0x0F
        
        # Special handling: CONNECT messages typically have mqtt_conflags set
        # If conflags is non-zero and hdrflags indicates it, it's likely a CONNECT
        connect_mask = (conflags_int > 0) & (msgtype == 1)
        msgtype = np.where(connect_mask, 1, msgtype)
        
        # If msgtype is 0 but we have MQTT data, infer from context
        # CONNACK is typically type 2, PUBLISH is type 3
        zero_mask = (msgtype == 0) & (hdrflags_int > 0)
        
        # If mqtt_qos is set and msgtype is 0, it's likely a PUBLISH
        # This is a heuristic - adjust based on actual data patterns
        msgtype = np.where(zero_mask & (hdrflags_int > 16), 3, msgtype)
        
        return msgtype
    
    def generate_temporal_features(self, df):
        """
        Generate temporal features:
        - Inter-Arrival Time (IAT)
        - Message Rate (1s, 10s, 60s windows)
        - Rolling Statistics (mean, std of mqtt_msg over windows)
        """
        print("\n=== Generating Temporal Features ===")
        df_temp = df.copy()
        
        # Ensure sorted by timestamp
        df_temp = df_temp.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate Inter-Arrival Time (IAT)
        # Use tcp_time_delta if available, otherwise calculate from timestamp
        if 'tcp_time_delta' in df_temp.columns:
            # Calculate IAT from consecutive tcp_time_delta values
            df_temp['iat'] = df_temp['tcp_time_delta'].diff().fillna(0)
        else:
            # Calculate from timestamp
            df_temp['timestamp_epoch'] = df_temp['timestamp'].astype('int64') / 1e9
            df_temp['iat'] = df_temp['timestamp_epoch'].diff().fillna(0)
        
        # Convert timestamp to numeric seconds for window calculations
        if 'timestamp_epoch' not in df_temp.columns:
            df_temp['timestamp_epoch'] = df_temp['timestamp'].astype('int64') / 1e9
        
        # Calculate message rates using sliding windows
        df_temp = self._calculate_message_rates(df_temp, windows=[1, 10, 60])
        
        # Calculate rolling statistics (before outlier capping for stability)
        df_temp = self._calculate_rolling_statistics(df_temp, windows=[1, 10, 60])
        
        print("Temporal features generated: IAT, message rates (1s, 10s, 60s), rolling stats")
        
        return df_temp
    
    def _calculate_message_rates(self, df, windows=[1, 10, 60]):
        """Calculate message/packet rates over sliding time windows"""
        df_rates = df.copy()
        timestamp_epoch = df_rates['timestamp_epoch'].values
        
        for window_sec in windows:
            window_name = f'msg_rate_{window_sec}s'
            rates = []
            
            for i in range(len(df_rates)):
                current_time = timestamp_epoch[i]
                window_start = current_time - window_sec
                
                # Count packets in the window
                count = np.sum((timestamp_epoch >= window_start) & (timestamp_epoch <= current_time))
                rates.append(count / window_sec)  # Rate per second
            
            df_rates[window_name] = rates
        
        return df_rates
    
    def _calculate_rolling_statistics(self, df, windows=[1, 10, 60]):
        """Calculate rolling mean and std of mqtt_msg (payload length) over time windows"""
        df_stats = df.copy()
        timestamp_epoch = df_stats['timestamp_epoch'].values
        payload_len = df_stats['mqtt_payload_len'].fillna(0).values
        
        for window_sec in windows:
            mean_name = f'rolling_mean_payload_{window_sec}s'
            std_name = f'rolling_std_payload_{window_sec}s'
            
            means = []
            stds = []
            
            for i in range(len(df_stats)):
                current_time = timestamp_epoch[i]
                window_start = current_time - window_sec
                
                # Get payloads in the window
                mask = (timestamp_epoch >= window_start) & (timestamp_epoch <= current_time)
                window_payloads = payload_len[mask]
                
                if len(window_payloads) > 0:
                    means.append(np.mean(window_payloads))
                    stds.append(np.std(window_payloads) if len(window_payloads) > 1 else 0)
                else:
                    means.append(0)
                    stds.append(0)
            
            df_stats[mean_name] = means
            df_stats[std_name] = stds
        
        return df_stats
    
    def cap_outliers(self, df, columns=['tcp_len', 'mqtt_msg', 'iat'], method='iqr', factor=3.0):
        """
        Cap extreme values to prevent outliers from skewing statistical windows
        
        Args:
            df: DataFrame
            columns: List of columns to cap
            method: 'iqr' (interquartile range) or 'zscore'
            factor: Multiplier for IQR or threshold for z-score
        """
        print(f"\n=== Capping Outliers (method: {method}, factor: {factor}) ===")
        df_capped = df.copy()
        
        for col in columns:
            if col not in df_capped.columns:
                continue
            
            if method == 'iqr':
                Q1 = df_capped[col].quantile(0.25)
                Q3 = df_capped[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                # Cap values
                capped_lower = (df_capped[col] < lower_bound).sum()
                capped_upper = (df_capped[col] > upper_bound).sum()
                df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)
                
                print(f"  {col}: Capped {capped_lower} lower, {capped_upper} upper outliers")
            
            elif method == 'zscore':
                mean = df_capped[col].mean()
                std = df_capped[col].std()
                z_scores = np.abs((df_capped[col] - mean) / std)
                
                outliers = (z_scores > factor).sum()
                df_capped.loc[z_scores > factor, col] = np.sign(df_capped.loc[z_scores > factor, col] - mean) * factor * std + mean
                
                print(f"  {col}: Capped {outliers} outliers (|z| > {factor})")
        
        return df_capped
    
    def apply_encoding(self, df):
        """
        Apply categorical encoding using metadata
        Note: mqtt_topic is not available in the dataset
        """
        print("\n=== Applying Categorical Encoding ===")
        df_encoded = df.copy()
        
        # Encode mqtt_qos if it's categorical
        # mqtt_qos is already numeric (0, 1, 2), but we can ensure it's properly encoded
        if 'mqtt_qos' in df_encoded.columns:
            df_encoded['mqtt_qos'] = pd.to_numeric(df_encoded['mqtt_qos'], errors='coerce').fillna(0).astype(int)
            print(f"mqtt_qos: Unique values = {df_encoded['mqtt_qos'].unique()}")
        
        # Note: mqtt_topic encoding would go here if topic column existed
        # if 'mqtt_topic' in df_encoded.columns:
        #     # Use label encoding from metadata
        #     pass
        
        return df_encoded
    
    def apply_normalization(self, df, columns_to_normalize=None):
        """
        Apply Z-score standardization to temporal and length features
        
        Args:
            df: DataFrame
            columns_to_normalize: List of columns to normalize. If None, uses default set
        """
        print("\n=== Applying Z-score Standardization ===")
        
        if columns_to_normalize is None:
            columns_to_normalize = [
                'tcp_len', 'mqtt_msg', 'iat',
                'frame_length', 'mqtt_payload_len',
                'msg_rate_1s', 'msg_rate_10s', 'msg_rate_60s',
                'rolling_mean_payload_1s', 'rolling_mean_payload_10s', 'rolling_mean_payload_60s',
                'rolling_std_payload_1s', 'rolling_std_payload_10s', 'rolling_std_payload_60s'
            ]
        
        # Filter to only columns that exist
        columns_to_normalize = [col for col in columns_to_normalize if col in df.columns]
        
        df_norm = df.copy()
        
        # Fit and transform
        normalized_data = self.scaler.fit_transform(df_norm[columns_to_normalize])
        df_norm[columns_to_normalize] = normalized_data
        
        print(f"Normalized {len(columns_to_normalize)} columns: {columns_to_normalize}")
        
        return df_norm
    
    def filter_ml_features(self, df):
        """
        Filter DataFrame to keep only features needed for machine learning.
        Removes original/intermediate columns that have been mapped/derived.
        
        Args:
            df: DataFrame with all columns
            
        Returns:
            DataFrame with only ML-relevant features
        """
        # Columns to keep (features for ML)
        feature_columns = [
            # Packet-level features (mapped/derived)
            'frame_length',
            'protocol_flags',
            
            # MQTT-level features (mapped/derived)
            'mqtt_msgtype',
            'mqtt_qos',
            'mqtt_payload_len',
            'mqtt_retain',
            'mqtt_client_id_hash',
            
            # Temporal features
            'iat',
            'msg_rate_1s',
            'msg_rate_10s',
            'msg_rate_60s',
            'rolling_mean_payload_1s',
            'rolling_std_payload_1s',
            'rolling_mean_payload_10s',
            'rolling_std_payload_10s',
            'rolling_mean_payload_60s',
            'rolling_std_payload_60s',
            
            # Target label
            'Target'
        ]
        
        # Filter to only columns that exist in the DataFrame
        available_features = [col for col in feature_columns if col in df.columns]
        df_filtered = df[available_features].copy()
        
        removed_cols = [col for col in df.columns if col not in available_features]
        print(f"\n=== Filtering ML Features ===")
        print(f"Keeping {len(available_features)} feature columns")
        print(f"Removed {len(removed_cols)} columns: {removed_cols[:10]}{'...' if len(removed_cols) > 10 else ''}")
        
        return df_filtered
    
    def time_aware_split(self, df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        """
        Split data by timestamp (time-aware, no shuffling)
        
        Args:
            df: DataFrame sorted by timestamp
            train_ratio: Proportion for training (default 0.6)
            val_ratio: Proportion for validation (default 0.2)
            test_ratio: Proportion for test (default 0.2)
        """
        print(f"\n=== Time-aware Data Splitting ===")
        print(f"Ratios: Train={train_ratio}, Val={val_ratio}, Test={test_ratio}")
        
        # Ensure sorted by timestamp
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        n_total = len(df_sorted)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_df = df_sorted.iloc[:n_train].copy()
        val_df = df_sorted.iloc[n_train:n_train + n_val].copy()
        test_df = df_sorted.iloc[n_train + n_val:].copy()
        
        print(f"Total samples: {n_total}")
        print(f"Train: {len(train_df)} samples ({len(train_df)/n_total*100:.1f}%)")
        print(f"  Time range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
        print(f"Validation: {len(val_df)} samples ({len(val_df)/n_total*100:.1f}%)")
        print(f"  Time range: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
        print(f"Test: {len(test_df)} samples ({len(test_df)/n_total*100:.1f}%)")
        print(f"  Time range: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
        
        return train_df, val_df, test_df
    
    def process(self, output_dir=None, cap_outliers_before_stats=True):
        """
        Run the complete preprocessing pipeline
        
        Args:
            output_dir: Directory to save processed data
            cap_outliers_before_stats: If True, cap outliers before calculating rolling stats
        """
        print("=" * 60)
        print("IoT Dataset Preprocessing Pipeline")
        print("=" * 60)
        
        # Load data and metadata
        df = self.load_data()
        self.load_metadata()
        
        # Step 1: Feature mapping
        df = self.map_features(df)
        
        # Step 2: Temporal feature generation
        # Note: If capping outliers before stats, we need to do it here
        if cap_outliers_before_stats:
            df = self.cap_outliers(df, columns=['tcp_len', 'mqtt_msg'])
        
        df = self.generate_temporal_features(df)
        
        # Step 3: Additional outlier capping (if not done before)
        if not cap_outliers_before_stats:
            df = self.cap_outliers(df, columns=['tcp_len', 'mqtt_msg', 'iat'])
        else:
            df = self.cap_outliers(df, columns=['iat'])
        
        # Step 4: Encoding
        df = self.apply_encoding(df)
        
        # Step 5: Normalization
        df = self.apply_normalization(df)
        
        # Step 6: Time-aware splitting
        train_df, val_df, test_df = self.time_aware_split(df)
        
        # Step 7: Filter columns - keep only features needed for ML
        train_df = self.filter_ml_features(train_df)
        val_df = self.filter_ml_features(val_df)
        test_df = self.filter_ml_features(test_df)
        
        # Save results
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            train_df.to_csv(output_dir / 'train.csv', index=False)
            val_df.to_csv(output_dir / 'validation.csv', index=False)
            test_df.to_csv(output_dir / 'test.csv', index=False)
            
            print(f"\n=== Saved Processed Data ===")
            print(f"Train: {output_dir / 'train.csv'}")
            print(f"Validation: {output_dir / 'validation.csv'}")
            print(f"Test: {output_dir / 'test.csv'}")
        
        return train_df, val_df, test_df


def main():
    """Main execution function"""
    # Paths
    data_path = 'MQTTEEB-D_Final_Dataset/Preprocessed_Data/MQTTEEB-D_dataset_All_Processed.csv'
    metadata_dir = 'MQTTEEB-D_Final_Dataset/Preprocessed_Data'
    output_dir = 'processed_output'
    
    # Create preprocessor
    preprocessor = IoTDatasetPreprocessor(data_path, metadata_dir)
    
    # Run preprocessing
    train_df, val_df, test_df = preprocessor.process(output_dir=output_dir, cap_outliers_before_stats=True)
    
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

