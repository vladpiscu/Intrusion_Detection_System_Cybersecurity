"""
Isolation Forest Anomaly Detection Script for IoT Cybersecurity Dataset
Simple binary anomaly detection (normal vs anomaly) with default parameters
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score
)
import joblib
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class IsolationForestTrainer:
    """
    Simple Isolation Forest anomaly detector trainer
    Uses default parameters with auto contamination detection
    Note: Isolation Forest is unsupervised, but we use labels for evaluation only
    """
    
    def __init__(self, train_path, val_path, test_path, output_dir='models', normal_class=3):
        """
        Initialize trainer
        
        Args:
            train_path: Path to training CSV file
            val_path: Path to validation CSV file
            test_path: Path to test CSV file
            output_dir: Directory to save models and results
            normal_class: Class label for normal/legitimate traffic (default: 3)
        """
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.test_path = Path(test_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.normal_class = normal_class
        
        self.model = None
        self.contamination = None
        self.best_max_samples = None
        self.best_n_estimators = None
        self.labels_flipped = False  # Track if we flipped labels (normal becomes minority)
        self.custom_threshold = None  # Custom threshold for better anomaly detection
        
    def load_data(self):
        """Load train, validation, and test datasets"""
        print("=" * 60)
        print("Loading Data")
        print("=" * 60)
        
        train_df = pd.read_csv(self.train_path)
        val_df = pd.read_csv(self.val_path)
        test_df = pd.read_csv(self.test_path)
        
        print(f"Train set: {len(train_df)} samples, {len(train_df.columns)} features")
        print(f"Validation set: {len(val_df)} samples, {len(val_df.columns)} features")
        print(f"Test set: {len(test_df)} samples, {len(test_df.columns)} features")
        
        feature_cols = [col for col in train_df.columns if col != 'Target']
        
        X_train = train_df[feature_cols].values
        y_train = train_df['Target'].values
        X_val = val_df[feature_cols].values
        y_val = val_df['Target'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['Target'].values
        
        print(f"\nFeatures: {len(feature_cols)}")
        
        # Convert labels to binary (normal vs anomaly)
        y_train_binary = np.where(y_train == self.normal_class, 1, -1)
        y_val_binary = np.where(y_val == self.normal_class, 1, -1)
        y_test_binary = np.where(y_test == self.normal_class, 1, -1)
        
        # Check binary distribution
        normal_count = np.sum(y_train_binary == 1)
        anomaly_count = np.sum(y_train_binary == -1)
        print(f"\nTrain binary distribution:")
        print(f"  Normal (1): {normal_count} ({normal_count/len(y_train_binary)*100:.2f}%)")
        print(f"  Anomaly (-1): {anomaly_count} ({anomaly_count/len(y_train_binary)*100:.2f}%)")
        
        return X_train, y_train_binary, X_val, y_val_binary, X_test, y_test_binary, feature_cols
    
    def test_max_samples(self, X_train, y_train_binary, X_val, y_val_binary, max_samples_options=None, n_estimators_options=None):
        """
        Test different max_samples and n_estimators values and select the best combination based on F1 score
        
        Args:
            X_train: Training features
            y_train_binary: Binary training labels (may be flipped)
            X_val: Validation features
            y_val_binary: Binary validation labels (in original space: 1=normal, -1=anomaly)
            max_samples_options: List of max_samples values to test. If None, uses defaults.
            n_estimators_options: List of n_estimators values to test. If None, uses defaults.
        
        Returns:
            Best max_samples value
        """
        print("\n" + "=" * 60)
        print("Testing Hyperparameters (max_samples and n_estimators)")
        print("=" * 60)
        
        if max_samples_options is None:
            # Default options based on dataset size
            n_samples = len(X_train)
            max_samples_options = ['auto']
            if n_samples >= 256:
                max_samples_options.append(256)
            if n_samples >= 512:
                max_samples_options.append(512)
            if n_samples >= 1024:
                max_samples_options.append(1024)
            if n_samples >= 2048:
                max_samples_options.append(2048)
        
        if n_estimators_options is None:
            n_estimators_options = [100, 200, 300]
        
        print(f"Testing {len(max_samples_options)} max_samples values: {max_samples_options}")
        print(f"Testing {len(n_estimators_options)} n_estimators values: {n_estimators_options}")
        print(f"Total combinations: {len(max_samples_options) * len(n_estimators_options)}")
        
        # Ensure contamination is valid (must be <= 0.5)
        if self.contamination is None or self.contamination > 0.5:
            raise ValueError(f"Contamination must be set and <= 0.5, got {self.contamination}")
        
        print(f"Using contamination: {self.contamination:.4f}")
        if self.labels_flipped:
            print("Note: Labels are flipped (normal class is treated as contamination)")
        
        best_f1 = -1
        best_max_samples = None
        best_n_estimators = None
        best_precision = 0
        best_recall = 0
        
        for n_estimators in n_estimators_options:
            for max_samples in max_samples_options:
                model = IsolationForest(
                    contamination=self.contamination,
                    max_samples=max_samples,
                    n_estimators=n_estimators,
                    random_state=42,
                    n_jobs=-1
                )
                
                model.fit(X_train)
                y_pred = model.predict(X_val)
                
                # If labels were flipped during training, flip predictions back
                if self.labels_flipped:
                    y_pred = np.where(y_pred == 1, -1, 1)
                
                # Calculate macro averages
                prec_anomaly = precision_score(y_val_binary, y_pred, pos_label=-1, zero_division=0)
                rec_anomaly = recall_score(y_val_binary, y_pred, pos_label=-1, zero_division=0)
                f1_anomaly = f1_score(y_val_binary, y_pred, pos_label=-1, zero_division=0)
                
                prec_normal = precision_score(y_val_binary, y_pred, pos_label=1, zero_division=0)
                rec_normal = recall_score(y_val_binary, y_pred, pos_label=1, zero_division=0)
                f1_normal = f1_score(y_val_binary, y_pred, pos_label=1, zero_division=0)
                
                prec_macro = (prec_normal + prec_anomaly) / 2.0
                rec_macro = (rec_normal + rec_anomaly) / 2.0
                f1_macro = (f1_normal + f1_anomaly) / 2.0
                
                print(f"  n_estimators={n_estimators}, max_samples={max_samples}: "
                      f"Precision (macro)={prec_macro:.4f}, Recall (macro)={rec_macro:.4f}, F1 (macro)={f1_macro:.4f}")
                
                if f1_macro > best_f1:
                    best_f1 = f1_macro
                    best_max_samples = max_samples
                    best_n_estimators = n_estimators
                    best_precision = prec_macro
                    best_recall = rec_macro
        
        self.best_max_samples = best_max_samples
        self.best_n_estimators = best_n_estimators
        print(f"\nBest combination:")
        print(f"  n_estimators: {best_n_estimators}")
        print(f"  max_samples: {best_max_samples}")
        print(f"  F1 (macro)={best_f1:.4f}, Precision (macro)={best_precision:.4f}, Recall (macro)={best_recall:.4f}")
        
        return best_max_samples
    
    def tune_threshold(self, X_val, y_val_binary, min_precision=None):
        """
        Tune the decision threshold on validation set to optimize F1 score for anomaly detection.
        Uses score_samples to find optimal threshold.
        
        Args:
            X_val: Validation features
            y_val_binary: Binary validation labels (1=normal, -1=anomaly) in original space
            min_precision: Optional minimum precision constraint
        
        Returns:
            Best threshold value
        """
        print("\n" + "=" * 60)
        print("Threshold Tuning for Anomaly Detection")
        print("=" * 60)
        
        if self.model is None:
            raise ValueError("Model must be trained first.")
        
        # Get anomaly scores (lower scores = more anomalous)
        y_scores = self.model.score_samples(X_val)
        
        # If labels were flipped, we need to adjust our approach
        # The model predicts -1 for the minority class (which is normal if flipped)
        # We want to detect anomalies, so we need to find the right threshold
        
        # Test percentile thresholds from 1% to 50%
        percentiles = np.concatenate([
            np.arange(1, 10, 0.5),
            np.arange(10, 25, 1.0),
            np.arange(25, 50, 2.0)
        ])
        
        print(f"Testing {len(percentiles)} threshold values...")
        if min_precision is not None:
            print(f"Minimum precision constraint: {min_precision}")
        
        best_f1 = -1
        best_threshold = None
        best_precision = 0
        best_recall = 0
        
        for percentile in percentiles:
            threshold = np.percentile(y_scores, percentile)
            # Lower scores indicate anomalies, so predict -1 if score <= threshold
            y_pred = np.where(y_scores <= threshold, -1, 1)
            
            # If labels were flipped during training, flip predictions back
            if self.labels_flipped:
                y_pred = np.where(y_pred == 1, -1, 1)
            
            # Calculate macro averages
            prec_anomaly = precision_score(y_val_binary, y_pred, pos_label=-1, zero_division=0)
            rec_anomaly = recall_score(y_val_binary, y_pred, pos_label=-1, zero_division=0)
            f1_anomaly = f1_score(y_val_binary, y_pred, pos_label=-1, zero_division=0)
            
            prec_normal = precision_score(y_val_binary, y_pred, pos_label=1, zero_division=0)
            rec_normal = recall_score(y_val_binary, y_pred, pos_label=1, zero_division=0)
            f1_normal = f1_score(y_val_binary, y_pred, pos_label=1, zero_division=0)
            
            prec_macro = (prec_normal + prec_anomaly) / 2.0
            rec_macro = (rec_normal + rec_anomaly) / 2.0
            f1_macro = (f1_normal + f1_anomaly) / 2.0
            
            # Apply minimum precision constraint (on macro average)
            if min_precision is not None and prec_macro < min_precision:
                continue
            
            if f1_macro > best_f1:
                best_f1 = f1_macro
                best_threshold = threshold
                best_precision = prec_macro
                best_recall = rec_macro
        
        if best_threshold is None:
            print("Warning: No threshold found satisfying constraints. Using default model threshold.")
            self.custom_threshold = None
            return None
        
        self.custom_threshold = best_threshold
        
        print(f"\nBest threshold (percentile): {best_threshold:.4f}")
        print(f"  Precision (macro): {best_precision:.4f}")
        print(f"  Recall (macro):    {best_recall:.4f}")
        print(f"  F1-Score (macro):  {best_f1:.4f}")
        
        return best_threshold
    
    def train(self, X_train, y_train_binary):
        """
        Train the Isolation Forest model on training data
        
        Args:
            X_train: Training features
            y_train_binary: Binary training labels (for calculating contamination only)
        """
        print("\n" + "=" * 60)
        print("Training Isolation Forest Model")
        print("=" * 60)
        
        # Contamination should already be set, but calculate if not
        if self.contamination is None:
            contamination_raw = np.sum(y_train_binary == -1) / len(y_train_binary)
            # Cap at 0.5 if it exceeds the limit
            if contamination_raw > 0.5:
                self.contamination = 0.5
            else:
                self.contamination = contamination_raw
        
        print(f"Using contamination (anomaly rate): {self.contamination:.4f}")
        
        # Use best hyperparameters if found, otherwise use defaults
        max_samples = self.best_max_samples if self.best_max_samples is not None else 'auto'
        n_estimators = self.best_n_estimators if self.best_n_estimators is not None else 100
        print(f"Using max_samples: {max_samples}")
        print(f"Using n_estimators: {n_estimators}")
        
        # Train with computed contamination and selected hyperparameters
        self.model = IsolationForest(
            contamination=self.contamination,
            max_samples=max_samples,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        
        print(f"\nTraining Isolation Forest with {len(X_train)} samples...")
        self.model.fit(X_train)
        print("Training complete!")
        
        return self.model
    
    def evaluate(self, X, y_binary, dataset_name='Test'):
        """
        Evaluate model on a dataset for binary anomaly detection
        
        Args:
            X: Features
            y_binary: Binary labels in original space (1=normal, -1=anomaly)
            dataset_name: Name of the dataset for printing
        """
        print("\n" + "=" * 60)
        print(f"Evaluation on {dataset_name} Set")
        print("=" * 60)
        
        if self.model is None:
            raise ValueError("Model must be trained first.")
        
        # Use custom threshold if available, otherwise use model's default prediction
        if self.custom_threshold is not None:
            y_scores = self.model.score_samples(X)
            y_pred = np.where(y_scores <= self.custom_threshold, -1, 1)
            # If labels were flipped during training, flip predictions back
            if self.labels_flipped:
                y_pred = np.where(y_pred == 1, -1, 1)
            print(f"Using custom threshold: {self.custom_threshold:.4f}")
        else:
            # Use model's default prediction
            y_pred = self.model.predict(X)
            # If labels were flipped during training, flip predictions back to original space
            if self.labels_flipped:
                y_pred = np.where(y_pred == 1, -1, 1)  # Flip: 1->-1, -1->1
        
        # Calculate metrics
        accuracy = accuracy_score(y_binary, y_pred)
        cm = confusion_matrix(y_binary, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Calculate per-class metrics for macro averaging
        # Normal class (label=1): precision, recall, f1
        normal_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        normal_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        normal_f1 = 2 * (normal_precision * normal_recall) / (normal_precision + normal_recall) if (normal_precision + normal_recall) > 0 else 0.0
        
        # Anomaly class (label=-1): precision, recall, f1
        anomaly_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        anomaly_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        anomaly_f1 = 2 * (anomaly_precision * anomaly_recall) / (anomaly_precision + anomaly_recall) if (anomaly_precision + anomaly_recall) > 0 else 0.0
        
        # Calculate macro averages
        precision_macro = (normal_precision + anomaly_precision) / 2.0
        recall_macro = (normal_recall + anomaly_recall) / 2.0
        f1_macro = (normal_f1 + anomaly_f1) / 2.0
        
        # Also keep per-class metrics for reporting
        precision = anomaly_precision  # For backward compatibility
        recall = anomaly_recall
        f1 = anomaly_f1
        
        # Calculate ROC-AUC
        y_scores = self.model.score_samples(X)
        y_scores_normalized = -y_scores
        y_scores_normalized = (y_scores_normalized - y_scores_normalized.min()) / (y_scores_normalized.max() - y_scores_normalized.min() + 1e-10)
        y_binary_01 = np.where(y_binary == -1, 1, 0)
        
        try:
            roc_auc = roc_auc_score(y_binary_01, y_scores_normalized)
            avg_precision = average_precision_score(y_binary_01, y_scores_normalized)
        except Exception as e:
            print(f"ROC-AUC calculation error: {e}")
            roc_auc = None
            avg_precision = None
        
        # Print metrics
        print(f"\nBinary Classification Metrics (Normal vs Anomaly):")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"\nPer-Class Metrics:")
        print(f"  Normal - Precision: {normal_precision:.4f}, Recall: {normal_recall:.4f}, F1: {normal_f1:.4f}")
        print(f"  Anomaly - Precision: {anomaly_precision:.4f}, Recall: {anomaly_recall:.4f}, F1: {anomaly_f1:.4f}")
        print(f"\nMacro Averages:")
        print(f"  Precision (macro): {precision_macro:.4f}")
        print(f"  Recall (macro):    {recall_macro:.4f}")
        print(f"  F1-Score (macro):  {f1_macro:.4f}")
        if roc_auc is not None:
            print(f"  ROC-AUC: {roc_auc:.4f}")
        if avg_precision is not None:
            print(f"  Average Precision: {avg_precision:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Normal  Anomaly")
        print(f"Actual Normal  {tn:6d}  {fp:6d}")
        print(f"       Anomaly {fn:6d}  {tp:6d}")
        
        report = classification_report(y_binary, y_pred, target_names=['Normal', 'Anomaly'], zero_division=0)
        print(f"\nClassification Report:")
        print(report)
        
        # Prepare results dictionary with macro averages
        results = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_score_macro': float(f1_macro),
            'precision_per_class': [float(normal_precision), float(anomaly_precision)],
            'recall_per_class': [float(normal_recall), float(anomaly_recall)],
            'f1_per_class': [float(normal_f1), float(anomaly_f1)],
            'precision_anomaly': float(anomaly_precision),  # For backward compatibility
            'recall_anomaly': float(anomaly_recall),
            'f1_score_anomaly': float(anomaly_f1),
            'confusion_matrix': cm.tolist(),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'average_precision': float(avg_precision) if avg_precision is not None else None,
            'classification_report': report
        }
        
        return results
    
    def save_model(self, filename='isolation_forest_model.joblib'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_path = self.output_dir / filename
        joblib.dump(self.model, model_path)
        print(f"\nModel saved to: {model_path}")
    
    def save_results(self, results, filename='evaluation_results.json'):
        """Save evaluation results to JSON"""
        results_path = self.output_dir / filename
        
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results_serializable[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                results_serializable[key] = float(value) if not np.isnan(value) else None
            elif value is None:
                results_serializable[key] = None
            else:
                results_serializable[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Results saved to: {results_path}")
    
    def run_full_pipeline(self, save_model=True):
        """
        Run the complete training pipeline
        
        Args:
            save_model: Whether to save the trained model
        """
        print("=" * 60)
        print("Isolation Forest Anomaly Detection Pipeline")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Normal class (legitimate traffic): {self.normal_class}")
        
        # Load data
        (X_train, y_train_binary, X_val, y_val_binary,
         X_test, y_test_binary, feature_names) = self.load_data()
        
        # Combine train and validation for training
        X_train_full = np.vstack([X_train, X_val])
        y_train_full_binary = np.hstack([y_train_binary, y_val_binary])
        print(f"\nCombined training set: {len(X_train_full)} samples")
        
        # Calculate contamination from full training set
        anomaly_rate = np.sum(y_train_full_binary == -1) / len(y_train_full_binary)
        normal_rate = np.sum(y_train_full_binary == 1) / len(y_train_full_binary)
        
        # Store original labels for evaluation (keep in original space: 1=normal, -1=anomaly)
        y_val_binary_original = y_val_binary.copy()
        y_test_binary_original = y_test_binary.copy()
        
        # IsolationForest only accepts contamination in (0.0, 0.5] or 'auto'
        # If anomaly rate > 0.5, flip labels and use normal class as contamination
        if anomaly_rate > 0.5:
            print(f"\nAnomaly rate ({anomaly_rate:.4f}) exceeds 0.5 limit.")
            print(f"Flipping labels: treating normal class ({normal_rate:.4f}) as contamination.")
            self.labels_flipped = True
            # Flip labels for training: normal becomes -1, anomalies become 1
            y_train_full_binary = np.where(y_train_full_binary == 1, -1, 1)
            y_val_binary = np.where(y_val_binary == 1, -1, 1)
            y_test_binary = np.where(y_test_binary == 1, -1, 1)
            self.contamination = normal_rate
            print(f"Using contamination (normal class rate): {self.contamination:.4f}")
        else:
            self.labels_flipped = False
            self.contamination = anomaly_rate
            print(f"Using contamination (anomaly rate): {self.contamination:.4f}")
        
        # Test different hyperparameters on validation set
        self.test_max_samples(X_train_full, y_train_full_binary, X_val, y_val_binary_original)
        
        # Train final model with best parameters
        self.train(X_train_full, y_train_full_binary)
        
        # Tune threshold on validation set to optimize anomaly detection
        self.tune_threshold(X_val, y_val_binary_original, min_precision=0.1)
        
        # Evaluate on validation and test sets
        val_results = self.evaluate(X_val, y_val_binary_original, 'Validation')
        test_results = self.evaluate(X_test, y_test_binary_original, 'Test')
        
        # Save model and results
        if save_model:
            self.save_model()
        
        self.save_results(test_results, 'isolation_forest_test_evaluation_results.json')
        
        # Save summary
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'normal_class': int(self.normal_class),
            'labels_flipped': self.labels_flipped,
            'model_parameters': {
                'contamination': float(self.contamination),
                'max_samples': self.best_max_samples if self.best_max_samples is not None else 'auto',
                'n_estimators': self.best_n_estimators if self.best_n_estimators is not None else 100,
                'custom_threshold': float(self.custom_threshold) if self.custom_threshold is not None else None,
                'random_state': 42
            },
            'test_metrics': {
                'accuracy': float(test_results['accuracy']),
                'precision_macro': float(test_results['precision_macro']),
                'recall_macro': float(test_results['recall_macro']),
                'f1_score_macro': float(test_results['f1_score_macro']),
                'precision_per_class': test_results.get('precision_per_class', []),
                'recall_per_class': test_results.get('recall_per_class', []),
                'f1_per_class': test_results.get('f1_per_class', []),
                'roc_auc': test_results.get('roc_auc'),
                'average_precision': test_results.get('average_precision')
            }
        }
        
        self.save_results(summary, 'isolation_forest_training_summary.json')
        
        print("\n" + "=" * 60)
        print("Training Pipeline Complete!")
        print("=" * 60)
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return test_results


def main():
    """Main execution function"""
    train_path = 'processed_output/train.csv'
    val_path = 'processed_output/validation.csv'
    test_path = 'processed_output/test.csv'
    output_dir = 'models'
    
    trainer = IsolationForestTrainer(train_path, val_path, test_path, output_dir, normal_class=3)
    
    test_results = trainer.run_full_pipeline(save_model=True)
    
    print("\nFinal Test Results:")
    print(f"  Accuracy:  {test_results['accuracy']:.4f}")
    print(f"  Precision (macro): {test_results['precision_macro']:.4f}")
    print(f"  Recall (macro):    {test_results['recall_macro']:.4f}")
    print(f"  F1-Score (macro):  {test_results['f1_score_macro']:.4f}")
    if test_results.get('roc_auc'):
        print(f"  ROC-AUC: {test_results['roc_auc']:.4f}")
    if test_results.get('average_precision'):
        print(f"  Average Precision: {test_results['average_precision']:.4f}")


if __name__ == '__main__':
    main()
