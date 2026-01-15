"""
Evaluate trained models with fault injection
Applies fault injection to test data and evaluates model performance
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
from datetime import datetime
import warnings
import sys
warnings.filterwarnings('ignore')

# Import from same package
# Handle both direct execution and module execution
# Add parent directory to path to allow imports when running as script
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from fault_injection import FaultInjector, get_fault_injection_params_from_config, generate_output_folder_name
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.ensemble import IsolationForest


class FaultInjectionEvaluator:
    """
    Evaluates trained models with fault injection applied to test data
    """
    
    def __init__(self, model_path, test_path, output_dir='evaluation_results', normal_class=3):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained model file (.joblib)
            test_path: Path to test CSV file
            output_dir: Directory to save evaluation results
            normal_class: Class label for normal/legitimate traffic (for Isolation Forest, default: 3)
        """
        self.model_path = Path(model_path)
        self.test_path = Path(test_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.normal_class = normal_class
        
        self.model = None
        self.test_df = None
        self.feature_names = None
        self.is_isolation_forest = False
        
    def load_model(self):
        """Load the trained model"""
        print(f"Loading model from: {self.model_path}")
        self.model = joblib.load(self.model_path)
        
        # Detect if model is Isolation Forest
        self.is_isolation_forest = isinstance(self.model, IsolationForest)
        if self.is_isolation_forest:
            print("Model type: Isolation Forest (binary anomaly detection)")
        else:
            print("Model type: Multi-class classifier (e.g., Random Forest)")
        
        print("Model loaded successfully")
        return self.model
    
    def load_data(self):
        """Load test data"""
        print(f"\nLoading test data from: {self.test_path}")
        self.test_df = pd.read_csv(self.test_path)
        
        # Separate features and target
        self.feature_names = [col for col in self.test_df.columns if col != 'Target']
        X_test = self.test_df[self.feature_names].values
        y_test_original = self.test_df['Target'].values
        
        print(f"Test set: {len(X_test)} samples, {len(self.feature_names)} features")
        
        # Check class distribution
        print(f"\nTest class distribution (original):")
        unique, counts = np.unique(y_test_original, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count} ({count/len(y_test_original)*100:.2f}%)")
        
        # For Isolation Forest, convert to binary labels
        if self.is_isolation_forest:
            y_test = np.where(y_test_original == self.normal_class, 1, -1)
            print(f"\nConverted to binary labels for Isolation Forest:")
            print(f"  Normal (1): {np.sum(y_test == 1)} ({np.sum(y_test == 1)/len(y_test)*100:.2f}%)")
            print(f"  Anomaly (-1): {np.sum(y_test == -1)} ({np.sum(y_test == -1)/len(y_test)*100:.2f}%)")
        else:
            y_test = y_test_original
        
        return X_test, y_test
    
    def apply_fault_injection(
        self,
        X_test,
        corruption_rate=0.2,
        noise_std=0.1,
        payload_loss_rate=0.15,
        payload_feature='mqtt_payload_len',
        drop_features=None,
        random_state=42
    ):
        """
        Apply fault injection to test data
        
        Args:
            X_test: Test features
            corruption_rate: Percentage of features to corrupt
            noise_std: Standard deviation of Gaussian noise
            payload_loss_rate: Percentage of samples to affect with payload loss
            payload_feature: Name of payload length feature
            drop_features: Additional features to drop/zero
            random_state: Random seed
            
        Returns:
            Tuple of (corrupted_X_test, fault_info)
        """
        print("\n" + "=" * 60)
        print("Applying Fault Injection")
        print("=" * 60)
        
        injector = FaultInjector(random_state=random_state)
        
        X_corrupted, fault_info = injector.inject_combined_faults(
            data=X_test,
            feature_names=self.feature_names,
            corruption_rate=corruption_rate,
            noise_std=noise_std,
            payload_loss_rate=payload_loss_rate,
            payload_feature=payload_feature,
            drop_features=drop_features,
            exclude_features=['Target']
        )
        
        print(f"\nFault Injection Summary:")
        print(f"  Corrupted features: {fault_info['corruption']['n_features_corrupted']}")
        print(f"    Features: {fault_info['corruption']['corrupted_features']}")
        print(f"    Corruption rate: {fault_info['corruption']['corruption_rate']:.2%}")
        print(f"  Affected samples: {fault_info['payload_loss']['affected_samples']}")
        print(f"    Sample rate: {fault_info['payload_loss']['sample_rate']:.2%}")
        if fault_info['payload_loss']['dropped_features']:
            print(f"    Dropped features: {fault_info['payload_loss']['dropped_features']}")
        
        return X_corrupted, fault_info
    
    def evaluate(self, X_test, y_test, dataset_name='Test'):
        """
        Evaluate model on test data
        
        Args:
            X_test: Test features
            y_test: True labels
            dataset_name: Name of dataset for printing
            
        Returns:
            Dictionary with evaluation results
        """
        print("\n" + "=" * 60)
        print(f"Evaluation on {dataset_name} Set")
        print("=" * 60)
        
        if self.model is None:
            raise ValueError("Model must be loaded first. Call load_model() first.")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Check if model supports predict_proba
        try:
            y_pred_proba = self.model.predict_proba(X_test)
            has_proba = True
        except:
            y_pred_proba = None
            has_proba = False
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Handle binary (Isolation Forest) vs multi-class (Random Forest) differently
        if self.is_isolation_forest:
            # Binary classification for Isolation Forest
            all_classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
            # Ensure we have both -1 and 1
            if -1 not in all_classes:
                all_classes = [-1] + all_classes
            if 1 not in all_classes:
                all_classes = all_classes + [1]
            all_classes = sorted(all_classes)
            
            # Binary metrics
            precision_binary = precision_score(y_test, y_pred, average='binary', zero_division=0, pos_label=1)
            recall_binary = recall_score(y_test, y_pred, average='binary', zero_division=0, pos_label=1)
            f1_binary = f1_score(y_test, y_pred, average='binary', zero_division=0, pos_label=1)
            precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            # Per-class metrics
            precision_per_class = precision_score(y_test, y_pred, labels=all_classes, average=None, zero_division=0)
            recall_per_class = recall_score(y_test, y_pred, labels=all_classes, average=None, zero_division=0)
            f1_per_class = f1_score(y_test, y_pred, labels=all_classes, average=None, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=all_classes)
            
            # Binary class names
            class_names = ['Anomaly', 'Normal']
            class_label_map = {-1: 'Anomaly', 1: 'Normal'}
            
            # Classification report
            report = classification_report(
                y_test, y_pred,
                labels=all_classes,
                target_names=class_names,
                zero_division=0
            )
            
            # Per-class metrics dictionary
            per_attack_metrics = {}
            for idx, class_label in enumerate(all_classes):
                class_idx = all_classes.index(class_label)
                class_name = class_label_map.get(int(class_label), f'class_{int(class_label)}')
                per_attack_metrics[class_name] = {
                    'class_label': int(class_label),
                    'precision': float(precision_per_class[class_idx]),
                    'recall': float(recall_per_class[class_idx]),
                    'f1_score': float(f1_per_class[class_idx]),
                    'support': int(np.sum(y_test == class_label))
                }
            
            # Print results
            print(f"\nOverall Metrics:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision (binary, pos_label=1): {precision_binary:.4f}")
            print(f"  Recall (binary, pos_label=1):    {recall_binary:.4f}")
            print(f"  F1-Score (binary, pos_label=1):  {f1_binary:.4f}")
            print(f"  Precision (macro): {precision_macro:.4f}")
            print(f"  Recall (macro):    {recall_macro:.4f}")
            print(f"  F1-Score (macro):  {f1_macro:.4f}")
            
            print(f"\nPer-Class Metrics:")
            print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
            print("-" * 65)
            for class_label in all_classes:
                class_idx = all_classes.index(class_label)
                class_name = class_label_map.get(int(class_label), f'class_{int(class_label)}')
                support = int(np.sum(y_test == class_label))
                print(f"{class_name:<15} {precision_per_class[class_idx]:<12.4f} {recall_per_class[class_idx]:<12.4f} {f1_per_class[class_idx]:<12.4f} {support:<10}")
            
            print(f"\nConfusion Matrix:")
            print(cm)
            
            print(f"\nClassification Report:")
            print(report)
            
            # ROC-AUC for binary classification
            roc_auc = None
            if has_proba and y_pred_proba is not None:
                try:
                    # For binary, use the positive class probabilities
                    if y_pred_proba.shape[1] == 2:
                        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1], average='macro')
                    else:
                        roc_auc = roc_auc_score(y_test, y_pred_proba, average='macro')
                    print(f"\nROC-AUC Score: {roc_auc:.4f}")
                except Exception as e:
                    print(f"\nROC-AUC calculation skipped: {e}")
            else:
                # Try using decision function for Isolation Forest
                try:
                    decision_scores = self.model.decision_function(X_test)
                    # Convert y_test to 0/1 for roc_auc_score
                    y_test_binary = (y_test == 1).astype(int)
                    roc_auc = roc_auc_score(y_test_binary, decision_scores)
                    print(f"\nROC-AUC Score (from decision function): {roc_auc:.4f}")
                except Exception as e:
                    print(f"\nROC-AUC calculation skipped: {e}")
            
            results = {
                'model_type': 'isolation_forest',
                'accuracy': float(accuracy),
                'precision_binary': float(precision_binary),
                'recall_binary': float(recall_binary),
                'f1_binary': float(f1_binary),
                'precision_macro': float(precision_macro),
                'recall_macro': float(recall_macro),
                'f1_macro': float(f1_macro),
                'precision_per_class': precision_per_class.tolist(),
                'recall_per_class': recall_per_class.tolist(),
                'f1_per_class': f1_per_class.tolist(),
                'per_attack_type_metrics': per_attack_metrics,
                'confusion_matrix': cm.tolist(),
                'roc_auc': float(roc_auc) if roc_auc is not None else None,
                'classification_report': report
            }
        else:
            # Multi-class classification for Random Forest
            # Attack type mapping
            attack_type_map = {
                0: 'bruteforce',
                1: 'dos',
                2: 'flood',
                3: 'legitimate',
                4: 'malformed',
                5: 'slowite'
            }
            
            # Get all unique classes from both y_test and y_pred
            all_classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
            unique_classes = sorted(np.unique(y_test))
            
            # Per-class metrics
            precision_per_class = precision_score(y_test, y_pred, labels=all_classes, average=None, zero_division=0)
            recall_per_class = recall_score(y_test, y_pred, labels=all_classes, average=None, zero_division=0)
            f1_per_class = f1_score(y_test, y_pred, labels=all_classes, average=None, zero_division=0)
            
            precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=all_classes)
            
            # Create class names for all classes
            class_names = [f'Class_{i}' for i in all_classes]
            
            # Classification report
            report = classification_report(
                y_test, y_pred,
                labels=all_classes,
                target_names=class_names,
                zero_division=0
            )
            
            # Create per-attack-type metrics dictionary
            per_attack_metrics = {}
            for idx, class_label in enumerate(all_classes):
                class_idx = all_classes.index(class_label)
                attack_name = attack_type_map.get(int(class_label), f'class_{int(class_label)}')
                per_attack_metrics[attack_name] = {
                    'class_label': int(class_label),
                    'precision': float(precision_per_class[class_idx]),
                    'recall': float(recall_per_class[class_idx]),
                    'f1_score': float(f1_per_class[class_idx]),
                    'support': int(np.sum(y_test == class_label))
                }
            
            # Print results
            print(f"\nOverall Metrics:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision (macro): {precision_macro:.4f}")
            print(f"  Recall (macro):    {recall_macro:.4f}")
            print(f"  F1-Score (macro):  {f1_macro:.4f}")
            
            print(f"\nPer-Attack-Type Metrics:")
            print(f"{'Attack Type':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
            print("-" * 65)
            for class_label in all_classes:
                class_idx = all_classes.index(class_label)
                attack_name = attack_type_map.get(int(class_label), f'class_{int(class_label)}')
                support = int(np.sum(y_test == class_label))
                print(f"{attack_name:<15} {precision_per_class[class_idx]:<12.4f} {recall_per_class[class_idx]:<12.4f} {f1_per_class[class_idx]:<12.4f} {support:<10}")
            
            print(f"\nConfusion Matrix:")
            print(cm)
            
            print(f"\nClassification Report:")
            print(report)
            
            # Calculate ROC-AUC if probabilities available
            roc_auc = None
            if has_proba and y_pred_proba is not None:
                try:
                    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
                    print(f"\nROC-AUC Score (macro, OVR): {roc_auc:.4f}")
                except Exception as e:
                    print(f"\nROC-AUC calculation skipped: {e}")
            
            results = {
                'model_type': 'multi_class',
                'accuracy': float(accuracy),
                'precision_macro': float(precision_macro),
                'recall_macro': float(recall_macro),
                'f1_macro': float(f1_macro),
                'precision_per_class': precision_per_class.tolist(),
                'recall_per_class': recall_per_class.tolist(),
                'f1_per_class': f1_per_class.tolist(),
                'per_attack_type_metrics': per_attack_metrics,
                'confusion_matrix': cm.tolist(),
                'roc_auc': float(roc_auc) if roc_auc is not None else None,
                'classification_report': report
            }
        
        return results
    
    def save_results(self, results, fault_info, filename='evaluation_results.json'):
        """Save evaluation results to JSON"""
        results_path = self.output_dir / filename
        
        # Combine results with fault injection info
        full_results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': str(self.model_path),
            'test_path': str(self.test_path),
            'fault_injection': fault_info,
            'evaluation': results
        }
        
        with open(results_path, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")
        return results_path
    
    def run_evaluation(
        self,
        corruption_rate=0.2,
        noise_std=0.1,
        payload_loss_rate=0.15,
        payload_feature='mqtt_payload_len',
        drop_features=None,
        fault_injection_config=None,
        random_state=42
    ):
        """
        Run complete evaluation pipeline with fault injection
        
        Args:
            corruption_rate: Percentage of features to corrupt
            noise_std: Standard deviation of Gaussian noise
            payload_loss_rate: Percentage of samples to affect with payload loss
            payload_feature: Name of payload length feature
            drop_features: Additional features to drop/zero
            fault_injection_config: Path to fault injection config JSON file (optional)
            random_state: Random seed
            
        Returns:
            Dictionary with evaluation results
        """
        print("=" * 60)
        print("Fault Injection Evaluation Pipeline")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load model (must be loaded before load_data to detect model type)
        self.load_model()
        
        # Load data
        X_test, y_test = self.load_data()
        
        # Load fault injection config if provided
        if fault_injection_config is not None:
            try:
                config_params = get_fault_injection_params_from_config(fault_injection_config)
                corruption_rate = config_params['corruption_rate']
                noise_std = config_params['noise_std']
                payload_loss_rate = config_params['payload_loss_rate']
                payload_feature = config_params['payload_feature']
                drop_features = config_params.get('drop_features', [])
                random_state = config_params.get('random_state', random_state)
                print(f"\nLoaded fault injection configuration from: {fault_injection_config}")
            except Exception as e:
                warnings.warn(f"Failed to load fault injection config: {e}. Using provided parameters.")
        
        # Generate output directory name based on fault injection parameters
        new_output_dir = generate_output_folder_name(
            base_dir=str(self.output_dir),
            enable_fault_injection=True,
            corruption_rate=corruption_rate,
            noise_std=noise_std,
            payload_loss_rate=payload_loss_rate,
            fault_injection_target='test'
        )
        self.output_dir = Path(new_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput directory: {self.output_dir}")
        
        # Apply fault injection
        X_corrupted, fault_info = self.apply_fault_injection(
            X_test=X_test,
            corruption_rate=corruption_rate,
            noise_std=noise_std,
            payload_loss_rate=payload_loss_rate,
            payload_feature=payload_feature,
            drop_features=drop_features,
            random_state=random_state
        )
        
        # Evaluate on corrupted data
        results = self.evaluate(X_corrupted, y_test, 'Corrupted Test')
        
        # Save results
        self.save_results(results, fault_info)
        
        print("\n" + "=" * 60)
        print("Evaluation Complete!")
        print("=" * 60)
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return results


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained model with fault injection')
    parser.add_argument('--model', type=str, default='models/random_forest_model.joblib',
                       help='Path to trained model file')
    parser.add_argument('--test', type=str, default='processed_output/test.csv',
                       help='Path to test CSV file')
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to fault injection config JSON file (optional)')
    parser.add_argument('--corruption-rate', type=float, default=0.2,
                       help='Corruption rate (0.0-1.0)')
    parser.add_argument('--noise-std', type=float, default=0.1,
                       help='Noise standard deviation')
    parser.add_argument('--payload-loss-rate', type=float, default=0.15,
                       help='Payload loss rate (0.0-1.0)')
    parser.add_argument('--payload-feature', type=str, default='mqtt_payload_len',
                       help='Name of payload length feature')
    parser.add_argument('--normal-class', type=int, default=3,
                       help='Class label for normal/legitimate traffic (for Isolation Forest, default: 3)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = FaultInjectionEvaluator(
        model_path=args.model,
        test_path=args.test,
        output_dir=args.output,
        normal_class=args.normal_class
    )
    
    # Run evaluation
    results = evaluator.run_evaluation(
        corruption_rate=args.corruption_rate,
        noise_std=args.noise_std,
        payload_loss_rate=args.payload_loss_rate,
        payload_feature=args.payload_feature,
        fault_injection_config=args.config,
        random_state=42
    )
    
    print("\nFinal Results:")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision (macro): {results['precision_macro']:.4f}")
    print(f"  Recall (macro):    {results['recall_macro']:.4f}")
    print(f"  F1-Score (macro):  {results['f1_macro']:.4f}")


if __name__ == '__main__':
    main()
