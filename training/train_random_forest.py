"""
Random Forest Classifier Training Script for IoT Cybersecurity Dataset
Includes hyperparameter tuning, cross-validation, and comprehensive evaluation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import joblib
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Note: Fault injection has been moved to evaluate_with_fault_injection.py
# Training scripts now focus solely on training models


class RandomForestTrainer:
    """
    Random Forest classifier trainer with hyperparameter tuning and evaluation
    """
    
    def __init__(self, train_path, val_path, test_path, output_dir='models'):
        """
        Initialize trainer
        
        Args:
            train_path: Path to training CSV file
            val_path: Path to validation CSV file
            test_path: Path to test CSV file
            output_dir: Directory to save models and results
        """
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.test_path = Path(test_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.best_params = None
        self.cv_results = None
        
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
        
        # Separate features and target
        feature_cols = [col for col in train_df.columns if col != 'Target']
        
        X_train = train_df[feature_cols].values
        y_train = train_df['Target'].values
        X_val = val_df[feature_cols].values
        y_val = val_df['Target'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['Target'].values
        
        print(f"\nFeatures: {len(feature_cols)}")
        print(f"Feature names: {feature_cols}")
        
        # Check class distribution
        print(f"\nTrain class distribution:")
        unique, counts = np.unique(y_train, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count} ({count/len(y_train)*100:.2f}%)")
        
        print(f"\nValidation class distribution:")
        unique, counts = np.unique(y_val, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count} ({count/len(y_val)*100:.2f}%)")
        
        print(f"\nTest class distribution:")
        unique, counts = np.unique(y_test, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count} ({count/len(y_test)*100:.2f}%)")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
    
    def hyperparameter_tuning(self, X_train, y_train, method='randomized', n_iter=50, cv=5):
        """
        Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV
        
        Args:
            X_train: Training features
            y_train: Training labels
            method: 'grid' or 'randomized'
            n_iter: Number of iterations for RandomizedSearchCV
            cv: Number of cross-validation folds
        """
        print("\n" + "=" * 60)
        print(f"Hyperparameter Tuning ({method.upper()})")
        print("=" * 60)
        
        # Define parameter grid for Random Forest
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'class_weight': [None, 'balanced']
        }
        
        # Base model
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Use StratifiedKFold for cross-validation to handle class imbalance
        cv_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        if method == 'grid':
            print("Using GridSearchCV (this may take a while...)")
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_fold,
                scoring='f1_macro',  # Use macro F1 for multi-class
                n_jobs=-1,
                verbose=1
            )
        else:
            print(f"Using RandomizedSearchCV with {n_iter} iterations...")
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=cv_fold,
                scoring='f1_macro',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
        
        # Fit the search
        print("\nStarting hyperparameter search...")
        search.fit(X_train, y_train)
        
        self.best_params = search.best_params_
        self.cv_results = search.cv_results_
        
        print(f"\nBest parameters found:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        print(f"\nBest cross-validation score (F1-macro): {search.best_score_:.4f}")
        
        # Create model with best parameters
        self.model = search.best_estimator_
        
        return self.model
    
    def cross_validate(self, X_train, y_train, cv=5):
        """
        Perform cross-validation on training data
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv: Number of folds
        """
        print("\n" + "=" * 60)
        print(f"Cross-Validation ({cv}-fold)")
        print("=" * 60)
        
        if self.model is None:
            raise ValueError("Model must be trained first. Call hyperparameter_tuning() or train() first.")
        
        cv_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Evaluate multiple metrics
        scoring_metrics = {
            'accuracy': 'accuracy',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro',
            'f1_macro': 'f1_macro'
        }
        
        cv_scores = {}
        for metric_name, metric_scorer in scoring_metrics.items():
            scores = cross_val_score(self.model, X_train, y_train, cv=cv_fold, scoring=metric_scorer, n_jobs=-1)
            cv_scores[metric_name] = scores
            print(f"\n{metric_name.upper()}:")
            print(f"  Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            print(f"  Scores: {scores}")
        
        return cv_scores
    
    def evaluate(self, X, y, dataset_name='Test'):
        """
        Evaluate model on a dataset
        
        Args:
            X: Features
            y: True labels
            dataset_name: Name of the dataset for printing
        """
        print("\n" + "=" * 60)
        print(f"Evaluation on {dataset_name} Set")
        print("=" * 60)
        
        if self.model is None:
            raise ValueError("Model must be trained first.")
        
        # Predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision_macro = precision_score(y, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y, y_pred, average='macro', zero_division=0)
        
        # Attack type mapping
        attack_type_map = {
            0: 'bruteforce',
            1: 'dos',
            2: 'flood',
            3: 'legitimate',
            4: 'malformed',
            5: 'slowite'
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y, y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Get unique classes in sorted order
        unique_classes = sorted(np.unique(y))
        class_names = [f'Class_{i}' for i in unique_classes]
        report = classification_report(y, y_pred, target_names=class_names, zero_division=0)
        
        # Create per-attack-type metrics dictionary
        per_attack_metrics = {}
        for idx, class_label in enumerate(unique_classes):
            attack_name = attack_type_map.get(int(class_label), f'class_{int(class_label)}')
            per_attack_metrics[attack_name] = {
                'class_label': int(class_label),
                'precision': float(precision_per_class[idx]),
                'recall': float(recall_per_class[idx]),
                'f1_score': float(f1_per_class[idx]),
                'support': int(np.sum(y == class_label))
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
        for idx, class_label in enumerate(unique_classes):
            attack_name = attack_type_map.get(int(class_label), f'class_{int(class_label)}')
            support = int(np.sum(y == class_label))
            print(f"{attack_name:<15} {precision_per_class[idx]:<12.4f} {recall_per_class[idx]:<12.4f} {f1_per_class[idx]:<12.4f} {support:<10}")
        
        print(f"\nConfusion Matrix:")
        print(cm)
        
        print(f"\nClassification Report:")
        print(report)
        
        # Calculate ROC-AUC (for multi-class, use 'ovr' or 'ovo')
        try:
            roc_auc = roc_auc_score(y, y_pred_proba, multi_class='ovr', average='macro')
            print(f"\nROC-AUC Score (macro, OVR): {roc_auc:.4f}")
        except Exception as e:
            print(f"\nROC-AUC calculation skipped: {e}")
            roc_auc = None
        
        results = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'per_attack_type_metrics': per_attack_metrics,
            'confusion_matrix': cm.tolist(),
            'roc_auc': roc_auc,
            'classification_report': report
        }
        
        return results
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the model on training data and evaluate on validation set
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        print("\n" + "=" * 60)
        print("Training Final Model")
        print("=" * 60)
        
        if self.model is None:
            # Use default parameters if no tuning was done
            print("No hyperparameter tuning performed. Using default parameters.")
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        
        # Train the model
        print(f"\nTraining Random Forest with {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_results = self.evaluate(X_val, y_val, 'Validation')
        
        return val_results
    
    def get_feature_importance(self, feature_names):
        """
        Get and display feature importance
        
        Args:
            feature_names: List of feature names
        """
        if self.model is None:
            raise ValueError("Model must be trained first.")
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\n" + "=" * 60)
        print("Feature Importance (Top 20)")
        print("=" * 60)
        
        for i in range(min(20, len(feature_names))):
            idx = indices[i]
            print(f"{i+1:2d}. {feature_names[idx]:<35} {importances[idx]:.6f}")
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': [feature_names[i] for i in indices],
            'importance': importances[indices]
        })
        
        return importance_df
    
    def save_model(self, filename='random_forest_model.joblib'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_path = self.output_dir / filename
        joblib.dump(self.model, model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Save best parameters
        if self.best_params:
            params_path = self.output_dir / 'best_parameters.json'
            with open(params_path, 'w') as f:
                json.dump(self.best_params, f, indent=2)
            print(f"Best parameters saved to: {params_path}")
    
    def save_results(self, results, filename='evaluation_results.json'):
        """Save evaluation results to JSON"""
        results_path = self.output_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results_serializable[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                results_serializable[key] = float(value)
            else:
                results_serializable[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Results saved to: {results_path}")
    
    def run_full_pipeline(
        self, 
        tuning_method='randomized', 
        n_iter=50, 
        cv=5, 
        save_model=True
    ):
        """
        Run the complete training pipeline
        
        Args:
            tuning_method: 'grid' or 'randomized' for hyperparameter tuning
            n_iter: Number of iterations for RandomizedSearchCV
            cv: Number of cross-validation folds
            save_model: Whether to save the trained model
            
        Note: For fault injection evaluation, use evaluate_with_fault_injection.py
        """
        print("=" * 60)
        print("Random Forest Training Pipeline")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load data
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names = self.load_data()
        
        # Combine train and validation for hyperparameter tuning and cross-validation
        # (as per best practices: use all available training data)
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.hstack([y_train, y_val])
        print(f"\nCombined training set for tuning: {len(X_train_full)} samples")
        
        # Hyperparameter tuning
        self.hyperparameter_tuning(X_train_full, y_train_full, method=tuning_method, n_iter=n_iter, cv=cv)
        
        # Cross-validation
        cv_scores = self.cross_validate(X_train_full, y_train_full, cv=cv)
        
        # Train final model on all training data
        self.train(X_train_full, y_train_full, X_val, y_val)
        
        # Evaluate on test set
        test_results = self.evaluate(X_test, y_test, 'Test')
        
        # Feature importance
        importance_df = self.get_feature_importance(feature_names)
        importance_path = self.output_dir / 'feature_importance.csv'
        importance_df.to_csv(importance_path, index=False)
        print(f"\nFeature importance saved to: {importance_path}")
        
        # Save model and results
        if save_model:
            self.save_model()
        
        self.save_results(test_results, 'test_evaluation_results.json')
        
        # Save summary
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'best_parameters': self.best_params,
            'cv_scores': {k: {'mean': float(v.mean()), 'std': float(v.std())} for k, v in cv_scores.items()},
            'test_metrics': {
                'accuracy': float(test_results['accuracy']),
                'precision_macro': float(test_results['precision_macro']),
                'recall_macro': float(test_results['recall_macro']),
                'f1_macro': float(test_results['f1_macro'])
            }
        }
        
        self.save_results(summary, 'training_summary.json')
        
        print("\n" + "=" * 60)
        print("Training Pipeline Complete!")
        print("=" * 60)
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return test_results


def main():
    """Main execution function"""
    # Paths
    train_path = 'processed_output/train.csv'
    val_path = 'processed_output/validation.csv'
    test_path = 'processed_output/test.csv'
    output_dir = 'models'
    
    # Create trainer
    trainer = RandomForestTrainer(train_path, val_path, test_path, output_dir)
    
    # Run full pipeline
    # Note: Use 'randomized' for faster tuning (recommended), or 'grid' for exhaustive search
    # For fault injection evaluation, use evaluate_with_fault_injection.py
    test_results = trainer.run_full_pipeline(
        tuning_method='randomized',  # 'randomized' or 'grid'
        n_iter=10,  # Number of iterations for RandomizedSearchCV
        cv=5,  # Number of cross-validation folds
        save_model=True
    )
    
    print("\nFinal Test Results:")
    print(f"  Accuracy:  {test_results['accuracy']:.4f}")
    print(f"  Precision (macro): {test_results['precision_macro']:.4f}")
    print(f"  Recall (macro):    {test_results['recall_macro']:.4f}")
    print(f"  F1-Score (macro):  {test_results['f1_macro']:.4f}")


if __name__ == '__main__':
    main()

