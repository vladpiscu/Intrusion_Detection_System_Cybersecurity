#!/usr/bin/env python3
"""
Visualization script for Isolation Forest evaluation results.

This script reads the isolation forest evaluation results JSON file and creates:
1. A confusion matrix heatmap for binary anomaly detection (Normal vs Anomaly)

Usage:
    python visualization/visualize_isolation_forest.py
    python visualization/visualize_isolation_forest.py --input models/isolation_forest_test_evaluation_results.json
    python visualization/visualize_isolation_forest.py --input models/isolation_forest_test_evaluation_results.json --output results_plots/
"""

import json
import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any


def load_results(json_path: str) -> Dict[str, Any]:
    """Load evaluation results from a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_confusion_matrix(results: Dict[str, Any], output_path: str = None):
    """
    Create a confusion matrix heatmap for binary anomaly detection.
    
    Args:
        results: Dictionary containing evaluation results
        output_path: Optional path to save the plot
    """
    cm = np.array(results.get('confusion_matrix', []))
    
    if cm.size == 0:
        raise ValueError("Missing confusion matrix in results file")
    
    # For binary classification: Normal vs Anomaly
    class_names = ['Normal', 'Anomaly']
    
    # Extract metrics for display
    accuracy = results.get('accuracy', None)
    precision = results.get('precision', None)
    recall = results.get('recall', None)
    f1_score = results.get('f1_score', None)
    
    # Create figure with two subplots: one for raw counts, one for normalized
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Raw confusion matrix with counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax1,
                linewidths=1, linecolor='gray', vmin=0)
    
    ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax1.set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold', pad=15)
    
    # Add annotations for TP, TN, FP, FN
    tn, fp, fn, tp = results.get('true_negatives', 0), \
                     results.get('false_positives', 0), \
                     results.get('false_negatives', 0), \
                     results.get('true_positives', 0)
    
    # Add text annotations
    ax1.text(0.5, -0.15, f'TN: {tn:,} | FP: {fp:,}', 
             transform=ax1.transAxes, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax1.text(0.5, -0.22, f'FN: {fn:,} | TP: {tp:,}', 
             transform=ax1.transAxes, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Plot 2: Normalized confusion matrix (percentages)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'}, ax=ax2,
                linewidths=1, linecolor='gray', vmin=0, vmax=1)
    
    ax2.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Isolation Forest evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualization/visualize_isolation_forest.py
  python visualization/visualize_isolation_forest.py --input models/isolation_forest_test_evaluation_results.json
  python visualization/visualize_isolation_forest.py --input models/isolation_forest_test_evaluation_results.json --output results_plots/
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='models/isolation_forest_test_evaluation_results.json',
        help='Path to the JSON file containing isolation forest evaluation results (default: models/isolation_forest_test_evaluation_results.json)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results_plots/isolation_forest_confusion_matrix.png',
        help='Output path for saving the confusion matrix plot (default: results_plots/isolation_forest_confusion_matrix.png)'
    )
    
    args = parser.parse_args()
    
    # Load results
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        return
    
    print(f"Loading results from: {args.input}")
    results = load_results(args.input)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Generate confusion matrix plot
    try:
        print("Generating confusion matrix plot...")
        plot_confusion_matrix(results, args.output)
        print("Visualization complete!")
        
    except Exception as e:
        print(f"Error generating visualization: {e}")
        raise


if __name__ == '__main__':
    main()
