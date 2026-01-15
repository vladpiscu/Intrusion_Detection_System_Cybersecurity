#!/usr/bin/env python3
"""
Visualization script for fault injection evaluation results.

This script reads JSON files containing fault injection evaluation results and creates:
1. A bar plot comparing precision, recall, and F1 score per class
2. A confusion matrix heatmap
3. Displays fault injection parameters

Usage:
    python visualization/visualize_fault_injection.py --input evaluation_results_corruption0.1_noise0.1_payload0.1_test/evaluation_results.json
    python visualization/visualize_fault_injection.py --input evaluation_results_corruption0.1_noise0.1_payload0.1_test/evaluation_results.json --output results_plots/
"""

import json
import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any


def load_results(json_path: str) -> Dict[str, Any]:
    """Load evaluation results from a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def get_class_names(results: Dict[str, Any]) -> List[str]:
    """Extract class names from the results."""
    # Check if results are nested under 'evaluation' key
    eval_data = results.get('evaluation', results)
    
    if 'per_attack_type_metrics' in eval_data:
        # Sort by class_label to ensure correct order
        metrics = eval_data['per_attack_type_metrics']
        sorted_classes = sorted(metrics.items(), key=lambda x: x[1]['class_label'])
        return [name.capitalize() for name, _ in sorted_classes]
    else:
        # Fallback: use generic class names
        num_classes = len(eval_data.get('precision_per_class', []))
        return [f'Class_{i}' for i in range(num_classes)]


def get_fault_injection_info(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract fault injection configuration information."""
    fault_info = results.get('fault_injection', {})
    info = {}
    
    if 'corruption' in fault_info:
        corr = fault_info['corruption']
        info['corruption'] = {
            'rate': corr.get('corruption_rate', 0),
            'noise_std': corr.get('noise_std', 0),
            'features': corr.get('corrupted_features', []),
            'n_features': corr.get('n_features_corrupted', 0)
        }
    
    if 'payload_loss' in fault_info:
        payload = fault_info['payload_loss']
        info['payload_loss'] = {
            'sample_rate': payload.get('sample_rate', 0),
            'affected_samples': payload.get('affected_samples', 0),
            'feature': payload.get('payload_feature', ''),
            'zeroed': payload.get('payload_zeroed', False)
        }
    
    return info


def plot_metrics_comparison(results: Dict[str, Any], output_path: str = None):
    """
    Create a bar plot comparing precision, recall, and F1 score per class.
    
    Args:
        results: Dictionary containing evaluation results
        output_path: Optional path to save the plot
    """
    # Get evaluation data (may be nested)
    eval_data = results.get('evaluation', results)
    
    precision = eval_data.get('precision_per_class', [])
    recall = eval_data.get('recall_per_class', [])
    f1 = eval_data.get('f1_per_class', [])
    class_names = get_class_names(results)
    
    if not precision or not recall or not f1:
        raise ValueError("Missing required metrics in results file")
    
    # Set up the plot
    x = np.arange(len(class_names))
    width = 0.25  # Width of bars
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create bars
    bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8, color='#3498db')
    bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8, color='#2ecc71')
    bars3 = ax.bar(x + width, f1, width, label='F1 Score', alpha=0.8, color='#e74c3c')
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    # Customize the plot
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Precision, Recall, and F1 Score per Class (Fault Injection)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add overall metrics as text
    accuracy = eval_data.get('accuracy', None)
    precision_macro = eval_data.get('precision_macro', None)
    recall_macro = eval_data.get('recall_macro', None)
    f1_macro = eval_data.get('f1_macro', None)
    
    metrics_text = []
    if accuracy is not None:
        metrics_text.append(f'Accuracy: {accuracy:.4f}')
    if precision_macro is not None:
        metrics_text.append(f'Precision (macro): {precision_macro:.4f}')
    if recall_macro is not None:
        metrics_text.append(f'Recall (macro): {recall_macro:.4f}')
    if f1_macro is not None:
        metrics_text.append(f'F1-Score (macro): {f1_macro:.4f}')
    
    if metrics_text:
        metrics_str = '\n'.join(metrics_text)
        ax.text(0.02, 0.98, metrics_str, 
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Add fault injection info
    fault_info = get_fault_injection_info(results)
    if fault_info:
        fault_text = []
        if 'corruption' in fault_info:
            corr = fault_info['corruption']
            fault_text.append(f"Corruption: {corr['rate']:.2%} (noise_std={corr['noise_std']})")
            if corr['features']:
                fault_text.append(f"Features: {', '.join(corr['features'])}")
        if 'payload_loss' in fault_info:
            payload = fault_info['payload_loss']
            fault_text.append(f"Payload Loss: {payload['sample_rate']:.2%} ({payload['affected_samples']} samples)")
        
        if fault_text:
            fault_str = '\n'.join(fault_text)
            ax.text(0.98, 0.98, fault_str,
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(results: Dict[str, Any], output_path: str = None):
    """
    Create a normalized confusion matrix heatmap.
    
    Args:
        results: Dictionary containing evaluation results
        output_path: Optional path to save the plot
    """
    # Get evaluation data (may be nested)
    eval_data = results.get('evaluation', results)
    
    cm = np.array(eval_data.get('confusion_matrix', []))
    
    if cm.size == 0:
        raise ValueError("Missing confusion matrix in results file")
    
    class_names = get_class_names(results)
    
    # Normalize the confusion matrix by row (each row sums to 1)
    # This shows the percentage of each true class that was predicted as each class
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use seaborn for better-looking heatmap with percentage formatting
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'}, ax=ax,
                linewidths=0.5, linecolor='gray', vmin=0, vmax=1)
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Normalized Confusion Matrix (Fault Injection)', fontsize=14, fontweight='bold', pad=20)
    
    # Add fault injection info as subtitle or text
    fault_info = get_fault_injection_info(results)
    if fault_info:
        fault_text = []
        if 'corruption' in fault_info:
            corr = fault_info['corruption']
            fault_text.append(f"Corruption: {corr['rate']:.2%}, Noise std: {corr['noise_std']}")
        if 'payload_loss' in fault_info:
            payload = fault_info['payload_loss']
            fault_text.append(f"Payload Loss: {payload['sample_rate']:.2%}")
        
        if fault_text:
            fault_str = ' | '.join(fault_text)
            ax.text(0.5, -0.08, fault_str,
                   transform=ax.transAxes, fontsize=10,
                   horizontalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize fault injection evaluation results from JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualization/visualize_fault_injection.py --input evaluation_results_corruption0.1_noise0.1_payload0.1_test/evaluation_results.json
  python visualization/visualize_fault_injection.py --input evaluation_results_corruption0.1_noise0.1_payload0.1_test/evaluation_results.json --output results_plots/
  python visualization/visualize_fault_injection.py --input evaluation_results_*/evaluation_results.json --output plots/
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to the JSON file containing fault injection evaluation results'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for saving plots. If not specified, plots will be displayed.'
    )
    
    parser.add_argument(
        '--metrics-only',
        action='store_true',
        help='Only generate the metrics comparison plot'
    )
    
    parser.add_argument(
        '--confusion-only',
        action='store_true',
        help='Only generate the confusion matrix plot'
    )
    
    args = parser.parse_args()
    
    # Load results
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        return
    
    print(f"Loading results from: {args.input}")
    results = load_results(args.input)
    
    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        metrics_path = os.path.join(args.output, 'fault_injection_metrics_comparison.png')
        cm_path = os.path.join(args.output, 'fault_injection_confusion_matrix.png')
    else:
        metrics_path = None
        cm_path = None
    
    # Generate plots
    try:
        if not args.confusion_only:
            print("Generating metrics comparison plot...")
            plot_metrics_comparison(results, metrics_path)
        
        if not args.metrics_only:
            print("Generating confusion matrix plot...")
            plot_confusion_matrix(results, cm_path)
        
        print("Visualization complete!")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        raise


if __name__ == '__main__':
    main()
