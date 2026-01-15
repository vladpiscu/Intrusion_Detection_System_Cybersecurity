#!/usr/bin/env python3
"""
Visualization script for feature importance.

This script reads a CSV file containing feature importance scores and creates:
1. A horizontal bar plot showing feature importance (sorted by importance)
2. Optionally shows cumulative importance

Usage:
    python visualization/visualize_feature_importance.py
    python visualization/visualize_feature_importance.py --input models/feature_importance.csv
    python visualization/visualize_feature_importance.py --input models/feature_importance.csv --output results_plots/
"""

import pandas as pd
import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


def load_feature_importance(csv_path: str) -> pd.DataFrame:
    """Load feature importance from a CSV file."""
    df = pd.read_csv(csv_path)
    
    # Sort by importance (descending)
    df = df.sort_values('importance', ascending=True)
    
    return df


def plot_feature_importance(df: pd.DataFrame, output_path: Optional[str] = None, 
                            top_n: Optional[int] = None, show_cumulative: bool = False):
    """
    Create a horizontal bar plot showing feature importance.
    
    Args:
        df: DataFrame with 'feature' and 'importance' columns
        output_path: Optional path to save the plot
        top_n: Optional number of top features to show (if None, shows all)
        show_cumulative: Whether to show cumulative importance line
    """
    # Select top N features if specified
    if top_n is not None and top_n < len(df):
        df_plot = df.tail(top_n)  # tail because sorted ascending
    else:
        df_plot = df
    
    # Calculate cumulative importance
    if show_cumulative:
        df_plot = df_plot.copy()
        df_plot['cumulative'] = df_plot['importance'].cumsum()
        total_importance = df_plot['importance'].sum()
    
    # Create figure with subplots
    if show_cumulative:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(8, len(df_plot) * 0.4)))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, max(8, len(df_plot) * 0.4)))
    
    # Plot 1: Horizontal bar chart
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_plot)))
    bars = ax1.barh(df_plot['feature'], df_plot['importance'], color=colors, alpha=0.8)
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, df_plot['importance'])):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.4f}',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Customize the plot
    ax1.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax1.set_title('Feature Importance', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add statistics
    total_importance = df_plot['importance'].sum()
    max_importance = df_plot['importance'].max()
    mean_importance = df_plot['importance'].mean()
    
    stats_text = [
        f'Total Features: {len(df_plot)}',
        f'Total Importance: {total_importance:.4f}',
        f'Max Importance: {max_importance:.4f}',
        f'Mean Importance: {mean_importance:.4f}'
    ]
    
    stats_str = '\n'.join(stats_text)
    ax1.text(0.98, 0.02, stats_str,
            transform=ax1.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Plot 2: Cumulative importance (if requested)
    if show_cumulative:
        ax2.plot(df_plot['cumulative'], df_plot['feature'], marker='o', linewidth=2, markersize=6)
        ax2.set_xlabel('Cumulative Importance', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Feature', fontsize=12, fontweight='bold')
        ax2.set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold', pad=20)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add percentage markers
        for i, (feature, cumul) in enumerate(zip(df_plot['feature'], df_plot['cumulative'])):
            pct = (cumul / total_importance) * 100 if total_importance > 0 else 0
            ax2.text(cumul, i, f' {pct:.1f}%',
                    va='center', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_feature_importance_simple(df: pd.DataFrame, output_path: Optional[str] = None, 
                                   top_n: Optional[int] = None):
    """
    Create a simple horizontal bar plot showing feature importance.
    
    Args:
        df: DataFrame with 'feature' and 'importance' columns
        output_path: Optional path to save the plot
        top_n: Optional number of top features to show (if None, shows all)
    """
    # Select top N features if specified
    if top_n is not None and top_n < len(df):
        df_plot = df.tail(top_n)  # tail because sorted ascending
    else:
        df_plot = df
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(8, len(df_plot) * 0.4)))
    
    # Create color gradient based on importance
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_plot)))
    bars = ax.barh(df_plot['feature'], df_plot['importance'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar, importance in zip(bars, df_plot['importance']):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.4f}',
                ha='left', va='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))
    
    # Customize the plot
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance Ranking', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add statistics
    total_importance = df_plot['importance'].sum()
    max_importance = df_plot['importance'].max()
    mean_importance = df_plot['importance'].mean()
    median_importance = df_plot['importance'].median()
    
    # Calculate top features contribution
    top_5_importance = df_plot.tail(5)['importance'].sum() if len(df_plot) >= 5 else total_importance
    top_5_pct = (top_5_importance / total_importance) * 100 if total_importance > 0 else 0
    
    stats_text = [
        f'Total Features: {len(df_plot)}',
        f'Total Importance: {total_importance:.4f}',
        f'Max: {max_importance:.4f}',
        f'Mean: {mean_importance:.4f}',
        f'Median: {median_importance:.4f}',
        f'Top 5 Features: {top_5_pct:.1f}%'
    ]
    
    stats_str = '\n'.join(stats_text)
    ax.text(0.98, 0.02, stats_str,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize feature importance from CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualization/visualize_feature_importance.py
  python visualization/visualize_feature_importance.py --input models/feature_importance.csv
  python visualization/visualize_feature_importance.py --input models/feature_importance.csv --output results_plots/
  python visualization/visualize_feature_importance.py --input models/feature_importance.csv --top-n 10
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='models/feature_importance.csv',
        help='Path to the CSV file containing feature importance (default: models/feature_importance.csv)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results_plots/feature_importance.png',
        help='Output path for saving the plot (default: results_plots/feature_importance.png)'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=None,
        help='Number of top features to display (if not specified, shows all)'
    )
    
    parser.add_argument(
        '--cumulative',
        action='store_true',
        help='Show cumulative importance plot alongside the bar chart'
    )
    
    args = parser.parse_args()
    
    # Load feature importance
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        return
    
    print(f"Loading feature importance from: {args.input}")
    df = load_feature_importance(args.input)
    
    print(f"Loaded {len(df)} features")
    print(f"\nTop 5 features:")
    for idx, row in df.tail(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.6f}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Generate plot
    try:
        print("Generating feature importance plot...")
        if args.cumulative:
            plot_feature_importance(df, args.output, top_n=args.top_n, show_cumulative=True)
        else:
            plot_feature_importance_simple(df, args.output, top_n=args.top_n)
        print("Visualization complete!")
        
    except Exception as e:
        print(f"Error generating visualization: {e}")
        raise


if __name__ == '__main__':
    main()
