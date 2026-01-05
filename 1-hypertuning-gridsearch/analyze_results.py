"""
Analysis and Visualization Script for Hyperparameter Tuning Results
This script analyzes the experiment results from modellogs and creates visualizations.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import toml
from pathlib import Path
import numpy as np
from datetime import datetime


def parse_toml_files(logdir: str = "modellogs"):
    """Parse all TOML files in the modellogs directory"""
    results = []
    logdir_path = Path(logdir)
    
    if not logdir_path.exists():
        print(f"Warning: {logdir} directory not found!")
        return pd.DataFrame()
    
    # Iterate through all subdirectories (each experiment run)
    for exp_dir in logdir_path.iterdir():
        if not exp_dir.is_dir():
            continue
        
        model_toml = exp_dir / "model.toml"
        settings_toml = exp_dir / "settings.toml"
        
        if model_toml.exists() and settings_toml.exists():
            try:
                model_data = toml.load(model_toml)
                settings_data = toml.load(settings_toml)
                
                # Combine data
                result = {
                    'experiment': exp_dir.name,
                    'timestamp': exp_dir.name,
                }
                
                # Extract model parameters
                if 'model' in model_data:
                    result.update({
                        'units1': model_data['model'].get('units1', None),
                        'units2': model_data['model'].get('units2', None),
                        'depth': model_data['model'].get('depth', 2),
                    })
                
                # Extract settings
                if 'model' in settings_data:
                    result.update({
                        'epochs': settings_data['model'].get('epochs', None),
                        'train_steps': settings_data['model'].get('train_steps', None),
                        'valid_steps': settings_data['model'].get('valid_steps', None),
                    })
                    
                    # Extract optimizer parameters
                    if 'optimizer_kwargs' in settings_data['model']:
                        result['learning_rate'] = settings_data['model']['optimizer_kwargs'].get('lr', None)
                
                results.append(result)
                
            except Exception as e:
                print(f"Error parsing {exp_dir}: {e}")
    
    return pd.DataFrame(results)


def create_heatmap(df: pd.DataFrame, x_col: str, y_col: str, value_col: str, title: str, output_file: str):
    """Create a heatmap for two variables"""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        print(f"Cannot create heatmap: missing columns {x_col} or {y_col}")
        return
    
    # Create pivot table
    pivot_data = df.pivot_table(values=value_col, index=y_col, columns=x_col, aggfunc='mean')
    
    if pivot_data.empty:
        print(f"No data available for heatmap: {title}")
        return
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlOrRd', cbar_kws={'label': value_col})
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def create_bar_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str, output_file: str):
    """Create a bar plot for a single variable"""
    if df.empty or x_col not in df.columns:
        print(f"Cannot create bar plot: missing column {x_col}")
        return
    
    plt.figure(figsize=(10, 6))
    df_sorted = df.sort_values(by=x_col)
    plt.bar(range(len(df_sorted)), df_sorted[y_col])
    plt.xticks(range(len(df_sorted)), df_sorted[x_col], rotation=45, ha='right')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def create_line_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str, output_file: str):
    """Create a line plot showing trends"""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        print(f"Cannot create line plot: missing columns {x_col} or {y_col}")
        return
    
    plt.figure(figsize=(10, 6))
    df_sorted = df.sort_values(by=x_col)
    plt.plot(df_sorted[x_col], df_sorted[y_col], marker='o', linewidth=2, markersize=8)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def analyze_experiments():
    """Main analysis function"""
    print("="*70)
    print("ANALYZING HYPERPARAMETER TUNING RESULTS")
    print("="*70 + "\n")
    
    # Parse results
    print("Parsing TOML files from modellogs...")
    df = parse_toml_files()
    
    if df.empty:
        print("No experiment results found! Please run grid_search_experiments.py first.")
        return
    
    print(f"Found {len(df)} experiments\n")
    print("Summary Statistics:")
    print(df.describe())
    print("\n")
    
    # Create visualizations directory
    viz_dir = Path("visualizations")
    viz_dir.mkdir(exist_ok=True)
    
    # Note: These visualizations assume you have performance metrics
    # Since we're working with TOML files that may not have final metrics,
    # we'll create placeholder visualizations based on configurations
    
    print("\nCreating visualizations...")
    
    # 1. Units heatmap (if we have units data)
    if 'units1' in df.columns and 'units2' in df.columns:
        # For demonstration, we'll create a count heatmap
        units_df = df.dropna(subset=['units1', 'units2'])
        if not units_df.empty:
            plt.figure(figsize=(10, 8))
            pivot = units_df.pivot_table(values='epochs', index='units2', columns='units1', aggfunc='count', fill_value=0)
            sns.heatmap(pivot, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Number of Experiments'})
            plt.title('Experiment Count: Units1 vs Units2')
            plt.xlabel('Units in Layer 1')
            plt.ylabel('Units in Layer 2')
            plt.tight_layout()
            plt.savefig(viz_dir / 'units_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {viz_dir / 'units_heatmap.png'}")
    
    # 2. Learning rate vs epochs
    if 'learning_rate' in df.columns and 'epochs' in df.columns:
        lr_df = df.dropna(subset=['learning_rate'])
        if not lr_df.empty:
            plt.figure(figsize=(10, 6))
            plt.scatter(lr_df['learning_rate'], lr_df['epochs'], s=100, alpha=0.6)
            plt.xscale('log')
            plt.xlabel('Learning Rate')
            plt.ylabel('Epochs')
            plt.title('Learning Rate vs Epochs Configuration')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(viz_dir / 'learning_rate_scatter.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {viz_dir / 'learning_rate_scatter.png'}")
    
    # 3. Depth distribution
    if 'depth' in df.columns:
        depth_df = df.dropna(subset=['depth'])
        if not depth_df.empty:
            plt.figure(figsize=(8, 6))
            depth_counts = depth_df['depth'].value_counts().sort_index()
            plt.bar(depth_counts.index, depth_counts.values, color='skyblue', edgecolor='black')
            plt.xlabel('Model Depth (Number of Layers)')
            plt.ylabel('Number of Experiments')
            plt.title('Distribution of Model Depths Tested')
            plt.xticks(depth_counts.index)
            plt.tight_layout()
            plt.savefig(viz_dir / 'depth_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {viz_dir / 'depth_distribution.png'}")
    
    # 4. Configuration summary
    plt.figure(figsize=(14, 10))
    
    subplot_idx = 1
    if 'epochs' in df.columns and not df['epochs'].dropna().empty:
        plt.subplot(2, 3, subplot_idx)
        df['epochs'].value_counts().sort_index().plot(kind='bar', color='coral')
        plt.title('Epochs Distribution')
        plt.xlabel('Epochs')
        plt.ylabel('Count')
        subplot_idx += 1
    
    if 'units1' in df.columns and not df['units1'].dropna().empty:
        plt.subplot(2, 3, subplot_idx)
        df['units1'].value_counts().sort_index().plot(kind='bar', color='lightblue')
        plt.title('Units1 Distribution')
        plt.xlabel('Units')
        plt.ylabel('Count')
        subplot_idx += 1
    
    if 'units2' in df.columns and not df['units2'].dropna().empty:
        plt.subplot(2, 3, subplot_idx)
        df['units2'].value_counts().sort_index().plot(kind='bar', color='lightgreen')
        plt.title('Units2 Distribution')
        plt.xlabel('Units')
        plt.ylabel('Count')
        subplot_idx += 1
    
    if 'learning_rate' in df.columns and not df['learning_rate'].dropna().empty:
        plt.subplot(2, 3, subplot_idx)
        df['learning_rate'].value_counts().sort_index().plot(kind='bar', color='orange')
        plt.title('Learning Rate Distribution')
        plt.xlabel('Learning Rate')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        subplot_idx += 1
    
    if 'depth' in df.columns and not df['depth'].dropna().empty:
        plt.subplot(2, 3, subplot_idx)
        df['depth'].value_counts().sort_index().plot(kind='bar', color='purple')
        plt.title('Depth Distribution')
        plt.xlabel('Depth')
        plt.ylabel('Count')
        subplot_idx += 1
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'configuration_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {viz_dir / 'configuration_summary.png'}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nVisualization saved in: {viz_dir.absolute()}")
    print("\nNext steps:")
    print("1. Review visualizations in the 'visualizations' directory")
    print("2. Open TensorBoard to see training curves:")
    print("   tensorboard --logdir=modellogs")
    print("3. Use the insights to write your report")
    
    return df


if __name__ == "__main__":
    df = analyze_experiments()
    
    # Save summary to CSV
    if not df.empty:
        output_csv = "experiment_summary.csv"
        df.to_csv(output_csv, index=False)
        print(f"\nExperiment summary saved to: {output_csv}")
