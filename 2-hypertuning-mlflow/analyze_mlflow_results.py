"""
MLflow Analysis and Visualization Script
Extract results from MLflow and create visualizations
"""

import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def get_mlflow_runs():
    """Extract all runs from MLflow"""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("fashion_mnist_cnn_tuning")
    
    if experiment is None:
        print("No experiment found! Run mlflow_experiments.py first.")
        return None
    
    runs = client.search_runs(experiment.experiment_id)
    
    data = []
    for run in runs:
        row = {
            'run_id': run.info.run_id,
            'run_name': run.data.tags.get('mlflow.runName', 'unknown'),
            'status': run.info.status
        }
        
        # Add all parameters
        row.update(run.data.params)
        
        # Add all metrics
        row.update(run.data.metrics)
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Convert numeric columns
    numeric_columns = ['dropout_rate', 'epochs', 'batchsize', 'num_conv_layers', 
                      'learning_rate', 'final_val_accuracy', 'final_train_accuracy',
                      'final_val_loss', 'best_val_accuracy', 'total_params']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def plot_dropout_impact(df):
    """Visualize impact of dropout rate"""
    dropout_data = df[df['run_name'].str.contains('dropout_', na=False)]
    
    if dropout_data.empty:
        print("No dropout experiments found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy vs Dropout
    axes[0].plot(dropout_data['dropout_rate'], dropout_data['final_val_accuracy'], 
                 marker='o', linewidth=2, markersize=8, color='#2E86C1')
    axes[0].set_xlabel('Dropout Rate', fontsize=12)
    axes[0].set_ylabel('Validation Accuracy (%)', fontsize=12)
    axes[0].set_title('Impact of Dropout Rate on Accuracy', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Loss vs Dropout
    axes[1].plot(dropout_data['dropout_rate'], dropout_data['final_val_loss'], 
                 marker='s', linewidth=2, markersize=8, color='#E74C3C')
    axes[1].set_xlabel('Dropout Rate', fontsize=12)
    axes[1].set_ylabel('Validation Loss', fontsize=12)
    axes[1].set_title('Impact of Dropout Rate on Loss', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/dropout_impact.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/dropout_impact.png")
    plt.close()


def plot_batch_norm_comparison(df):
    """Compare with/without batch normalization"""
    bn_data = df[df['run_name'].str.contains('batchnorm_', na=False)]
    
    if bn_data.empty:
        print("No batch norm experiments found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Without BatchNorm', 'With BatchNorm']
    accuracies = []
    losses = []
    
    for use_bn in ['False', 'True']:
        row = bn_data[bn_data['use_batch_norm'] == use_bn]
        if not row.empty:
            accuracies.append(row['final_val_accuracy'].values[0])
            losses.append(row['final_val_loss'].values[0])
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='#27AE60')
    ax.bar(x + width/2, [l*20 for l in losses], width, label='Loss (×20)', color='#E67E22')
    
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Batch Normalization Impact', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('visualizations/batchnorm_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/batchnorm_comparison.png")
    plt.close()


def plot_conv_depth_impact(df):
    """Visualize impact of convolutional depth"""
    depth_data = df[df['run_name'].str.contains('conv_depth_', na=False)]
    
    if depth_data.empty:
        print("No conv depth experiments found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy vs Depth
    axes[0].plot(depth_data['num_conv_layers'], depth_data['final_val_accuracy'], 
                 marker='D', linewidth=2, markersize=10, color='#8E44AD')
    axes[0].set_xlabel('Number of Convolutional Layers', fontsize=12)
    axes[0].set_ylabel('Validation Accuracy (%)', fontsize=12)
    axes[0].set_title('Impact of CNN Depth', fontsize=14, fontweight='bold')
    axes[0].set_xticks([1, 2, 3])
    axes[0].grid(True, alpha=0.3)
    
    # Parameters vs Depth
    axes[1].plot(depth_data['num_conv_layers'], depth_data['total_params'], 
                 marker='o', linewidth=2, markersize=10, color='#16A085')
    axes[1].set_xlabel('Number of Convolutional Layers', fontsize=12)
    axes[1].set_ylabel('Total Parameters', fontsize=12)
    axes[1].set_title('Model Size vs CNN Depth', fontsize=14, fontweight='bold')
    axes[1].set_xticks([1, 2, 3])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/conv_depth_impact.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/conv_depth_impact.png")
    plt.close()


def plot_interactions_heatmap(df):
    """Plot heatmap of dropout x batch normalization interactions"""
    interaction_data = df[df['run_name'].str.contains('interaction_', na=False)]
    
    if interaction_data.empty:
        print("No interaction experiments found")
        return
    
    # Create pivot table
    pivot_data = interaction_data.pivot_table(
        values='final_val_accuracy',
        index='dropout_rate',
        columns='use_batch_norm',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlGnBu', 
                cbar_kws={'label': 'Validation Accuracy (%)'}, ax=ax)
    ax.set_xlabel('Use Batch Normalization', fontsize=12)
    ax.set_ylabel('Dropout Rate', fontsize=12)
    ax.set_title('Dropout × Batch Normalization Interaction', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/interactions_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/interactions_heatmap.png")
    plt.close()


def plot_top_configs(df, top_n=10):
    """Plot top N configurations by accuracy"""
    top_runs = df.nlargest(top_n, 'best_val_accuracy')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(top_runs))
    accuracies = top_runs['best_val_accuracy'].values
    names = top_runs['run_name'].values
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_runs)))
    bars = ax.barh(y_pos, accuracies, color=colors)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Best Validation Accuracy (%)', fontsize=12)
    ax.set_title(f'Top {top_n} Configurations by Accuracy', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(acc + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{acc:.2f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('visualizations/top_configurations.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/top_configurations.png")
    plt.close()


def create_summary_table(df):
    """Create summary table of all experiments"""
    summary = df[[
        'run_name', 'dropout_rate', 'use_batch_norm', 'num_conv_layers',
        'use_pooling', 'final_val_accuracy', 'final_val_loss', 'best_val_accuracy'
    ]].copy()
    
    summary = summary.sort_values('best_val_accuracy', ascending=False)
    summary.to_csv('results/experiment_summary.csv', index=False)
    print("\nSummary saved to: results/experiment_summary.csv")
    print("\nTop 5 Configurations:")
    print(summary.head().to_string(index=False))


def main():
    """Main analysis function"""
    print("="*70)
    print("MLflow EXPERIMENT ANALYSIS")
    print("="*70)
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Create output directories
    Path("visualizations").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    # Get data
    print("\nFetching MLflow runs...")
    df = get_mlflow_runs()
    
    if df is None or df.empty:
        print("No data found. Run mlflow_experiments.py first!")
        return
    
    print(f"Found {len(df)} experiment runs")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_dropout_impact(df)
    plot_batch_norm_comparison(df)
    plot_conv_depth_impact(df)
    plot_interactions_heatmap(df)
    plot_top_configs(df)
    
    # Create summary
    print("\nCreating summary table...")
    create_summary_table(df)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - visualizations/*.png (5 plots)")
    print("  - results/experiment_summary.csv")
    print("\nTo view MLflow UI:")
    print("  mlflow ui")
    print("  Then open: http://localhost:5000")
    print("="*70)


if __name__ == "__main__":
    main()
