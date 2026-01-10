"""
Analyze Ray Tune Results
Extracts metrics from Ray Tune experiment logs and creates visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import json
from loguru import logger

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def extract_trial_data(experiment_path: Path) -> pd.DataFrame:
    """
    Extract trial data from Ray Tune experiment directory.
    
    Args:
        experiment_path: Path to Ray Tune experiment (e.g., logs/ray_results/flowers)
    
    Returns:
        DataFrame with trial results
    """
    trials_data = []
    
    # Find all trial directories
    trial_dirs = [d for d in experiment_path.iterdir() if d.is_dir() and d.name.startswith('trial_')]
    
    logger.info(f"Found {len(trial_dirs)} trial directories")
    
    for trial_dir in trial_dirs:
        try:
            # Read params.json
            params_file = trial_dir / "params.json"
            if not params_file.exists():
                logger.warning(f"No params.json in {trial_dir.name}")
                continue
                
            with open(params_file, 'r') as f:
                params = json.load(f)
            
            # Read result.json (final result) - Ray Tune writes one JSON per line
            result_file = trial_dir / "result.json"
            if not result_file.exists():
                logger.warning(f"No result.json in {trial_dir.name}")
                continue
                
            # Read the last line which has the final result
            with open(result_file, 'r') as f:
                lines = f.readlines()
                if not lines:
                    logger.warning(f"Empty result.json in {trial_dir.name}")
                    continue
                result = json.loads(lines[-1])  # Last line has final result
            
            # Combine params and results
            trial_data = {
                'trial_id': trial_dir.name,
                'num_conv_layers': params.get('num_conv_layers'),
                'start_filters': params.get('start_filters'),
                'fc_units': params.get('fc_units'),
                'dropout': params.get('dropout'),
                'lr': params.get('lr'),
                'batch_size': params.get('batch_size'),
                'final_accuracy': result.get('accuracy'),
                'final_loss': result.get('loss'),
                'epochs_completed': result.get('training_iteration'),
                'stopped_early': result.get('training_iteration', 10) < 10,
                'training_time': result.get('time_total_s', 0),
            }
            
            trials_data.append(trial_data)
            
        except Exception as e:
            logger.error(f"Error processing {trial_dir.name}: {e}")
            continue
    
    df = pd.DataFrame(trials_data)
    logger.info(f"Successfully extracted {len(df)} trials")
    
    return df


def create_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create comprehensive visualizations of hyperparameter tuning results.
    
    Args:
        df: DataFrame with trial results
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Overall Performance Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(df['final_accuracy'], bins=20, edgecolor='black', alpha=0.7)
    axes[0].axvline(df['final_accuracy'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {df["final_accuracy"].mean():.3f}')
    axes[0].axvline(df['final_accuracy'].median(), color='green', linestyle='--',
                    label=f'Median: {df["final_accuracy"].median():.3f}')
    axes[0].set_xlabel('Final Validation Accuracy')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Final Accuracies')
    axes[0].legend()
    
    axes[1].hist(df['epochs_completed'], bins=range(1, 12), edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Epochs Completed')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('ASHA Early Stopping Distribution')
    axes[1].set_xticks(range(1, 11))
    
    plt.tight_layout()
    plt.savefig(output_dir / '01_overall_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Created overall performance plot")
    
    # 2. Parameter Impact - Boxplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # num_conv_layers
    df_sorted = df.sort_values('num_conv_layers')
    axes[0, 0].boxplot([df[df['num_conv_layers'] == val]['final_accuracy'].values 
                        for val in sorted(df['num_conv_layers'].unique())],
                       labels=sorted(df['num_conv_layers'].unique()))
    axes[0, 0].set_xlabel('Number of Conv Layers')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Impact of Network Depth')
    
    # start_filters
    axes[0, 1].boxplot([df[df['start_filters'] == val]['final_accuracy'].values 
                        for val in sorted(df['start_filters'].unique())],
                       labels=sorted(df['start_filters'].unique()))
    axes[0, 1].set_xlabel('Starting Filters')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Impact of Filter Size')
    
    # fc_units
    axes[0, 2].boxplot([df[df['fc_units'] == val]['final_accuracy'].values 
                        for val in sorted(df['fc_units'].unique())],
                       labels=sorted(df['fc_units'].unique()))
    axes[0, 2].set_xlabel('FC Units')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].set_title('Impact of FC Layer Width')
    
    # dropout
    axes[1, 0].scatter(df['dropout'], df['final_accuracy'], alpha=0.6, s=50)
    axes[1, 0].set_xlabel('Dropout Rate')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Impact of Dropout')
    
    # learning rate
    axes[1, 1].scatter(df['lr'], df['final_accuracy'], alpha=0.6, s=50)
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_xlabel('Learning Rate (log scale)')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Impact of Learning Rate')
    
    # batch_size
    axes[1, 2].boxplot([df[df['batch_size'] == val]['final_accuracy'].values 
                        for val in sorted(df['batch_size'].unique())],
                       labels=sorted(df['batch_size'].unique()))
    axes[1, 2].set_xlabel('Batch Size')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].set_title('Impact of Batch Size')
    
    plt.tight_layout()
    plt.savefig(output_dir / '02_parameter_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Created parameter impact plot")
    
    # 3. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    corr_cols = ['num_conv_layers', 'start_filters', 'fc_units', 'dropout', 
                 'lr', 'batch_size', 'final_accuracy']
    corr_matrix = df[corr_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Hyperparameter Correlation Matrix', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / '03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Created correlation heatmap")
    
    # 4. Top vs Bottom Performers
    fig, ax = plt.subplots(figsize=(12, 6))
    
    top_5 = df.nlargest(5, 'final_accuracy')
    bottom_5 = df.nsmallest(5, 'final_accuracy')
    
    x_top = range(len(top_5))
    x_bottom = range(len(top_5), len(top_5) + len(bottom_5))
    
    ax.bar(x_top, top_5['final_accuracy'], color='green', alpha=0.7, label='Top 5')
    ax.bar(x_bottom, bottom_5['final_accuracy'], color='red', alpha=0.7, label='Bottom 5')
    
    ax.set_xlabel('Trial Rank')
    ax.set_ylabel('Final Accuracy')
    ax.set_title('Top 5 vs Bottom 5 Performers')
    ax.legend()
    ax.axhline(df['final_accuracy'].mean(), color='blue', linestyle='--', 
               label=f'Mean: {df["final_accuracy"].mean():.3f}')
    
    plt.tight_layout()
    plt.savefig(output_dir / '04_top_vs_bottom.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Created top vs bottom plot")
    
    # 5. Early Stopping Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    stopped_early = df[df['stopped_early'] == True]
    full_training = df[df['stopped_early'] == False]
    
    axes[0].scatter(stopped_early['final_accuracy'], stopped_early['epochs_completed'], 
                   alpha=0.6, s=100, label='Stopped Early', color='red')
    axes[0].scatter(full_training['final_accuracy'], full_training['epochs_completed'], 
                   alpha=0.6, s=100, label='Full Training', color='green')
    axes[0].set_xlabel('Final Accuracy')
    axes[0].set_ylabel('Epochs Completed')
    axes[0].set_title('Early Stopping vs Accuracy')
    axes[0].legend()
    
    early_stop_data = [
        len(df[df['epochs_completed'] <= 2]),
        len(df[(df['epochs_completed'] > 2) & (df['epochs_completed'] < 10)]),
        len(df[df['epochs_completed'] == 10])
    ]
    axes[1].bar(['Stopped ≤2', 'Stopped 3-9', 'Full (10)'], early_stop_data,
               color=['red', 'orange', 'green'], alpha=0.7)
    axes[1].set_ylabel('Number of Trials')
    axes[1].set_title('ASHA Early Stopping Distribution')
    
    plt.tight_layout()
    plt.savefig(output_dir / '05_early_stopping.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Created early stopping analysis plot")
    
    # 6. Parameter Interactions (2D heatmaps)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # num_conv_layers vs start_filters
    pivot1 = df.pivot_table(values='final_accuracy', 
                            index='num_conv_layers', 
                            columns='start_filters', 
                            aggfunc='mean')
    sns.heatmap(pivot1, annot=True, fmt='.3f', cmap='viridis', ax=axes[0], 
               cbar_kws={'label': 'Mean Accuracy'})
    axes[0].set_title('Conv Layers vs Start Filters')
    axes[0].set_xlabel('Start Filters')
    axes[0].set_ylabel('Num Conv Layers')
    
    # fc_units vs batch_size
    pivot2 = df.pivot_table(values='final_accuracy', 
                            index='fc_units', 
                            columns='batch_size', 
                            aggfunc='mean')
    sns.heatmap(pivot2, annot=True, fmt='.3f', cmap='viridis', ax=axes[1],
               cbar_kws={'label': 'Mean Accuracy'})
    axes[1].set_title('FC Units vs Batch Size')
    axes[1].set_xlabel('Batch Size')
    axes[1].set_ylabel('FC Units')
    
    plt.tight_layout()
    plt.savefig(output_dir / '06_parameter_interactions.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Created parameter interaction plots")


def generate_summary_stats(df: pd.DataFrame) -> Dict:
    """Generate summary statistics."""
    
    stats = {
        'total_trials': len(df),
        'best_accuracy': df['final_accuracy'].max(),
        'mean_accuracy': df['final_accuracy'].mean(),
        'std_accuracy': df['final_accuracy'].std(),
        'median_accuracy': df['final_accuracy'].median(),
        'worst_accuracy': df['final_accuracy'].min(),
        'trials_stopped_early': (df['stopped_early'] == True).sum(),
        'trials_full_training': (df['stopped_early'] == False).sum(),
        'avg_training_time': df['training_time'].mean(),
        'total_training_time': df['training_time'].sum(),
    }
    
    # Best configuration
    best_trial = df.loc[df['final_accuracy'].idxmax()]
    stats['best_config'] = {
        'num_conv_layers': int(best_trial['num_conv_layers']),
        'start_filters': int(best_trial['start_filters']),
        'fc_units': int(best_trial['fc_units']),
        'dropout': float(best_trial['dropout']),
        'lr': float(best_trial['lr']),
        'batch_size': int(best_trial['batch_size']),
    }
    
    return stats


def main():
    """Main execution."""
    # Paths
    experiment_path = Path("logs/ray_results/flowers")
    output_dir = Path("visualizations")
    results_dir = Path("results")
    
    if not experiment_path.exists():
        logger.error(f"Experiment path not found: {experiment_path}")
        logger.info("Please run hypertune.py first to generate results")
        return
    
    # Extract trial data
    logger.info("Extracting trial data from Ray Tune logs...")
    df = extract_trial_data(experiment_path)
    
    if df.empty:
        logger.error("No trial data found!")
        return
    
    # Save to CSV
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "experiment_summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")
    
    # Generate summary statistics
    stats = generate_summary_stats(df)
    
    logger.info("\n" + "="*60)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*60)
    logger.info(f"Total Trials: {stats['total_trials']}")
    logger.info(f"Best Accuracy: {stats['best_accuracy']:.4f}")
    logger.info(f"Mean Accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
    logger.info(f"Median Accuracy: {stats['median_accuracy']:.4f}")
    logger.info(f"Worst Accuracy: {stats['worst_accuracy']:.4f}")
    logger.info(f"\nEarly Stopping:")
    logger.info(f"  Stopped Early: {stats['trials_stopped_early']}")
    logger.info(f"  Full Training: {stats['trials_full_training']}")
    logger.info(f"\nTraining Time:")
    logger.info(f"  Average per trial: {stats['avg_training_time']:.1f}s")
    logger.info(f"  Total: {stats['total_training_time']/60:.1f} min")
    logger.info(f"\nBest Configuration:")
    for key, value in stats['best_config'].items():
        logger.info(f"  {key}: {value}")
    logger.info("="*60)
    
    # Save stats to JSON
    stats_path = results_dir / "summary_stats.json"
    # Convert numpy types to Python types for JSON serialization
    stats_json = json.loads(json.dumps(stats, default=lambda x: float(x) if hasattr(x, 'item') else int(x) if isinstance(x, (np.integer, np.int64)) else x))
    with open(stats_path, 'w') as f:
        json.dump(stats_json, f, indent=2)
    logger.info(f"Saved summary statistics to {stats_path}")
    
    # Create visualizations
    logger.info("\nCreating visualizations...")
    create_visualizations(df, output_dir)
    
    logger.info(f"\n✅ Analysis complete! Check {output_dir}/ for visualizations")


if __name__ == "__main__":
    main()
