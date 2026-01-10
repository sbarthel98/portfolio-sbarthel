"""
Analyze RNN Experiment Results from MLflow
Generate visualizations and summary report
"""

import pandas as pd
import mlflow
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.tracking.MlflowClient()

# Get experiment
experiment = client.get_experiment_by_name("gesture_recognition_rnn")
if experiment is None:
    print("No experiments found yet!")
    exit()

# Get all runs
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"]
)

# Extract data
data = []
for run in runs:
    # Get all metrics for this run
    metrics = run.data.metrics
    
    # Find best validation accuracy (could be named differently)
    best_valid_acc = (
        metrics.get('best_valid_accuracy') or
        metrics.get('best_valid_accuracy') or
        metrics.get('valid_accuracy') or
        metrics.get('val_accuracy')
    )
    
    run_data = {
        'run_name': run.data.tags.get('mlflow.runName', 'unknown'),
        'model_type': run.data.tags.get('model_type', 'unknown'),
        'description': run.data.tags.get('description', ''),
        'hidden_size': run.data.params.get('hidden_size', None),
        'num_layers': run.data.params.get('num_layers', None),
        'dropout': run.data.params.get('dropout', None),
        'total_params': run.data.params.get('total_params', None),
        'best_valid_accuracy': best_valid_acc,
        'train_accuracy': metrics.get('train_accuracy'),
        'valid_accuracy': metrics.get('valid_accuracy'),
        'valid_loss': metrics.get('valid_loss'),
        'start_time': run.info.start_time,
    }
    data.append(run_data)

df = pd.DataFrame(data)

# Convert numeric columns
numeric_cols = ['hidden_size', 'num_layers', 'dropout', 'total_params', 
                'best_valid_accuracy', 'valid_accuracy', 'valid_loss', 'train_accuracy']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Save to CSV
df.to_csv('results/rnn_experiment_summary.csv', index=False)
print(f"✓ Saved results to results/rnn_experiment_summary.csv")
print(f"  Total experiments: {len(df)}")

# Filter completed experiments with accuracy metrics
completed = df[df['best_valid_accuracy'].notna()].copy()
if len(completed) == 0:
    print("No completed experiments with accuracy metrics yet!")
    exit()

print(f"  Completed with results: {len(completed)}")
print(f"  Best accuracy: {completed['best_valid_accuracy'].max():.4f}")

print(f"\n{'='*70}")
print("TOP 10 CONFIGURATIONS BY VALIDATION ACCURACY")
print(f"{'='*70}")
top10 = completed.nlargest(10, 'best_valid_accuracy')[
    ['run_name', 'hidden_size', 'num_layers', 'dropout', 'best_valid_accuracy', 'total_params']
]
print(top10.to_string(index=False))

# Create visualizations directory
Path("results").mkdir(exist_ok=True)

# ============================================================================
# Visualization 1: Hidden Size Impact (Experiment 1)
# ============================================================================
baseline_gru = completed[completed['run_name'].str.contains('baseline_gru', na=False)]
if len(baseline_gru) > 0:
    plt.figure(figsize=(10, 6))
    baseline_gru = baseline_gru.sort_values('hidden_size')
    plt.plot(baseline_gru['hidden_size'], baseline_gru['best_valid_accuracy'] * 100, 
             marker='o', linewidth=2, markersize=10, color='#2E86AB')
    plt.xlabel('Hidden Size', fontsize=12)
    plt.ylabel('Best Validation Accuracy (%)', fontsize=12)
    plt.title('Experiment 1: Impact of Hidden Size on GRU Performance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    for _, row in baseline_gru.iterrows():
        plt.annotate(f"{row['best_valid_accuracy']*100:.2f}%", 
                    (row['hidden_size'], row['best_valid_accuracy']*100),
                    textcoords="offset points", xytext=(0,10), ha='center')
    plt.tight_layout()
    plt.savefig('results/hidden_size_impact.png', dpi=300, bbox_inches='tight')
    print("\n✓ Created: results/hidden_size_impact.png")
    plt.close()

# ============================================================================
# Visualization 2: Depth Impact (Experiment 2)
# ============================================================================
depth_gru = completed[completed['run_name'].str.contains('gru_depth', na=False)]
if len(depth_gru) > 0:
    plt.figure(figsize=(10, 6))
    depth_gru = depth_gru.sort_values('num_layers')
    
    x = depth_gru['num_layers']
    y = depth_gru['best_valid_accuracy'] * 100
    
    plt.bar(x, y, color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.xlabel('Number of Layers', fontsize=12)
    plt.ylabel('Best Validation Accuracy (%)', fontsize=12)
    plt.title('Experiment 2: Impact of RNN Depth', fontsize=14, fontweight='bold')
    plt.xticks(x)
    plt.grid(True, alpha=0.3, axis='y')
    
    for i, (idx, row) in enumerate(depth_gru.iterrows()):
        plt.text(row['num_layers'], row['best_valid_accuracy']*100 + 0.5, 
                f"{row['best_valid_accuracy']*100:.2f}%", 
                ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/depth_impact.png', dpi=300, bbox_inches='tight')
    print("✓ Created: results/depth_impact.png")
    plt.close()

# ============================================================================
# Visualization 3: Dropout Impact (Experiment 3)
# ============================================================================
dropout_gru = completed[completed['run_name'].str.contains('gru_dropout', na=False)]
if len(dropout_gru) > 0:
    plt.figure(figsize=(10, 6))
    dropout_gru = dropout_gru.sort_values('dropout')
    plt.plot(dropout_gru['dropout'], dropout_gru['best_valid_accuracy'] * 100, 
             marker='s', linewidth=2, markersize=10, color='#F18F01')
    plt.xlabel('Dropout Rate', fontsize=12)
    plt.ylabel('Best Validation Accuracy (%)', fontsize=12)
    plt.title('Experiment 3: Impact of Dropout Regularization', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    for _, row in dropout_gru.iterrows():
        plt.annotate(f"{row['best_valid_accuracy']*100:.2f}%", 
                    (row['dropout'], row['best_valid_accuracy']*100),
                    textcoords="offset points", xytext=(0,10), ha='center')
    plt.tight_layout()
    plt.savefig('results/dropout_impact.png', dpi=300, bbox_inches='tight')
    print("✓ Created: results/dropout_impact.png")
    plt.close()

# ============================================================================
# Visualization 4: LSTM vs GRU (Experiment 4)
# ============================================================================
lstm_gru = completed[completed['run_name'].str.contains('lstm_|gru_[0-9]', regex=True, na=False)]
if len(lstm_gru) > 0:
    plt.figure(figsize=(10, 6))
    
    # Group by model type
    gru_data = lstm_gru[lstm_gru['run_name'].str.contains('gru', na=False)]
    lstm_data = lstm_gru[lstm_gru['run_name'].str.contains('lstm', na=False)]
    
    x = np.arange(len(lstm_gru))
    width = 0.35
    
    models = []
    accuracies = []
    colors = []
    
    for _, row in lstm_gru.iterrows():
        models.append(row['run_name'])
        accuracies.append(row['best_valid_accuracy'] * 100)
        if 'gru' in row['run_name'].lower():
            colors.append('#2E86AB')
        else:
            colors.append('#C73E1D')
    
    plt.bar(x, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Best Validation Accuracy (%)', fontsize=12)
    plt.title('Experiment 4: LSTM vs GRU Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.5, f"{acc:.2f}%", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/lstm_vs_gru.png', dpi=300, bbox_inches='tight')
    print("✓ Created: results/lstm_vs_gru.png")
    plt.close()

# ============================================================================
# Visualization 5: Top Configurations Comparison
# ============================================================================
plt.figure(figsize=(14, 8))
top_configs = completed.nlargest(10, 'best_valid_accuracy')

y_pos = np.arange(len(top_configs))
accuracies = top_configs['best_valid_accuracy'].values * 100
names = top_configs['run_name'].values

# Color based on accuracy
colors = plt.cm.RdYlGn(accuracies / 100)

bars = plt.barh(y_pos, accuracies, color=colors, edgecolor='black', linewidth=1.5)
plt.yticks(y_pos, names)
plt.xlabel('Best Validation Accuracy (%)', fontsize=12)
plt.title('Top 10 RNN Configurations', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')

# Add accuracy labels
for i, (acc, params) in enumerate(zip(accuracies, top_configs['total_params'].values)):
    plt.text(acc + 0.3, i, f"{acc:.2f}% ({int(params):,} params)", 
            va='center', fontweight='bold')

# Add 90% target line
plt.axvline(x=90, color='red', linestyle='--', linewidth=2, label='90% Target')
plt.legend()

plt.tight_layout()
plt.savefig('results/top_configurations.png', dpi=300, bbox_inches='tight')
print("✓ Created: results/top_configurations.png")
plt.close()

# ============================================================================
# Visualization 6: Parameter Efficiency
# ============================================================================
plt.figure(figsize=(10, 8))

# Scatter plot: params vs accuracy
plt.scatter(completed['total_params'] / 1000, completed['best_valid_accuracy'] * 100,
           s=100, alpha=0.6, edgecolors='black', linewidth=1.5, c=completed['best_valid_accuracy'], 
           cmap='RdYlGn', vmin=0.7, vmax=1.0)

# Add labels for top models
top5 = completed.nlargest(5, 'best_valid_accuracy')
for _, row in top5.iterrows():
    plt.annotate(row['run_name'], 
                (row['total_params'] / 1000, row['best_valid_accuracy'] * 100),
                textcoords="offset points", xytext=(5,5), ha='left',
                fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

plt.xlabel('Total Parameters (thousands)', fontsize=12)
plt.ylabel('Best Validation Accuracy (%)', fontsize=12)
plt.title('Model Parameter Efficiency', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.colorbar(label='Accuracy')

# Add 90% target line
plt.axhline(y=90, color='red', linestyle='--', linewidth=2, label='90% Target')
plt.legend()

plt.tight_layout()
plt.savefig('results/parameter_efficiency.png', dpi=300, bbox_inches='tight')
print("✓ Created: results/parameter_efficiency.png")
plt.close()

print(f"\n{'='*70}")
print("ANALYSIS COMPLETE!")
print(f"{'='*70}")
print(f"\nBest Model: {completed.iloc[completed['best_valid_accuracy'].idxmax()]['run_name']}")
print(f"Best Accuracy: {completed['best_valid_accuracy'].max()*100:.2f}%")
print(f"\nTotal Experiments: {len(completed)}")
print(f"Models > 90%: {len(completed[completed['best_valid_accuracy'] >= 0.90])}")
print(f"Models > 85%: {len(completed[completed['best_valid_accuracy'] >= 0.85])}")

# Calculate statistics by experiment type
print(f"\n{'='*70}")
print("STATISTICS BY EXPERIMENT TYPE")
print(f"{'='*70}")

experiment_groups = {
    'Baseline GRU': completed[completed['run_name'].str.contains('baseline_gru', na=False)],
    'GRU Depth': completed[completed['run_name'].str.contains('gru_depth', na=False)],
    'Dropout': completed[completed['run_name'].str.contains('gru_dropout', na=False)],
    'LSTM': completed[completed['run_name'].str.contains('lstm', na=False)],
    'Bidirectional': completed[completed['run_name'].str.contains('bidirectional', na=False)],
    'Conv1D+GRU': completed[completed['run_name'].str.contains('conv1d', na=False)],
    'Deep GRU': completed[completed['run_name'].str.contains('deep_gru', na=False)],
    'Optimal': completed[completed['run_name'].str.contains('optimal', na=False)],
}

for exp_name, exp_df in experiment_groups.items():
    if len(exp_df) > 0:
        print(f"\n{exp_name}:")
        print(f"  Runs: {len(exp_df)}")
        print(f"  Best: {exp_df['best_valid_accuracy'].max()*100:.2f}%")
        print(f"  Mean: {exp_df['best_valid_accuracy'].mean()*100:.2f}%")
        print(f"  Std: {exp_df['best_valid_accuracy'].std()*100:.2f}%")

print("\n✓ All visualizations saved to results/")
print("✓ Summary data saved to results/rnn_experiment_summary.csv\n")
