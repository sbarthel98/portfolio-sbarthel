"""
Extract metrics from TensorBoard event files and update TOML result files
"""

import toml
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

def extract_metrics_from_tensorboard(event_file_path):
    """Extract final metrics from a TensorBoard event file"""
    try:
        ea = event_accumulator.EventAccumulator(str(event_file_path))
        ea.Reload()
        
        metrics = {}
        
        # Get available tags
        scalar_tags = ea.Tags()['scalars']
        
        # Extract final values for each metric
        for tag in scalar_tags:
            events = ea.Scalars(tag)
            if events:
                # Get the last value
                final_value = events[-1].value
                final_step = events[-1].step
                
                # Clean up tag name for storage
                clean_tag = tag.replace('/', '_').replace(' ', '_').lower()
                metrics[f'final_{clean_tag}'] = float(final_value)
                metrics[f'final_{clean_tag}_step'] = int(final_step)
        
        return metrics
    except Exception as e:
        print(f"Error extracting metrics from {event_file_path}: {e}")
        return {}

def update_toml_files_with_metrics():
    """Update all TOML result files with metrics from TensorBoard"""
    modellogs_dir = Path("modellogs")
    
    if not modellogs_dir.exists():
        print("modellogs directory not found!")
        return
    
    updated_count = 0
    
    for exp_dir in sorted(modellogs_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        # Find TensorBoard event file
        event_files = list(exp_dir.glob("events.out.tfevents.*"))
        if not event_files:
            continue
        
        event_file = event_files[0]
        
        # Find corresponding TOML result file
        toml_files = list(exp_dir.glob("*_results.toml"))
        if not toml_files:
            print(f"No TOML file found in {exp_dir.name}")
            continue
        
        toml_file = toml_files[0]
        
        # Extract metrics from TensorBoard
        print(f"Processing {exp_dir.name}...")
        metrics = extract_metrics_from_tensorboard(event_file)
        
        if not metrics:
            print(f"  No metrics extracted from {exp_dir.name}")
            continue
        
        # Load existing TOML data
        try:
            with open(toml_file, 'r') as f:
                data = toml.load(f)
            
            # Update with metrics
            data.update(metrics)
            
            # Save back to TOML
            with open(toml_file, 'w') as f:
                toml.dump(data, f)
            
            print(f"  Updated {toml_file.name} with {len(metrics)} metrics")
            updated_count += 1
            
        except Exception as e:
            print(f"  Error updating {toml_file}: {e}")
    
    print(f"\n{'='*70}")
    print(f"Updated {updated_count} TOML files with TensorBoard metrics")
    print(f"{'='*70}")
    print("\nRun analyze_results.py again to see the updated metrics!")

if __name__ == "__main__":
    print("="*70)
    print("EXTRACTING METRICS FROM TENSORBOARD")
    print("="*70)
    print()
    update_toml_files_with_metrics()
