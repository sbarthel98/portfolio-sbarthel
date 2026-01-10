"""
Monitor RNN Experiment Progress
Check MLflow database for completed runs
"""

import mlflow
import time
from datetime import datetime

mlflow.set_tracking_uri("sqlite:///mlflow.db")

def check_progress():
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("gesture_recognition_rnn")
        
        if experiment is None:
            print("No experiments found yet!")
            return 0
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            order_by=["start_time DESC"]
        )
        
        completed = 0
        running = 0
        failed = 0
        
        print(f"\n{'='*70}")
        print(f"RNN EXPERIMENT PROGRESS - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*70}")
        
        for run in runs:
            status = run.info.status
            name = run.data.tags.get('mlflow.runName', 'unknown')
            acc = run.data.metrics.get('best_accuracy', None)
            
            if status == 'FINISHED':
                completed += 1
                acc_str = f"{acc*100:.2f}%" if acc else "N/A"
                print(f"✓ {name:40s} - {acc_str}")
            elif status == 'RUNNING':
                running += 1
                print(f"⟳ {name:40s} - Running...")
            elif status == 'FAILED':
                failed += 1
                print(f"✗ {name:40s} - Failed")
        
        print(f"{'='*70}")
        print(f"Completed: {completed} | Running: {running} | Failed: {failed}")
        print(f"{'='*70}\n")
        
        return completed
        
    except Exception as e:
        print(f"Error checking progress: {e}")
        return 0

if __name__ == "__main__":
    print("Monitoring RNN experiments... (Ctrl+C to stop)")
    print("Expected experiments: ~26 total\n")
    
    last_count = 0
    while True:
        try:
            count = check_progress()
            if count > last_count:
                print(f"✓ Progress update: {count} experiments completed!")
                last_count = count
            
            if count >= 26:
                print("\n🎉 All experiments complete!")
                break
                
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break
