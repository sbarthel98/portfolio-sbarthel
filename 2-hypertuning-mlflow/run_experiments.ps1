# MLflow Hyperparameter Tuning - Quick Start Menu
# Interactive helper script for running experiments

function Show-Menu {
    Clear-Host
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  MLflow Hyperparameter Tuning Menu" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "What would you like to do?" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "[1] Run ALL experiments (30+ runs, ~2-3 hours)"
    Write-Host "[2] Run QUICK test (single experiment, ~2 min)"
    Write-Host "[3] Analyze results and create visualizations"
    Write-Host "[4] Launch MLflow UI"
    Write-Host "[5] View experiment summary"
    Write-Host "[6] Open journal template"
    Write-Host "[7] Open report template"
    Write-Host "[Q] Quit"
    Write-Host ""
}

function Run-AllExperiments {
    Write-Host "`nRunning ALL experiments..." -ForegroundColor Green
    Write-Host "This will run 30+ experiments and may take 2-3 hours."
    $confirm = Read-Host "Continue? (y/n)"
    
    if ($confirm -eq 'y') {
        Write-Host "`nStarting experiments..." -ForegroundColor Yellow
        & ..\.venv\Scripts\python.exe mlflow_experiments.py
    }
}

function Run-QuickTest {
    Write-Host "`nRunning quick test experiment..." -ForegroundColor Green
    Write-Host "This will run a single experiment to verify everything works."
    
    $testScript = @"
import sys
sys.path.insert(0, '.')
from mlflow_experiments import run_experiment, setup_data
import mlflow

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("fashion_mnist_cnn_tuning")

config = {
    'epochs': 2,
    'batchsize': 64,
    'num_conv_layers': 2,
    'conv_channels': [32, 64],
    'kernel_size': 3,
    'use_pooling': True,
    'use_batch_norm': True,
    'dropout_rate': 0.5,
    'fc_units': [128],
    'learning_rate': 0.001,
    'optimizer': 'Adam'
}

run_experiment(config, 'quick_test')
print("\n✓ Quick test completed successfully!")
"@
    
    $testScript | & ..\.venv\Scripts\python.exe
}

function Analyze-Results {
    Write-Host "`nAnalyzing results..." -ForegroundColor Green
    & ..\.venv\Scripts\python.exe analyze_mlflow_results.py
}

function Launch-MLflowUI {
    Write-Host "`nLaunching MLflow UI..." -ForegroundColor Green
    Write-Host "MLflow UI will open at http://localhost:5000" -ForegroundColor Yellow
    Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
    Write-Host ""
    & ..\.venv\Scripts\python.exe -m mlflow ui
}

function Show-Summary {
    Write-Host "`nExperiment Summary" -ForegroundColor Green
    Write-Host "="*50
    
    if (Test-Path "results\experiment_summary.csv") {
        $summary = Import-Csv "results\experiment_summary.csv" | Select-Object -First 10
        $summary | Format-Table -AutoSize
        Write-Host "`nShowing top 10 of $(($summary | Measure-Object).Count) experiments"
        Write-Host "Full summary: results\experiment_summary.csv"
    } else {
        Write-Host "No summary found. Run experiments first (option 1 or 2)." -ForegroundColor Red
    }
}

function Open-Journal {
    Write-Host "`nOpening experiment journal..." -ForegroundColor Green
    if (Test-Path "experiment_journal.md") {
        code experiment_journal.md
    } else {
        Write-Host "Journal not found. Creating from template..." -ForegroundColor Yellow
        Copy-Item "experiment_journal_template.md" "experiment_journal.md" -ErrorAction SilentlyContinue
        code experiment_journal.md
    }
}

function Open-Report {
    Write-Host "`nOpening report template..." -ForegroundColor Green
    if (Test-Path "report_template.md") {
        code report_template.md
    } else {
        Write-Host "Report template not found!" -ForegroundColor Red
    }
}

# Main loop
do {
    Show-Menu
    $choice = Read-Host "Enter your choice"
    
    switch ($choice) {
        '1' { Run-AllExperiments }
        '2' { Run-QuickTest }
        '3' { Analyze-Results }
        '4' { Launch-MLflowUI }
        '5' { Show-Summary }
        '6' { Open-Journal }
        '7' { Open-Report }
        'Q' { 
            Write-Host "`nExiting..." -ForegroundColor Yellow
            break
        }
        default {
            Write-Host "`nInvalid choice. Press any key to continue..." -ForegroundColor Red
            $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        }
    }
    
    if ($choice -ne 'Q') {
        Write-Host "`nDone! Press any key to continue..." -ForegroundColor Green
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    }
} while ($choice -ne 'Q')
