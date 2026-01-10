# PowerShell script for running Ray Tune experiments
# Interactive menu for experiment management

function Show-Menu {
    Clear-Host
    Write-Host "===============================================" -ForegroundColor Cyan
    Write-Host "   Ray Tune Hyperparameter Optimization" -ForegroundColor Cyan
    Write-Host "===============================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "1. Check GPU Availability" -ForegroundColor Green
    Write-Host "2. Run Hyperparameter Tuning (hypertune.py)" -ForegroundColor Green
    Write-Host "3. Analyze Results & Create Visualizations" -ForegroundColor Green
    Write-Host "4. View Summary Statistics" -ForegroundColor Green
    Write-Host "5. Clean Ray Results (delete logs/ray_results)" -ForegroundColor Yellow
    Write-Host "6. Open Experiment Journal" -ForegroundColor Cyan
    Write-Host "7. Open Report" -ForegroundColor Cyan
    Write-Host "Q. Quit" -ForegroundColor Red
    Write-Host ""
}

function Check-GPU {
    Write-Host "Checking GPU availability..." -ForegroundColor Cyan
    python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
    Write-Host ""
    Read-Host "Press Enter to continue"
}

function Run-Hypertuning {
    Write-Host "Starting Ray Tune hyperparameter optimization..." -ForegroundColor Green
    Write-Host "This will run 10 trials with ASHA scheduler and HyperOpt search." -ForegroundColor Yellow
    Write-Host ""
    
    $confirm = Read-Host "Continue? (y/n)"
    if ($confirm -eq 'y' -or $confirm -eq 'Y') {
        python hypertune.py
        Write-Host ""
        Write-Host "Experiments complete!" -ForegroundColor Green
    } else {
        Write-Host "Cancelled." -ForegroundColor Yellow
    }
    
    Read-Host "Press Enter to continue"
}

function Analyze-Results {
    Write-Host "Analyzing results and creating visualizations..." -ForegroundColor Cyan
    
    if (-not (Test-Path "logs\ray_results\flowers")) {
        Write-Host "Error: No results found! Run experiments first." -ForegroundColor Red
        Read-Host "Press Enter to continue"
        return
    }
    
    python analyze_results.py
    Write-Host ""
    Write-Host "Analysis complete! Check visualizations/ folder." -ForegroundColor Green
    
    Read-Host "Press Enter to continue"
}

function View-Summary {
    Write-Host "Viewing summary statistics..." -ForegroundColor Cyan
    
    if (-not (Test-Path "results\summary_stats.json")) {
        Write-Host "Error: No summary stats found! Run analysis first." -ForegroundColor Red
        Read-Host "Press Enter to continue"
        return
    }
    
    $stats = Get-Content "results\summary_stats.json" | ConvertFrom-Json
    
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Green
    Write-Host "         SUMMARY STATISTICS" -ForegroundColor Green
    Write-Host "======================================" -ForegroundColor Green
    Write-Host "Total Trials: $($stats.total_trials)"
    Write-Host "Best Accuracy: $([math]::Round($stats.best_accuracy, 4))"
    Write-Host "Mean Accuracy: $([math]::Round($stats.mean_accuracy, 4)) ± $([math]::Round($stats.std_accuracy, 4))"
    Write-Host "Median Accuracy: $([math]::Round($stats.median_accuracy, 4))"
    Write-Host ""
    Write-Host "Early Stopping:"
    Write-Host "  Stopped Early: $($stats.trials_stopped_early)"
    Write-Host "  Full Training: $($stats.trials_full_training)"
    Write-Host ""
    Write-Host "Best Configuration:" -ForegroundColor Yellow
    Write-Host "  num_conv_layers: $($stats.best_config.num_conv_layers)"
    Write-Host "  start_filters: $($stats.best_config.start_filters)"
    Write-Host "  fc_units: $($stats.best_config.fc_units)"
    Write-Host "  dropout: $([math]::Round($stats.best_config.dropout, 4))"
    Write-Host "  lr: $([math]::Round($stats.best_config.lr, 6))"
    Write-Host "  batch_size: $($stats.best_config.batch_size)"
    Write-Host "======================================" -ForegroundColor Green
    Write-Host ""
    
    Read-Host "Press Enter to continue"
}

function Clean-Results {
    Write-Host "This will delete all Ray Tune results in logs/ray_results/" -ForegroundColor Yellow
    $confirm = Read-Host "Are you sure? (y/n)"
    
    if ($confirm -eq 'y' -or $confirm -eq 'Y') {
        if (Test-Path "logs\ray_results") {
            Remove-Item -Recurse -Force "logs\ray_results"
            Write-Host "Results cleaned!" -ForegroundColor Green
        } else {
            Write-Host "No results directory found." -ForegroundColor Yellow
        }
    } else {
        Write-Host "Cancelled." -ForegroundColor Yellow
    }
    
    Read-Host "Press Enter to continue"
}

function Open-Journal {
    Write-Host "Opening experiment journal..." -ForegroundColor Cyan
    if (Test-Path "experiment_journal.md") {
        Start-Process "experiment_journal.md"
    } else {
        Write-Host "Error: experiment_journal.md not found!" -ForegroundColor Red
    }
    Read-Host "Press Enter to continue"
}

function Open-Report {
    Write-Host "Opening report..." -ForegroundColor Cyan
    if (Test-Path "report.md") {
        Start-Process "report.md"
    } else {
        Write-Host "Error: report.md not found!" -ForegroundColor Red
    }
    Read-Host "Press Enter to continue"
}

# Main loop
do {
    Show-Menu
    $choice = Read-Host "Select an option"
    
    switch ($choice) {
        '1' { Check-GPU }
        '2' { Run-Hypertuning }
        '3' { Analyze-Results }
        '4' { View-Summary }
        '5' { Clean-Results }
        '6' { Open-Journal }
        '7' { Open-Report }
        'Q' { 
            Write-Host "Exiting..." -ForegroundColor Cyan
            exit 
        }
        default {
            Write-Host "Invalid option. Please try again." -ForegroundColor Red
            Start-Sleep -Seconds 1
        }
    }
} while ($true)
