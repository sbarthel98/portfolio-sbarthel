# Quick Start Script for Hyperparameter Tuning Experiments
# This script helps you run the complete experimental pipeline

Write-Host "================================" -ForegroundColor Cyan
Write-Host "Hyperparameter Tuning Quick Start" -ForegroundColor Cyan
Write-Host "================================`n" -ForegroundColor Cyan

# Check if in correct directory
$currentDir = Split-Path -Leaf (Get-Location)
if ($currentDir -ne "1-hypertuning-gridsearch") {
    Write-Host "Please run this script from the 1-hypertuning-gridsearch directory!" -ForegroundColor Red
    Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
    Write-Host "`nRun: cd c:\Git\portfolio-sbarthel\1-hypertuning-gridsearch" -ForegroundColor Yellow
    exit 1
}

# Display menu
Write-Host "What would you like to do?`n" -ForegroundColor White
Write-Host "[1] Run ALL experiments (takes longest - comprehensive)" -ForegroundColor Green
Write-Host "[2] Run QUICK experiments (skips unit grid - faster)" -ForegroundColor Green
Write-Host "[3] Analyze existing results and create visualizations" -ForegroundColor Yellow
Write-Host "[4] Launch TensorBoard to view results" -ForegroundColor Cyan
Write-Host "[5] Open report template for editing" -ForegroundColor Magenta
Write-Host "[6] Show experiment status" -ForegroundColor White
Write-Host "[Q] Quit`n" -ForegroundColor Red

$choice = Read-Host "Enter your choice"

switch ($choice.ToUpper()) {
    "1" {
        Write-Host "`nRunning ALL experiments..." -ForegroundColor Green
        Write-Host "This will test all hyperparameter combinations including the full units grid." -ForegroundColor Yellow
        Write-Host "This may take 30-60 minutes depending on your hardware.`n" -ForegroundColor Yellow
        
        $confirm = Read-Host "Continue? (y/n)"
        if ($confirm -eq "y") {
            # Uncomment experiment 2 in the script
            Write-Host "Enabling all experiments..." -ForegroundColor Cyan
            & ..\\.venv\Scripts\python.exe grid_search_experiments.py
        } else {
            Write-Host "Cancelled." -ForegroundColor Red
        }
    }
    
    "2" {
        Write-Host "`nRunning QUICK experiments..." -ForegroundColor Green
        Write-Host "This will test epochs, batch size, depth, learning rate, optimizers, and combined configs." -ForegroundColor Yellow
        Write-Host "Estimated time: 10-20 minutes.`n" -ForegroundColor Yellow
        
        & ..\\.venv\Scripts\python.exe grid_search_experiments.py
    }
    
    "3" {
        Write-Host "`nAnalyzing results..." -ForegroundColor Yellow
        
        # Check if modellogs exists
        if (Test-Path "modellogs") {
            & ..\\.venv\Scripts\python.exe analyze_results.py
            
            Write-Host "`n================================" -ForegroundColor Cyan
            Write-Host "Analysis complete!" -ForegroundColor Green
            Write-Host "================================" -ForegroundColor Cyan
            Write-Host "`nCheck the following:" -ForegroundColor White
            Write-Host "- visualizations/ directory for plots" -ForegroundColor Yellow
            Write-Host "- experiment_summary.csv for data" -ForegroundColor Yellow
            Write-Host "`nNext: Fill in report_template.md with your findings!" -ForegroundColor Cyan
        } else {
            Write-Host "No modellogs directory found!" -ForegroundColor Red
            Write-Host "Please run experiments first (option 1 or 2)." -ForegroundColor Yellow
        }
    }
    
    "4" {
        Write-Host "`nLaunching TensorBoard..." -ForegroundColor Cyan
        
        if (Test-Path "modellogs") {
            Write-Host "TensorBoard will open at http://localhost:6006" -ForegroundColor Green
            Write-Host "Press Ctrl+C to stop TensorBoard`n" -ForegroundColor Yellow
            tensorboard --logdir=modellogs
        } else {
            Write-Host "No modellogs directory found!" -ForegroundColor Red
            Write-Host "Please run experiments first (option 1 or 2)." -ForegroundColor Yellow
        }
    }
    
    "5" {
        Write-Host "`nOpening report template..." -ForegroundColor Magenta
        code report_template.md
        Write-Host "Report template opened in VS Code!" -ForegroundColor Green
    }
    
    "6" {
        Write-Host "`nExperiment Status:" -ForegroundColor Cyan
        Write-Host "================================" -ForegroundColor Cyan
        
        if (Test-Path "modellogs") {
            $expCount = (Get-ChildItem "modellogs" -Directory).Count
            Write-Host "Total experiments completed: $expCount" -ForegroundColor Green
            
            if (Test-Path "visualizations") {
                $vizCount = (Get-ChildItem "visualizations" -Filter "*.png").Count
                Write-Host "Visualizations generated: $vizCount" -ForegroundColor Green
            } else {
                Write-Host "Visualizations: Not yet generated" -ForegroundColor Yellow
            }
            
            if (Test-Path "experiment_summary.csv") {
                Write-Host "Summary CSV: Generated" -ForegroundColor Green
            } else {
                Write-Host "Summary CSV: Not yet generated" -ForegroundColor Yellow
            }
        } else {
            Write-Host "No experiments run yet!" -ForegroundColor Yellow
            Write-Host "Choose option 1 or 2 to run experiments." -ForegroundColor Cyan
        }
        
        Write-Host "================================`n" -ForegroundColor Cyan
    }
    
    "Q" {
        Write-Host "`nGoodbye!" -ForegroundColor Cyan
        exit 0
    }
    
    default {
        Write-Host "`nInvalid choice!" -ForegroundColor Red
        Write-Host "Please run the script again and choose 1-6 or Q." -ForegroundColor Yellow
    }
}

Write-Host "`nDone! Run this script again for more options.`n" -ForegroundColor Green
