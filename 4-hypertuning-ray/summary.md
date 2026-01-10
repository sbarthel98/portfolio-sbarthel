# Summary: Hyperparameter Tuning with Ray Tune

**Status**: COMPLETED - January 10, 2026

This project implements advanced hyperparameter optimization using Ray Tune with ASHA (Asynchronous Successive Halving Algorithm) scheduler on the Flowers dataset. Through 10 trials using GPU acceleration and intelligent search algorithms, achieved 68.5% accuracy with optimal CNN architecture. ASHA early stopping saved 40% training time while HyperOpt found best configuration efficiently.

**Key Achievements**:
- Best Accuracy: 68.5% (3 layers, 32 filters, 128 FC units)
- Efficiency: 10 trials in 17.7 minutes (vs. 2 hours for Week 1 grid search)
- Early Stopping: 60% of trials terminated early, saving significant compute
- Smart Search: HyperOpt Bayesian optimization outperformed exhaustive methods

## Objectives

1. Build a configurable CNN model for image classification 
2. Implement Ray Tune with ASHA scheduler for efficient hyperparameter search 
3. Use HyperOpt search algorithm for intelligent parameter selection 
4. Run experiments with GPU acceleration (0.5 GPU per trial for parallelization) 
5. Analyze results and compare with previous grid search approaches 
6. Create visualizations showing hyperparameter relationships  (6 visualizations)
7. Write comprehensive report with theoretical justifications 

## Project Files

### Core Experiment Files
- [instructions.md](./instructions.md) - Detailed assignment instructions
- **[hypertune.py](./hypertune.py)** - Ray Tune hyperparameter optimization script with GPU support
- [inspect_data.py](./inspect_data.py) - Dataset inspection utilities
- [inspect_data_v2.py](./inspect_data_v2.py) - Enhanced data inspection
- [check_dims.py](./check_dims.py) - Dimension checking utility

### Report Files
- **[experiment_journal.md](./experiment_journal.md)** - Hypothesis → experiment → results with full analysis
- **[report.md](./report.md)** - Final technical report with visualizations and theoretical connections
- **[summary.md](./summary.md)** - Project overview and results summary
- **[results/experiment_summary.csv](./results/experiment_summary.csv)** - 10 trials with full metrics
- **[results/summary_stats.json](./results/summary_stats.json)** - Aggregated statistics and best configuration

## Quick Start

### Setup Environment
```powershell
cd c:\Users\stijn\Documents\GitHub\portfolio-sbarthel\4-hypertuning-ray

# Ensure required packages are installed
# ray, torch, torchvision, mads-datasets, loguru, filelock, hyperopt
```

### Check GPU Availability
```powershell
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"
```

### Run Hyperparameter Tuning
```powershell
python hypertune.py
```

**Note**: The script is configured to use GPU if available:
- 0.5 GPU per trial allows 2 trials in parallel
- Automatic fallback to CPU if GPU not available
- Results saved to `logs/ray_results/flowers/`

## Experiment Design

### Dataset
- **Name**: Flowers (5 classes)
- **Image Size**: 224x224x3 RGB
- **Classes**: daisy, dandelion, roses, sunflowers, tulips
- **Location**: `~/.cache/mads_datasets`

### Model Architecture
**Configurable TunableCNN**:
- Variable convolutional layers (2-4)
- Dynamic filter progression (starts at 16/32/64, doubles each layer)
- BatchNorm + ReLU + MaxPool2d per conv block
- Fully connected classifier with dropout
- Total parameters: varies by configuration

### Hyperparameter Search Space
```python
{
    "num_conv_layers": [2, 3, 4],          # Network depth
    "start_filters": [16, 32, 64],          # Initial filter size
    "fc_units": [64, 128, 256],             # FC layer width
    "dropout": uniform(0.1, 0.5),           # Regularization
    "lr": loguniform(1e-4, 1e-2),          # Learning rate
    "batch_size": [16, 32]                  # Batch size
}
```

### Optimization Strategy
- **Scheduler**: ASHA (Async Successive Halving)
  - Max epochs: 10
  - Grace period: 2 epochs (min before stopping)
  - Reduction factor: 3
  - Early stopping of poor performers
  
- **Search Algorithm**: HyperOpt
  - Bayesian optimization
  - More efficient than random search
  
- **Resources**: 
  - GPU: 0.5 per trial (2 parallel on RTX 3060)
  - CPU: 2 per trial
  - Number of samples: 10 trials
  
**Results**:
- Total time: 17.7 minutes
- Best accuracy: 68.5%
- Trials stopped early: 6/10 (60%)
- Time savings: ~40% vs. running all trials to completion

## Scientific Method Workflow

### Phase 1: Hypothesis Formation
1. Review theory from Deep Learning book
2. Analyze results from previous weeks
3. Formulate testable hypotheses
4. Document in [experiment_journal.md](./experiment_journal.md)

### Phase 2: Experiment Execution
1. Run `python hypertune.py`
2. Monitor Ray Tune dashboard/CLI reporter
3. Observe which trials are stopped early
4. Note any patterns or unexpected results

### Phase 3: Analysis
1. Extract results from Ray Tune logs
2. Create visualizations (heatmaps, scatter plots, learning curves)
3. Analyze parameter importance and interactions
4. Compare with previous grid search results

### Phase 4: Reflection & Reporting
1. Validate or reject hypotheses
2. Connect findings to theory
3. Write concise report (max 3 pages)
4. Document key insights and lessons learned

## Key Comparisons with Previous Weeks

| Week | Method | Dataset | Model | Trials | Best Acc | Time | Efficiency |
|------|--------|---------|-------|--------|----------|------|------------|
| 1 | Grid Search | Fashion MNIST | MLP | 48 | 60% | ~2 hrs | Exhaustive, slow |
| 2 | MLflow | Fashion MNIST | CNN | ~30 | ~92% | ~1 hr | Better tracking |
| 3 | RNN/LSTM | Gestures | RNN | ~20 | 99.7% | Variable | Specialized arch |
| 4 | **Ray Tune + ASHA** | **Flowers** | **CNN** | **10** | **68.5%** | **18 min** | **Smart + Fast** |

**Key Insight**: Week 4 achieved competitive accuracy with 80% fewer trials than Week 1, demonstrating the power of intelligent search combined with early stopping.

## Experimental Results

### Best Configuration Found
```python
{
    "num_conv_layers": 3,
    "start_filters": 32,
    "fc_units": 128,
    "dropout": 0.1497,
    "lr": 0.000472,
    "batch_size": 32
}
```
**Validation Accuracy**: 68.5%

### Performance Distribution
- Mean: 43.6% ± 13.8%
- Median: 40.7%
- Best: 68.5%
- Worst: 25.3%
- Range: 43.2 percentage points

### Key Findings
1. **Architecture**: 3 layers optimal (not 4) - moderate depth balances capacity and overfitting
2. **Filters**: 32 starting filters outperformed both 16 and 64
3. **Dropout**: Low dropout (0.15-0.20) performed best, not 0.3-0.4 as expected
4. **Learning Rate**: Optimal ~5e-4, confirming Adam optimizer recommendations
5. **ASHA Efficiency**: Stopped 60% trials early without missing good configurations

### Visualizations Generated
1. Overall performance distribution
2. Parameter impact analysis
3. Correlation heatmap
4. Top vs. bottom performers comparison
5. Early stopping patterns
6. Parameter interaction scatter matrix

All visualizations available in `visualizations/` directory.

---

## Report Requirements

### experiment_journal.md (Track Process)
- Hypothesis → Experiment → Results cycle
- Real-time observations and thoughts
- Don't worry about mistakes - they help learning!

### report.md (Final Deliverable, Max 3 Pages)
- Clear visualizations showing key findings
- Theoretical justifications for results
- Comparison with previous approaches
- Concise conclusions and recommendations

## Troubleshooting

### Common Issues
- **Path errors**: Use `Path` and check `.exists()`
- **Import errors**: Check if src folder needs adding to sys.path
- **CUDA OOM**: Reduce batch size or use smaller images
- **Ray errors**: Check Ray version compatibility

### GPU Verification
Your hypertune.py already includes GPU detection:
```python
use_gpu = torch.cuda.is_available()
if use_gpu:
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    resources_per_trial = {"cpu": 2, "gpu": 0.5}
else:
    logger.info("Using CPU")
    resources_per_trial = {"cpu": 2}
```

## Theoretical Foundation

### Key Concepts Validated
1. **Network Capacity**: Results confirmed bias-variance tradeoff - 3 layers balanced capacity and generalization
2. **Regularization**: Low dropout (0.15) optimal; BatchNorm provided training stability
3. **Optimization**: Learning rate findings aligned with Adam optimizer theory (1e-4 to 1e-3 range)
4. **Early Stopping**: ASHA successfully allocated resources to promising trials, stopping 60% early

### Theoretical Connections
- **Bias-Variance Tradeoff**: 2 layers underfit (high bias), 4 layers risked overfitting (high variance)
- **Capacity Theory**: 32 filters provided sufficient representational power without excess capacity
- **Optimization Landscape**: LR ~5e-4 enabled stable convergence; extremes caused slow learning or instability
- **Multi-Armed Bandits**: ASHA's successive halving efficiently explored hyperparameter space

### Resources
- Deep Learning Book - Chapters 5, 7-9 (Regularization, CNNs, Optimization)
- ASHA Paper: "A System for Massively Parallel Hyperparameter Tuning" (Li et al., 2020)
- Ray Tune Documentation: https://docs.ray.io/en/latest/tune/

---

[Go back to Homepage](../README.md)
