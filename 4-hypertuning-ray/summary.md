# Summary: Hyperparameter Tuning with Ray Tune

**Status**: 🚧 **IN PROGRESS** - January 10, 2026

This project implements advanced hyperparameter optimization using Ray Tune with ASHA (Asynchronous Successive Halving Algorithm) scheduler on the Flowers dataset. The project uses GPU acceleration for efficient training and intelligent search algorithms to find optimal CNN architectures.

## 🎯 Objectives

1. ✅ Build a configurable CNN model for image classification
2. ⬜ Implement Ray Tune with ASHA scheduler for efficient hyperparameter search
3. ⬜ Use HyperOpt search algorithm for intelligent parameter selection
4. ⬜ Run experiments with GPU acceleration (0.5 GPU per trial for parallelization)
5. ⬜ Analyze results and compare with previous grid search approaches
6. ⬜ Create visualizations showing hyperparameter relationships
7. ⬜ Write comprehensive report with theoretical justifications

## 📁 Project Files

### Core Experiment Files
- [instructions.md](./instructions.md) - Detailed assignment instructions
- **[hypertune.py](./hypertune.py)** - ✅ Ray Tune hyperparameter optimization script with GPU support
- [inspect_data.py](./inspect_data.py) - Dataset inspection utilities
- [inspect_data_v2.py](./inspect_data_v2.py) - Enhanced data inspection
- [check_dims.py](./check_dims.py) - Dimension checking utility

### Report Files
- **[experiment_journal.md](./experiment_journal.md)** - ✅ CREATED - Track hypothesis → experiment → results cycle
- **[report.md](./report.md)** - ✅ CREATED - Final report template (max 3 pages)
- **[summary.md](./summary.md)** - This file - Project overview and quick reference
- **[experiment_summary.csv](./experiment_summary.csv)** - ✅ CREATED - Results data file

## 🚀 Quick Start

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

## 📊 Experiment Design

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
  - GPU: 0.5 per trial (2 parallel)
  - CPU: 2 per trial
  - Number of samples: 10 trials

## 🔬 Scientific Method Workflow

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

## 📈 Key Comparisons with Previous Weeks

| Week | Method | Dataset | Model | Efficiency |
|------|--------|---------|-------|------------|
| 1 | Grid Search | Fashion MNIST | MLP | Exhaustive, slow |
| 2 | MLflow | Fashion MNIST | MLP | Better tracking |
| 3 | RNN/LSTM | [Sequence data] | RNN | Specialized architecture |
| 4 | **Ray Tune + ASHA** | **Flowers** | **CNN** | **Smart search, early stopping** |

## 📝 Report Requirements

### experiment_journal.md (Track Process)
- Hypothesis → Experiment → Results cycle
- Real-time observations and thoughts
- Don't worry about mistakes - they help learning!

### report.md (Final Deliverable, Max 3 Pages)
- Clear visualizations showing key findings
- Theoretical justifications for results
- Comparison with previous approaches
- Concise conclusions and recommendations

### Tips for Excellence
✅ **DO**:
- Make specific, testable hypotheses
- Use theory to explain results
- Create clear, uncluttered visualizations
- Be honest about mistakes and learnings
- Compare ASHA efficiency vs. grid search

❌ **DON'T**:
- Use ChatGPT for reflections (use theory!)
- Create heatmaps with HyperBand (mixed epochs)
- Exceed 3 pages for report
- Blindly copy configurations

## 🛠️ Troubleshooting

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

## 📚 Theoretical Foundation

### Key Concepts to Explore
1. **Network Capacity**: How architecture affects representational power
2. **Regularization**: Dropout, BatchNorm effects
3. **Optimization**: Learning rate impact on convergence
4. **Early Stopping**: ASHA's efficiency in exploring hyperparameter space

### Resources
- Deep Learning Book - Chapters 7-9 (CNNs, Optimization, Regularization)
- ASHA Paper: "A System for Massively Parallel Hyperparameter Tuning"
- Ray Tune Documentation

## 🎓 Grading Criteria

- **0**: Needs significant improvement
- **1**: Good work, on the right track
- **2**: Excellent, exceeded expectations

**Focus Areas**:
- Scientific method application
- Theoretical understanding
- Clear communication
- Quality of visualizations

---

[Go back to Homepage](../README.md)
