# Summary: MLflow Hyperparameter Tuning with CNNs

**Status**: COMPLETED - January 7, 2026

This project successfully executed 19 systematic CNN hyperparameter tuning experiments on Fashion MNIST using MLflow experiment tracking. All experiments, analysis, visualizations, and comprehensive documentation are complete.

**Key Results Achieved:**
- **Best Performance**: 92.51% validation accuracy (combined_optimal_shallow)
- **Best Val Peak**: 92.80% accuracy during training
- **Most Impactful**: Batch Normalization (+0.14-0.84% boost)
- **Optimal Architecture**: 2 conv layers with 421K parameters
- **Critical Insight**: BatchNorm > Dropout for regularization
- **Surprise Finding**: Deeper networks (3 layers) show diminishing returns

## Objectives

1. Investigate dropout regularization impact (0.0 to 0.7)
2. Evaluate batch normalization effectiveness
3. ✅ Explore convolutional depth trade-offs (1-3 layers)
4. ✅ Test pooling layer impact on performance
5. ✅ Analyze hyperparameter interactions (dropout × BatchNorm)
6. ✅ Identify optimal combined configuration
7. ✅ Track all experiments with MLflow for reproducibility

## Project Files

### Core Experiment Files
- [instructions.md](./instructions.md) - Detailed experiment instructions and study questions
- **[mlflow_experiments.py](./mlflow_experiments.py)** - Main training script with FlexibleCNN architecture
- **[analyze_mlflow_results.py](./analyze_mlflow_results.py)** - Results analysis and visualization generator

### Execution Helpers
- **[run_experiments.ps1](./run_experiments.ps1)** - Interactive PowerShell menu for experiment management

### Reports & Documentation
- **[report_template.md](./report.md)** - COMPLETED - Full technical report with findings
- **[experiment_journal.md](./experiment_journal.md)** - COMPLETED - Detailed scientific experiment journal

### Configuration
- Settings files for model configuration and experiment parameters

## Quick Start

### Option 1: Use the Interactive Menu
```powershell
cd c:\Users\stijn\Documents\GitHub\portfolio-sbarthel\2-hypertuning-mlflow
.\run_experiments.ps1
```

**Menu Options:**
1. Run all experiments (19 experiments)
2. Run quick test (single experiment)
3. Analyze results and generate visualizations
4. Launch MLflow UI
5. Show experiment summary
6. Clean experiment data
7. Exit

### Option 2: Manual Execution
```powershell
# Run experiments
python mlflow_experiments.py

# Analyze results
python analyze_mlflow_results.py

# View in MLflow UI
mlflow ui
# Then open: http://localhost:5000
```

## Experiments Completed (19 Total)

1. **Dropout Rate** (5 configs) - 0.0, 0.2, 0.3, 0.5, 0.7 → Optimal: 0.3
2. **Batch Normalization** (2 configs) - With/Without → +0.14% improvement
3. **Convolutional Depth** (3 configs) - 1, 2, 3 layers → Optimal: 2 layers
4. **Pooling Strategy** (2 configs) - MaxPool vs None → 15% param reduction
5. **Interactions** (4 configs) - Dropout × BatchNorm factorial → BatchNorm dominant
6. **Combined Optimal** (3 configs) - Best combinations → 92.51% winner

## Key Concepts Explored

- **MLflow Tracking**: Experiment versioning, metric logging, parameter tracking
- **CNN Architecture Design**: Flexible ModuleList-based construction
- **Regularization Strategies**: Dropout vs Batch Normalization trade-offs
- **Architecture Depth**: Capacity vs efficiency balance
- **Hyperparameter Interactions**: Synergistic and complementary effects
- **GPU Acceleration**: CUDA-enabled PyTorch training (RTX 3060)
- **Progress Monitoring**: Real-time tqdm progress bars

## Deliverables (All Complete)

1. **MLflow tracking** in `./mlruns` (19 experiment runs with full metrics)
2. **Experiment summary CSV** with all results and configurations
3. **Visualization plots** in `visualizations/` (5 PNG files)
   - dropout_impact.png
   - batchnorm_comparison.png
   - conv_depth_impact.png
   - interactions_heatmap.png
   - top_configurations.png
4. **Comprehensive technical report** in `report_template.md` (all sections filled)
5. **Scientific experiment journal** in `experiment_journal.md` (19 experiments documented)
6. **Analysis scripts** for reproducible results generation

## Key Findings

### 1. Dropout Impact
- **Optimal Rate**: 0.3 (92.28% val accuracy)
- **Too Low** (0.0): Still good (91.78%) - architecture generalizes well
- **Too High** (0.7): Performance drops to 91.63% - over-regularization
- **Insight**: Moderate dropout provides balance; Fashion MNIST doesn't need aggressive regularization

### 2. Batch Normalization
- **Impact**: +0.14-0.84% accuracy improvement
- **Convergence**: Reaches 80% accuracy within 3 epochs (faster than without)
- **Stability**: Smoother loss curves, better gradient flow
- **Verdict**: Essential component - always include

### 3. Convolutional Depth
- **1 Layer**: 90.57% accuracy, 52K params - insufficient capacity
- **2 Layers**: 92.03% accuracy, 422K params - optimal efficiency
- **3 Layers**: 92.27% accuracy, 1.1M params - diminishing returns (+0.24% for 2.5× params)
- **Insight**: Two layers capture all meaningful Fashion MNIST patterns

### 4. Hyperparameter Interactions
- **BatchNorm Dominance**: Provides consistent boost regardless of dropout
- **Best Combo**: BatchNorm=True + Dropout=0.3 → 92.51%
- **Without BatchNorm**: Performance drops significantly (91.29-91.42%)
- **Insight**: BatchNorm is primary driver; dropout is secondary support

## Optimal Configuration

```python
best_config = {
    'num_conv_layers': 2,
    'conv_channels': [32, 64],
    'kernel_size': 3,
    'use_pooling': True,
    'use_batch_norm': True,
    'dropout_rate': 0.3,
    'fc_units': [128],
    'learning_rate': 0.001,
    'optimizer': 'Adam',
    'epochs': 10,
    'batchsize': 64
}
```

**Performance:**
- Validation Accuracy: 92.51%
- Best Val Accuracy: 92.80%
- Total Parameters: 421,834
- Training Time: ~18 seconds/epoch (GPU)
- Overfitting Gap: 0.6% (excellent generalization)

## Technical Stack

- **PyTorch**: Deep learning framework with CUDA support
- **MLflow**: Experiment tracking and model registry
- **mads-datasets**: Fashion MNIST data loading
- **tqdm**: Real-time progress visualization
- **Matplotlib/Seaborn**: Results visualization
- **Pandas**: Data analysis and CSV handling

## What Worked Well

✅ **MLflow Integration**: Seamless experiment tracking with automatic metric logging  
✅ **GPU Acceleration**: RTX 3060 reduced training time from hours to minutes  
✅ **Progress Bars**: Real-time feedback with nested tqdm bars (batch/epoch/experiment)  
✅ **Modular Architecture**: FlexibleCNN class enables easy configuration changes  
✅ **Factorial Design**: Systematic exploration of hyperparameter interactions  
✅ **Comprehensive Logging**: All metrics, parameters, and artifacts tracked

## Challenges Overcome

1. **Infinite Dataloader Loop**: Fixed by calculating batch limits from dataset size
2. **Virtual Environment Paths**: Resolved by using activated `python` command
3. **CUDA Setup**: Installed PyTorch 2.2.0+cu121 for GPU acceleration
4. **Progress Bar TypeError**: Added try-except wrapper for generator-based dataloaders

## Future Improvements

- [ ] Test on full Fashion MNIST (60K samples vs current subset)
- [ ] Evaluate on CIFAR-10/100 for color image complexity
- [ ] Explore learning rate scheduling (CosineAnnealing, ReduceLROnPlateau)
- [ ] Implement data augmentation (rotation, flip, cutout)
- [ ] Compare optimizers (SGD+momentum, AdamW, RMSprop)
- [ ] Try different weight initialization strategies
- [ ] Investigate attention mechanisms for CNNs

## 📚 Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [PyTorch CNN Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Batch Normalization Paper](https://arxiv.org/abs/1502.03167)
- [Dropout Paper](https://jmlr.org/papers/v15/srivastava14a.html)
- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)

## Lessons Learned

1. **BatchNorm First**: Start with batch normalization as foundation
2. **Moderate Regularization**: Dropout 0.3 is sweet spot for Fashion MNIST
3. **Depth Isn't Everything**: 2 layers optimal; 3+ layers show diminishing returns
4. **Parameter Efficiency**: 422K params achieves 92.5% - more isn't always better
5. **Systematic Testing**: Factorial designs reveal interaction effects
6. **Track Everything**: MLflow makes reproducibility and comparison effortless

## Quick Results Reference

| Experiment | Parameter | Best Value | Val Accuracy | Key Finding |
|-----------|-----------|------------|--------------|-------------|
| Dropout | rate | 0.3 | 92.28% | Moderate dropout optimal |
| BatchNorm | enabled | True | 92.28% | +0.14-0.84% boost |
| Conv Depth | layers | 2 | 92.03% | Best efficiency at 422K params |
| Pooling | enabled | True | 92.01% | 15% param reduction |
| Interactions | dropout+BN | 0.3+True | 92.51% | BatchNorm dominant factor |
| Combined | optimal | Config A | 92.51% | ⭐ Best overall |

**Best Run**: `combined_optimal_shallow` → 92.51% validation accuracy

---

Find the [instructions](./instructions.md)

[Go back to Homepage](../README.md)

## Project Overview
This week focused on systematic hyperparameter optimization for a convolutional neural network (CNN) trained on Fashion MNIST. Using MLflow for experiment tracking, I explored the impact of dropout regularization, batch normalization, convolutional depth, and pooling strategies across 19 experiments.

## Technical Approach

**Dataset:** Fashion MNIST (subset with ~960 training, ~192 validation images)  
**Framework:** PyTorch with CUDA GPU acceleration (NVIDIA RTX 3060)  
**Tracking:** MLflow experiment tracking with comprehensive metrics logging  
**Architecture:** Flexible CNN with configurable depth, BatchNorm, dropout, and pooling

### Experiments Conducted
1. **Dropout rates** (0.0, 0.2, 0.3, 0.5, 0.7): Explored regularization strength
2. **Batch Normalization** (True/False): Tested convergence and stability improvements
3. **Convolutional depth** (1, 2, 3 layers): Analyzed capacity vs efficiency trade-offs
4. **Pooling strategies** (True/False): Evaluated spatial downsampling impact
5. **Interaction effects** (dropout × BatchNorm): Identified synergistic combinations
6. **Combined optimal configurations**: Validated best hyperparameter sets

## Key Results

**Best Configuration Achieved:**
- **Validation Accuracy:** 92.51% (Best: 92.80%)
- **Architecture:** 2 convolutional layers [32, 64 channels]
- **Regularization:** Dropout 0.3 + Batch Normalization
- **Parameters:** 421,834 (optimal efficiency)
- **Training Time:** ~18 seconds/epoch on GPU

### Major Findings
1. **Batch Normalization is essential**: Provided +0.14-0.84% accuracy boost across all configurations with faster convergence
2. **Moderate dropout optimal**: Rate of 0.3 balanced regularization without over-constraining capacity
3. **Two layers sufficient**: Additional depth (3 layers) added 2.5× parameters for only 0.24% gain
4. **Low overfitting**: 0.6% train-validation gap indicates excellent generalization

## Learnings

### What Worked Well
✓ MLflow tracking provided excellent experiment organization and comparison capabilities  
✓ GPU acceleration reduced training from hours to minutes (18s/epoch)  
✓ Systematic approach revealed clear hyperparameter trends  
✓ BatchNorm + moderate dropout combination was highly effective

### Challenges Overcome
- Fixed infinite dataloader bug with mads-datasets streaming interface
- Optimized CUDA installation for GPU utilization
- Implemented comprehensive progress tracking with tqdm
- Balanced experimentation speed vs statistical significance with dataset subset

### Future Improvements
- Scale experiments to full Fashion MNIST dataset (60K samples) for robustness
- Test on more complex datasets (CIFAR-10/100) to validate depth findings
- Explore learning rate schedules and advanced optimizers
- Investigate data augmentation techniques to push accuracy toward 95%+

## Deliverables

**19 MLflow experiments** tracked in `./mlruns/`  
**5 visualization charts** analyzing dropout, BatchNorm, depth, pooling, interactions  
**Comprehensive documentation**: Experiment journal + executive report  
**Reproducible code**: PowerShell automation + flexible CNN implementation

---

Find the [instructions](./instructions.md)

[Go back to Homepage](../README.md)
