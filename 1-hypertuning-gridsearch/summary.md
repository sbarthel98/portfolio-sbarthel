# Summary: Hyperparameter Tuning with Grid Search

**Status**: COMPLETED - January 5, 2026

This project successfully executed 61 systematic hyperparameter tuning experiments on a neural network trained on the Fashion MNIST dataset. All experiments, analysis, visualizations, and comprehensive reports are complete.

**Key Results Achieved:**
- **Best Performance**: 85.5% accuracy (epochs_10)
- **Best Combined Config**: 84.2% accuracy (combined_config_2)  
- **Most Impactful**: Optimizer choice (65.9% performance swing)
- **Critical Insight**: Vanilla SGD failed completely (17% accuracy)
- **Surprise Finding**: Shallow networks (1 layer) beat deeper ones

## Objectives

1. Experiment with various hyperparameters (epochs, units, batch size, depth, learning rate, optimizer)
2. Analyze and visualize the relationships between hyperparameters
3. Reflect on findings using the scientific method (hypothesis → experiment → analysis → conclusion)
4. Create data-driven reports documenting insights and recommendations

## Project Files

### Core Experiment Files
- [instructions.md](./instructions.md) - Detailed experiment instructions and study questions
- **[grid_search_experiments.py](./grid_search_experiments.py)** - Automated experiment runner with SSL fixes
- **[analyze_results.py](./analyze_results.py)** - Results analysis and visualization generator
- **[extract_tensorboard_metrics.py](./extract_tensorboard_metrics.py)** - Extracts metrics from TensorBoard logs

### Execution Helpers
- **[run_experiments.ps1](./run_experiments.ps1)** - Interactive PowerShell menu for experiment management

### Report Templates & Completed Reports
- **[report.md](./report.md)** - COMPLETED - Full technical report with all findings
- **[experiment_journal.md](./experiment_journal.md)** - COMPLETED - Detailed experiment journal

### Legacy
- [notebook.ipynb](./notebook.ipynb) - Original interactive notebook (reference only)
- [model.toml](./model.toml) - Configuration file
- [settings.toml](./settings.toml) - Settings file

## Quick Start

### Option 1: Use the Interactive Menu
```powershell
cd c:\Git\portfolio-sbarthel\1-hypertuning-gridsearch
.\run_experiments.ps1
```

### Option 2: Manual Execution
```powershell
# Run experiments
python grid_search_experiments.py

# Analyze results
python analyze_results.py

# View in TensorBoard
tensorboard --logdir=modellogs
```

## Experiments Completed (61 Total)

1. **Epochs** (3 configs) - Training duration impact: 79.5% → 85.5%
2. **Hidden Units** (36 configs) - Network capacity exploration: 79.0% → 84.5%
3. **Batch Size** (6 configs) - Training dynamics: 70.5% → 84.4%
4. **Model Depth** (3 configs) - Architecture complexity: Depth 1 won
5. **Learning Rate** (6 configs) - Optimization sensitivity: 57.0% → 83.9%
6. **Optimizers** (4 configs) - Algorithm comparison: SGD failed, RMSprop won
7. **Combined Configurations** (3 configs) - Optimal combinations: 84.2% best

## Key Concepts Explored

- **Grid Search**: Systematic exploration of hyperparameter space
- **Factors of 2**: Efficient parameter scanning strategy
- **Train Steps**: Understanding epochs vs. batches
- **Hyperparameter Interactions**: How parameters influence each other
- **Overfitting vs Underfitting**: Balancing model capacity
- **Convergence**: Finding optimal learning dynamics

## Deliverables (All Complete)

1. **Experiment logs** in `modellogs/` (61 directories) with TensorBoard event files
2. **TOML result files** with extracted metrics (train/valid loss, accuracy)
3. **Visualization plots** in `visualizations/` (heatmaps, scatter plots, summaries)
4. **Summary data** in `experiment_summary.csv` (61 rows with full metrics)
5. **Comprehensive report** in `REPORT_FILLED.md` (all study questions answered)
6. **Experiment journal** in `EXPERIMENT_JOURNAL_FILLED.md` (scientific documentation)
7. **Metric extraction** from TensorBoard logs via `extract_tensorboard_metrics.py`

## Study Questions (All Answered in REPORT_FILLED.md)

- What is the upside and downside of increasing epochs?
- When do you need more epochs to find the best configuration?
- What are the advantages and disadvantages of using factors of 2?
- How do hyperparameters interact with each other?
- Why did SGD fail so catastrophically?
- What is the optimal learning rate for Adam optimizer?
- Why do asymmetric network architectures work well?

**See [REPORT_FILLED.md](./REPORT_FILLED.md) for complete answers with data and analysis.**

## Technical Stack

- **PyTorch**: Deep learning framework
- **mads-datasets**: Fashion MNIST data loading
- **mltrainer**: Training pipeline with built-in logging
- **TensorBoard**: Real-time visualization
- **tomlserializer**: Configuration management
- **Matplotlib/Seaborn**: Results visualization

## Resources

- [TensorBoard Guide](https://www.tensorflow.org/tensorboard)
- [PyTorch Optimizers](https://pytorch.org/docs/stable/optim.html)
- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [Grid Search Best Practices](https://scikit-learn.org/stable/modules/grid_search.html)

## Next Steps

This exercise is complete. You now understand:
1. Why grid search is "naive" (exhaustive but inefficient - we ran 61 experiments)
2. The massive importance of optimizer and learning rate choices
3. How to systematically test hyperparameters
4. The scientific method applied to machine learning

**Ready to progress to:**
1. [MLflow-based hypertuning](../2-hypertuning-mlflow/) - Better tracking and comparison
2. [RNN hypertuning](../3-hypertuning-rnn/) - Sequence models
3. [Ray Tune](../4-hypertuning-ray/) - Efficient automated hyperparameter optimization

## Quick Results Reference

| Experiment | Best Config | Accuracy | Key Finding |
|-----------|-------------|----------|-------------|
| Epochs | 10 epochs | 85.5% | More is better, no overfitting |
| Units | 512/128 | 84.5% | Asymmetric architectures work well |
| Batch Size | 128 | 84.4% | Larger batches = better results |
| Depth | 1 layer | 82.9% | Simpler is better for Fashion MNIST |
| Learning Rate | 0.001 | 83.9% | Default Adam LR is optimal |
| Optimizer | RMSprop | 82.9% | Adaptive optimizers essential |
| Combined | Config 2 | 84.2% | Synergy of optimal parameters |

**Worst**: SGD optimizer → 17.0% (complete failure without momentum)

---

[Go back to Homepage](../README.md)
