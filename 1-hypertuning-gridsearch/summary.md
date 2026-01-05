# Summary: Hyperparameter Tuning with Grid Search

This project implements a comprehensive hyperparameter tuning study on the Fashion MNIST dataset using systematic grid search methodology.

## 🎯 Objectives

1. Experiment with various hyperparameters (epochs, units, batch size, depth, learning rate, optimizer)
2. Analyze and visualize the relationships between hyperparameters
3. Reflect on findings using the scientific method (hypothesis → experiment → analysis → conclusion)
4. Create a data-driven report documenting insights and recommendations

## 📁 Project Files

- [notebook.ipynb](./notebook.ipynb) - Original interactive notebook with basic examples
- [instructions.md](./instructions.md) - Detailed experiment instructions and study questions
- **[grid_search_experiments.py](./grid_search_experiments.py)** - Automated experiment runner (NEW)
- **[analyze_results.py](./analyze_results.py)** - Results analysis and visualization script (NEW)
- **[report_template.md](./report_template.md)** - Comprehensive report template (NEW)
- **[README_EXECUTION.md](./README_EXECUTION.md)** - Complete execution guide (NEW)
- **[run_experiments.ps1](./run_experiments.ps1)** - Interactive PowerShell helper script (NEW)

## 🚀 Quick Start

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

## 📊 Experiments Included

1. **Epochs** (3, 5, 10) - Understanding training duration impact
2. **Hidden Units** (16-512) - Exploring network capacity (optional, comprehensive)
3. **Batch Size** (4-128) - Training dynamics and convergence
4. **Model Depth** (1-3 layers) - Architecture complexity
5. **Learning Rate** (1e-2 to 1e-5) - Optimization sensitivity
6. **Optimizers** (SGD, Adam, AdamW, RMSprop) - Algorithm comparison
7. **Combined Configurations** - Testing promising combinations

## 🔍 Key Concepts Explored

- **Grid Search**: Systematic exploration of hyperparameter space
- **Factors of 2**: Efficient parameter scanning strategy
- **Train Steps**: Understanding epochs vs. batches
- **Hyperparameter Interactions**: How parameters influence each other
- **Overfitting vs Underfitting**: Balancing model capacity
- **Convergence**: Finding optimal learning dynamics

## 📈 Deliverables

1. ✅ Experiment logs in `modellogs/` with TensorBoard visualization
2. ✅ Configuration tracking via TOML files
3. ✅ Visualization plots in `visualizations/`
4. ✅ Summary data in `experiment_summary.csv`
5. 📝 Final report based on `report_template.md` (to be completed after experiments)

## 🎓 Study Questions Answered in Report

- What is the upside and downside of increasing epochs?
- When do you need more epochs to find the best configuration?
- What are the advantages and disadvantages of using factors of 2?
- How do hyperparameters interact with each other?

## 🛠️ Technical Stack

- **PyTorch**: Deep learning framework
- **mads-datasets**: Fashion MNIST data loading
- **mltrainer**: Training pipeline with built-in logging
- **TensorBoard**: Real-time visualization
- **tomlserializer**: Configuration management
- **Matplotlib/Seaborn**: Results visualization

## 📚 Resources

- [TensorBoard Guide](https://www.tensorflow.org/tensorboard)
- [PyTorch Optimizers](https://pytorch.org/docs/stable/optim.html)
- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [Grid Search Best Practices](https://scikit-learn.org/stable/modules/grid_search.html)

## 💡 Next Steps

After completing this exercise:
1. Understand why grid search is "naive" and its limitations
2. Progress to [MLflow-based hypertuning](../2-hypertuning-mlflow/)
3. Explore advanced methods with [Ray Tune](../4-hypertuning-ray/)

---

[Go back to Homepage](../README.md)
