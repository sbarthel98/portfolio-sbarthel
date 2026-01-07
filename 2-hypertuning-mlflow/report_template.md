# CNN Hyperparameter Tuning Report
**Fashion MNIST Classification with Dropout, Batch Normalization, and Convolutional Layers**

---

## Objective
Systematically investigate the impact of dropout regularization, batch normalization, convolutional depth, and pooling strategies on CNN performance for Fashion MNIST classification using MLflow experiment tracking.

---

## Methodology

**Dataset:** Fashion MNIST (60k train, 10k validation, 10 classes)  
**Framework:** PyTorch with MLflow tracking  
**Architecture:** Flexible CNN with ModuleList (Conv2d → BatchNorm → ReLU → MaxPool)

### Experiments Conducted

| Experiment | Variable | Values Tested | Runs |
|------------|----------|---------------|------|
| 1. Dropout | dropout_rate | 0.0, 0.2, 0.3, 0.5, 0.7 | 5 |
| 2. BatchNorm | use_batch_norm | True, False | 2 |
| 3. Conv Depth | num_conv_layers | 1, 2, 3 | 3 |
| 4. Pooling | use_pooling | True, False | 2 |
| 5. Interactions | dropout × batch_norm | 2×2 factorial | 4 |
| 6. Combined | Optimal configs | Top 3 | 3 |

**Total Experiments:** 19+ runs

---

## Key Findings

### 1. Dropout Impact
[Insert visualization/chart: dropout_impact.png]

**Result:** Optimal dropout rate = **[VALUE]**
- No dropout (0.0): [observation about overfitting]
- Moderate dropout (0.3-0.5): [observation about balance]
- High dropout (0.7): [observation about underfitting]

**Insight:** [1-2 sentence key takeaway]

### 2. Batch Normalization
[Insert visualization/chart: batch_norm_comparison.png]

**Result:** BatchNorm provided **[X]%** accuracy improvement
- Convergence speed: [faster/slower]
- Training stability: [more/less stable]
- Final validation accuracy: [WITH]% vs [WITHOUT]%

**Insight:** [1-2 sentence key takeaway]

### 3. Convolutional Depth
[Insert visualization/chart: conv_depth_impact.png]

**Result:** Optimal depth = **[VALUE] layers**
- 1 layer ([32]): [accuracy]%, [params]
- 2 layers ([32,64]): [accuracy]%, [params]
- 3 layers ([32,64,128]): [accuracy]%, [params]

**Insight:** [1-2 sentence key takeaway about depth vs performance trade-off]

### 4. Hyperparameter Interactions
[Insert visualization/chart: interactions_heatmap.png]

**Result:** Best combination = dropout **[VALUE]** + BatchNorm **[True/False]**
- Synergistic effects: [observation]
- Conflicting effects: [observation]

**Insight:** [1-2 sentence key takeaway]

---

## Optimal Configuration

```python
best_config = {
    'num_conv_layers': [VALUE],
    'conv_channels': [VALUES],
    'use_pooling': True/False,
    'use_batch_norm': True/False,
    'dropout_rate': [VALUE],
    'fc_units': [VALUES],
}
```

### Performance Metrics
- **Validation Accuracy:** [XX.X]%
- **Training Accuracy:** [XX.X]%
- **Total Parameters:** [X,XXX]
- **Training Time:** [X] minutes/epoch
- **Overfitting Gap:** [X.X]% (train - val)

### Comparison to Baseline
- Baseline (no regularization): [accuracy]%
- Optimal configuration: [accuracy]%
- **Improvement:** +[X.X]%

---

## Conclusions

1. **Regularization Trade-offs:** [Key finding about dropout and BatchNorm balance]

2. **Architecture Depth:** [Key finding about conv layers and complexity]

3. **Training Efficiency:** [Key finding about BatchNorm and convergence]

4. **Generalization:** [Key finding about val vs train performance]

### Practical Recommendations

✓ Use dropout rate of **[VALUE]** for Fashion MNIST-scale problems  
✓ Always include BatchNorm for faster convergence  
✓ [2/3] conv layers provide best accuracy-parameter trade-off  
✓ MaxPooling improves generalization without significant accuracy loss

### Future Work
- Test configurations on CIFAR-10/100 for generalization
- Explore learning rate schedules and advanced optimizers
- Investigate data augmentation impact with current architecture

---

## MLflow Experiment Details

**Tracking URI:** `./mlruns`  
**Experiment:** `fashion_mnist_cnn_tuning`  
**Total Runs:** [NUMBER]  
**Best Run ID:** `[MLflow run ID]`

View full results: `mlflow ui` → http://localhost:5000

---

**Report Date:** [DATE]  
**Author:** [YOUR NAME]  
**Code Repository:** `2-hypertuning-mlflow/`
