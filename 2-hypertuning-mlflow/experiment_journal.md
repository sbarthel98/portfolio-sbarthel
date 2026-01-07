# MLflow CNN Hyperparameter Tuning - Experiment Journal

**Date Started:** [DATE]  
**Author:** [YOUR NAME]  
**Project:** Fashion MNIST CNN with Dropout, BatchNorm, and Convolutional Layers  
**Framework:** PyTorch + MLflow

---

## Research Question

How do dropout, batch normalization, convolutional depth, and pooling layers affect CNN performance on Fashion MNIST classification? What are the optimal hyperparameter combinations?

---

## Hypothesis

**Dropout Impact:** Higher dropout rates (0.4-0.6) will reduce overfitting and improve validation accuracy, but excessive dropout (>0.6) will hurt training.

**Batch Normalization:** Models with BatchNorm will converge faster and achieve higher accuracy by stabilizing gradients.

**Convolutional Depth:** Deeper networks (2-3 conv layers) will capture more complex features, but may require more data/epochs.

**Pooling Layers:** MaxPooling will reduce parameters and improve generalization, but may lose fine-grained spatial information.

**Interactions:** BatchNorm + moderate dropout (0.3-0.5) will provide best results by combining regularization and training stability.

---

## Experiment 1: Dropout Rate Exploration

### Setup
- **Parameter:** `dropout_rate` = [0.0, 0.2, 0.3, 0.5, 0.7]
- **Fixed Config:** 2 conv layers, BatchNorm enabled, MaxPooling enabled
- **Runs:** 5 experiments

### Hypothesis
Moderate dropout (0.3-0.5) will provide best validation accuracy by reducing overfitting without excessive regularization.

### Results

| Dropout Rate | Train Acc | Val Acc | Train Loss | Val Loss | Overfitting? |
|--------------|-----------|---------|------------|----------|--------------|
| 0.0          | %         | %       |            |          |              |
| 0.2          | %         | %       |            |          |              |
| 0.3          | %         | %       |            |          |              |
| 0.5          | %         | %       |            |          |              |
| 0.7          | %         | %       |            |          |              |

**Best Configuration:** dropout_rate = [VALUE]

### Analysis
[After running experiments, fill in:]
- At what dropout rate did overfitting start to appear?
- Which rate provided the best balance?
- Did very high dropout (0.7) hurt performance significantly?
- How did loss curves behave across different rates?

### Visualization Reference
See: `visualizations/dropout_impact.png`

### Conclusion
[Summary of findings and implications for next experiments]

---

## Experiment 2: Batch Normalization Impact

### Setup
- **Parameter:** `use_batch_norm` = [True, False]
- **Fixed Config:** dropout=0.5, 2 conv layers, MaxPooling enabled
- **Runs:** 2 experiments

### Hypothesis
BatchNorm will improve convergence speed and final accuracy by normalizing activations.

### Results

| BatchNorm | Train Acc | Val Acc | Epochs to 80% | Final Loss |
|-----------|-----------|---------|---------------|------------|
| Enabled   | %         | %       |               |            |
| Disabled  | %         | %       |               |            |

**Performance Gain:** [X]% improvement with BatchNorm

### Analysis
[After running experiments, fill in:]
- Did BatchNorm speed up convergence?
- Was there a significant accuracy improvement?
- How did training stability differ?
- Did BatchNorm affect overfitting?

### Visualization Reference
See: `visualizations/batch_norm_comparison.png`

### Conclusion
[Summary of findings]

---

## Experiment 3: Convolutional Depth

### Setup
- **Parameter:** `num_conv_layers` = [1, 2, 3]
- **Fixed Config:** dropout=0.5, BatchNorm enabled, MaxPooling enabled
- **Channel Progression:**
  - 1 layer: [32]
  - 2 layers: [32, 64]
  - 3 layers: [32, 64, 128]
- **Runs:** 3 experiments

### Hypothesis
Deeper networks will capture more complex features, but may require longer training or risk overfitting.

### Results

| Depth | Channels    | Params | Train Acc | Val Acc | Training Time |
|-------|-------------|--------|-----------|---------|---------------|
| 1     | [32]        |        | %         | %       | min           |
| 2     | [32, 64]    |        | %         | %       | min           |
| 3     | [32, 64, 128]|       | %         | %       | min           |

**Best Configuration:** [VALUE] layers

### Analysis
[After running experiments, fill in:]
- Did deeper networks improve accuracy?
- Was there a point of diminishing returns?
- How did parameter count affect training time?
- Did overfitting increase with depth?

### Visualization Reference
See: `visualizations/conv_depth_impact.png`

### Conclusion
[Summary of findings]

---

## Experiment 4: Pooling Layer Impact

### Setup
- **Parameter:** `use_pooling` = [True, False]
- **Fixed Config:** dropout=0.5, BatchNorm enabled, 2 conv layers
- **Runs:** 2 experiments

### Hypothesis
MaxPooling will reduce parameters and improve generalization through spatial downsampling.

### Results

| Pooling  | Params | Train Acc | Val Acc | Overfitting Gap |
|----------|--------|-----------|---------|-----------------|
| Enabled  |        | %         | %       | %               |
| Disabled |        | %         | %       | %               |

**Parameter Reduction:** [X]% fewer params with pooling

### Analysis
[After running experiments, fill in:]
- Did pooling improve generalization?
- What was the trade-off between parameters and performance?
- Did removing pooling lead to overfitting?
- How did spatial resolution affect feature learning?

### Visualization Reference
See: `visualizations/pooling_comparison.png` (if created)

### Conclusion
[Summary of findings]

---

## Experiment 5: Dropout × BatchNorm Interactions

### Setup
- **Parameters:** 2×2 factorial design
  - `dropout_rate` = [0.3, 0.5]
  - `use_batch_norm` = [True, False]
- **Fixed Config:** 2 conv layers, MaxPooling enabled
- **Runs:** 4 experiments

### Hypothesis
BatchNorm + moderate dropout will provide best results. High dropout without BatchNorm may hurt training.

### Results

| Dropout | BatchNorm | Train Acc | Val Acc | Best? |
|---------|-----------|-----------|---------|-------|
| 0.3     | True      | %         | %       |       |
| 0.3     | False     | %         | %       |       |
| 0.5     | True      | %         | %       |       |
| 0.5     | False     | %         | %       |       |

**Best Combination:** dropout=[VALUE], batch_norm=[VALUE]

### Analysis
[After running experiments, fill in:]
- Was there a synergistic effect between dropout and BatchNorm?
- Which regularization technique had stronger impact?
- Did BatchNorm compensate for high dropout?
- What was the optimal balance?

### Visualization Reference
See: `visualizations/interactions_heatmap.png`

### Conclusion
[Summary of findings]

---

## Experiment 6: Combined Optimal Configuration

### Setup
Based on experiments 1-5, test top 3 promising configurations:
- **Config A:** [describe]
- **Config B:** [describe]
- **Config C:** [describe]

### Hypothesis
Combining insights from previous experiments will yield best overall performance.

### Results

| Config | Description | Train Acc | Val Acc | Params | Best? |
|--------|-------------|-----------|---------|--------|-------|
| A      |             | %         | %       |        |       |
| B      |             | %         | %       |        |       |
| C      |             | %         | %       |        |       |

**Winner:** Config [LETTER] with [X]% validation accuracy

### Analysis
[After running experiments, fill in:]
- Did the combined configuration outperform individual experiments?
- Were there any unexpected interactions?
- What trade-offs were made (accuracy vs parameters vs training time)?

### Visualization Reference
See: `visualizations/top_configs.png`

### Conclusion
[Summary of findings]

---

## Overall Conclusions

### Key Findings

1. **Dropout Impact:** [Summary]

2. **Batch Normalization:** [Summary]

3. **Architecture Depth:** [Summary]

4. **Pooling Strategy:** [Summary]

5. **Hyperparameter Interactions:** [Summary]

### Best Model Configuration

```python
optimal_config = {
    'num_conv_layers': [VALUE],
    'conv_channels': [VALUES],
    'use_pooling': [VALUE],
    'use_batch_norm': [VALUE],
    'dropout_rate': [VALUE],
    'fc_units': [VALUES],
}
```

**Performance:**
- Validation Accuracy: [X]%
- Training Time: [X] minutes
- Total Parameters: [X]
- Overfitting Gap: [X]%

### Lessons Learned

1. [Key insight about dropout]
2. [Key insight about batch normalization]
3. [Key insight about architecture design]
4. [Key insight about hyperparameter interactions]

### Future Directions

- [ ] Test on more complex datasets (CIFAR-10, CIFAR-100)
- [ ] Explore learning rate scheduling
- [ ] Try different optimizers (SGD with momentum, AdamW)
- [ ] Experiment with data augmentation
- [ ] Investigate weight initialization strategies

---

## MLflow Tracking Notes

**Experiment Name:** `fashion_mnist_cnn_tuning`  
**Tracking URI:** `file:./mlruns`  
**Total Runs:** [NUMBER]

### How to View Results

```powershell
# Launch MLflow UI
python -m mlflow ui

# Then open: http://localhost:5000
```

### Metrics Tracked
- `train_loss` (per epoch)
- `train_accuracy` (per epoch)
- `val_loss` (per epoch)
- `val_accuracy` (per epoch)
- `final_train_accuracy`
- `final_val_accuracy`
- `total_params`

### Parameters Logged
All config values: epochs, batch size, learning rate, optimizer, architecture details

---

## Appendix: Technical Details

### Environment
- Python: 3.12.10
- PyTorch: 2.9.1
- MLflow: 3.8.1
- Device: [CPU/CUDA]

### Dataset
- Fashion MNIST
- Training samples: 60,000
- Validation samples: 10,000
- Classes: 10
- Image size: 28×28 grayscale

### Training Configuration
- Epochs: 10
- Batch size: 64
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: CrossEntropyLoss

### Architecture Notes
- Base CNN with ModuleList for flexible layer construction
- Each conv block: Conv2d → BatchNorm2d (optional) → ReLU → MaxPool2d (optional)
- FC layers with Dropout between
- Output: 10 classes (Fashion MNIST categories)

---

**Journal Completed:** [DATE]
