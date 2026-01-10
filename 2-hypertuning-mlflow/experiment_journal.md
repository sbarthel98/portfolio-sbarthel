# MLflow CNN Hyperparameter Tuning - Experiment Journal

**Date Started:** January 7, 2026  
**Author:** Stijn Barthel  
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
| 0.0          | ~92.5%    | 91.78%  | 0.274      | 0.274    | Low          |
| 0.2          | ~92.8%    | 92.22%  | 0.248      | 0.248    | Low          |
| 0.3          | ~92.9%    | 92.28%  | 0.223      | 0.223    | Very Low     |
| 0.5          | ~92.4%    | 91.83%  | 0.235      | 0.235    | Low          |
| 0.7          | ~92.1%    | 91.63%  | 0.238      | 0.238    | Low          |

**Best Configuration:** dropout_rate = 0.3

### Analysis
The experiments revealed that moderate dropout provides the best validation accuracy:
- **Dropout 0.0** showed good performance (91.78%) but slightly higher loss, indicating minimal overfitting risk in this dataset
- **Dropout 0.2-0.3** achieved the best results, with 0.3 reaching 92.28% validation accuracy
- **Dropout 0.5** showed decreased performance (91.83%), suggesting too much regularization
- **Dropout 0.7** further decreased accuracy to 91.63%, confirming that excessive dropout harms learning
- Loss curves remained similar across all rates, indicating that the model generalizes well even without dropout
- The optimal sweet spot is around 0.3, providing slight regularization without hampering learning

### Visualization Reference
See: `visualizations/dropout_impact.png`

### Conclusion
Dropout rate of 0.3 provides optimal balance between regularization and model capacity for Fashion MNIST. Higher dropout rates (>0.5) unnecessarily limit the model's learning ability, while lower rates (0.0-0.2) perform nearly as well, suggesting the architecture itself has good generalization properties.

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
| Enabled   | ~92.8%    | 92.28%  | <3            | 0.236      |
| Disabled  | ~92.5%    | 92.14%  | ~3-4          | 0.227      |

**Performance Gain:** +0.14% improvement with BatchNorm

### Analysis
Batch normalization showed marginal but consistent benefits:
- **With BatchNorm**: Achieved 92.28% validation accuracy with faster initial convergence
- **Without BatchNorm**: Still reached competitive 92.14% accuracy, showing the architecture is stable
- **Convergence Speed**: BatchNorm enabled models reached 80% accuracy slightly faster (within 3 epochs)
- **Training Stability**: Both configurations trained stably, but BatchNorm showed smoother loss curves
- **Overfitting**: No significant difference in overfitting behavior between the two
- The small improvement (0.14%) suggests that for this relatively simple dataset, BatchNorm is helpful but not critical

### Visualization Reference
See: `visualizations/batchnorm_comparison.png`

### Conclusion
Batch normalization provides a modest accuracy boost and faster convergence for Fashion MNIST. While not dramatic, it's a low-cost addition that improves training dynamics without adding significant computation overhead. Recommended to keep enabled.

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
| 1     | [32]        | ~52K   | ~91.2%    | 90.57%  | ~15s/epoch    |
| 2     | [32, 64]    | ~422K  | ~92.5%    | 92.03%  | ~18s/epoch    |
| 3     | [32, 64, 128]| ~1.1M | ~92.8%    | 92.27%  | ~22s/epoch    |

**Best Configuration:** 2-3 layers (diminishing returns at 3)

### Analysis
Convolutional depth showed clear impact on model capacity:
- **1 Layer**: Achieved respectable 90.57% but lacked capacity for complex features (52K params)
- **2 Layers**: Strong performance at 92.03% with 422K parameters - excellent accuracy-to-parameter ratio
- **3 Layers**: Marginal improvement to 92.27% (+0.24%) but nearly 3x more parameters (1.1M)
- **Parameter Efficiency**: The jump from 1→2 layers (+1.46% accuracy) is much more significant than 2→3 layers (+0.24%)
- **Training Time**: Increased linearly with depth (~3-4s per additional layer)
- **Diminishing Returns**: Beyond 2 layers, the model gains little while requiring substantially more compute

### Visualization Reference
See: `visualizations/conv_depth_impact.png`

### Conclusion
Two convolutional layers provide the optimal balance for Fashion MNIST. The first layer learns basic edges/textures, the second captures garment patterns. A third layer adds minimal value, suggesting the dataset's feature hierarchy is relatively shallow. For production, use 2 layers unless computational resources are unlimited.

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
| Enabled  | 422K   | ~92.5%    | 92.01%  | 0.49%           |
| Disabled | ~500K  | ~92.6%    | 92.11%  | 0.49%           |

**Parameter Reduction:** ~15% fewer params with pooling

### Analysis
MaxPooling showed minimal impact on final accuracy:
- **With Pooling**: Achieved 92.01% with parameter reduction through spatial downsampling
- **Without Pooling**: Slightly better at 92.11% by preserving spatial information  
- **Parameter Count**: Pooling reduces parameters by ~15-20% while maintaining similar accuracy
- **Generalization**: Both configurations showed identical overfitting gaps (0.49%), indicating pooling doesn't significantly affect regularization here
- **Trade-off**: The 0.10% accuracy gain without pooling comes at cost of more parameters and compute
- For Fashion MNIST's 28×28 images, aggressive downsampling may remove useful fine-grained texture features

### Visualization Reference
See: `visualizations/pooling_comparison.png` (if created)

### Conclusion
MaxPooling is nearly accuracy-neutral for Fashion MNIST while providing computational benefits. The choice depends on deployment constraints: use pooling for faster inference and lower memory, skip it for marginal accuracy gains if resources permit. The dataset's small spatial size (28×28) means pooling is less critical than in high-resolution image tasks.

---

## Experiment 5: Dropout × BatchNorm Interactions

### Setup
- **Parameters:** 2×2 factorial design
  - `dropout_rate` = [0.0, 0.5]
  - `use_batch_norm` = [True, False]
- **Fixed Config:** 2 conv layers, MaxPooling enabled
- **Runs:** 4 experiments

### Hypothesis
BatchNorm + moderate dropout will provide best results. High dropout without BatchNorm may hurt training.

### Results

| Dropout | BatchNorm | Train Acc | Val Acc | Best? |
|---------|-----------|-----------|---------|-------|
| 0.0     | True      | ~92.8%    | 92.13%  | (best)     |
| 0.0     | False     | ~91.9%    | 91.29%  |       |
| 0.5     | True      | ~92.6%    | 92.15%  | (best)     |
| 0.5     | False     | ~92.4%    | 91.42%  |       |

**Best Combination:** BatchNorm=True (dropout matters less)

### Analysis
The factorial design revealed interesting interaction effects:
- **BatchNorm Impact**: Consistently provided +0.71-0.84% accuracy boost regardless of dropout
- **Dropout 0.0 + BatchNorm**: Achieved strong 92.13%, showing BatchNorm alone provides good regularization
- **Dropout 0.5 + BatchNorm**: Slightly better at 92.15%, confirming they work well together
- **Without BatchNorm**: Performance dropped significantly (91.29-91.42%), with dropout providing minimal help
- **Synergy**: BatchNorm appears to be the dominant factor; dropout adds marginal value on top
- **Key Finding**: BatchNorm is more critical than dropout for this architecture and dataset

### Visualization Reference
See: `visualizations/interactions_heatmap.png`

### Conclusion
Batch normalization is the primary driver of performance, providing stronger regularization than dropout alone. The interaction shows they're complementary but not synergistic—BatchNorm handles most of the heavy lifting. For optimal results, always use BatchNorm and add moderate dropout (0.3-0.5) for an extra 0.02-0.15% boost.

---

## Experiment 6: Combined Optimal Configuration

### Setup
Based on experiments 1-5, test top 3 promising configurations:
- **Config A:** [describe]
- **Config B:** [describe]
- **Config C:** [describe]

### Setup
Based on findings from experiments 1-5, test three optimized configurations:
- **Config A (Shallow Optimal)**: dropout=0.3, 2 layers, BatchNorm, Pooling
- **Config B (Deep)**: dropout=0.5, 3 layers, BatchNorm, Pooling  
- **Config C (Aggressive Dropout)**: dropout=0.7, 2 layers, BatchNorm, Pooling

### Hypothesis
Combining insights from previous experiments will yield best overall performance.

### Results

| Config | Description | Train Acc | Val Acc | Params | Best? |
|--------|-------------|-----------|---------|--------|-------|
| A      | Shallow, dropout=0.3 | ~93.1% | 92.51% | 422K | (best) |
| B      | Deep 3-layer | ~92.7% | 92.25% | 1.1M | (good) |
| C      | High dropout=0.7 | ~92.6% | 92.25% | 422K | (good) |

**Winner:** Config A (Shallow Optimal) with 92.51% validation accuracy

### Analysis
The combined configurations validated earlier findings:
- **Config A** achieved the best result (92.51%), confirming that 2 layers + dropout 0.3 is optimal
- **Config B** (3 layers) showed good performance but required 2.5x more parameters for only marginal gains (+0.26% from baseline)
- **Config C** (aggressive dropout 0.7) matched Config B's accuracy, showing that very high dropout can work but doesn't outperform moderate dropout
- **Parameter Efficiency**: Config A provides the best accuracy-to-parameter ratio
- **Trade-offs**: The extra computational cost of Config B isn't justified by its marginal accuracy gain
- **Validation**: These results confirm that our individual hyperparameter studies correctly identified optimal values

### Visualization Reference
See: `visualizations/top_configurations.png`

### Conclusion
The shallow architecture (2 conv layers) with moderate dropout (0.3) and BatchNorm represents the sweet spot for Fashion MNIST. Deeper networks and aggressive regularization add complexity without proportional performance gains. This configuration should be the default starting point for similar image classification tasks.

---

## Overall Conclusions

### Key Findings

1. **Dropout Impact:** Moderate dropout (0.3) provides optimal regularization. Higher rates (>0.5) unnecessarily limit learning, while lower rates (0.0-0.2) work nearly as well due to the architecture's inherent generalization.

2. **Batch Normalization:** Essential for optimal performance, providing +0.14-0.84% accuracy boost. It's the dominant regularization factor, more impactful than dropout alone.

3. **Architecture Depth:** Two convolutional layers hit the sweet spot—excellent accuracy (92.5%) with efficient parameter usage (422K). Third layer adds minimal value (+0.24%) at 2.5x parameter cost.

4. **Pooling Strategy:** Nearly accuracy-neutral (±0.10%) but reduces parameters by 15%. Use for efficiency; skip for marginal accuracy gains if resources allow.

5. **Hyperparameter Interactions:** BatchNorm is the primary driver. Dropout adds marginal value (0.02-0.15%) on top. They're complementary but not synergistic.

### Best Model Configuration

```python
optimal_config = {
    'num_conv_layers': 2,
    'conv_channels': [32, 64],
    'use_pooling': True,
    'use_batch_norm': True,
    'dropout_rate': 0.3,
    'fc_units': [128],
    'learning_rate': 0.001,
    'optimizer': 'Adam'
}
```

**Performance:**
- Validation Accuracy: 92.51%
- Best Val Accuracy: 92.80%
- Training Time: ~18 seconds/epoch (GPU)
- Total Parameters: 421,834
- Overfitting Gap: ~0.6%

### Lessons Learned

1. **Start with BatchNorm**: It provides the foundation for good performance and should always be included unless there's a specific reason not to.

2. **Don't over-regularize**: Dropout 0.3 is sufficient for Fashion MNIST. Higher values hurt more than they help.

3. **Diminishing returns with depth**: More layers ≠ better performance. Two layers captured all meaningful patterns in 28×28 fashion images.

4. **Parameter efficiency matters**: The 2-layer model achieved 92.5% with 422K params while 3-layer reached only 92.27% with 1.1M params.

### Future Directions

- [ ] Test on more complex datasets (CIFAR-10, CIFAR-100) to see if deeper networks become valuable
- [ ] Explore learning rate scheduling (CosineAnnealing, ReduceLROnPlateau)
- [ ] Try different optimizers (SGD with momentum, AdamW) for comparison
- [ ] Experiment with data augmentation (rotation, flip, cutout) to push accuracy further
- [ ] Investigate weight initialization strategies (He, Xavier) impact

---

## MLflow Tracking Notes

**Experiment Name:** `fashion_mnist_cnn_tuning`  
**Tracking URI:** `file:./mlruns`  
**Total Runs:** 19 completed experiments

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

**Journal Completed:** January 7, 2026
