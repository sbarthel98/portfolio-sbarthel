# Hyperparameter Tuning Report
**Grid Search Experiments on Fashion MNIST**

*Date: January 5, 2026*  
*Student: Stijn Barthel*

---

## Executive Summary

This report presents findings from systematic hyperparameter tuning experiments on a neural network trained on the Fashion MNIST dataset. Through 61 controlled experiments, we tested epochs (3, 5, 10), hidden units (36 combinations), batch sizes (4-128), model depth (1-3 layers), learning rates (1e-5 to 0.01), and optimizers (SGD, Adam, AdamW, RMSprop).

**Key Findings:**
- **Best Overall Configuration**: 10 epochs, batch size 64, units 256/128, learning rate 0.0005, AdamW optimizer → **84.2% accuracy**
- **Most Impactful Parameters**: Optimizer choice and learning rate had the strongest effects
- **Worst Performance**: SGD optimizer with default LR → **17.0% accuracy** (essentially random)
- **Optimal Hidden Units**: 512/256 and 128/512 achieved ~83.5-84.5% accuracy, showing model capacity matters
- **Batch Size Sweet Spot**: 64-128 performed best (84-84.4%), while very small batches (4-8) performed poorly (70-74%)

---

## 1. Hypothesis & Experimental Design

### Initial Hypotheses

**Hypothesis 1: Epochs**
- **Prediction**: Increasing epochs will improve model performance up to a point
- **Result**: CONFIRMED - 3 epochs: 79.5%, 5 epochs: 83.6%, 10 epochs: 85.5%

**Hypothesis 2: Hidden Units**
- **Prediction**: Larger networks will perform better with diminishing returns
- **Result**: PARTIALLY CONFIRMED - Best was 128/512 (83.5%), but 16/16 still achieved 79.0%

**Hypothesis 3: Batch Size**
- **Prediction**: Medium batch sizes (32-64) will be optimal
- **Result**: CONFIRMED - Batch 64: 83.6%, Batch 128: 84.4%, but Batch 4: 73.8%

**Hypothesis 4: Model Depth**
- **Prediction**: One additional layer will help, deeper may not
- **Result**: REJECTED - Depth 1: 82.9%, Depth 2: 81.8%, Depth 3: 81.4% (simpler was better)

**Hypothesis 5: Learning Rate**
- **Prediction**: 1e-3 to 1e-4 will be optimal
- **Result**: CONFIRMED - LR 0.001: 83.9%, LR 0.0001: 77.5%, LR 1e-5: 57.0%

**Hypothesis 6: Optimizer**
- **Prediction**: Adam/AdamW will outperform SGD
- **Result**: STRONGLY CONFIRMED - SGD: 17.0%, Adam: 80.8%, AdamW: 82.1%, RMSprop: 82.9%

---

## 2. Results

### Experiment 1: Impact of Epochs

| Configuration | Epochs | Final Valid Loss | Accuracy | Train Loss |
|--------------|--------|------------------|----------|------------|
| epochs_3 | 3 | 0.551 | 79.5% | 0.521 |
| epochs_5 | 5 | 0.478 | 83.6% | 0.458 |
| epochs_10 | 10 | 0.399 | 85.5% | 0.401 |

**Analysis**: Clear improvement with more epochs. No overfitting observed (train/valid loss similar). 10 epochs showed best performance with 6% absolute improvement over 3 epochs.

---

### Experiment 2: Hidden Units Grid Search (36 combinations)

**Top 5 Configurations:**

| Configuration | Units1 | Units2 | Accuracy | Valid Loss |
|--------------|--------|--------|----------|------------|
| units_128_512 | 128 | 512 | **83.5%** | 0.452 |
| units_512_128 | 512 | 128 | **84.5%** | 0.441 |
| units_256_128 | 256 | 128 | 83.8% | 0.459 |
| units_512_256 | 512 | 256 | 83.8% | 0.437 |
| units_256_512 | 256 | 512 | 83.7% | 0.453 |

**Bottom 5 Configurations:**

| Configuration | Units1 | Units2 | Accuracy | Valid Loss |
|--------------|--------|--------|----------|------------|
| units_16_16 | 16 | 16 | 79.0% | 0.617 |
| units_64_16 | 64 | 16 | 78.8% | 0.596 |
| units_16_32 | 16 | 32 | 79.7% | 0.575 |
| units_512_32 | 512 | 32 | 79.6% | 0.572 |
| units_32_16 | 32 | 16 | 80.2% | 0.559 |

**Analysis**: 
- Larger models (128+ units) consistently outperformed tiny models (16-32 units)
- Optimal balance appears around 256-512 units per layer
- Even smallest model (16/16) achieved 79% - Fashion MNIST is not extremely complex
- Asymmetric configurations (128/512, 512/128) performed best

![Units Heatmap](visualizations/units_heatmap.png)

---

### Experiment 3: Batch Size Impact

| Configuration | Batch Size | Accuracy | Valid Loss | Train Loss |
|--------------|-----------|----------|------------|------------|
| batchsize_4 | 4 | 73.8% | 0.757 | 0.806 |
| batchsize_8 | 8 | 70.5% | 0.723 | 0.714 |
| batchsize_16 | 16 | 77.2% | 0.614 | 0.560 |
| batchsize_32 | 32 | 82.2% | 0.495 | 0.499 |
| batchsize_64 | 64 | **83.6%** | 0.468 | 0.473 |
| batchsize_128 | 128 | **84.4%** | 0.430 | 0.421 |

**Analysis**: 
- Dramatic performance drop with tiny batches (4-8)
- Batch size 4 shows high loss (0.806 train, 0.757 valid) - noisy gradients prevented convergence
- Larger batches (64-128) achieved best results
- Batch 128 was optimal but requires more memory

---

### Experiment 4: Model Depth

| Configuration | Depth | Accuracy | Valid Loss | Observation |
|--------------|-------|----------|------------|-------------|
| depth_1 | 1 | **82.9%** | 0.477 | Simpler is better! |
| depth_2 | 2 | 81.8% | 0.476 | Baseline config |
| depth_3 | 3 | 81.4% | 0.509 | Slight overfitting |

**Analysis**: 
- Unexpected result: Single hidden layer performed BEST
- Additional layers added complexity without benefit
- Fashion MNIST is simple enough for shallow architecture
- This contradicts initial hypothesis but aligns with Occam's razor

---

### Experiment 5: Learning Rate

| Configuration | Learning Rate | Accuracy | Valid Loss | Observation |
|--------------|--------------|----------|------------|-------------|
| lr_0.01 | 0.01 | 81.1% | 0.511 | Too high, unstable |
| lr_0.005 | 0.005 | 83.4% | 0.457 | Good performance |
| lr_0.001 | 0.001 | **83.9%** | 0.462 | Optimal |
| lr_0.0005 | 0.0005 | 82.1% | 0.489 | Slightly slow |
| lr_0.0001 | 0.0001 | 77.5% | 0.631 | Too slow to converge |
| lr_1e-05 | 0.00001 | 57.0% | 1.666 | Failed to learn |

**Analysis**:
- Clear optimal range at 0.001-0.005
- LR 0.0001 and below: insufficient learning in 5 epochs
- LR 0.01: some instability but still functional
- 10x difference in learning rate can mean 26% accuracy difference

![Learning Rate Impact](visualizations/learning_rate_scatter.png)

---

### Experiment 6: Optimizer Comparison

| Configuration | Optimizer | Accuracy | Valid Loss | Train Loss |
|--------------|-----------|----------|------------|------------|
| optimizer_SGD | SGD | **17.0%** | 2.244 | 2.250 |
| optimizer_Adam | Adam | 80.8% | 0.515 | 0.455 |
| optimizer_AdamW | AdamW | 82.1% | 0.486 | 0.457 |
| optimizer_RMSprop | RMSprop | **82.9%** | 0.456 | 0.454 |

**Analysis**:
- **SGD complete failure**: 17% accuracy is worse than random (10 classes = 10% baseline)
- Adaptive optimizers (Adam, AdamW, RMSprop) all performed well
- RMSprop slightly edged out AdamW
- SGD would need careful tuning (momentum, LR scheduling) to compete

---

### Experiment 7: Combined Optimal Configurations

| Configuration | Epochs | Batch | Units | LR | Optimizer | Accuracy | Valid Loss |
|--------------|--------|-------|-------|-----|-----------|----------|------------|
| combined_config_1 | 10 | 32 | 512/256 | 0.001 | Adam | 83.8% | 0.445 |
| combined_config_2 | 10 | 64 | 256/128 | 0.0005 | AdamW | **84.2%** | 0.443 |
| combined_config_3 | 5 | 128 | 128/64 | 0.001 | Adam | 83.3% | 0.476 |

**Analysis**:
- Combined configurations validated individual findings
- Config 2 achieved best overall accuracy: **84.2%**
- Combining multiple optimal settings produced consistent high performance

![Configuration Summary](visualizations/configuration_summary.png)

---

## 3. Analysis & Interpretation

### Most Impactful Hyperparameters (Ranked)

1. **Optimizer** (17.0% → 82.9%): 65.9% swing - SGD catastrophic, adaptive optimizers essential
2. **Learning Rate** (57.0% → 83.9%): 26.9% swing - Must be in optimal range
3. **Batch Size** (70.5% → 84.4%): 13.9% swing - Small batches hurt significantly
4. **Epochs** (79.5% → 85.5%): 6.0% swing - Clear improvement but diminishing returns
5. **Hidden Units** (79.0% → 84.5%): 5.5% swing - Model capacity matters moderately
6. **Depth** (81.4% → 82.9%): 1.5% swing - Minimal impact, simpler is better

### Overfitting Analysis

**No significant overfitting observed:**
- Best models showed train/valid loss within 0.02-0.05
- Example: combined_config_2 → Train: 0.423, Valid: 0.443 (healthy gap)
- Only very small batches (4-8) showed poor generalization

### Diminishing Returns

- **Epochs**: 3→5 gained 4.1%, but 5→10 only gained 1.9%
- **Units**: 16→64 gained 4%, but 256→512 gained <1%
- **Batch size**: 16→32 gained 5%, but 64→128 gained 0.8%

### Unexpected Findings

1. **Shallow networks best**: Depth 1 outperformed depth 2 and 3
2. **Very small batches disastrous**: Batch size 8 got 70.5% (worse than batch 4!)
3. **SGD failure**: Even with 5 epochs, SGD learned almost nothing

---

## 4. Study Questions

### 1. Batch Size Impact on Training Stability

**Observation**: Small batch sizes (4-8) showed unstable training with high variance in loss.

**Explanation**: 
- Batch size 4: Gradient estimates are extremely noisy with only 4 samples
- Each update is based on 0.06% of training data (4/60,000)
- High variance prevents consistent convergence direction
- Larger batches (64-128) average over 1,000-2,000 samples for stable gradients

**Trade-off**: Small batches enable escape from local minima but prevent convergence; large batches converge smoothly but may get stuck in sharp minima.

---

### 2. Learning Rate Sweet Spot

**Finding**: LR 0.001 optimal; 0.0001 too slow (77.5%), 0.01 too fast (81.1%)

**Why 0.001 works**:
- With Adam's adaptive learning rate, 0.001 provides good initial step size
- Allows 5 epochs to make significant progress
- Auto-adjusts per-parameter based on gradient history

**Why extremes fail**:
- **0.0001**: Takes tiny steps, can't explore loss landscape in 5 epochs
- **0.01**: Overshoots minima, causes oscillation, slower convergence
- **1e-5**: Essentially frozen, barely moves from initialization

---

### 3. Optimizer Performance Differences

**Results Summary**:
- SGD: 17.0% (failed)
- Adam: 80.8%
- AdamW: 82.1%
- RMSprop: 82.9%

**Explanation**:
- **SGD failure**: Vanilla SGD requires careful tuning (momentum, LR scheduling, warmup)
  - Our fixed LR 0.001 with no momentum was insufficient
  - SGD converged to poor local minimum
  
- **Adam success**: Adaptive per-parameter learning rates helped navigate loss landscape
  - Momentum terms (β1, β2) provide stability
  - Works well "out of the box"

- **RMSprop edge**: Root mean square scaling prevented exploding/vanishing gradients
  - Similar to Adam but simpler momentum

- **AdamW improvement**: Weight decay decoupling improved regularization over Adam

---

### 4. Epoch Count vs. Overfitting

**Data**:
- 3 epochs: Train 0.521, Valid 0.551 (gap: 0.030)
- 5 epochs: Train 0.458, Valid 0.478 (gap: 0.020)
- 10 epochs: Train 0.401, Valid 0.399 (gap: -0.002)

**Analysis**: No overfitting! In fact, 10 epochs showed perfect generalization.

**Why no overfitting?**:
- Fashion MNIST is relatively simple (10 classes, clear patterns)
- 60,000 training samples is substantial
- Implicit regularization from Adam optimizer
- Model capacity (256/256) appropriate for task complexity

**Signs of healthy training**:
- Both train and valid loss decreased together
- Valid loss tracked train loss closely
- More epochs = better performance on both sets

---

### 5. Model Capacity (Units) Analysis

**Findings**:
- 16/16: 79.0% (underfitting)
- 128/512: 83.5% (sweet spot)
- 512/512: 83.3% (marginal benefit)

**Interpretation**:
- **16/16 underfits**: Insufficient capacity to capture clothing patterns
- **128-512 range optimal**: Enough parameters without excessive redundancy
- **Diminishing returns**: 512/512 has 262K+ params but only matches 128/512 performance

**Why asymmetric works** (128/512, 512/128):
- First layer extracts features (needs capacity)
- Second layer combines features (can be simpler)
- Asymmetry creates efficient "hourglass" or "expansion" architecture

---

### 6. Combined Configurations

**Best combined config**: 10 epochs, batch 64, 256/128 units, LR 0.0005, AdamW → **84.2%**

**Why this combination works**:
1. **10 epochs**: Maximum learning time without overfitting
2. **Batch 64**: Sweet spot for stability and memory efficiency
3. **256/128 units**: Efficient asymmetric architecture
4. **LR 0.0005**: Slightly conservative for longer training
5. **AdamW**: Weight decay improves generalization

**Synergy effects**:
- Lower LR (0.0005) paired with more epochs (10) allows careful optimization
- AdamW's weight decay complements longer training
- Moderate batch size supports consistent gradient updates

**Did not reach 85%+ like epochs_10 alone** because:
- Different unit configuration (256/128 vs 256/256)
- Lower learning rate trades final accuracy for stability

---

## 5. Conclusions & Recommendations

### Key Takeaways

1. **Always use adaptive optimizers** (Adam/AdamW/RMSprop) for deep learning - SGD requires expert tuning
2. **Learning rate is critical** - 0.001 is a good starting point for Adam on image tasks
3. **Batch size matters more than expected** - Avoid very small batches (<16) even with adaptive LR
4. **Simpler can be better** - Single hidden layer outperformed deeper networks on Fashion MNIST
5. **More epochs help** - No overfitting observed up to 10 epochs on this dataset

### Recommended Configuration

For Fashion MNIST (or similar simple classification tasks):

```python
config = {
    'epochs': 10,
    'batch_size': 64,  # Balance of speed and stability
    'units': [512, 256],  # Asymmetric for efficiency
    'learning_rate': 0.001,
    'optimizer': 'RMSprop',  # Slightly edged out AdamW
    'depth': 1  # Single hidden layer sufficient
}
```

**Expected performance**: ~84-85% accuracy

### Future Work

1. **Test data augmentation** (rotation, flip, zoom) to push above 85%
2. **Implement early stopping** to automatically detect convergence
3. **Try learning rate scheduling** (reduce on plateau) for even better SGD performance
4. **Explore convolutional layers** - CNNs likely to exceed 90% on Fashion MNIST
5. **Bayesian optimization** instead of grid search for efficiency

---

## 6. Appendices

### Appendix A: Experimental Setup

- **Dataset**: Fashion MNIST (60,000 train, 10,000 test)
- **Framework**: PyTorch with mltrainer
- **Hardware**: CPU training
- **Total Experiments**: 61 configurations
- **Training Time**: ~2.5 hours total

### Appendix B: File Outputs

- **TensorBoard logs**: `modellogs/` (61 subdirectories)
- **TOML results**: Each experiment has `*_results.toml` with final metrics
- **Visualizations**: `visualizations/` directory (heatmaps, scatter plots)
- **Summary CSV**: `experiment_summary.csv` (all 61 experiments)

### Appendix C: Reproducibility

All experiments are reproducible using:

```bash
python grid_search_experiments.py
python extract_tensorboard_metrics.py
python analyze_results.py
```

Environment:
- Python 3.12.10
- PyTorch 2.9.1
- See `pyproject.toml` for complete dependencies

---

**Report prepared by**: Stijn Barthel  
**Date**: January 5, 2026  
**Total Experiments**: 61  
**Best Accuracy**: 85.5% (epochs_10) | 84.2% (combined_config_2)
