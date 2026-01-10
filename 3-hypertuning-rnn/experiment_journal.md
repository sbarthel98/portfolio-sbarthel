# Experiment Journal - RNN Hyperparameter Tuning
**Gesture Recognition with Recurrent Neural Networks**

**Student:** Stijn Barthel  
**Date:** January 10, 2026  
**Project:** Hypertuning RNN for SmartWatch Gestures  
**Framework:** PyTorch + MLflow

---

## Experiment Log Overview

This journal documents the systematic exploration of RNN architectures for gesture recognition. The goal was to achieve >90% accuracy using the SmartWatch Gestures dataset (20 classes).

### Phase 1: Baseline Establishment (GRU Hidden Sizes)
*Objective: Determine optimal model capacity*

| Experiment ID | Configuration | Parameters | Val Accuracy | Notes |
|---|---|---|---|---|
| `baseline_gru_h64` | GRU (64 units, 1 layer) | 14,548 | 87.03% | Underfitting, too simple |
| `baseline_gru_h128` | GRU (128 units, 1 layer) | 53,652 | 96.72% | Excellent performance, efficient |
| `baseline_gru_h256` | GRU (256 units, 1 layer) | 205,588 | 99.22% | Near perfect, slightly slower training |

**Finding:** Increasing hidden size from 64 to 128 yielded massive gains (+9.7%). 256 units provided diminishing returns but highest accuracy.

### Phase 2: Network Depth
*Objective: Test if deeper networks capture more complex temporal features*

| Experiment ID | Configuration | Parameters | Val Accuracy | Notes |
|---|---|---|---|---|
| `gru_depth_l1` | GRU-128 (1 layer) | 53,652 | 97.34% | Strong baseline |
| `gru_depth_l2` | GRU-128 (2 layers) | 152,724 | 99.38% | Slight improvement (+2.0%) |

**Finding:** Adding a second layer improved accuracy to near-perfect levels, suggesting hierarchical temporal features are useful but 1 layer is already very strong.

### Phase 3: Regularization (Dropout)
*Objective: Prevent overfitting in deeper networks*

| Experiment ID | Configuration | Val Accuracy | Notes |
|---|---|---|---|
| `gru_dropout_0.0` | Dropout 0.0 | 97.81% | Good, slight overfitting possible |
| `gru_dropout_0.2` | Dropout 0.2 | 98.91% | Better generalization |
| `gru_dropout_0.3` | Dropout 0.3 | 99.06% | Best balance |

**Finding:** Moderate dropout (0.2-0.3) helps generalization.

### Phase 4: Architecture Comparison
*Objective: Compare GRU vs LSTM and Advanced Architectures*

| Experiment ID | Architecture | Val Accuracy | Parameters | Notes |
|---|---|---|---|---|
| `gru_128` | Standard GRU | 99.22% | 152K | Fast, accurate |
| `lstm_128` | Standard LSTM | 96.25% | 202K | Slower, lower accuracy |
| `bidirectional_gru` | Bidirectional GRU | 99.69% | 403K | **Best Performer** |
| `conv1d_gru` | Conv1D + GRU | 99.06% | 158K | Very parameter efficient |
| `deep_gru` | GRU + LayerNorm | 99.53% | 252K | Very stable training |

**Finding:** Bidirectional GRU achieved the highest accuracy (99.69%). Conv1D+GRU was surprisingly effective, matching pure RNNs with efficient feature extraction.

---

## Key Learnings

1. **GPU Acceleration**: Switching from CPU (mltrainer) to native PyTorch GPU training reduced epoch time from ~90s to ~3s (30x speedup).
2. **Architecture Matters**: Bidirectional processing added significant value (+0.5-2%) by seeing future context.
3. **Capacity Threshold**: 64 hidden units was insufficient; 128 was the "sweet spot" for efficiency/performance.
4. **Hybrid Models**: Convolutional layers can effectively extract features from raw sensor data before the RNN processes temporal dynamics.

## Final Optimal Configuration

The best model found was the **Bidirectional GRU** with:
- Hidden Size: 128
- Layers: 2
- Dropout: 0.2
- Accuracy: **99.69%**

This far exceeds the project goal of >90% accuracy.
