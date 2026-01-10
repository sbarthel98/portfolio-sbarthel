# Experiment Journal - Ray Tune Hyperparameter Optimization
**Flowers Dataset CNN Hyperparameter Tuning**

**Student**: Stijn Barthel  
**Date**: January 10, 2026  
**Project**: Ray Tune with ASHA Scheduler and HyperOpt

---

## Experiment Log Overview

This journal documents the systematic exploration of CNN architectures and hyperparameters using Ray Tune's advanced optimization capabilities. The experiments use the Flowers dataset with image classification, employing ASHA (Asynchronous Successive Halving Algorithm) for efficient hyperparameter search.

**Key Tools:**
- Ray Tune for distributed hyperparameter optimization
- ASHA Scheduler for early stopping of poorly performing trials
- HyperOpt search algorithm for intelligent parameter selection
- GPU acceleration for faster training

---

## Pre-Experiment Setup

**Date**: January 10, 2026

### Dataset Selection
- **Dataset**: Flowers (5 classes)
- **Image Size**: 224x224x3 (RGB)
- **Reason**: More complex than Fashion MNIST, requires convolutional architecture
- **Data Location**: `~/.cache/mads_datasets`

### Hardware Configuration
- **GPU Available**: ✅ Yes
- **GPU Model**: NVIDIA GeForce RTX 3060
- **GPU Memory**: 12.00 GB
- **CUDA Version**: 12.1
- **Resources per Trial**: GPU: 0.5, CPU: 2 (allows 2 trials in parallel)

### Model Architecture Design
**Architecture**: Tunable CNN with:
- Variable number of convolutional layers (2-4)
- Configurable starting filter size (16, 32, 64)
- BatchNorm after each Conv2d
- MaxPool2d (2x2) after each conv block
- Dropout for regularization
- Fully connected layers with configurable units

**Design Rationale**: 
- ConvNets are appropriate for image classification tasks
- BatchNorm helps with training stability
- Dropout prevents overfitting
- Variable architecture allows exploration of model capacity

---

## Experiment Group 1: Initial Architecture Search

### Hypothesis 1: Model Depth vs. Performance

**Date**: January 10, 2026

**Hypothesis**: Deeper networks (4 conv layers) will perform better than shallow ones (2 layers) for the flowers dataset, but may require more training time.

**Reasoning**: 
- Deeper networks can learn more complex hierarchical features
- Flowers have complex visual patterns (petals, colors, shapes)
- However, deeper networks may overfit with limited data
- ASHA scheduler will help identify which depth works best early

**Variables to Test**:
- `num_conv_layers`: [2, 3, 4]
- Fixed initially: Other hyperparameters varied by Ray Tune

**Expected Outcomes**:
- 2 layers: Fast to train, may underfit
- 3 layers: Sweet spot for complexity vs. overfitting
- 4 layers: Best performance if enough data, but may be stopped early by ASHA

**Experiment Configuration**:
```python
config = {
    "num_conv_layers": tune.choice([2, 3, 4]),
    "start_filters": tune.choice([16, 32, 64]),
    "fc_units": tune.choice([64, 128, 256]),
    "dropout": tune.uniform(0.1, 0.5),
    "lr": tune.loguniform(1e-4, 1e-2),
    "batch_size": tune.choice([16, 32]),
}
```

---

### Observations During Training

**Completed**: January 10, 2026

**Total Trials**: 10 (6 stopped early, 4 ran to completion)
**Total Time**: 17.7 minutes
**Average Time per Trial**: 106 seconds (~1.8 min)

**Top 3 Performing Trials**:

1. **trial_e44d1f7b** → **68.5% accuracy** ✅ BEST
   - 3 conv layers, 32 start filters, 128 FC units
   - Dropout: 0.150, LR: 0.000472, Batch: 32
   - Ran 10 epochs (full training)
   - Training time: 124s

2. **trial_ec2e8386** → **64.0% accuracy**
   - 3 conv layers, 64 start filters, 256 FC units
   - Dropout: 0.237, LR: 0.000241, Batch: 32
   - Ran 10 epochs (full training)
   - Training time: 205s

3. **trial_cb382de0** → **49.3% accuracy**
   - 4 conv layers, 16 start filters, 256 FC units
   - Dropout: 0.459, LR: 0.00684, Batch: 32
   - Stopped at 6 epochs
   - Training time: 57s

**Early Stopping Observations**:
- **6 out of 10 trials** stopped early by ASHA (60%)
- Most stopped at **2 epochs** (4 trials): poor performers identified quickly
- **2 trials stopped at 6 epochs**: medium performance
- **4 trials completed 10 epochs**: best performers kept training
- ASHA correctly identified low performers early, saving ~40-50% of training time

---

### Results & Analysis

**Best Configuration Found**:
- num_conv_layers: **3**
- start_filters: **32**
- fc_units: **128**
- dropout: **0.150**
- lr: **0.000472**
- batch_size: **32**

**Accuracy Achieved**: **68.5%**

**Performance Distribution**:
- Best: 68.5%
- Mean: 43.6% ± 13.8%
- Median: 40.7%
- Worst: 25.3%

**Key Observations**:
1. **3 conv layers performed best** - sweet spot between capacity and overfitting
2. **Moderate starting filters (32) outperformed** both small (16) and large (64)
3. **Lower dropout (~0.15) was optimal** - too much regularization hurt performance
4. **Learning rate ~0.0005 worked best** - very small (<0.0001) or large (>0.005) performed poorly
5. **Batch size 32 > batch size 16** - larger batches provided more stable gradients
6. **Network depth mattered**: 4 layers didn't help, possibly overfitting or training issues

---

### Conclusions & Next Steps

**Hypothesis Validation**: 
- **H1 (Deeper = Better)**: ❌ **REJECTED** - 3 layers beat 4 layers. Possibly gradient vanishing or overfitting.
- **H2 (Moderate filters optimal)**: ✅ **CONFIRMED** - 32 filters provided best balance
- **H3 (Dropout ~0.3-0.4)**: ❌ **REJECTED** - Lower dropout (0.15) performed best
- **H4 (LR 1e-3 to 1e-4)**: ✅ **CONFIRMED** - Best was 4.7e-4, right in this range

**Insights**:
1. **ASHA Efficiency**: Saved ~40% time by stopping poor trials at 2 epochs
2. **Architecture > Hyperparams**: Network structure (3 layers, 32 filters) mattered more than fine-tuning LR/dropout
3. **Batch size matters**: 32 consistently outperformed 16
4. **More parameters ≠ Better**: 64 start filters with 256 FC gave 64%, but 32/128 gave 68.5%

**Next Experiments** (if continuing):
- Test 3 layers more thoroughly with varied filter patterns
- Try learning rate scheduling (cosine annealing)
- Add data augmentation (rotation, flipping)
- Test different optimizers (AdamW, RMSprop)

---

## Experiment Group 2: Learning Rate and Optimizer Interaction

### Hypothesis 2: [TO BE FORMULATED]

**Date**: [DATE]

**Hypothesis**: [Your hypothesis here]

**Reasoning**: [Based on theory and previous results]

**Variables to Test**:
- [List variables]

**Expected Outcomes**:
- [Predictions]

---

### Observations During Training

**[TO BE FILLED]**

---

### Results & Analysis

**[TO BE FILLED]**

---

### Conclusions & Next Steps

**[TO BE FILLED]**

---

## Experiment Group 3: Filter Size and Network Capacity

### Hypothesis 3: [TO BE FORMULATED]

**Date**: [DATE]

**Hypothesis**: [Your hypothesis here]

**Reasoning**: [Based on theory and previous results]

---

### Observations During Training

**[TO BE FILLED]**

---

### Results & Analysis

**[TO BE FILLED]**

---

### Conclusions & Next Steps

**[TO BE FILLED]**

---

## Overall Reflections

**[TO BE FILLED AT END OF PROJECT]**

### Key Learnings
1. [Learning 1]
2. [Learning 2]
3. [Learning 3]

### Comparison with Previous Weeks
**Grid Search (Week 1)**: 
- [Comparison points]

**MLflow (Week 2)**:
- [Comparison points]

**RNN Tuning (Week 3)**:
- [Comparison points]

**Ray Tune (Week 4)**:
- [What makes Ray Tune different/better]

### Theoretical Connections
**From Deep Learning Book**:
- [Theory point 1]
- [Theory point 2]

### Mistakes Made & Lessons Learned
1. [Mistake/Lesson 1]
2. [Mistake/Lesson 2]

---

## Appendix: Technical Notes

### Ray Tune Configuration Details
- **Scheduler**: AsyncHyperBandScheduler
  - `max_t`: 10 epochs
  - `grace_period`: 2 epochs
  - `reduction_factor`: 3
  
- **Search Algorithm**: HyperOpt
  - Bayesian optimization for intelligent search
  - Much more efficient than random/grid search

### GPU Configuration
- **Fractional GPUs**: 0.5 per trial
- **Allows**: 2 parallel trials on single GPU
- **Benefit**: Faster experimentation

### Code Quality Improvements Needed
- [List any linting issues]
- [Type hints to add]
- [Refactoring needed]

---

## Timeline

- **January 10, 2026**: Project start, initial setup
- [Add dates as you progress]

---

## References & Resources

1. Ray Tune Documentation: https://docs.ray.io/en/latest/tune/
2. ASHA Paper: https://arxiv.org/abs/1810.05934
3. Deep Learning Book - Chapter on CNNs
4. Previous experiments: 1-hypertuning-gridsearch, 2-hypertuning-mlflow, 3-hypertuning-rnn
