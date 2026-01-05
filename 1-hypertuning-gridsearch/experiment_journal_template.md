# Experiment Journal Template
*Following the scientific method: hypothesis → experiment → analysis → conclusion*

---

## Date: [Current Date]

### Session Goals
- [ ] Define initial hypotheses
- [ ] Run baseline experiments
- [ ] Analyze preliminary results
- [ ] Form new hypotheses based on findings

---

## Hypothesis Formation

### Hypothesis 1: [Topic]
**Prediction**: 
[What do you expect to happen?]

**Reasoning**: 
[Why do you expect this? What theory supports it?]

**How to test**: 
[What experiment will validate/invalidate this?]

---

### Hypothesis 2: [Topic]
**Prediction**: 
[What do you expect to happen?]

**Reasoning**: 
[Why do you expect this? What theory supports it?]

**How to test**: 
[What experiment will validate/invalidate this?]

---

## Experiment Design

### Experiment [Name/Number]
**Date/Time**: 
**Goal**: 
**Configuration**:
- Parameter 1: [value]
- Parameter 2: [value]
- Control variables: [list]

**Expected outcome**: 
**How to measure success**: 

---

## Observations During Training

### [Timestamp]
**What I'm seeing**:
[Real-time observations from TensorBoard or console output]

**Interesting patterns**:
[Anything unexpected or noteworthy]

**Questions raised**:
[New questions that emerge from observations]

---

## Results Analysis

### Experiment [Name/Number] Results
**Date completed**: 
**Final metrics**:
- Training accuracy: 
- Validation accuracy: 
- Loss: 
- Convergence speed: 

**Observations**:
[What patterns do you see? How do results compare to expectations?]

**Visualization notes**:
[What do the plots tell you? Any interesting curves or patterns?]

---

## Reflection & Insights

### What worked well?
[Successful strategies or configurations]

### What didn't work?
[Failed approaches and why they failed]

### Surprising findings
[Results that contradicted expectations]

### Connection to theory
[How do results relate to ML theory? Bias-variance tradeoff, optimization landscape, etc.]

---

## Conclusions

### Hypothesis validation
- Hypothesis 1: ✅ Confirmed / ❌ Rejected / ⚠️ Partial
  - Evidence: [what supports this conclusion?]
  
- Hypothesis 2: ✅ Confirmed / ❌ Rejected / ⚠️ Partial
  - Evidence: [what supports this conclusion?]

### Key learnings
1. [Learning 1]
2. [Learning 2]
3. [Learning 3]

### New questions
[What new questions emerged from this work?]

---

## Next Steps

### Immediate next experiments
- [ ] [Next experiment to run]
- [ ] [Parameter to investigate]
- [ ] [Hypothesis to test]

### Longer-term ideas
- [Ideas for future exploration]

---

## Notes & Reminders

[Any additional thoughts, reminders, or ideas]

---

## Example Entry (for reference)

### Hypothesis: Larger batch sizes will train faster but generalize worse

**Prediction**: Batch size 128 will converge quickly but have lower validation accuracy than batch size 32.

**Reasoning**: Larger batches provide more stable gradients (faster convergence) but may find sharper minima (worse generalization).

### Experiment: Compare batch sizes 32, 64, 128
**Configuration**: All other params constant (epochs=5, units=256/256, lr=0.001, Adam)

### Observations:
- Batch 128: Loss decreases very smoothly, almost monotonic
- Batch 32: More noise in loss curve, but interesting final validation accuracy
- Training time per epoch: 32 > 64 > 128 (expected)

### Results:
- Batch 32: Val accuracy = 88.5%
- Batch 64: Val accuracy = 88.2%  
- Batch 128: Val accuracy = 87.1%

### Conclusion:
✅ Hypothesis CONFIRMED! Larger batches do train faster but achieve slightly worse validation accuracy. The "sweet spot" appears to be batch=64 for this problem.

### Insight:
This demonstrates the classic exploration vs exploitation tradeoff in optimization. Smaller batches provide noisier gradients that help escape local minima.

### Next: Test if this holds across different learning rates...

---

*Journal Tip: Don't overthink it! Quick notes during experimentation are valuable. You can always refine later for the final report.*
