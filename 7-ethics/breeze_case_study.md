# Case Study: Breeze Dating App Algorithmic Discrimination

**Date**: January 10, 2026  
**Student**: Stijn Barthel  
**Assignment**: Algorithmic Bias Analysis and Recommendations

---

## Case Context

The Dutch dating app **Breeze** suspected its matching algorithm was discriminating against users of non-Dutch origin and darker skin tones. The algorithm, optimized to maximize match success, learned from existing user behaviors and consequently showed profiles of minority users less frequently. 

In September 2023, the **College voor de Rechten van de Mens** (Dutch Human Rights Institute) ruled that Breeze not only may but **must** adjust its algorithm to prevent this indirect discrimination. The ruling clarified that correcting algorithmic bias is not preferential treatment but rather restoration of baseline fairness.

---

## 1. Initial Analysis

### Core Ethical Issue

This case represents a classic example of **automation bias** and the **proxy problem** in machine learning. The discrimination did not stem from explicitly programmed racism but from optimizing for "efficiency" (maximum match rate) using biased training data.

### Key Problem

The algorithm treated **human bias** (users disproportionately swiping left on minority profiles) as "ground truth" for desirability, thereby automating and amplifying existing social inequality. Rather than providing a neutral platform, the system actively manufactured invisibility for certain groups.

### Ethical Tension

The case involves competing principles:

**User Autonomy**: Individual users have the right to choose their dating preferences without interference.

**Non-Discrimination**: The platform has a duty not to render certain demographic groups structurally invisible through its algorithmic design.

### Initial Misconception

Breeze initially feared that adjusting the algorithm would constitute illegal "positive discrimination." The ruling corrected this misunderstanding: correcting a biased baseline is an act of **neutrality**, not favoritism. The algorithm created a structural disadvantage; removing that disadvantage restores fair treatment.

---

## 2. Directed Acyclic Graph (DAG) Analysis

### Causal Structure

The following DAG illustrates the feedback loop creating and amplifying discrimination:

```
graph TD
    A[Societal Biases & User Preferences] -->|Input| B(Historical Interaction Data)
    B -->|Training Data| C{Algorithm Optimization}
    C -->|Output| D[Match Probability Score]
    D -->|Decision| E[Suggestion Frequency]
    E -->|Feedback Loop| F[User Interactions/Swipes]
    F -->|Reinforcement| B
    
    G[Sensitive Attributes] -.->|Latent Variable| A
    G -.->|Proxy Correlation| D
    
    style E fill:#f9f,stroke:#333,stroke-width:2px
    style F fill:#f9f,stroke:#333,stroke-width:2px
```

### Node Descriptions

- **Node A**: Pre-existing societal biases in dating preferences
- **Node B**: Historical swipe data collected by platform
- **Node C**: Machine learning optimization process
- **Node D**: Algorithmic match probability scores
- **Node E**: Frequency with which profiles are shown (critical discrimination point)
- **Node F**: User interactions (swipes) generating new training data
- **Node G**: Protected attributes (ethnicity, appearance) acting as latent variables

---

## 3. Post-DAG Reflection

### Critical Insight: The Amplification Loop

Drawing the DAG revealed a dynamic system that my initial impression overlooked. The problem is not merely:

**Static Bias**: Input Bias → Output Bias

But rather:

**Dynamic Amplification**: A self-reinforcing feedback loop creating a "death spiral" of invisibility:

1. **Initial Discrimination** (Node D): Algorithm assigns lower scores to minority profiles based on biased historical data
2. **Reduced Visibility** (Node E): Lower scores result in showing these profiles to fewer users
3. **Reduced Engagement** (Node F): Being shown less frequently results in fewer absolute likes/matches
4. **Reinforced Bias** (Node B): Algorithm interprets fewer likes as validation that these profiles are "less desirable"
5. **Amplified Discrimination** (Node D): Scores decrease further in next iteration

### Key Realization

This structure means the algorithm is not just reproducing bias—it is **actively manufacturing and amplifying invisibility**. Minority users are denied the opportunity to overcome initial bias because they never receive sufficient exposure for the algorithm to learn otherwise. The system creates a structural barrier that individual user preferences alone cannot explain.

### Systemic Implications

The feedback loop demonstrates why "letting the data speak for itself" is ethically problematic in systems with:
- Historical bias in training data
- Reinforcement learning or continuous retraining
- Unequal initial conditions between groups
- Lack of exploration (only exploitation of known patterns)

---

## 4. Recommendations for Data Scientists

If assigned to address this issue, implement the following:

### 4.1 Break the Feedback Loop

**Recommendation**: Implement exploration-exploitation trade-off mechanisms.

**Specific Actions**:
- Deploy **Epsilon-Greedy** or **Thompson Sampling** algorithms that randomly boost visibility of lower-scored profiles
- Ensure all profiles receive minimum baseline exposure independent of predicted match probability
- Collect new interaction data across diverse user populations to verify whether low scores reflect true preference or just low exposure

**Rationale**: Without exploration, the algorithm cannot learn whether its predictions are accurate or self-fulfilling prophecies.

### 4.2 Define Fairness Mathematically

**Recommendation**: Implement formal fairness constraints in the optimization objective.

**Specific Metrics**:
- **Demographic Parity**: Ensure suggestion frequency (Node E) is independent of protected attributes (Node G)
- **Equalized Odds**: Ensure true positive rates (successful matches) are comparable across demographic groups
- **Calibration**: Ensure match probability scores are accurate across all groups

**Implementation**:
```python
# Constraint: Suggestion frequency must be independent of ethnicity
P(shown | ethnicity=A) ≈ P(shown | ethnicity=B)

# NOT forcing users to match, but ensuring equal opportunity to BE seen
```

**Rationale**: The College ruling requires preventing indirect discrimination. This translates to ensuring visibility is not structurally reduced for protected groups.

### 4.3 Conduct Human Rights Impact Assessment (IAMA)

**Recommendation**: Perform formal impact assessment before deployment.

**Key Components**:
1. **Necessity Justification**: Document why processing sensitive data (ethnicity) is necessary to prevent discrimination
2. **Data Minimization**: Ensure ethnicity data is used solely for fairness adjustment, not commercial profiling
3. **Technical Safeguards**: Implement data siloing between fairness correction module and commercial matching logic
4. **Transparency**: Provide users with explanation of fairness measures
5. **Monitoring**: Establish ongoing auditing of fairness metrics across demographic groups

**Rationale**: GDPR restricts processing of sensitive data. However, Article 9(2)(g) permits processing when necessary for substantial public interest (preventing discrimination). Proper IAMA documentation establishes legal justification.

### 4.4 Reframe Correction as Neutrality

**Recommendation**: Communicate internally and externally that algorithmic adjustment is restoration of fairness, not preferential treatment.

**Key Points**:
- **What it is**: Removing structural disadvantage created by the system itself
- **What it is NOT**: Forcing users to match with people they don't prefer
- **Analogy**: Similar to accessibility requirements in physical spaces—providing equal access, not special advantages

**Rationale**: The College ruling explicitly states this position. Proceeding with confidence prevents internal resistance and potential legal challenges.

### 4.5 Technical Implementation Details

**Specific Approaches**:

1. **Pre-Processing**: Re-weight training data to balance exposure across groups
2. **In-Processing**: Add fairness constraints to loss function during model training
3. **Post-Processing**: Adjust prediction thresholds to equalize outcomes across groups

**Recommended Approach**: Combination of (1) and (2):
- Re-weight historical data to simulate equal exposure baseline
- Add soft fairness constraint to optimization: `loss = match_accuracy_loss + λ * fairness_penalty`

### 4.6 Monitoring and Auditing

**Ongoing Requirements**:
- Track suggestion frequency distributions across demographic groups weekly
- Monitor match success rates across groups monthly
- Conduct annual fairness audits with external review
- Establish feedback mechanism for users to report perceived bias
- Document all findings and adjustments for regulatory compliance

---

## 5. Ethical Framework Application

### Using Markkula Framework

| Lens | Analysis | Recommendation |
|------|----------|----------------|
| **Rights** | Minority users' right to equal platform access violated | Restore equal visibility |
| **Justice** | Distributive injustice in exposure allocation | Implement demographic parity |
| **Utilitarian** | Overall utility decreased (minorities excluded, majority choices limited) | Increase diversity of exposure |
| **Common Good** | Platform reinforcing societal discrimination harms social cohesion | Break amplification loop |
| **Virtue** | Platform fails duty of care and fairness | Demonstrate ethical leadership |

### Conclusion from Framework

All five lenses converge on the same recommendation: **algorithmic adjustment is ethically required**.

---

## 6. Conclusion

The Breeze case demonstrates that algorithmic fairness is not just a technical challenge but a legal and ethical obligation. The feedback loop identified in the DAG analysis reveals why passive data collection and optimization inevitably amplify existing biases. 

**Key Takeaway**: Machine learning systems trained on biased data in reinforcement loops do not merely reflect reality—they actively construct it. Data scientists have an ethical duty to break these amplification cycles through deliberate fairness interventions.

The College ruling provides legal clarity: correcting algorithmic bias is not preferential treatment but the restoration of baseline neutrality that the system itself disrupted.

---

## References

1. College voor de Rechten van de Mens. (2023, September 6). *Dating-app Breeze mag (en moet) algoritme aanpassen om discriminatie te voorkomen*. Retrieved from https://mensenrechten.nl/

2. Barocas, S., Hardt, M., & Narayanan, A. (2019). *Fairness and Machine Learning: Limitations and Opportunities*. MIT Press.

3. European Union. (2016). *General Data Protection Regulation (GDPR)*. Article 9: Processing of special categories of personal data.

4. Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A survey on bias and fairness in machine learning. *ACM Computing Surveys*, 54(6), 1-35.

---

[Back to Ethics Summary](./summary.md) | [Back to Portfolio](../README.md)
