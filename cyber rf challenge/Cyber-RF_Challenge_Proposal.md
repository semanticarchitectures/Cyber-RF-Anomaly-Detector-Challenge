# Cyber-RF Anomaly Detector Challenge - Comprehensive Proposal

## Executive Summary

This proposal outlines a multi-phase approach to develop a high-performance machine learning system for detecting anomalous Zigbee transmissions (replay attacks and rogue transmitters) using both IQ signal features and network traffic characteristics.

**Key Challenge Characteristics:**
- Binary classification: Legitimate vs. Anomalous Zigbee transmissions
- 730 training samples (52.2% anomalous, 47.8% legitimate)
- 52 features (43 IQ-based, 9 network traffic-based)
- Power-balanced dataset (prevents simple energy detection)
- Two attack types: Replay attacks and Rogue transmitters

**Success Metrics:**
- Accuracy: (TP + TN) / Total
- Recall: TP / (TP + FN) - Critical for detecting attacks
- Precision: TP / (TP + FP) - Minimize false alarms
- FPR: FP / (FP + TN) - Keep false positives low

---

## Phase 1: Data Understanding & Exploratory Analysis

### 1.1 Dataset Characteristics

**IQ-Based Features (43 features):**
- Amplitude measurements (7 features): min, max, variance, skewness, range, kurtosis, entropy
- Phase measurements (4 features): min, variance, skewness, entropy
- RMS measurements (6 features): max, skewness, range, kurtosis, entropy
- Signal Power measurements (7 features): min, max, variance, skewness, range, kurtosis, entropy, std error
- FFT measurements (10 features): min, max, avg, median, variance, skewness, range, kurtosis, entropy, std error
- Periodogram measurements (7 features): max, avg, variance, skewness, range, kurtosis, entropy

**Network Traffic Features (9 features):**
- Duration_Avg: Average communication duration
- SumNoPktsSent: Total packets transmitted
- numPktSent_avg: Average packets per flow
- NoBytesSnt_avg: Average bytes per flow
- minPktSize_min: Minimum packet size
- maxPktSize_max: Maximum packet size
- avgPktSz_avg: Average packet size
- pktps_avg: Packets per second
- bytps_avg: Bytes per second
- maxIAT_max: Maximum inter-arrival time
- avgIAT_avg: Average inter-arrival time

### 1.2 Critical Analysis Questions

**Before proceeding, we need to address:**

1. **Test Set Availability:** Is there a separate test set (remaining 20%)? If not, we need a robust validation strategy.

2. **Attack Type Labels:** Are the anomalous samples labeled by attack type (replay vs. rogue)? This could inform:
   - Multi-class classification approach
   - Ensemble strategies targeting specific attack patterns
   - Feature importance per attack type

3. **Operational Requirements:**
   - What's the acceptable false positive rate for operational deployment?
   - Is real-time detection required? (impacts model complexity)
   - What's the priority: maximizing recall (catching all attacks) or precision (minimizing false alarms)?

4. **Feature Engineering Constraints:**
   - Are we allowed to create new features from the raw IQ/network data?
   - Or must we work only with the provided 52 features?

5. **Packet Proportion Information:** The PDF mentions different malicious/legitimate packet proportions (50/50, 60/40, 70/30, 80/20) for rogue transmitters. Is this information available as metadata?

---

## Phase 2: Data Preprocessing & Feature Engineering

### 2.1 Data Quality Assessment

```python
# Comprehensive checks:
1. Verify no missing values
2. Detect outliers using IQR and Z-score methods
3. Check for duplicate samples
4. Analyze feature distributions (normal, skewed, bimodal)
5. Identify features with near-zero variance
6. Check for multicollinearity (VIF analysis)
```

### 2.2 Feature Scaling Strategy

Given the diverse feature ranges (signal power, entropy, packet counts), we'll implement:

**Approach A: StandardScaler (Z-score normalization)**
- Good for features with Gaussian-like distributions
- Preserves relative distances

**Approach B: RobustScaler**
- Better for features with outliers
- Uses median and IQR instead of mean and std

**Approach C: MinMaxScaler**
- For tree-based models (optional, less critical)

**Decision:** Use StandardScaler as baseline, compare with RobustScaler if outliers are significant.

### 2.3 Feature Engineering Opportunities

**Derived Features:**
1. **IQ Feature Ratios:**
   - Amplitude/RMS ratios
   - Signal Power to FFT energy ratios
   - Phase entropy to amplitude entropy ratios

2. **Network Traffic Patterns:**
   - Packet size variance (max - min)
   - Throughput efficiency (bytps_avg / pktps_avg)
   - IAT coefficient of variation

3. **Cross-Domain Features:**
   - Correlation between signal quality (SP_entropy) and network performance (pktps_avg)
   - Anomaly scores from isolation on IQ features vs. network features

4. **Statistical Moments:**
   - Combine skewness and kurtosis into shape descriptors
   - Create "abnormality scores" for statistical outliers

### 2.4 Feature Selection

**Multi-Method Approach:**

1. **Filter Methods:**
   - Mutual Information scores
   - Chi-square test for independence
   - ANOVA F-value

2. **Wrapper Methods:**
   - Recursive Feature Elimination (RFE)
   - Forward/Backward selection

3. **Embedded Methods:**
   - L1 regularization (Lasso)
   - Tree-based feature importance (Random Forest, XGBoost)

4. **Domain-Driven Selection:**
   - Consult RF/SDR experts on which IQ features are most discriminative
   - Analyze which network features differ between replay vs. rogue attacks

---

## Phase 3: Model Development Strategy

### 3.1 Baseline Models

**Purpose:** Establish performance floor and understand data complexity

1. **Logistic Regression** (L2 regularized)
   - Interpretable coefficients
   - Fast training
   - Good baseline for linear separability

2. **Decision Tree** (max_depth=5)
   - Interpretable rules
   - Captures non-linear patterns
   - Identifies important feature interactions

3. **Naive Bayes** (Gaussian)
   - Fast, probabilistic
   - Good for feature independence assumption testing

**Expected Performance:** 70-80% accuracy baseline

### 3.2 Advanced Models

**Ensemble Methods (Primary Focus):**

1. **Random Forest**
   ```python
   Strengths:
   - Robust to overfitting
   - Handles feature interactions
   - Provides feature importance
   - Good with imbalanced data
   
   Hyperparameters to tune:
   - n_estimators: [100, 200, 500]
   - max_depth: [10, 20, 30, None]
   - min_samples_split: [2, 5, 10]
   - min_samples_leaf: [1, 2, 4]
   - max_features: ['sqrt', 'log2', None]
   ```

2. **XGBoost**
   ```python
   Strengths:
   - State-of-the-art gradient boosting
   - Handles class imbalance well
   - Built-in regularization
   - Fast training with GPU
   
   Hyperparameters to tune:
   - n_estimators: [100, 200, 500]
   - max_depth: [3, 5, 7, 9]
   - learning_rate: [0.01, 0.05, 0.1, 0.3]
   - subsample: [0.6, 0.8, 1.0]
   - colsample_bytree: [0.6, 0.8, 1.0]
   - gamma: [0, 0.1, 0.3]
   - scale_pos_weight: [balance ratio]
   ```

3. **LightGBM**
   ```python
   Strengths:
   - Faster than XGBoost
   - Better with categorical features
   - Leaf-wise tree growth
   
   Hyperparameters to tune:
   - Similar to XGBoost
   - num_leaves: [31, 50, 100]
   - min_child_samples: [20, 50, 100]
   ```

4. **CatBoost**
   ```python
   Strengths:
   - Handles categorical features natively
   - Built-in overfitting detection
   - Symmetric tree structure
   
   Good for if we encode attack types or experimental parameters
   ```

**Neural Network Approaches:**

5. **Multi-Layer Perceptron (MLP)**
   ```python
   Architecture:
   Input (52) -> Dense(128, ReLU) -> Dropout(0.3) -> 
   Dense(64, ReLU) -> Dropout(0.3) -> 
   Dense(32, ReLU) -> Dense(1, Sigmoid)
   
   Training:
   - Optimizer: Adam
   - Loss: Binary crossentropy
   - Batch size: 32
   - Epochs: 100 with early stopping
   ```

6. **Autoencoder + Classifier**
   ```python
   Two-stage approach:
   Stage 1: Unsupervised feature learning
   - Autoencoder to learn compressed representation
   - Train on both legitimate and anomalous
   
   Stage 2: Classification on learned features
   - Use bottleneck layer as features
   - Train classifier on compressed representation
   ```

**Distance-Based Methods:**

7. **Support Vector Machine (SVM)**
   - RBF kernel for non-linear boundaries
   - Tune C and gamma parameters
   - Good for high-dimensional data

8. **k-Nearest Neighbors (k-NN)**
   - Simple but effective
   - Tune k parameter
   - Use weighted voting

### 3.3 Specialized Approaches

**One-Class Classification (if applicable):**
- If we can assume "normal" is well-defined
- One-Class SVM
- Isolation Forest
- Local Outlier Factor (LOF)

**Multi-Modal Ensemble:**
```python
Strategy: Separate models for IQ and network features

Model A: Random Forest on IQ features (43)
Model B: XGBoost on network features (9)
Model C: Combined model on all features (52)

Ensemble: Weighted voting or stacking
- If IQ patterns are strong → higher weight to Model A
- If network patterns are strong → higher weight to Model B
- Model C provides holistic view
```

---

## Phase 4: Training & Validation Strategy

### 4.1 Cross-Validation Approach

Given 730 samples, we need careful validation:

**Stratified K-Fold Cross-Validation (k=5 or k=10)**
```python
Rationale:
- Maintains class distribution in each fold
- Provides robust performance estimates
- 5-fold: 146 test samples per fold (20% of data)
- 10-fold: 73 test samples per fold (10% of data)

Recommendation: Use 5-fold for faster iteration, 10-fold for final evaluation
```

**Nested Cross-Validation (for hyperparameter tuning)**
```python
Outer loop: 5-fold CV for performance estimation
Inner loop: 3-fold CV for hyperparameter selection

Prevents optimistic bias from using same data for tuning and evaluation
```

### 4.2 Handling Class Imbalance

While relatively balanced (52% vs 48%), we should still consider:

1. **Stratified Sampling:** Maintain proportion in train/test splits

2. **Class Weights:** For tree-based models
   ```python
   class_weight = {0: 1.0, 1: 349/381}  # Inverse proportion
   ```

3. **SMOTE (Synthetic Minority Over-sampling):**
   - Only if needed after baseline results
   - Apply only to training folds, not validation

4. **Threshold Tuning:**
   - Default threshold: 0.5
   - Optimize threshold based on cost matrix or F1-score

### 4.3 Hyperparameter Optimization

**Grid Search** (exhaustive but slow):
- Use for smaller hyperparameter spaces
- Guarantees optimal combination within grid

**Randomized Search** (faster):
- Sample from parameter distributions
- Good for initial exploration
- Typical iterations: 50-100

**Bayesian Optimization** (most efficient):
- Use Optuna or scikit-optimize
- Builds surrogate model of objective function
- Intelligently explores parameter space
- Recommended for XGBoost/LightGBM

**Hyperband** (adaptive):
- Early stopping for poorly performing configurations
- Efficient resource allocation

### 4.4 Evaluation Metrics

**Primary Metrics:**
1. **Recall (Sensitivity):** Most critical - we must catch attacks
2. **Precision:** Important - minimize false alarms
3. **F1-Score:** Harmonic mean of precision and recall
4. **FPR:** False Positive Rate - operational constraint

**Secondary Metrics:**
5. **AUC-ROC:** Overall discriminative ability
6. **AUC-PR:** Better for imbalanced scenarios
7. **Confusion Matrix:** Detailed error analysis

**Cost-Based Evaluation:**
```python
# Define operational costs
cost_matrix = {
    'FN': 100,  # Missing an attack is very costly
    'FP': 1,    # False alarm is annoying but acceptable
    'TP': 0,    # Correct detection
    'TN': 0     # Correct normal classification
}

total_cost = (FN * 100) + (FP * 1)
```

---

## Phase 5: Model Interpretation & Analysis

### 5.1 Feature Importance Analysis

**For Tree-Based Models:**
1. **Gini Importance:** Built-in feature importance
2. **Permutation Importance:** More reliable, tests actual impact
3. **SHAP Values:** 
   - Explains individual predictions
   - Shows feature contribution to each decision
   - Identifies feature interactions

**Expected Insights:**
- Which IQ features best distinguish attacks?
- Are network traffic features redundant given IQ features?
- Do replay attacks and rogue transmitters have different signatures?

### 5.2 Error Analysis

**Misclassification Investigation:**
```python
Questions to answer:
1. What characteristics do false positives share?
   - Are they legitimate signals at unusual frequencies?
   - Do they have borderline statistical properties?

2. What characteristics do false negatives share?
   - Are replay attacks more subtle than rogue transmitters?
   - Do certain packet proportions evade detection?

3. Are errors clustered by experimental conditions?
   - Specific center frequencies (2.47, 2.48, 2.49 GHz)?
   - Specific Tx gain settings (20, 25, 30)?
```

### 5.3 Decision Boundary Visualization

**Dimensionality Reduction:**
- PCA: Linear projection to 2D/3D
- t-SNE: Non-linear, preserves local structure
- UMAP: Faster than t-SNE, preserves global structure

**Visualizations:**
- Plot legitimate vs. anomalous in reduced space
- Overlay misclassified samples
- Identify potential decision boundary issues

---

## Phase 6: Ensemble & Stacking Strategy

### 6.1 Simple Ensemble (Voting)

**Hard Voting:**
```python
Ensemble = {Random Forest, XGBoost, LightGBM}
Prediction = majority vote of 3 models
```

**Soft Voting:**
```python
Ensemble = {Random Forest, XGBoost, LightGBM}
Prediction = average of probability outputs
Threshold = 0.5 (or optimized)
```

### 6.2 Stacked Ensemble

**Level 0 Models (Base Learners):**
1. Random Forest
2. XGBoost
3. LightGBM
4. Neural Network
5. SVM

**Level 1 Model (Meta-Learner):**
- Logistic Regression (simple, interpretable)
- Or another XGBoost (more complex)

**Training Process:**
```python
1. Train base learners on training data
2. Generate out-of-fold predictions using CV
3. Use OOF predictions as features for meta-learner
4. Train meta-learner on OOF features
5. For test set: base learner predictions → meta-learner
```

### 6.3 Specialized Ensemble

**Attack-Type Specific Models:**
```python
IF we have attack type labels:
  Model 1: Optimized for replay attack detection
  Model 2: Optimized for rogue transmitter detection
  Gating Network: Decides which model to trust more
```

**Feature-Group Specific Models:**
```python
Model 1: IQ features only (43 features)
Model 2: Network features only (9 features)
Model 3: All features (52 features)
Ensemble: Weighted combination based on validation performance
```

---

## Phase 7: Production Considerations

### 7.1 Model Deployment Requirements

**Real-Time Constraints:**
- Inference latency: < 10ms per sample (typical requirement)
- Throughput: Process continuous 2-minute captures
- Memory footprint: Embedded systems may have constraints

**Model Selection for Deployment:**
1. **Random Forest:** Fast inference, parallel predictions
2. **XGBoost:** Slightly slower but better accuracy
3. **Neural Network:** GPU acceleration possible
4. **SVM:** Fast for moderate feature dimensions

### 7.2 Model Compression

If deployment requires smaller models:
1. **Pruning:** Remove less important trees/features
2. **Quantization:** Reduce numerical precision
3. **Knowledge Distillation:** Train smaller model to mimic larger ensemble

### 7.3 Monitoring & Maintenance

**Performance Monitoring:**
- Track prediction distributions over time
- Detect dataset drift (input features changing)
- Monitor false positive/negative rates

**Model Updates:**
- Retrain periodically with new data
- Implement A/B testing for model updates
- Maintain version control for models

---

## Phase 8: Risk Mitigation & Contingency Plans

### 8.1 Potential Challenges

**Challenge 1: Overfitting (small dataset)**
```python
Mitigation strategies:
- Aggressive cross-validation
- Regularization (L1/L2, dropout, early stopping)
- Ensemble methods (reduce variance)
- Feature selection (reduce complexity)
```

**Challenge 2: Poor generalization to new attack variants**
```python
Mitigation strategies:
- Focus on robust features (not specific to current attacks)
- Anomaly detection component
- Regular model updates with new attack data
```

**Challenge 3: High false positive rate in production**
```python
Mitigation strategies:
- Optimize threshold based on operational costs
- Implement confidence scores
- Two-stage detection (fast filter + thorough analysis)
```

### 8.2 Fallback Approaches

**If supervised learning underperforms:**

1. **Semi-Supervised Learning:**
   - Use large amounts of unlabeled legitimate data
   - Self-training or pseudo-labeling

2. **Anomaly Detection:**
   - Treat legitimate as "normal" class
   - Use one-class SVM or Isolation Forest
   - Any deviation from normal = anomalous

3. **Transfer Learning:**
   - If similar datasets exist (WiFi, Bluetooth)
   - Pre-train on related data, fine-tune on Zigbee

---

## Phase 9: Implementation Timeline

### Week 1: Data Analysis & Preparation
- [ ] Load and explore training dataset
- [ ] Statistical analysis of all 52 features
- [ ] Correlation analysis and multicollinearity check
- [ ] Outlier detection and treatment strategy
- [ ] Feature scaling implementation
- [ ] Train/validation split strategy finalized

### Week 2: Baseline Models
- [ ] Implement Logistic Regression
- [ ] Implement Decision Tree
- [ ] Implement Naive Bayes
- [ ] Evaluate baseline performance (5-fold CV)
- [ ] Initial feature importance analysis
- [ ] Document baseline results

### Week 3: Feature Engineering & Selection
- [ ] Create derived features
- [ ] Mutual information analysis
- [ ] RFE with Random Forest
- [ ] Compare feature subsets
- [ ] Select optimal feature set
- [ ] Update preprocessing pipeline

### Week 4: Advanced Models - Part 1
- [ ] Implement Random Forest with hyperparameter tuning
- [ ] Implement XGBoost with Bayesian optimization
- [ ] Implement LightGBM
- [ ] Compare tree-based models
- [ ] SHAP analysis for best model
- [ ] Error analysis and visualization

### Week 5: Advanced Models - Part 2
- [ ] Implement Neural Network (MLP)
- [ ] Implement SVM with RBF kernel
- [ ] Implement k-NN
- [ ] Compare all models
- [ ] Select top 3-5 models for ensemble

### Week 6: Ensemble Development
- [ ] Implement voting ensemble
- [ ] Implement stacked ensemble
- [ ] Hyperparameter tuning for ensemble
- [ ] Nested CV evaluation
- [ ] Final model selection

### Week 7: Optimization & Validation
- [ ] Threshold optimization
- [ ] Cost-sensitive learning
- [ ] Final cross-validation runs (10-fold)
- [ ] Bootstrap confidence intervals
- [ ] Performance on test set (if available)

### Week 8: Documentation & Deployment Prep
- [ ] Comprehensive results documentation
- [ ] Model interpretation report
- [ ] Deployment guide
- [ ] Inference pipeline implementation
- [ ] Performance benchmarks
- [ ] Final presentation preparation

---

## Expected Outcomes

### Performance Targets

**Conservative Estimates (Baseline):**
- Accuracy: 85-90%
- Recall: 85-90%
- Precision: 85-90%
- F1-Score: 85-90%

**Optimistic Estimates (With Optimization):**
- Accuracy: 92-96%
- Recall: 92-96%
- Precision: 90-95%
- F1-Score: 92-95%

**Stretch Goals:**
- Accuracy: >97%
- Recall: >96%
- Precision: >95%
- F1-Score: >96%

### Deliverables

1. **Trained Models:**
   - Best single model (likely XGBoost or Random Forest)
   - Best ensemble model
   - Lightweight model for deployment

2. **Code Repository:**
   - Data preprocessing scripts
   - Feature engineering pipeline
   - Model training scripts
   - Evaluation scripts
   - Inference pipeline

3. **Documentation:**
   - Technical report with methodology
   - Model performance analysis
   - Feature importance analysis
   - Error analysis and insights
   - Deployment guide

4. **Visualizations:**
   - Feature distributions
   - Correlation heatmaps
   - ROC and PR curves
   - Confusion matrices
   - SHAP summary plots
   - Decision boundary visualizations

---

## Tools & Technology Stack

### Core ML Frameworks
- **scikit-learn:** Preprocessing, baseline models, evaluation
- **XGBoost:** Gradient boosting
- **LightGBM:** Fast gradient boosting
- **CatBoost:** Categorical boosting (if needed)

### Deep Learning
- **TensorFlow/Keras** or **PyTorch:** Neural networks

### Hyperparameter Optimization
- **Optuna:** Bayesian optimization
- **scikit-optimize:** Gaussian processes

### Model Interpretation
- **SHAP:** Feature importance and interpretability
- **LIME:** Local interpretability (if needed)

### Visualization
- **matplotlib:** Basic plots
- **seaborn:** Statistical visualizations
- **plotly:** Interactive visualizations

### Data Processing
- **pandas:** Data manipulation
- **numpy:** Numerical operations

### Experiment Tracking
- **MLflow:** Experiment tracking and model registry
- **Weights & Biases:** Alternative tracking (if preferred)

---

## Questions for Stakeholders

Before proceeding with full implementation, please clarify:

1. **Test Data:** Is there a separate test set (20% holdout), or should we create our own validation strategy?

2. **Attack Type Labels:** Are individual samples labeled as "replay" vs. "rogue", or only "anomalous"?

3. **Operational Priorities:** 
   - Is minimizing false negatives (high recall) most critical?
   - Or is minimizing false positives (high precision) equally important?
   - What's the acceptable trade-off?

4. **Real-Time Requirements:** 
   - Does detection need to be real-time (<10ms latency)?
   - Or can it be near-real-time (batch processing every few seconds)?

5. **Deployment Environment:**
   - Cloud-based (ample compute resources)?
   - Edge device (resource-constrained)?
   - Embedded system (strict memory/CPU limits)?

6. **Feature Engineering:** 
   - Are we restricted to the 52 provided features?
   - Or can we create new derived features?

7. **Additional Data:** 
   - Are there more captures available for training?
   - Can we collect more data if needed?

8. **Experimental Metadata:** 
   - Do we have labels for center frequency, Tx gain, packet proportion?
   - This could inform stratified validation or conditional modeling

---

## Conclusion

This proposal outlines a systematic, rigorous approach to the Cyber-RF Anomaly Detector Challenge. By combining:
- Thorough data analysis
- Multiple modeling approaches
- Ensemble methods
- Robust validation
- Comprehensive interpretation

We expect to develop a high-performance anomaly detection system that reliably distinguishes between legitimate and malicious Zigbee transmissions. The modular approach allows for iteration and optimization based on intermediate results, while the focus on interpretability ensures the model's decisions are explainable and trustworthy for operational deployment.

**Recommended Next Steps:**
1. Clarify outstanding questions
2. Begin Phase 1 (Data Analysis)
3. Implement baseline models
4. Iterate based on results
