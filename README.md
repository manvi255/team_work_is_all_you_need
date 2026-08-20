# Ensemble Learning: Bagging vs Boosting

## Overview

This project implements **ensemble learning methods from scratch using NumPy**, using a small neural network as the base learner.

The two ensemble methods explored are:

- **Bagging**
- **Boosting**

The main goal was to understand how ensemble methods work, compare them with a single neural network, and determine **when ensemble methods help and when they are unnecessary**.

---

## 1. Base Neural Network

The base learner is a small neural network implemented from scratch using NumPy.

### Architecture

```text
8 Input Features
       ↓
8 ReLU Neurons
       ↓
1 Output Neuron
```

The neural network implements:

- Forward propagation
- ReLU activation
- Backpropagation
- Squared-error loss
- Gradient descent
- Linear output for regression

The network was intentionally kept small so that it could act as a **weak learner** for the ensemble experiments.

---

## 2. Bagging

### What is Bagging?

Bagging stands for **Bootstrap Aggregating**.

Multiple models are trained **independently** using different bootstrap samples of the training dataset.

```text
                Original Dataset
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
    Bootstrap 1   Bootstrap 2   Bootstrap 3
          ↓            ↓            ↓
       Model 1      Model 2      Model 3
          ↓            ↓            ↓
          └────────────┼────────────┘
                       ↓
                  Average
                       ↓
                Final Prediction
```

The important property is:

> **The models do not depend on each other during training.**

The main purpose of Bagging is to **reduce variance and make predictions more stable**.

---

## 3. Bagging Experiment

### Single Neural Network

```text
Accuracy  = 0.8173
Precision = 0.6763
Recall    = 0.3338
F1        = 0.4470
```

### Bagging

```text
Accuracy  = 0.8172
Precision = 0.6657
Recall    = 0.3482
F1        = 0.4572
```

### Comparison

| Metric | Single NN | Bagging |
|---|---:|---:|
| Accuracy | 0.8173 | 0.8172 |
| Precision | 0.6763 | 0.6657 |
| Recall | 0.3338 | **0.3482** |
| F1 | 0.4470 | **0.4572** |

The F1-score increased only slightly:

```text
0.4470 → 0.4572
```

### Bagging Conclusion

Bagging **did not provide a major improvement** in this experiment.

This demonstrates an important point:

> Ensemble methods do not automatically improve performance.

Bagging is most useful when the base learner has **high variance** and is sensitive to changes in the training data.

---

## 4. Boosting

### What is Boosting?

Unlike Bagging, Boosting trains models **sequentially**.

Each new model attempts to learn the errors made by the current ensemble.

```text
Initial Prediction
       ↓
Calculate Residual
       ↓
Model 1 learns Residual
       ↓
Update Prediction
       ↓
Calculate New Residual
       ↓
Model 2 learns New Residual
       ↓
Update Prediction
       ↓
...
       ↓
Model 15
       ↓
Final Prediction
```

The residual was calculated as:

```python
residual = Y - predictions
```

Therefore, every new model focuses on:

> **What is the current ensemble still getting wrong?**

---

## 5. Important Boosting Implementation Change

Initially, the same learning rate was used for:

1. Neural-network training
2. Boosting corrections

This caused the boosting correction to be extremely small.

The implementation was changed to use **two separate learning rates**:

```text
Neural Network Learning Rate = 0.0001
Boosting Learning Rate        = 0.1
```

### Neural Network Learning Rate

Controls how quickly the individual neural network updates its weights.

### Boosting Learning Rate

Controls how strongly the newly trained model's correction contributes to the ensemble.

This separation was important for getting Boosting to work effectively.

---

## 6. Boosting Experiment

The final experiment used **15 sequential neural networks**.

The residual MSE decreased as follows:

| Round | Residual MSE |
|---:|---:|
| 1 | 1.0000 |
| 2 | 0.8716 |
| 3 | 0.7677 |
| 4 | 0.6838 |
| 5 | 0.6159 |
| 6 | 0.5611 |
| 7 | 0.5169 |
| 8 | 0.4811 |
| 9 | 0.4522 |
| 10 | 0.4289 |
| 11 | 0.4102 |
| 12 | 0.3949 |
| 13 | 0.3827 |
| 14 | 0.3727 |
| 15 | **0.3647** |

Residual MSE:

```text
1.0000 → 0.3647
```

This is approximately a **63.5% reduction** in residual MSE.

This provides direct evidence that the successive models were progressively reducing the remaining error.

---

## 7. Boosting Results

### Single Neural Network

```text
MAE  = 2.4099
RMSE = 2.5060
R²   = -3.7926
```

### 15-Model Boosting

```text
MAE  = 0.5125
RMSE = 0.7005
R²   = 0.6255
```

### Comparison

| Metric | Single NN | Boosting |
|---|---:|---:|
| MAE ↓ | 2.4099 | **0.5125** |
| RMSE ↓ | 2.5060 | **0.7005** |
| R² ↑ | -3.7926 | **0.6255** |

Boosting reduced:

- **MAE by ~78.7%**
- **RMSE by ~72.0%**

R² improved dramatically:

```text
-3.7926 → 0.6255
```

---

## 8. Why Did Boosting Work Better?

The base neural network was intentionally small:

```text
8 → 8 → 1
```

A single network had to learn the entire problem at once.

Boosting divided the problem into smaller correction tasks.

```text
Model 1
   ↓
First errors
   ↓
Model 2
   ↓
Remaining errors
   ↓
Model 3
   ↓
Remaining errors
   ↓
...
   ↓
Model 15
```

Each model does not need to solve the entire problem.

Instead, it asks:

> **"What is still wrong with the current prediction?"**

This made Boosting much more effective for this weak base learner.

---

## 9. Bagging vs Boosting

| Property | Bagging | Boosting |
|---|---|---|
| Training | Parallel / independent | Sequential |
| Data | Bootstrap samples | Residual/errors |
| Main goal | Reduce variance | Sequentially reduce errors |
| Models depend on each other? | No | Yes |
| Final combination | Average/vote | Weighted corrections |
| Best suited for | High-variance learners | Weak/high-bias learners |
| Experiment result | Small improvement | Large improvement |

### Bagging

```text
Model 1 ─┐
Model 2 ─┤
Model 3 ─┼──→ Average → Final prediction
Model 4 ─┤
Model 5 ─┘
```

### Boosting

```text
Model 1
   ↓
Residual
   ↓
Model 2
   ↓
Residual
   ↓
Model 3
   ↓
...
```

---

## 10. When Should You Use Bagging?

Use Bagging when:

- The base model has **high variance**.
- The model is sensitive to the training dataset.
- The model tends to overfit.
- You want more stable predictions.
- Multiple models can be trained independently.
- Reducing variance is the main objective.

---

## 11. When Should You Use Boosting?

Use Boosting when:

- The base learner is relatively weak.
- The learner has high bias.
- The problem contains complex patterns.
- Successive models can learn from previous errors.
- You want to progressively improve prediction accuracy.

This experiment is a good example:

```text
Weak Single NN
      ↓
Boosting
      ↓
Much stronger ensemble
```

---

## 12. When Should You NOT Use Ensemble Methods?

Ensemble learning is **not automatically the best choice**.

### Use a single model instead when:

#### 1. The single model is already good enough

If:

```text
Single model → Meets requirements
```

there may be little reason to add many models.

#### 2. Computational resources are limited

An ensemble may require:

```text
1 model
  ↓
15 / 50 / 100 models
```

This increases:

- Training time
- Memory usage
- Inference time

#### 3. Interpretability is important

A single model is generally easier to understand than an ensemble containing many models.

#### 4. The ensemble provides little improvement

This was demonstrated by the Bagging experiment:

```text
F1:
Single NN → 0.4470
Bagging   → 0.4572
```

The improvement is small.

If the additional complexity is significant, Bagging may not be worth it.

#### 5. Boosting begins to overfit

Adding more Boosting models does not always improve generalization.

```text
More models ≠ Always better
```

Validation performance should be monitored.

---

## 13. Decision Guide

```text
                 Start
                   │
                   ▼
       Is a single model good enough?
              /                        YES             NO
             │               │
             ▼               ▼
      Use single model   Is variance high?
                            /                                 YES        NO
                           │          │
                           ▼          ▼
                        BAGGING    Is learner weak
                                   / high bias?
                                      /                                          YES       NO
                                     │         │
                                     ▼         ▼
                                  BOOSTING   Compare methods
```

---

## 14. Key Lessons

### Lesson 1 — Ensembles are not automatically better

Bagging only produced a small improvement.

### Lesson 2 — Choose the ensemble based on the problem

```text
High variance → Bagging

Weak/high-bias learner → Boosting
```

### Lesson 3 — Boosting is sequential

Each model learns from the errors remaining after previous models.

### Lesson 4 — Hyperparameters matter

Separating the:

```text
NN learning rate
```

from the:

```text
Boosting learning rate
```

was essential to making the Boosting implementation effective.

### Lesson 5 — More models have diminishing returns

The residual MSE decreased rapidly initially and then more slowly:

```text
1.0000
 ↓
0.8716
 ↓
0.7677
 ↓
...
 ↓
0.3827
 ↓
0.3727
 ↓
0.3647
```

Additional models should therefore be selected using validation performance rather than simply maximizing the number of models.

---

## 15. Final Conclusion

This experiment demonstrates that **ensemble methods should be selected according to the weaknesses of the base learner**.

Bagging mainly helps by reducing variance. In this experiment, it produced only a small improvement, showing that Bagging is not useful for every problem.

Boosting performed substantially better with the weak neural network. The residual MSE decreased by approximately **63.5%** across 15 boosting rounds, while the final ensemble achieved:

```text
MAE  = 0.5125
RMSE = 0.7005
R²   = 0.6255
```

compared with:

```text
Single NN R² = -3.7926
```

### Main Takeaway

> **Use Bagging when variance and instability are the main problems. Use Boosting when weak learners need sequential error correction. Do not use an ensemble simply because it is an ensemble—if a single model is already sufficient or the ensemble provides only marginal improvement relative to its additional complexity, a single model is often the better choice.**
