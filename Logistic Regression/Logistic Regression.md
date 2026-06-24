---

title: Logistic Regression

tags:

- MachineLearning
    
- SupervisedLearning
    
- Classification
    
- LogisticRegression
    

aliases:

- Logistic Regression
    
- Logit Regression
    
- Maximum Entropy Classifier
    

created: 2026-06-24  
updated: 2026-06-24

status: Complete  
difficulty: Intermediate  
category: Classification Algorithm

prerequisites:

- Linear Regression
    
- Probability
    
- Calculus
    
- Gradient Descent
    
- Binary Cross Entropy
    

related:

- Sigmoid Function
    
- Binary Cross Entropy
    
- Softmax Regression
    
- Maximum Likelihood Estimation
    

---

# Logistic Regression

> [!quote]  
> **"Logistic Regression is a supervised machine learning algorithm used for classification problems that predicts probabilities using the Sigmoid function."**

---

# 1. Introduction

Logistic Regression is a **Supervised Machine Learning Classification Algorithm** used to predict **categorical outcomes**.

Unlike Linear Regression:

```text
Linear Regression → Continuous Values

Logistic Regression → Categorical Values
```

Examples:

```text
Spam Detection

Fraud Detection

Disease Prediction

Customer Churn Prediction

Pass / Fail Prediction
```

---

# Why Not Use Linear Regression for Classification?

Suppose:

```text
Pass → 1

Fail → 0
```

Linear Regression may predict:

```text
1.8

-0.5

2.7
```

These predictions are invalid because probabilities must lie between:

$$  
0 \le P(y) \le 1  
$$

Hence Logistic Regression is used.

---

# 2. Mathematical Intuition

Logistic Regression predicts:

```text
Probability of belonging to a class
```

For Binary Classification:

$$  
P(Y=1|X)  
$$

Examples:

```text
Probability of Spam

Probability of Disease

Probability of Fraud
```

The predicted probability is then converted into a class label.

---

# Decision Rule

Usually:

```text
If Probability ≥ 0.5 → Class 1

If Probability < 0.5 → Class 0
```

---

# 3. Sigmoid Function

The core of Logistic Regression is the:

# Sigmoid Function

Formula:

$$  
\sigma(z)=\frac{1}{1+e^{-z}}  
$$

Where:

$$  
z=\beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_nx_n  
$$

---

# Properties of Sigmoid

```text
Input Range  → (-∞, +∞)

Output Range → (0,1)
```

---

## Graph

```text
Probability
1 |                         ●
  |                     ●
  |                 ●
0.5|-------------●-------------
  |         ●
  |     ●
0 |●____________________________
         z
```

---

# Why Sigmoid?

Because:

```text
Any Real Number

↓

Probability between 0 and 1
```

---

# 4. Mathematical Formulation

Linear Combination:

$$  
z=\beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_nx_n  
$$

Apply Sigmoid:

$$  
P(Y=1|X)=\frac{1}{1+e^{-z}}  
$$

Complete Equation:

# $$  
P(Y=1|X)

\frac{1}  
{1+e^{-(\beta_0+\beta_1x_1+\cdots+\beta_nx_n)}}  
$$

---

# 5. Odds and Log Odds

Probability:

$$  
P  
$$

Odds:

$$  
Odds=\frac{P}{1-P}  
$$

Examples:

```text
P = 0.8

Odds = 0.8/0.2 = 4
```

Meaning:

```text
Event is 4 times more likely to occur.
```

---

Taking logarithm:

$$  
\log\left(\frac{P}{1-P}\right)  
$$

This is called:

# Logit Function

---

# Final Logistic Equation

# $$  
\log\left(\frac{P}{1-P}\right)

\beta_0+\beta_1x_1+\cdots+\beta_nx_n  
$$

---

# 6. Assumptions

## 1. Binary Dependent Variable

Target should be:

```text
0 or 1
```

---

## 2. Independent Observations

Observations should be independent.

---

## 3. Little or No Multicollinearity

Features should not be highly correlated.

---

## 4. Linear Relationship Between Features and Log Odds

Important:

```text
Features should be linearly related

to

Log Odds
```

Not necessarily with target itself.

---

## 5. Large Sample Size

Logistic Regression performs better with larger datasets.

---

# 7. Working Principle

```text
Input Features
        ↓
Linear Combination
        ↓
Compute z
        ↓
Apply Sigmoid
        ↓
Generate Probability
        ↓
Apply Threshold
        ↓
Predict Class
```

---

# Example

Suppose:

$$  
z=2  
$$

Probability:

$$  
P=\frac1{1+e^{-2}}  
$$

$$  
P=0.88  
$$

Threshold:

```text
0.88 > 0.5
```

Prediction:

```text
Class = 1
```

---

# 8. Mathematical Derivation

Linear Regression Cost:

```text
MSE
```

cannot be used because the Sigmoid function makes the cost non-convex.

Hence:

# Maximum Likelihood Estimation (MLE)

is used.

---

## Likelihood Function

For one sample:

$$  
P(y|x)=  
p^y(1-p)^{1-y}  
$$

For N samples:

# $$  
L(\beta)

\prod_{i=1}^{N}  
p_i^{y_i}(1-p_i)^{1-y_i}  
$$

Take logarithm:

# $$  
\log L(\beta)

\sum_{i=1}^{N}  
[y_i\log(p_i)+(1-y_i)\log(1-p_i)]  
$$

Negative Log Likelihood:

# $$  
J(\beta)

-\sum_{i=1}^{N}  
[y_i\log(p_i)+(1-y_i)\log(1-p_i)]  
$$

This is:

# Binary Cross Entropy Loss

---

# 9. Cost Function

Binary Cross Entropy:

# $$  
J(\beta)

-\frac1N  
\sum_{i=1}^{N}  
\left[  
y_i\log(\hat y_i)  
+  
(1-y_i)\log(1-\hat y_i)  
\right]  
$$

Goal:

$$  
\min J(\beta)  
$$

---

# 10. Optimization

Parameters are optimized using:

```text
Gradient Descent

Stochastic Gradient Descent

Newton's Method

LBFGS

SAG

SAGA
```

---

Gradient:

# $$  
\frac{\partial J}{\partial \beta}

X^T(\hat y-y)  
$$

Update Rule:

# $$  
\beta

\beta-\alpha\nabla J  
$$

where:

$$  
\alpha  
$$

is the learning rate.

---

# 11. Parameters

Parameters learned from data:

```text
β₀ → Intercept

β₁,β₂,...βₙ → Feature Weights
```

Interpretation:

Positive coefficient:

```text
Increases probability
```

Negative coefficient:

```text
Decreases probability
```

---

# 12. Hyperparameter Tuning

Important hyperparameters:

## Penalty

```python
penalty = 'l1'
penalty = 'l2'
penalty = 'elasticnet'
```

---

## Regularization Strength

```python
C = 0.01
C = 1
C = 100
```

Smaller C:

```text
Stronger Regularization
```

---

## Solver

```python
solver='liblinear'
solver='lbfgs'
solver='saga'
solver='newton-cg'
```

---

## Maximum Iterations

```python
max_iter = 100
```

---

# Hyperparameter Tuning Example

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

params = {
    "C":[0.01,0.1,1,10],
    "penalty":["l1","l2"],
    "solver":["liblinear"]
}

grid = GridSearchCV(
    LogisticRegression(),
    params,
    cv=5
)

grid.fit(X_train,y_train)

print(grid.best_params_)
```

---

# 13. Decision Boundary

Logistic Regression creates:

```text
Linear Decision Boundary
```

Equation:

$$  
\beta_0+\beta_1x_1+\cdots+\beta_nx_n=0  
$$

---

# 14. Advantages

## 1. Easy to Implement

---

## 2. Probabilistic Output

---

## 3. Fast Training

---

## 4. Highly Interpretable

---

## 5. Works Well for Linearly Separable Data

---

## 6. Less Prone to Overfitting

with Regularization.

---

# 15. Disadvantages

## 1. Assumes Linear Decision Boundary

---

## 2. Cannot Capture Complex Relationships

---

## 3. Sensitive to Outliers

---

## 4. Performance Degrades with Highly Non-Linear Data

---

# 16. Evaluation Metrics

Common metrics:

```text
Accuracy

Precision

Recall

F1 Score

ROC-AUC

Confusion Matrix
```

---

# 17. Code Implementation

## Scikit-Learn Implementation

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, y_pred))
```

---

## Probability Prediction

```python
probabilities = model.predict_proba(X_test)

print(probabilities)
```

---

# 18. Real World Applications

```text
Spam Detection

Fraud Detection

Credit Risk Analysis

Medical Diagnosis

Customer Churn Prediction

Sentiment Analysis
```

---

# 19. Interview Questions

## Basic

1. What is Logistic Regression?
    
2. Why can't we use Linear Regression for classification?
    
3. What is the Sigmoid Function?
    

---

## Intermediate

1. Explain Odds and Log Odds.
    
2. Why is BCE used?
    
3. What assumptions does Logistic Regression make?
    

---

## Advanced

1. Derive the Logistic Regression cost function.
    
2. Explain Maximum Likelihood Estimation.
    
3. Explain regularization in Logistic Regression.
    

---

# 20. Summary

> [!summary]
> 
> - Logistic Regression is a classification algorithm.
>     
> - It predicts probabilities using the Sigmoid function.
>     
> - It uses Binary Cross Entropy as the loss function.
>     
> - Parameters are estimated using Maximum Likelihood Estimation.
>     
> - It is highly interpretable and computationally efficient.
>     

---

# Mind Map

```text
Logistic Regression
│
├── Supervised Learning
├── Classification
├── Sigmoid Function
├── Probability Prediction
├── Binary Cross Entropy
├── Maximum Likelihood
├── Gradient Descent
└── Decision Boundary
```